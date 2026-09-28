"""Six-DOF Nezha-Mini development plant with published prototype parameters.

Table II and equations 4-9 supply mass, added mass, inertia, quadratic damping,
restoring forces and allocation. Figure 5 supplies separately archived curve
digitization. Surface interpolation and low-level gains are simulation design
choices, not a reproduction of the paper's controller or new hardware fits.

State: world position, body linear velocity, wxyz quaternion, body angular
velocity, seven actual motor speeds (rad/s), and discrete propulsion mode.
Mode is 0=air and 1=water. High-level actions request yaw-frame velocity and yaw rate.
Water sway requests are realized by course change, never lateral thrust.
"""
import json
from pathlib import Path
import numpy as np
from numba import njit

ROOT=Path(__file__).resolve().parents[1]
PARAMETERS=json.loads((ROOT/'configs/nezha_mini_published_v1.json').read_text())
DIGITIZATION=json.loads((ROOT/'configs/figure5_digitization.json').read_text())['fits']
MASS=PARAMETERS['mass_kg']
GRAVITY=PARAMETERS['gravity_m_s2']
BUOYANCY=PARAMETERS['volume_m3']*PARAMETERS['water_density_kg_m3']*GRAVITY
INERTIA=np.array(PARAMETERS['inertia_diagonal_kg_m2'])
ADDED=np.array(PARAMETERS['added_mass_positive_kg'])
ADDED_I=np.array(PARAMETERS['added_inertia_positive_kg_m2'])
DRAG=np.array(PARAMETERS['quadratic_translation_drag'])
ANGULAR_DRAG=np.array(PARAMETERS['quadratic_rotation_drag'])
POSITIONS=np.array(PARAMETERS['marine_thruster_positions_m'])
AIR_K=DIGITIZATION['air_thrust_g_all']['coefficient_rad_s']
AIR_D=DIGITIZATION['air_power_W_all']['coefficient_rad_s']/AIR_K
WATER_K_POS=DIGITIZATION['water_thrust_g_forward']['coefficient_rad_s']
WATER_K_NEG=DIGITIZATION['water_thrust_g_reverse']['coefficient_rad_s']
WATER_Q_POS=DIGITIZATION['water_power_W_forward']['coefficient_rad_s']
WATER_Q_NEG=DIGITIZATION['water_power_W_reverse']['coefficient_rad_s']
ARM=PARAMETERS['air_arm_length_m']/np.sqrt(2)
AIR_ALLOCATION=np.array([[1.,1.,1.,1.],[-ARM,ARM,ARM,-ARM],
                         [-ARM,-ARM,ARM,ARM],[-AIR_D,AIR_D,-AIR_D,AIR_D]])
AIR_INVERSE=np.linalg.inv(AIR_ALLOCATION)
AIR_OMEGA_MAX=18000.*2*np.pi/60
WATER_OMEGA_MIN=-400.*2*np.pi/60
WATER_OMEGA_MAX=500.*2*np.pi/60


@njit(cache=True)
def clamp(value, lower, upper):
    return min(max(value,lower),upper)


@njit(cache=True)
def rotation(q):
    w,x,y,z=q
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


@njit(cache=True)
def angles(q):
    w,x,y,z=q
    return np.array([np.arctan2(2*(w*x+y*z),1-2*(x*x+y*y)),
                     np.arcsin(clamp(2*(w*y-z*x),-1.,1.)),
                     np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))])


@njit(cache=True)
def wave_height(position,time,amplitude,phase):
    x,y=position[:2]
    return amplitude*(np.sin(.8*x+.4*y-1.1*time+phase)+.45*np.sin(-.4*x+1.2*y-1.5*time+2*phase))


@njit(cache=True)
def immersion(position,time,amplitude,phase):
    # Explicit smooth partial-volume approximation over the 91 mm body height.
    return clamp((wave_height(position,time,amplitude,phase)-position[2]+.0455)/.091,0.,1.)


@njit(cache=True)
def motor_wrench(state,time,amplitude,phase):
    r=rotation(state[6:10]); wet=immersion(state[:3],time,amplitude,phase)
    airborne=state[13:17]
    # Air propeller authority fades across a 30 mm interface band. The rotor
    # plane is approximated 25 mm above CG. This is an explicit model assumption.
    rotor_z=state[2]+.025*r[2,2]
    dry=clamp((rotor_z-wave_height(state[:3],time,amplitude,phase)+.015)/.03,0.,1.)
    air=AIR_ALLOCATION@(AIR_K*airborne*airborne*dry)
    force=np.array([0.,0.,air[0]])
    torque=air[1:].copy()
    for i in range(3):
        omega=state[17+i]
        k=WATER_K_POS if omega>=0 else WATER_K_NEG
        position=state[:3]+r@POSITIONS[i]
        contact=clamp((wave_height(position,time,amplitude,phase)-position[2]+.015)/.03,0.,1.)
        thrust=k*omega*abs(omega)*contact
        if i==0:
            thrust*=1.078*np.exp(-1.654*max(0.,state[3]))
            f=np.array([0.,0.,thrust])
            # Literal signed-power curve Q/omega from Eq. 6; zero at rest.
            qk=WATER_Q_POS if omega>=0 else WATER_Q_NEG
            torque[2]+=qk*omega*omega*contact
        else:
            f=np.array([thrust,0.,0.])
        force+=f
        torque+=np.cross(POSITIONS[i],f)
    return force,torque,wet


@njit(cache=True)
def controller(state,action,time,amplitude,phase):
    r=rotation(state[6:10]); euler=angles(state[6:10])
    velocity=state[3:6]; omega=state[10:13]
    water=state[20]>.5
    reference=np.array([1.2*action[0],1.2*action[1],.5*action[2]])
    yaw_rate=.6*action[3]
    command=np.zeros(7)
    if not water:
        c,s=np.cos(euler[2]),np.sin(euler[2])
        heading=np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
        acceleration=1.8*(reference-heading.T@(r@velocity))
        desired_roll=clamp(-acceleration[1]/GRAVITY,-.35,.35)
        desired_pitch=clamp(acceleration[0]/GRAVITY,-.35,.35)
        # Compensate gravity and partial buoyancy with measured attitude.
        wet=immersion(state[:3],time,amplitude,phase)
        rotor_z=state[2]+.025*r[2,2]
        dry=clamp((rotor_z-wave_height(state[:3],time,amplitude,phase)+.015)/.03,0.,1.)
        # During egress retain a bounded bow assist until aerial authority is
        # available. This overlap controller is OUR declared simulation design,
        # not a claim to reproduce the paper's unreported controller gains.
        assist=0.
        if wet>.05 and reference[2]>.03:
            assist=min(2.,2.*reference[2])*wet*(1.-dry)
            crossflow=1.078*np.exp(-1.654*max(0.,velocity[0]))
            command[4]=np.sqrt(assist/(WATER_K_POS*max(.1,crossflow)))
        vertical=max(0.,MASS*(GRAVITY+acceleration[2])-wet*BUOYANCY-assist*r[2,2])
        total=vertical/(max(.5,r[2,2])*max(.05,dry))
        desired=np.array([total,.25*(desired_roll-euler[0])-.075*omega[0],
                          .35*(desired_pitch-euler[1])-.1*omega[1],
                          .055*(yaw_rate-omega[2])])
        if wet>.05:
            restoring=np.cross(np.array([0.,0.,.0185]),r.T@np.array([0.,0.,wet*BUOYANCY]))
            desired[1:]-=restoring+np.cross(POSITIONS[0],np.array([0.,0.,assist]))
            desired[3]-=WATER_Q_POS*command[4]**2
            desired[1:]/=max(.05,dry)
        thrust=AIR_INVERSE@desired
        for i in range(4):command[i]=min(AIR_OMEGA_MAX,np.sqrt(max(0.,thrust[i])/AIR_K))
    else:
        # Forward-only pursuit, as in the published submerged operating mode.
        surge=clamp(reference[0],.12,.7)
        desired_pitch=clamp(np.arctan2(-reference[2],max(surge,.2)),-.43633,.43633)
        yaw_rate=clamp(yaw_rate+.7*np.arctan2(reference[1],max(surge,.2)),-.7,.7)
        # Near-surface ascent brings pitch toward zero to permit attitude-based
        # exit triggering. The gate uses own depth, never target future truth.
        clearance=state[2]-wave_height(state[:3],time,amplitude,phase)
        if reference[2]>.03 and clearance>-.175:
            desired_pitch*=clamp((-clearance-.025)/.15,0.,1.)
        wet=immersion(state[:3],time,amplitude,phase)
        restoring=np.cross(np.array([0.,0.,.0185]),r.T@np.array([0.,0.,BUOYANCY*wet]))
        fx=DRAG[0]*surge*abs(surge)+(MASS+ADDED[0])*1.5*(surge-velocity[0])
        fx=clamp(fx,-3.,8.)
        pitch_torque=.5*(desired_pitch-euler[1])-.16*omega[1]
        front=-(pitch_torque-restoring[1]+.01*fx)/.1403
        crossflow=1.078*np.exp(-1.654*max(0.,velocity[0]))
        static_front=front/max(.1,crossflow)
        k=WATER_K_POS if static_front>=0 else WATER_K_NEG
        bow_omega=np.sign(static_front)*np.sqrt(abs(static_front)/k)
        reaction=(WATER_Q_POS if bow_omega>=0 else WATER_Q_NEG)*bow_omega**2
        yaw_torque=.16*(yaw_rate-omega[2])
        difference=(yaw_torque-reaction)/.0545
        forces=np.array([static_front,.5*(fx-difference),.5*(fx+difference)])
        for i in range(3):
            k=WATER_K_POS if forces[i]>=0 else WATER_K_NEG
            command[4+i]=clamp(np.sign(forces[i])*np.sqrt(abs(forces[i])/k),WATER_OMEGA_MIN,WATER_OMEGA_MAX)
    return command


@njit(cache=True)
def derivative(state,time,amplitude,phase,inertia_scale=1.,drag_scale=1.):
    r=rotation(state[6:10]); velocity=state[3:6]; omega=state[10:13]
    force,torque,wet=motor_wrench(state,time,amplitude,phase)
    mass=MASS+wet*ADDED
    inertia=inertia_scale*INERTIA+wet*ADDED_I
    buoyancy=np.array([0.,0.,BUOYANCY*wet])
    force+=r.T@(buoyancy-np.array([0.,0.,MASS*GRAVITY]))
    force-=wet*drag_scale*DRAG*velocity*np.abs(velocity)
    torque+=np.cross(np.array([0.,0.,.0185]),r.T@buoyancy)
    torque-=wet*drag_scale*ANGULAR_DRAG*omega*np.abs(omega)
    linear=(force-np.cross(omega,mass*velocity))/mass
    angular=(torque-np.cross(omega,inertia*omega)+np.cross(wet*ADDED*velocity,velocity))/inertia
    return r@velocity,linear,angular


@njit(cache=True)
def step(state,action,time,dt=.1,amplitude=.02,phase=0.,substeps=50,inertia_scale=1.,drag_scale=1.):
    result=state.copy(); h=dt/substeps
    for substep in range(substeps):
        t=time+substep*h
        clearance=result[2]-wave_height(result[:3],t,amplitude,phase)
        attitude=angles(result[6:10])
        if result[20]<.5 and clearance<-.04 and action[2]<-.06:
            result[20]=1.
        elif (result[20]>.5 and clearance>-.03 and action[2]>.06 and
              abs(attitude[0])<.22 and abs(attitude[1])<.22 and
              np.linalg.norm(result[10:12])<.45):
            result[20]=0.
        command=controller(result,action,t,amplitude,phase)
        for i in range(7):
            # Local motor-response constants, explicitly distinct from Table II.
            tau=(.0125 if command[i]>result[13+i] else .025) if i<4 else .05
            result[13+i]=command[i]+(result[13+i]-command[i])*np.exp(-h/tau)
        position_dot,velocity_dot,omega_dot=derivative(result,t,amplitude,phase,inertia_scale,drag_scale)
        result[:3]+=h*position_dot
        result[3:6]+=h*velocity_dot
        result[10:13]+=h*omega_dot
        q=result[6:10].copy(); w=result[10:13]
        dq=np.array([-q[1]*w[0]-q[2]*w[1]-q[3]*w[2],
                     q[0]*w[0]+q[2]*w[2]-q[3]*w[1],
                     q[0]*w[1]+q[3]*w[0]-q[1]*w[2],
                     q[0]*w[2]+q[1]*w[1]-q[2]*w[0]])
        result[6:10]+=h*.5*dq
        result[6:10]/=np.linalg.norm(result[6:10])
    return result


def initial_state(position=(-1.,0.,.8),yaw=0.,speed=.25):
    state=np.zeros(21)
    state[:3]=position;state[3]=speed
    state[6]=np.cos(yaw/2);state[9]=np.sin(yaw/2)
    state[13:17]=np.sqrt(MASS*GRAVITY/(4*AIR_K))
    return state
