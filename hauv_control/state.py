"""Matched pursuit-and-dive HAUV state alternatives based only on common current estimates.

Physical observations have 32 values. The first 29 and previous four commands
are common. Only the final representation of estimated target position differs.
Predictors use own measured surface clearance as a locally flat surface and
constant estimated target velocity. Future target paths and waves are absent.
"""
import numpy as np
from numba import njit
from .direction import compact_direction
from .dynamics import step, rotation, angles, AIR_OMEGA_MAX, WATER_OMEGA_MAX

DIMENSIONS={'coordinates':36,'compact_dce':45,'motion_preview':39,'action_contrast':41}
DESIRED=np.array([1.2,0.,-1.0])
OFFSETS=np.vstack([np.zeros(4),np.stack([.5*np.eye(4),-.5*np.eye(4)],axis=1).reshape(8,4)])


@njit(cache=True)
def _heading(yaw):
    c,s=np.cos(yaw),np.sin(yaw)
    return np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])


@njit(cache=True)
def reconstruct(p):
    yaw=np.arctan2(p[10],p[11]); roll=np.arctan2(p[13],p[14]); pitch=np.arctan2(p[15],p[16])
    cy,sy=np.cos(yaw/2),np.sin(yaw/2)
    cp,sp=np.cos(pitch/2),np.sin(pitch/2)
    cr,sr=np.cos(roll/2),np.sin(roll/2)
    s=np.zeros(21);s[2]=2*p[28]
    s[6:10]=np.array([cr*cp*cy+sr*sp*sy,sr*cp*cy-cr*sp*sy,
                      cr*sp*cy+sr*cp*sy,cr*cp*sy-sr*sp*cy])
    h=_heading(yaw)
    s[3:6]=rotation(s[6:10]).T@(h@(3*p[7:10]))
    s[10:13]=2*p[17:20]
    s[13:17]=p[20:24]*AIR_OMEGA_MAX;s[17:20]=p[24:27]*WATER_OMEGA_MAX
    s[20]=p[27]
    target=s[:3]+h@(20*p[29:32]); velocity=h@(3*(p[1:4]+p[7:10]))
    return s,target,velocity


@njit(cache=True)
def forecast(p,action,horizon):
    state,target,vt=reconstruct(p)
    errors=np.empty((horizon,3));costs=np.empty(horizon)
    for k in range(horizon):
        # A 20 ms prediction step is separate from the 2 ms executed plant.
        # Its fixed-command development convergence audit is archived.
        state=step(state,action,0.,dt=.1,amplitude=0.,substeps=5)
        target+=.1*vt
        r=_heading(angles(state[6:10])[2]).T@(target-state[:3])
        e=r-DESIRED;dv=vt-rotation(state[6:10])@state[3:6]
        errors[k]=e
        costs[k]=.5*np.sum(e*e)+.1*np.sum(dv*dv)+.1*(1-np.cos(np.arctan2(r[1],r[0])))
    return errors,costs


@njit(cache=True)
def predictive_features(p,previous,contrast):
    if not contrast:
        errors,_=forecast(p,previous,15)
        selected=np.empty(6);selected[:3]=errors[4];selected[3:]=errors[14]
        return selected/(5.+np.abs(selected))
    costs=np.zeros(9)
    for i in range(9):
        candidate=np.clip(previous+OFFSETS[i],-1.,1.)
        _,cost=forecast(p,candidate,10)
        costs[i]=cost.mean()+.005*np.sum(candidate*candidate)+.002*np.sum((candidate-previous)**2)
    differences=costs[1:]-costs[0]
    return differences/(1.+np.abs(differences))


def encode_state(physical, previous, representation):
    p=np.asarray(physical,dtype=np.float64);old=np.asarray(previous,dtype=np.float64)
    if p.shape!=(32,) or old.shape!=(4,):raise ValueError('Expected physical(32) and previous action(4)')
    common=np.r_[p[:29],old]
    if representation=='coordinates':features=p[29:32]
    elif representation=='compact_dce':features=compact_direction(20*p[29:32],p[6] if p[5]<1 else 0.)
    elif representation in ('motion_preview','action_contrast'):
        features=predictive_features(p,old,representation=='action_contrast') if p[0]>0 else np.zeros(DIMENSIONS[representation]-33)
    else:raise ValueError(representation)
    result=np.r_[common,features].astype(np.float32)
    if not np.isfinite(result).all():raise FloatingPointError('Nonfinite state encoding')
    return result
