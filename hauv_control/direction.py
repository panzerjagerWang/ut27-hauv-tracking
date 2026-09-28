"""Analytic four-face encoding: 12 directional values plus common state/history."""
import numpy as np
FACES = {
    "front": ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
    "down": ((0, 0, -1), (0, 1, 0), (1, 0, 0)),
    "left": ((0, 1, 0), (-1, 0, 0), (0, 0, 1)),
    "right": ((0, -1, 0), (1, 0, 0), (0, 0, 1)),
}



BASIS = np.array([FACES[name] for name in ('front','down','left','right')], dtype=float)


def compact_direction(relative, amplitude):
    r = np.asarray(relative, dtype=float)
    code = np.zeros((4,3), dtype=np.float32)
    distance = np.linalg.norm(r)
    if distance < 1e-8 or amplitude <= 0:
        return code.ravel()
    components = BASIS@r
    positive=components[:,0]>0
    valid=components[positive]
    mass=np.clip(amplitude,0,1)*valid[:,0]/distance
    angles=np.arctan2(valid[:,1:],valid[:,:1])/(np.pi/2)
    code[positive]=np.column_stack((mass,mass[:,None]*angles))
    return code.ravel()

