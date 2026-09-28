"""Load the released actors and preserve the study's exact state/action interface."""
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .state import DIMENSIONS, encode_state

ROOT = Path(__file__).resolve().parents[1]
WIDTHS = {'coordinates': (128, 128), 'compact_dce': (120, 130), 'action_contrast': (125, 127)}


class Actor(nn.Module):
    def __init__(self, representation):
        super().__init__()
        first, second = WIDTHS[representation]
        self.net = nn.Sequential(nn.Linear(DIMENSIONS[representation], first), nn.ReLU(),
                                 nn.Linear(first, second), nn.ReLU(), nn.Linear(second, 4), nn.Tanh())

    def forward(self, state):
        return self.net(state)


def load_actor(representation='action_contrast', seed=60, root=ROOT):
    """Return a CPU inference actor and its release metadata."""
    root = Path(root)
    manifest = json.loads((root/'weights/manifest.json').read_text())
    records = [r for r in manifest['models'] if r['representation'] == representation and r['seed'] == seed]
    if len(records) != 1:
        raise ValueError('Choose coordinates, compact_dce or action_contrast and seed 60, 61 or 62')
    info = records[0]
    actor = Actor(representation)
    actor.load_state_dict(torch.load(root/info['file'], map_location='cpu', weights_only=True), strict=True)
    return actor.eval(), info


@torch.inference_mode()
def act(actor, physical, previous_action, representation):
    """32 measured/estimated fields plus four previous commands -> four commands."""
    encoded = encode_state(physical, previous_action, representation)
    return actor(torch.from_numpy(encoded).unsqueeze(0))[0].numpy()


def command_units(action):
    """Yaw-frame requested vx, vy, vz [m/s] and yaw rate [rad/s]."""
    action = np.asarray(action, dtype=float)
    if action.shape != (4,) or not np.isfinite(action).all():
        raise ValueError('Expected four finite normalized commands')
    return action * np.array([1.2, 1.2, .5, .6])
