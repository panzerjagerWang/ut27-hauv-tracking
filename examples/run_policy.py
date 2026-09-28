"""CPU inference on a measured/estimated observation retained from a recorded trial."""
import argparse
import json
from pathlib import Path

import torch

from hauv_control.policy import load_actor, act, command_units


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--representation', choices=['coordinates', 'compact_dce', 'action_contrast'], default='action_contrast')
    parser.add_argument('--seed', type=int, choices=[60, 61, 62], default=60)
    parser.add_argument('--sample', type=int, default=3, help='Index in examples/recorded_observations.json')
    args = parser.parse_args()
    torch.set_num_threads(1)
    samples = json.loads((Path(__file__).parent/'recorded_observations.json').read_text())
    sample = samples[args.sample]
    actor, info = load_actor(args.representation, args.seed)
    action = act(actor, sample['physical'], sample['previous_action'], args.representation)
    print(json.dumps({'representation': args.representation, 'seed': args.seed,
                      'sample_scenario': sample['scenario'], 'sample_time_s': sample['time_s'],
                      'selected_online_transitions': info['selected_online_transitions'],
                      'normalized_action': action.tolist(),
                      'requested_vx_vy_vz_yaw_rate': command_units(action).tolist(),
                      'units': ['m/s', 'm/s', 'm/s', 'rad/s']}, indent=2))


if __name__ == '__main__':
    main()
