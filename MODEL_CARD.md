# Control policy model card

## Intended use and evidence

These actors accompany the UT27 HAUV tracking component study. They request target-following motion across aerial and underwater modes from normalized measured/estimated inputs. Validation is software-in-the-loop simulation; the released reduced plant and interface supervisor are not a calibration of the modified physical vehicle. The shared visual front end used YOLO26n followed by stereo processing and a constant-velocity Kalman filter. ORB was evaluated as a detector substitution with frozen policies.

Nine selected actors are released under BSD-3-Clause. Coordinates uses a 36→128→128→4 network; DCE uses 45→120→130→4; AC uses 41→125→127→4. Hidden layers use ReLU and the output uses tanh. There is no recurrent hidden state. Four previous commands are explicit inputs.

## The common physical observation

Call `act(actor, physical, previous_action, representation)`. `physical` must have shape `(32,)`, and `previous_action` must have shape `(4,)`. Use normalized fields as below and clip the complete physical vector to [−1, 1], as in the study. All target quantities are estimates. World axes are z-up; the yaw frame uses x-forward and y-left.

| Index | Quantity | Normalization |
|---|---|---|
| 0 | Estimated relative range | / 20 m |
| 1–3 | Estimated target-minus-vehicle velocity in yaw frame | / 3 m/s |
| 4 | Position covariance trace | / 30 m² |
| 5 | Time since last accepted position | min(age / 5 s, 1) |
| 6 | Confidence with age decay | filter confidence × exp(−0.5 × age in s) |
| 7–9 | Vehicle velocity in yaw frame | / 3 m/s |
| 10–11 | sin(yaw), cos(yaw) | unscaled |
| 12 | Vehicle world height | / 10 m |
| 13–14 | sin(roll), cos(roll) | unscaled |
| 15–16 | sin(pitch), cos(pitch) | unscaled |
| 17–19 | Body angular velocity | / 2 rad/s |
| 20–23 | Four actual aerial motor speeds | / AIR_OMEGA_MAX |
| 24–26 | Three signed actual underwater motor speeds | / WATER_OMEGA_MAX |
| 27 | Propulsion mode | 0 aerial, 1 underwater |
| 28 | Vehicle clearance above local surface | / 2 m |
| 29–31 | Estimated target-minus-vehicle position in yaw frame | / 20 m |

`AIR_OMEGA_MAX = 18000 × 2π / 60 rad/s`; `WATER_OMEGA_MAX = 500 × 2π / 60 rad/s`. The common encoded prefix is the first 29 physical fields followed by the four previous normalized actions. At the beginning of a trial the previous action is zero.

The desired target-relative offset is `(1.2, 0, −1.0)` m. Its norm is about 1.56 m. It defines a non-contact following position, rather than an interception point.

## State-specific suffix

- **Coordinates** appends physical fields 29–31 directly. They are filter-derived coordinates, not target ground truth.
- **DCE** appends twelve values from four directional faces. A face contains confidence-weighted activation and two weighted angular coordinates. Confidence becomes zero when normalized age reaches one. A zero-confidence intervention removes the directional suffix.
- **AC** appends eight normalized predicted-cost differences. It evaluates the previous command and ±0.5 perturbations of each command dimension, clipped to [−1, 1]. Nine rollouts use a one-second horizon and a 20 ms reduced-model integration step. Estimated target velocity is held constant and current clearance defines a locally flat surface. Future target paths and waves are not inputs. Each contrast is Δcost / (1 + |Δcost|).

The code in `hauv_control/state.py` is the pursuit-and-dive encoder. It preserves the exact original computation with package-local imports. The scalar inputs in `examples/recorded_observations.json` are retained from the original trial; they are not synthetic fixtures.

## Actions and propulsion

The four actor outputs are normalized requests in [−1, 1]. `command_units` maps them to `(1.2 ax, 1.2 ay, 0.5 az)` m/s and `0.6 aψ` rad/s. These are requested yaw-frame velocities and yaw rate, not direct motor commands.

`hauv_control.dynamics.controller(state, action, time, amplitude, phase)` implements the shared simulated low-level controller. Its 21-value own-vehicle state contains world position (3), body linear velocity (3), wxyz quaternion (4), body angular velocity (3), actual aerial and underwater motor speeds (7), and propulsion mode (1). It returns seven motor-speed requests in rad/s. The underwater model has one vertical bow thruster and two rear surge thrusters; lateral requests affect heading rather than an independent lateral actuator. Surface transitions are implemented by the reduced plant's mode supervisor.

## Training and selection

All representations use 20,000 shared observation-based demonstration transitions, 2,000 behavior-cloning updates and 262,144 TD3 online transitions. Seeds are 60, 61 and 62. Eight separate validation trajectories select complete-mission success first, then penalized tracking error. Initialization remains eligible for selection. The selected steps are retained in `weights/manifest.json`; three selected actors are BC initializations. Actors were not selected again using test results when preparing this release.

The current comparison has 1,200 tracking trials and 100 trials per detector–representation–scenario cell. The original 50 trials per cell were retained and another 50 were added after reviewing those results, without changing frozen policies. Only three independently trained policies underlie each representation. Native and lake-derived appearance are simulation conditions; the latter is not a physical field trial.

## What is released

Tensor-only actor state dictionaries, their encoders, the shared reduced-model controller, prototype parameters, digitized propulsion fits, example observations, aggregate scores and provenance. Perception checkpoints, full training corpora and Isaac Sim scene assets are not part of this control-policy package. The website videos use the study's frozen YOLO26n checkpoint and retain its offline detections.
