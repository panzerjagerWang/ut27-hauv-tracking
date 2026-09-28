# HAUV aerial–underwater target tracking

**Prepared for the UT27 Student Poster Competition.**

**Jiaqing Wang** (first and corresponding author), Ruixuan Hu, Zheng Zeng  
School of Oceanography, Shanghai Jiao Tong University, Shanghai, China

[Project website](https://panzerjagerWang.github.io/ut27-hauv-tracking/) · [Two-page abstract](https://panzerjagerWang.github.io/ut27-hauv-tracking/downloads/ut27_spc_abstract.pdf) · [Control release ZIP](https://panzerjagerWang.github.io/ut27-hauv-tracking/downloads/ut27_control_release.zip) · [Model card](MODEL_CARD.md)

This project studies a hybrid aerial underwater vehicle (HAUV) that acquires a moving water target from the air, pursues it, enters the water through feedback control and continues following underwater. The release accompanies *Component Selection for HAUV Aerial–Underwater Target Tracking with Deep Reinforcement Learning*.

## Contents

- Nine validation-selected control actors: Coordinates, directional confidence encoding (DCE), and action contrast (AC), each with seeds 60, 61 and 62.
- Exact state encoders, the shared reduced-model propulsion controller, model metadata and recorded-input examples.
- Two synchronized third-person / onboard-YOLO scenario videos, a two-page abstract and its LaTeX figures.
- Audited aggregate results and a release-equivalence report.

## Run a policy

Requires Python 3.10 or later. The release was checked with Python 3.12, PyTorch 2.11.0, NumPy 2.3.1 and Numba 0.63.1 on CPU.

```bash
git clone https://github.com/panzerjagerWang/ut27-hauv-tracking.git
cd ut27-hauv-tracking
python -m pip install -r requirements.txt
python -m examples.run_policy --representation action_contrast --seed 60
```

The example uses an actual recorded observation and prints the four normalized commands plus requested yaw-frame velocities and yaw rate. `coordinates` and `compact_dce` select the other representations. Run from the repository root or the extracted control-release folder. The first AC call compiles the Numba predictor.

```python
from hauv_control.policy import load_actor, act, command_units

actor, metadata = load_actor("action_contrast", seed=60)
# physical: 32 normalized measured/estimated fields; previous_action: 4 commands.
action = act(actor, physical, previous_action, "action_contrast")
requested_vx_vy_vz_yaw_rate = command_units(action)
```

The [model card](MODEL_CARD.md) defines every input and output. The actor weights are tensor-only PyTorch `state_dict` files loaded with `weights_only=True`. They do not contain a serialized custom model object. Inference does not require Isaac Sim or YOLO; live tracking additionally requires the corresponding perception, navigation and estimation inputs.

## Tracking results

Complete-mission successes out of **100 trials per detector–representation–scenario cell**:

| Detector | State | Straight | Curved |
|---|---|---:|---:|
| YOLO26n | Coordinates | 58 | 42 |
| YOLO26n | DCE | 54 | 50 |
| YOLO26n | AC | 63 | 72 |
| ORB | Coordinates | 0 | 0 |
| ORB | DCE | 0 | 0 |
| ORB | AC | 0 | 0 |

There are 1,200 software-in-the-loop trials using actual Isaac Sim stereo images and an external reduced six-DOF plant. Each cell contains 50 native-water and 50 lake-appearance trials. Policy seeds contribute 34, 33 and 33 trials; seventeen target paths are represented. The first 50 trials per cell were retained and the extension added 50 with frozen settings after those results were reviewed. All policies share the estimation and low-level control stack. AC additionally uses nine predictive rollouts per action.

AC achieved the highest observed mission counts. The descriptive 97.5% crossed-bootstrap interval for its paired success difference against Coordinates spans zero (−1.5 to +36.4 percentage points). Three independent policy seeds do not establish a definitive representation ranking. See [data/results.json](data/results.json) for the original counts, uncertainty and extension disclosure.

## Videos

- [Straight pursuit, 60 s](https://panzerjagerWang.github.io/ut27-hauv-tracking/videos/straight_thirdperson_yolo_pip.mp4)
- [Curved pursuit, 60 s](https://panzerjagerWang.github.io/ut27-hauv-tracking/videos/curved_thirdperson_yolo_pip.mp4)

Both show recorded **Coordinates-policy** examples with policy seed 60 and target seed 83000. The external camera replays the original executed poses. The inset uses the original onboard video with offline YOLO26n boxes. Control time advances at 10 Hz; the camera samples at 5 Hz and holds the preceding image between captures. Cyan and yellow traces annotate the previous 20 seconds of HAUV and target positions. The curved example preserves the 30–32 s observation gap.

The straight example fails the original complete-mission criterion; the curved example passes. These fixed examples are not representative success-rate estimates and are not AC-policy rollouts. Original scores and file hashes are in [data/video_provenance.json](data/video_provenance.json).

Physical lake recordings included in the abstract are qualitative evidence of approach and water entry. Their clocks are unsynchronized and the available records do not support field success-rate scoring of the frozen policies.

## Checkpoint provenance

[weights/manifest.json](weights/manifest.json) records each original checkpoint hash, exported-file hash, architecture, training budget and selected transition count. Export did not change any actor parameter. AC seeds 60 and 62, and DCE seed 62, were selected at the behavior-cloning initialization (zero online transitions). All other actors were selected at the recorded online checkpoint.

The release was compared with the original implementation across nine actors and ten recorded aerial, interface and submerged inputs. Parameters, encoded inputs and actions match exactly in all 90 comparisons. [Verification report](data/release_verification.json).

## Abstract and figures

The [two-page abstract](https://panzerjagerWang.github.io/ut27-hauv-tracking/downloads/ut27_spc_abstract.pdf) contains Abstract, Introduction, Problem Formulation, Method, Experiment and References. All sixteen original graphics references are retained, including the intentionally repeated Figure 2 placeholder. [Source archive](https://panzerjagerWang.github.io/ut27-hauv-tracking/downloads/ut27_spc_abstract_source.zip).

```bash
cd paper
pdflatex -interaction=nonstopmode -halt-on-error abstract.tex
pdflatex -interaction=nonstopmode -halt-on-error abstract.tex
```

The title and author block identify preparation for the [UT27 Student Poster Competition](https://www.ut27.org/authors.html). This repository is the research-material release, not the conference submission portal.

## License and attribution

The original control code and exported actor weights are released under [BSD-3-Clause](LICENSE). Scientific figures, recordings, the manuscript and third-party components retain their respective rights; the software license does not relicense them. See [NOTICE.md](NOTICE.md).

Citation metadata is provided in [CITATION.cff](CITATION.cff).
