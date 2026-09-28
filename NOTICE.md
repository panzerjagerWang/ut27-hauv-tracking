# Attribution and license scope

The BSD-3-Clause license covers the original control implementation and exported actor weights in this release. Copyright (c) 2026 Jiaqing Wang, Shanghai Jiao Tong University.

Vehicle parameters and propulsion-curve digitization are attributed to Y. Bi, Y. Jin, C. Lyu, Z. Zeng and L. Lian, “Nezha-Mini: Design and locomotion of a miniature low-cost hybrid aerial underwater vehicle,” IEEE Robotics and Automation Letters, 7(3), 6669–6676, 2022, DOI [10.1109/LRA.2022.3176438](https://doi.org/10.1109/LRA.2022.3176438). Digitized values are approximate literature-derived data, not new thrust measurements. The reduced-model interface and controller are implementation choices described in the model card.

TD3 follows S. Fujimoto, H. van Hoof and D. Meger, [“Addressing Function Approximation Error in Actor-Critic Methods,” ICML 2018](https://proceedings.mlr.press/v80/fujimoto18a.html). Video detections use [Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26/). The software depends on NumPy, Numba and PyTorch, which retain their own licenses. These dependencies and the detector's upstream weights are not redistributed here.

The rendered tracking videos were generated with NVIDIA Isaac Sim using the recorded study trajectories and Nezha vehicle appearance. They are research illustrations with their original provenance, not new physical trials. Scientific figures, physical photographs, videos and manuscript text retain their respective rights; the control-software license does not grant additional rights in third-party material.

The included `paper/IEEEtran.cls` retains its original copyright and LaTeX Project Public License notices. The reference PDF used to confirm author order is not redistributed.
