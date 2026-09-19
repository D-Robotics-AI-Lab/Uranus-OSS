# Uranus

<!---
Copyright 2026 - The D-Robotics Large Model Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

<p align="center">
    <br>
    <img src="assets/figures/logo.svg" width="400"/>
    <br>
<p>
<p align="center">
    <a href="https://huggingface.co/D-Robotics"><img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-yellow"></a>
    <a href="https://www.modelscope.cn/models/D-Robotics/Uranus-1.3B"><img alt="ModelScope" src="https://img.shields.io/badge/%F0%9F%A4%96%20ModelScope-8A2BE2"></a>
    <a href="TODO_TEST_SAMPLES_URL"><img alt="Test Samples" src="https://img.shields.io/badge/%F0%9F%A7%AA%20Test%20Samples-00B4D8"></a>
    <a href="TODO_TECHNICAL_REPORT_URL"><img alt="Technical Report" src="https://img.shields.io/badge/Technical_Report-B31B1B?logo=arxiv&logoColor=white"></a>
    <a href="https://d-robotics-ai-lab.github.io/large-model-team/blog/uranus/"><img alt="Blog" src="https://img.shields.io/badge/Blog-FF7A00?logo=githubpages&logoColor=white"></a>
    <a href="TODO_WECHAT_URL"><img alt="WeChat" src="https://img.shields.io/badge/WeChat-07C160?logo=wechat&logoColor=white"></a>
    <a href="https://github.com/D-Robotics-AI-Lab/Uranus-OSS"><img alt="GitHub" src="https://img.shields.io/badge/OSS_Code-0077FF.svg?logo=github&logoColor=white"></a>
</p>

-----

In this repository, we present **Uranus**, a data-driven robot simulator built around a joint-trajectoryconditioned autoregressive diffusion model. Uranus offers three key capabilities:

- **streaming, open-ended rollout**, which receives future joint-position trajectories online and autoregressively
generates one latent frame per step, corresponding to four RGB frames, without a fixed horizon
- **low-latency generation**, achieving 24 FPS after inference optimization
- **scalable, extensible robot control**, providing a unified interface for synchronized multi-view generation
across diverse robot embodiments and camera configurations.

More technical details can be found in our [technical report](TODO_TECHNICAL_REPORT_URL), and visualization demos are available on our [blog](https://d-robotics-ai-lab.github.io/large-model-team/blog/uranus/).

## Quickstart

####  Installation

Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Create the environment and install dependencies

```bash
git clone https://github.com/D-Robotics-AI-Lab/Uranus-OSS.git
cd Uranus-OSS
uv sync
```

All commands in this repo are run through `uv run python <cmd>`, which resolves `.venv/` automatically without manual activation. 

#### Download model weights

| Models       | Download Link                                                                                                                                           |    Notes                      |
|--------------|---------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------|
| Uranus-1.3B | 🤗 [Huggingface](https://huggingface.co/D-Robotics/Uranus-1.3B)      🤖 [ModelScope](https://www.modelscope.cn/models/D-Robotics/Uranus-1.3B)             | SFT model, 384×640, 25 inference steps |
| Uranus-1.3B-Distillation | 🤗 [Huggingface](https://huggingface.co/D-Robotics/Uranus-1.3B-Distillation)    🤖 [ModelScope](https://www.modelscope.cn/models/D-Robotics/Uranus-1.3B-Distillation)     | Distilled model, 384×640, 4 inference steps |


Uranus consumes converted weights from a single directory. Expected layout:

```
weights_dir/
├── dit.pt                     # SpatialTemporalWanModel state_dict
├── vae.pt                     # WanVideoVAE state_dict (loaded strict=False)
├── text_encoder.pt            # WanTextEncoder (T5) state_dict
├── plucker_adapter.pt         # Plücker-conditioning adapter
├── vace_patch_embedding.pt    # skeleton controller patch embedding
├── tokenizer/                 # HuggingFace tokenizer directory
└── metadata.json              # model + inference metadata
```

#### Download test samples

We provide a set of ready-to-run test samples covering a variety of robot embodiments (ALOHA, ARX5, UR5, Franka, G1, DOS-W1, X5) and data sources (AgiBot World, DROID, RC-Table). Download and unpack them from [here](TODO_TEST_SAMPLES_URL) into `examples/data/`.

Each sample is a self-contained "XML-environment" directory:

```
sample_dir/
├── meta.json            # prompt, cameras, mjcf_path, end_effectors, skeleton, fps
├── temporal.json        # step_qpos: [{state, robot2world_transform}, ...]
├── mjcf/
│   └── <robot>.xml      # robot MJCF with baked-in camera calibration & mounts
├── ref_images/
│   └── <camera>.png     # one reference image per camera
└── gt/
    └── <camera>.mp4     # ground-truth video per camera
```

Your own robot data can be converted into this same format and fed to inference as well, as long as it provides the per-frame `qpos`, the robot MJCF with calibrated cameras, and one reference image per camera.


#### Run inference

Run generation on a test sample with the pretrained model:

```bash
uv run python main.py \
  --weights-dir <weights_dir> \
  --sample-dir examples/data/000000 \
  --output-dir ./output
```


## License

This project is released under the [Apache License 2.0](LICENSE). By using, distributing, or contributing to this repository, you agree to the terms and conditions of the license.


## Citation

🌟 If you find our work helpful, please leave us a star and cite our paper.

```
@article{drobotics2026uranus,
  author  = {D-Robotics Large Model Team},
  title   = {Uranus: Building the Next-Generation Simulation Infrastructure for Embodied AI},
  journal = {D-Robotics AI Blog},
  year    = {2026},
}
```

## Contact Us

If you would like to leave a message to our research or product teams, feel free to join our [Discord](TODO_DISCORD_URL) or [WeChat groups](TODO_WECHAT_URL)!