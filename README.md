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
    <img src="https://github.com/D-Robotics-AI-Lab/Uranus-OSS/blob/main/assets/figures/logo.png" width="400"/>
</p>
<p align="center">
    <a href="https://huggingface.co/collections/D-Robotics/uranus"><img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-yellow"></a>
    <a href="https://www.modelscope.cn/collections/D-Robotics/Uranus"><img alt="ModelScope" src="https://img.shields.io/badge/%F0%9F%A4%96%20ModelScope-8A2BE2"></a>
    <a href="https://huggingface.co/datasets/D-Robotics/Uranus-Demo-Data"><img alt="Test Samples" src="https://img.shields.io/badge/%F0%9F%A7%AA%20Test%20Samples-00B4D8"></a>
    <!-- <a href="TODO_TECHNICAL_REPORT_URL"><img alt="Technical Report" src="https://img.shields.io/badge/Technical_Report-B31B1B?logo=arxiv&logoColor=white"></a> -->
    <a href="https://d-robotics-ai-lab.github.io/large-model-team/blog/uranus/"><img alt="Blog" src="https://img.shields.io/badge/Blog-FF7A00?logo=githubpages&logoColor=white"></a>
    <a href="https://github.com/D-Robotics-AI-Lab/Uranus-OSS/raw/main/assets/figures/wechat.jpg"><img alt="WeChat" src="https://img.shields.io/badge/WeChat-07C160?logo=wechat&logoColor=white"></a>
    <a href="https://github.com/D-Robotics-AI-Lab/Uranus-OSS"><img alt="GitHub" src="https://img.shields.io/badge/OSS_Code-0077FF.svg?logo=github&logoColor=white"></a>
    <a href="https://github.com/D-Robotics-AI-Lab/Uranus-SDK"><img alt="GitHub" src="https://img.shields.io/badge/SDK_Code-0077FF.svg?logo=github&logoColor=white"></a>
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


You can download the weights with the Hugging Face CLI (`pip install -U huggingface_hub` if you don't have it):

```bash
# SFT model (17.5 GB) into ./weights/uranus-1.3b
hf download D-Robotics/Uranus-1.3B --local-dir ./weights/uranus-1.3b

# Distilled model (17.5 GB) into ./weights/uranus-1.3b-distillation
hf download D-Robotics/Uranus-1.3B-Distillation --local-dir ./weights/uranus-1.3b-distillation
```

The same repos are mirrored on ModelScope — use `modelscope download D-Robotics/Uranus-1.3B --local-dir ./weights/uranus-1.3b`.


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

We provide a set of ready-to-run test samples covering a variety of robot embodiments (ALOHA, ARX5, UR5, Franka, G1, DOS-W1, X5) and data sources (AgiBot World, DROID, RC-Table). Download and unpack them from [Huggingface](https://huggingface.co/datasets/D-Robotics/Uranus-Data) into `examples/data/`.

```bash
hf download D-Robotics/Uranus-Demo-Data --repo-type dataset --local-dir ./examples/data
```

Each episode lands in `./examples/data/<episode_id>/` and can be passed directly to `main.py --sample-dir`.

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

**Flag reference**

| Flag | Required | Description |
|------|----------|-------------|
| `--weights-dir` | yes | Path to the converted weights directory (see [Download model weights](#download-model-weights)). Must contain `metadata.json`. |
| `--sample-dir` | yes | Path to a single XML-environment sample (see [Download test samples](#download-test-samples)). |
| `--output-dir` | no | Root directory for generated videos. Defaults to `./output`. |
| `--num-chunks` | no | Number of autoregressive chunks to roll out. Omit to automatically use every frame in `temporal.json` (trailing partial chunk is padded with the last frame, then trimmed back to the original length in the output). |

**Hyperparameters are auto-resolved from `metadata.json`**

`num_inference_steps`, `step_length`, `default_height`, `default_width`, and `teacher_forcing_window_size` are read from the checkpoint's `metadata.json` automatically — no extra flags are needed. Any flag passed explicitly on the CLI takes precedence over the metadata. For example, the SFT model defaults to 25 denoising steps, while the distilled model defaults to 4.

**Outputs**

Each run writes four aligned video streams per camera under `--output-dir`:

- `gen/` — model-generated video
- `gt/` — ground-truth video (copied from the sample for comparison)
- `skeleton/` — rendered MuJoCo skeleton visualization
- `plucker/` — Plücker-coordinate conditioning visualization (force RGB + ray-direction RGB)

Plus a `preview.mp4` that stacks all four streams side by side for quick qualitative comparison.


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

If you would like to leave a message to our research or product teams, feel free to join our [WeChat](https://github.com/D-Robotics-AI-Lab/Uranus-OSS/raw/main/assets/figures/wechat.jpg) groups!

<p align="center">
    <img src="https://github.com/D-Robotics-AI-Lab/Uranus-OSS/blob/main/assets/figures/company.svg" alt="D-Robotics" width="400"/>
</p>