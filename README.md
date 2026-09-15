# Uranus

Export one Lance episode into the XML-environment format:

```bash
uv run python scripts/export_data.py \
  --repo agibot-challenge-2026 \
  --episode-index 0 \
  --output-root examples/data
```

The exported MJCF XML contains the robot and camera intrinsics/extrinsics.
Wrist cameras are mounted under their flange, while external cameras are
mounted under the robot base.

Every `state` in `temporal.json` is already a complete MuJoCo qpos vector in
`model.qpos` order. This is the only accepted temporal schema:

```json
{
  "step_qpos": [{
    "state": [],
    "robot2world_transform": [
      [1.0, 0.0, 0.0, 0.0],
      [0.0, 1.0, 0.0, 0.0],
      [0.0, 0.0, 1.0, 0.0],
      [0.0, 0.0, 0.0, 1.0]
    ]
  }]
}
```

There is no arm/body/gripper split and no compact-gripper expansion. The runner
passes `state` directly to `data.qpos[:]`.
`robot2world_transform` is the same per-frame 4x4
robot-to-world matrix used by `vincent/rm_robot_class`; fixed-base robots use
the identity matrix.

Running `main.py` writes aligned H.264/yuv420p videos under `gen/`, `gt/`,
`skeleton/`, and `plucker/`. A single top-level `preview.mp4` uses one row per
camera and the columns `GT | GEN | SKELETON | PLUCKER`. The Plücker view shows
moment RGB on its left half and ray-direction RGB on its right half.
