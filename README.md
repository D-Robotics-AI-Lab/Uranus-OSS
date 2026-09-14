# Uranus

Export one Lance episode into the XML-environment format:

```bash
uv run python scripts/export_data.py \
  --repo agibot-challenge-2026 \
  --episode-index 0 \
  --output-root examples/data
```

The exported MJCF XML contains the robot, camera intrinsics/extrinsics, camera
mounts, default joint state, joint ordering, and compact-gripper expansion.
Wrist cameras are mounted under their flange, while external cameras are
mounted under the robot base.

`temporal.json` starts with a machine-readable description of the joint-vector
layout, followed by uniform per-frame robot state:

```json
{
  "joint_groups": {
    "body": {"indices": [0, 1], "joints": ["body_joint_1", "body_joint_2"]},
    "head": {"indices": [2], "joints": ["head_joint_1"]},
    "left_arm": {"indices": [3], "joints": ["left_arm_joint_1"]},
    "right_arm": {"indices": [4], "joints": ["right_arm_joint_1"]}
  },
  "states": [{
    "joint_positions": [],
    "gripper": [],
    "robot_transform": {
      "xyz": [0.0, 0.0, 0.0],
      "quaternion": [0.0, 0.0, 0.0, 1.0]
    }
  }]
}
```

`joint_groups` is the JSON-compatible annotation for `joint_positions`; for
example, Agibot G2 exports `body`, `head`, `left_arm`, and `right_arm` slices.
`robot_transform` is the robot-base-to-environment pose and uses an `xyzw`
quaternion. Fixed-base robots use the identity pose. Joint-vector lengths remain
native to each robot; `gripper` contains one value per physical gripper.

Running `main.py` writes aligned H.264/yuv420p videos under `gen/`, `gt/`,
`skeleton/`, and `plucker/`. A single top-level `preview.mp4` uses one row per
camera and the columns `GT | GEN | SKELETON | PLUCKER`. The Plücker view shows
moment RGB on its left half and ray-direction RGB on its right half.
