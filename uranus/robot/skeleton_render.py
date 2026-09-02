import cv2
import mujoco
import numpy as np
import torch
from .sh_utils import eval_sh
from .unified_robot import precompute_fk

joint_colors = [
    "#FF0000", "#00FF00", "#0000FF", "#FFFF00",
    "#FF00FF", "#00FFFF", "#FFA500", "#800080",
    "#FFD700", "#008000", "#FF1493", "#00008B",
    "#7FFF00", "#4B0082", "#FF4500", "#00CED1",
    "#8B4513", "#ADFF2F", "#1E90FF", "#808000",
    "#FF6347", "#00FA9A", "#4682B4", "#D2691E"
]

link_colors = [
    "#FFFFFF", "#000000", "#FF00FF", "#006400",
    "#FFFF00", "#000080", "#00FF00", "#8B0000",
    "#00FFFF", "#4B0082", "#FFA500", "#2F4F4F",
    "#FFC0CB", "#008080", "#FFD700", "#191970",
    "#ADFF2F", "#800000", "#F0E68C", "#483D8B",
    "#7FFFD4", "#556B2F", "#E6E6FA", "#A52A2A"
]

_SH_COEFF_DEFAULT = np.array([
    [0.15, 0.15, 0.15],

    [ 2.5, -1.2, -1.2],
    [-1.2,  2.5, -1.2],
    [-1.2, -1.2,  2.5],

    [ 1.5,  0.5, -1.5],
    [-1.5,  1.5,  0.5],
    [ 0.5, -1.5,  1.5],

    [ 2.0, -0.5,  0.5],
    [ 0.5,  2.0, -0.5],
], dtype=np.float64)

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def render_mjcf_skeleton(model_path, qpos, intrinsic, extrinsic, raw_img_size=(720, 1280), target_img_size=None):
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    if len(qpos) < model.nq:
        full_qpos = np.zeros(model.nq)
        full_qpos[: len(qpos)] = qpos
        full_qpos[len(qpos): ] = qpos[-1]
        data.qpos = full_qpos
    else:
        data.qpos = qpos[: model.nq]

    mujoco.mj_forward(model, data)

    rotation_matrix = extrinsic[:3, :3]
    t = extrinsic[:3, 3]

    if target_img_size is not None:
        intrinsic[0, ...] *= target_img_size[1] / raw_img_size[1]
        intrinsic[1, ...] *= target_img_size[0] / raw_img_size[0]
    else:
        target_img_size = raw_img_size

    img = np.zeros((target_img_size[0], target_img_size[1], 3), dtype=np.uint8)
    body_positions_2d = {}
    positions_in_image = {}

    for i in range(model.nbody):
        pos_world = data.xpos[i]
        pos_cam = rotation_matrix @ pos_world + t

        if pos_cam[2] <= 0.01:
            continue

        u_homo = intrinsic @ pos_cam
        u = int(u_homo[0] / u_homo[2])
        v = int(u_homo[1] / u_homo[2])

        body_positions_2d[i] = (u, v)
        if 0 <= u < target_img_size[1] and 0 <= v < target_img_size[0]:
            positions_in_image[i] = (u, v)
            c = hex_to_rgb(joint_colors[i])
            radius = int(target_img_size[0] / 60)
            cv2.circle(img, (u, v), radius, c, -1) + 1
            # cv2.putText(img, str(i), (u, v), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    for i in range(1, model.nbody):
        parent_id = model.body_parentid[i]
        if i in positions_in_image or parent_id in positions_in_image:
            if parent_id in body_positions_2d and i in body_positions_2d:
                c = hex_to_rgb(link_colors[i])
                thickness = int(target_img_size[0] / 120) + 1
                cv2.line(img, body_positions_2d[parent_id], body_positions_2d[i], c, thickness)

    return img


def batch_render_sh_on_image(
    skeleton_imgs,
    ee_pos_world,
    ee_rot_world,
    intrinsic,
    R_cam,
    t_cam,
    radii,
    sh_coeffs=None,
    sh_degree=2,
):
    """Draw one 3DGS-style SH sphere per image for a flattened frame-camera batch."""
    assert isinstance(skeleton_imgs, np.ndarray) and skeleton_imgs.ndim == 4
    assert torch.is_tensor(ee_pos_world) and ee_pos_world.dtype == torch.float32
    assert torch.is_tensor(ee_rot_world) and ee_rot_world.dtype == torch.float32
    assert torch.is_tensor(intrinsic) and intrinsic.dtype == torch.float32
    assert torch.is_tensor(R_cam) and R_cam.dtype == torch.float32
    assert torch.is_tensor(t_cam) and t_cam.dtype == torch.float32
    assert torch.is_tensor(radii) and radii.dtype == torch.float32

    B, H, W = skeleton_imgs.shape[:3]
    device = ee_pos_world.device
    assert ee_pos_world.shape == (B, 3)
    assert ee_rot_world.shape == (B, 3, 3)
    assert intrinsic.shape == (B, 3, 3)
    assert R_cam.shape == (B, 3, 3)
    assert t_cam.shape == (B, 3)
    assert radii.shape == (B,)
    assert ee_rot_world.device == device
    assert intrinsic.device == device
    assert R_cam.device == device
    assert t_cam.device == device
    assert radii.device == device

    cam_pos_world = -torch.bmm(R_cam.transpose(1, 2), t_cam[..., None]).squeeze(-1)
    ee_pos_cam = torch.bmm(R_cam, ee_pos_world[..., None]).squeeze(-1) + t_cam
    center_h = torch.bmm(intrinsic, ee_pos_cam[..., None]).squeeze(-1)
    center_u = center_h[:, 0] / center_h[:, 2]
    center_v = center_h[:, 1] / center_h[:, 2]
    sphere_pr = intrinsic[:, 0, 0] * radii / ee_pos_cam[:, 2]

    dirs = ee_pos_world - cam_pos_world
    dirs = dirs / (dirs.norm(dim=1, keepdim=True) + 1e-8)
    dirs = torch.bmm(dirs[:, None, :], ee_rot_world).squeeze(1)

    coeff = (sh_degree + 1) ** 2
    if sh_coeffs is None:
        sh = torch.as_tensor(_SH_COEFF_DEFAULT.T, dtype=torch.float32, device=device)
    elif torch.is_tensor(sh_coeffs):
        sh = sh_coeffs.to(device=device, dtype=torch.float32)
        if sh.shape[-1] == 3:
            sh = torch.movedim(sh, -1, -2)
    else:
        sh_coeffs = np.asarray(sh_coeffs, dtype=np.float64)
        if sh_coeffs.shape[-1] == 3:
            sh_coeffs = np.moveaxis(sh_coeffs, -1, -2)
        sh = torch.as_tensor(sh_coeffs, dtype=torch.float32, device=device)
    sh = sh.view(1, 3, coeff).expand(B, 3, coeff)
    rgb = eval_sh(sh_degree, sh, dirs) + 0.5

    valid = (
        (radii >= 1e-4)
        & (ee_pos_cam[:, 2] > 0)
        & (center_u >= 0)
        & (center_u < W)
        & (center_v >= 0)
        & (center_v < H)
        & (sphere_pr >= 0.5)
    )
    rgb_uint8 = (torch.clamp(rgb, 0.0, 1.0) * 255.0).to(torch.uint8).cpu().numpy()
    center_u_np = center_u.detach().cpu().numpy()
    center_v_np = center_v.detach().cpu().numpy()
    sphere_pr_np = sphere_pr.detach().cpu().numpy()
    valid_np = valid.detach().cpu().numpy()

    for b in np.flatnonzero(valid_np):
        cv2.circle(
            skeleton_imgs[b],
            (int(round(center_u_np[b])), int(round(center_v_np[b]))),
            int(np.ceil(sphere_pr_np[b])),
            rgb_uint8[b].tolist(),
            -1,
        )
    return skeleton_imgs, rgb

def _render_keypoints(img, keypoints, R_cam, t_cam, K, target_size):
    H, W = target_size
    dot_r     = max(1, int(H / 60))
    thickness = int(H / 120) + 1
    proj, inside = [], []
    for kp in keypoints:
        p_cam = R_cam @ kp["pos"] + t_cam
        if p_cam[2] <= 0.01:
            proj.append(None); inside.append(False); continue
        uvw = K @ p_cam
        u, v = int(uvw[0] / uvw[2]), int(uvw[1] / uvw[2])
        proj.append((u, v))
        ok = 0 <= u < W and 0 <= v < H
        inside.append(ok)
        if ok:
            cv2.circle(img, (u, v), dot_r, hex_to_rgb(joint_colors[kp["color"] % len(joint_colors)]), -1)
    for i, kp in enumerate(keypoints):
        p_idx = kp["parent"]
        if p_idx is None or proj[i] is None or proj[p_idx] is None:
            continue
        if not (inside[i] or inside[p_idx]):
            continue
        cv2.line(img, proj[p_idx], proj[i], hex_to_rgb(link_colors[kp["color"] % len(link_colors)]), thickness)


def render_skeleton_frames(robot, frames, cam_name, raw_height, raw_width, *, fk_cache=None, target_size=None):
    """Render skeleton for a frame sequence via online FK. Returns [N, 3, H, W] uint8 tensor.

    Pass *fk_cache* = ``(ee_states_all, keypoints_all, robot_to_world_all)`` to
    skip the expensive FK step when rendering multiple cameras from the same sequence.

    Pass *target_size* = ``(H, W)`` to render at a different resolution; intrinsics
    are scaled from ``(raw_height, raw_width)`` to *target_size*.
    """
    if fk_cache is not None:
        ee_states_all, keypoints_all, robot_to_world_all = fk_cache
    else:
        ee_states_all, keypoints_all, robot_to_world_all = precompute_fk(robot, frames)

    H, W = target_size or (raw_height, raw_width)

    Ks, R_cams, t_cams = [], [], []
    for i, frame in enumerate(frames):
        extrinsic = np.asarray(frame[f"camera_extrinsics.{cam_name}"], dtype=np.float64)
        extrinsic = extrinsic @ np.linalg.inv(robot_to_world_all[i])
        K = frame[f"camera_intrinsics.{cam_name}"].copy().astype(np.float64)
        K[0, :] *= W / raw_width
        K[1, :] *= H / raw_height
        Ks.append(K)
        R_cams.append(extrinsic[:3, :3])
        t_cams.append(extrinsic[:3, 3])

    N = len(frames)
    imgs = np.zeros((N, H, W, 3), dtype=np.uint8)
    Ks_t = torch.as_tensor(np.stack(Ks), dtype=torch.float32)
    R_t  = torch.as_tensor(np.stack(R_cams), dtype=torch.float32)
    t_t  = torch.as_tensor(np.stack(t_cams), dtype=torch.float32)

    num_ee = len(ee_states_all[0])
    sh_corrections = robot.get_ee_sh_corrections()
    for ee_idx in range(num_ee):
        pos = torch.as_tensor(np.stack([ee_states_all[i][ee_idx][0] for i in range(N)]), dtype=torch.float32)
        rot = torch.as_tensor(np.stack([ee_states_all[i][ee_idx][1] for i in range(N)]), dtype=torch.float32)
        rad = torch.tensor([ee_states_all[i][ee_idx][2] for i in range(N)], dtype=torch.float32)
        corr = torch.as_tensor(sh_corrections[ee_idx], dtype=torch.float32)
        sh_rot = torch.bmm(rot, corr.unsqueeze(0).expand(N, -1, -1))
        imgs, _ = batch_render_sh_on_image(imgs, pos, sh_rot, Ks_t, R_t, t_t, rad)

    for i in range(N):
        _render_keypoints(imgs[i], keypoints_all[i], R_cams[i], t_cams[i], Ks[i], (H, W))

    return torch.from_numpy(imgs).permute(0, 3, 1, 2)  # [N, 3, H, W]
