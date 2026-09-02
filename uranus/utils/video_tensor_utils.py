import torch
from einops import repeat, reduce
from PIL import Image
import numpy as np

def preprocess_image(image, torch_dtype=None, device=None, pattern="B C H W", min_value=-1, max_value=1):
    if isinstance(image, Image.Image):
        # Transform a PIL.Image to torch.Tensor
        image = torch.tensor(np.array(image, dtype=np.float32))
        image = image.pin_memory().to(dtype=torch_dtype, device=device)
        image = image * ((max_value - min_value) / 255) + min_value
        image = repeat(image, f"H W C -> {pattern}", **({"B": 1} if "B" in pattern else {}))
        return image
    elif isinstance(image, torch.Tensor):
        image = image.pin_memory().to(dtype=torch_dtype, device=device)
        image = image * ((max_value - min_value) / 255) + min_value
        image = repeat(image, f"C H W-> {pattern}", **({"B": 1} if "B" in pattern else {}))
        return image


def preprocess_video(video, torch_dtype=None, device=None, pattern="B C T H W", min_value=-1, max_value=1):
    # Transform a list of PIL.Image to torch.Tensor
    video = [
        preprocess_image(image, torch_dtype=torch_dtype, device=device, min_value=min_value, max_value=max_value)
        for image in video]
    video = torch.stack(video, dim=pattern.index("T") // 2)
    return video


def preprocess_videos(videos, torch_dtype=None, device=None, min_value=-1, max_value=1):
    videos = [
        preprocess_video(video, torch_dtype=torch_dtype, device=device, min_value=min_value, max_value=max_value)
        for video in videos]
    # shape: B N_CAM C T H W
    videos = torch.stack(videos, dim=1)
    return videos
