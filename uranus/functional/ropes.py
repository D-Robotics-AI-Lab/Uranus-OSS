import torch

def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0, device: torch.device = "cpu"):
    # 3d rope precompute
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta).to(device=device)
    h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta).to(device=device)
    w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta).to(device=device)
    return f_freqs_cis, h_freqs_cis, w_freqs_cis


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0):
    # 1d rope precompute
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)
                   [: (dim // 2)].double() / dim))
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis