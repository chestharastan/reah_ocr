import torch
import torch.nn.functional as F


def chunk_image(image, chunk_width=100, overlap=16):
    """
    Slide a window of width `chunk_width` across the image with stride
    (chunk_width - overlap). The final chunk is right-padded with zeros if it
    runs off the end.

    image: (C, H, W)
    returns:
        chunks: (N, C, H, chunk_width)
        starts: list of left edges per chunk, length N
    """
    _, _, W = image.shape
    stride = chunk_width - overlap
    if stride <= 0:
        raise ValueError("overlap must be smaller than chunk_width")

    starts = []
    s = 0
    while True:
        starts.append(s)
        if s + chunk_width >= W:
            break
        s += stride

    chunks = []
    for s in starts:
        end = min(s + chunk_width, W)
        chunk = image[:, :, s:end]
        if chunk.shape[2] < chunk_width:
            chunk = F.pad(chunk, (0, chunk_width - chunk.shape[2]))
        chunks.append(chunk)

    return torch.stack(chunks, dim=0), starts


def chunk_batch(images, chunk_width=100, overlap=16):
    """
    Batched chunking. Assumes all images in the batch share the same width
    (true after the dataset's resize transform), so every sample produces the
    same number of chunks N.

    images: (B, C, H, W)
    returns:
        flat_chunks: (B*N, C, H, chunk_width), ordered as
                     [b0_c0, b0_c1, ..., b0_cN-1, b1_c0, ...]
        n_chunks: int N
    """
    B, C, H, W = images.shape
    stride = chunk_width - overlap
    if stride <= 0:
        raise ValueError("overlap must be smaller than chunk_width")

    starts = []
    s = 0
    while True:
        starts.append(s)
        if s + chunk_width >= W:
            break
        s += stride

    chunks = []
    for s in starts:
        end = min(s + chunk_width, W)
        chunk = images[:, :, :, s:end]
        if chunk.shape[3] < chunk_width:
            chunk = F.pad(chunk, (0, chunk_width - chunk.shape[3]))
        chunks.append(chunk)

    stacked = torch.stack(chunks, dim=1)  # (B, N, C, H, chunk_w)
    N = stacked.shape[1]
    flat = stacked.reshape(B * N, C, H, chunk_width)
    return flat, N
