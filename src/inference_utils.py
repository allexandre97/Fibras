import numpy as np
import torch
import torch.nn.functional as F


def pad_to_multiple_2d(tensor: torch.Tensor, multiple: int = 16):
    h, w = tensor.shape[-2:]
    pad_h = (multiple - (h % multiple)) % multiple
    pad_w = (multiple - (w % multiple)) % multiple
    return F.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect"), (h, w)


def normalize_image_percentile(image: np.ndarray, low: float = 0.5, high: float = 99.9) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    if image.size == 0:
        return image
    if float(image.max()) <= 1.0 and float(image.min()) >= 0.0:
        return image
    p_low, p_high = np.percentile(image, (low, high))
    return np.clip((image - p_low) / (p_high - p_low + 1e-8), 0.0, 1.0).astype(np.float32)


def tile_starts(size: int, tile_size: int, overlap: int):
    if tile_size <= 0:
        raise ValueError("tile_size must be greater than 0.")
    if overlap < 0 or overlap >= tile_size:
        raise ValueError("overlap must be in the interval [0, tile_size).")
    if size <= tile_size:
        return [0]

    stride = tile_size - overlap
    starts = list(range(0, size - tile_size + 1, stride))
    final_start = size - tile_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def blend_window_2d(height: int, width: int, overlap: int) -> np.ndarray:
    def axis_window(length: int):
        window = np.ones(length, dtype=np.float32)
        ramp = min(max(overlap // 2, 0), length // 2)
        if ramp > 0:
            values = np.linspace(0.05, 1.0, ramp, dtype=np.float32)
            window[:ramp] = values
            window[-ramp:] = values[::-1]
        return window

    return axis_window(height)[:, None] * axis_window(width)[None, :]


def predict_tiled_2d(
    model: torch.nn.Module,
    image: np.ndarray,
    device: torch.device,
    tile_size: int = 512,
    overlap: int = 128,
    output_channels: int = 4,
    multiple: int = 16,
    use_amp: bool = True,
) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D image, got shape {image.shape}.")

    height, width = image.shape
    y_starts = tile_starts(height, tile_size, overlap)
    x_starts = tile_starts(width, tile_size, overlap)
    output = np.zeros((output_channels, height, width), dtype=np.float32)
    weights = np.zeros((height, width), dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for y0 in y_starts:
            for x0 in x_starts:
                y1 = min(y0 + tile_size, height)
                x1 = min(x0 + tile_size, width)
                patch = image[y0:y1, x0:x1]
                tensor = torch.from_numpy(patch).to(device=device, dtype=torch.float32)[None, None]
                tensor, original_shape = pad_to_multiple_2d(tensor, multiple=multiple)

                if use_amp and device.type == "cuda":
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        pred = model(tensor)
                else:
                    pred = model(tensor)

                ph, pw = original_shape
                pred_np = pred[0, :, :ph, :pw].detach().cpu().float().numpy()
                window = blend_window_2d(ph, pw, overlap)
                output[:, y0:y1, x0:x1] += pred_np * window[None, :, :]
                weights[y0:y1, x0:x1] += window

    return output / np.maximum(weights[None, :, :], 1e-8)
