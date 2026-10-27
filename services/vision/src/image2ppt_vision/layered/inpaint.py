"""Background repair: LaMa for real holes, Telea only when LaMa is unavailable.

Pixels outside the mask are always returned unchanged; the inpainter's output is
composited back only inside the mask.
"""
import cv2
import numpy as np


def _pad8(array):
    h, w = array.shape[:2]
    ph, pw = (-h) % 8, (-w) % 8
    pad = ((0, ph), (0, pw)) + ((0, 0),)*(array.ndim-2)
    return np.pad(array, pad, mode='symmetric') if ph or pw else array


def lama(rgb, mask, model):
    import torch
    net, device = model
    h, w = mask.shape
    image = _pad8(rgb).astype(np.float32)/255.
    hole = _pad8(mask.astype(np.float32))
    with torch.inference_mode():
        a = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).to(device)
        b = torch.from_numpy((hole > 0).astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
        out = net(a, b)[0].permute(1, 2, 0).float().cpu().numpy()
    return np.clip(out*255, 0, 255).astype(np.uint8)[:h, :w]


def _regions(mask, margin):
    """Crop windows around clustered holes so LaMa sees local context at native scale."""
    grouped = cv2.dilate(mask.astype(np.uint8), np.ones((25, 25), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(grouped, 8)
    H, W = mask.shape
    boxes = []
    for i in range(1, n):
        x, y, w, h, _ = stats[i]
        m = max(margin, int(.35*max(w, h)))
        boxes.append([max(0, x-m), max(0, y-m), min(W, x+w+m), min(H, y+h+m)])
    merged = True
    while merged:
        merged = False
        for i in range(len(boxes)):
            for j in range(i+1, len(boxes)):
                a, b = boxes[i], boxes[j]
                if a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]:
                    boxes[i] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
                    boxes.pop(j); merged = True
                    break
            if merged:
                break
    return boxes


def inpaint(rgb, mask, model=None, *, margin=48):
    """Return (repaired RGB, method). ``mask`` is bool HxW; True means remove."""
    mask = mask.astype(bool)
    if not mask.any():
        return rgb.copy(), 'none'
    result = rgb.copy()
    if model is None:
        filled = cv2.inpaint(rgb, mask.astype(np.uint8)*255, 5, cv2.INPAINT_TELEA)
        result[mask] = filled[mask]
        return result, 'opencv_telea'
    for x0, y0, x1, y1 in _regions(mask, margin):
        window = mask[y0:y1, x0:x1]
        if not window.any():
            continue
        filled = lama(result[y0:y1, x0:x1], window, model)
        region = result[y0:y1, x0:x1]
        region[window] = filled[window]
    return result, 'lama'


def solid_color(rgb, tolerance=3., visible=12., max_fraction=.0005):
    """Six-digit colour if the image is visually uniform, else None.

    Small leftovers matter: any visible deviation beyond a few hundred pixels
    keeps the repaired image instead of flattening it to one colour.
    """
    median = np.median(rgb.reshape(-1, 3), 0)
    deviation = np.abs(rgb.reshape(-1, 3).astype(np.float32)-median).max(1)
    if (float(np.percentile(deviation, 99)) <= tolerance and
            float((deviation > visible).mean()) <= max_fraction):
        return ''.join(f'{int(v):02X}' for v in median)
    return None
