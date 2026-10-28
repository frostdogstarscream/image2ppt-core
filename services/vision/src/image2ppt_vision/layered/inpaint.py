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
    hole = _pad8(mask.astype(np.float32))
    # The hole must carry no trace of what was there, or the object comes back faded.
    image = _pad8(rgb).astype(np.float32)/255.*(1.-hole[..., None])
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


def pull_push(values, hole):
    """Smooth membrane fill of ``hole`` from the known pixels only (HxW or HxWxC)."""
    img = values.astype(np.float32)
    flat = img.ndim == 2
    if flat:
        img = img[..., None]
    known = (~hole).astype(np.float32)
    stack = []
    while min(img.shape[:2]) > 2 and (known == 0).any():
        stack.append((img, known))
        size = ((img.shape[1]+1)//2, (img.shape[0]+1)//2)
        den = cv2.resize(known, size, interpolation=cv2.INTER_AREA)
        num = cv2.resize(img*known[..., None], size, interpolation=cv2.INTER_AREA).reshape(size[1], size[0], -1)
        img = num/np.maximum(den, 1e-6)[..., None]
        known = (den > 1e-6).astype(np.float32)
    for fine, fine_known in reversed(stack):
        up = cv2.resize(img, (fine.shape[1], fine.shape[0]), interpolation=cv2.INTER_LINEAR).reshape(fine.shape)
        img = np.where(fine_known[..., None] > 0, fine, up)
    return img[..., 0] if flat else img


def calm_holes(rgb, mask, *, calm_grad=18., share=.8, skip=11):
    """Hole pixels whose surroundings are plain or a smooth gradient. LaMa
    invents structure in a large hole even there (a second header, a ghost
    card); a smooth fill from the surroundings cannot. Calmness of the known
    pixels is spread into the hole, so one hole can be smooth in one part and
    textured in another. The removed object's own shadow and border, just
    outside the hole, are not its surroundings and are skipped."""
    from .segment import gradient
    calm = (gradient(rgb) <= calm_grad).astype(np.float32)
    far = cv2.dilate(mask.astype(np.uint8), np.ones((skip, skip), np.uint8)) > 0
    return mask & (pull_push(calm, far) >= share)


def smooth_fill(rgb, mask, smooth, result, *, pad=32, outlier=40., large=80):
    """Fill each calm hole from the surface it lies on. A large hole (a chart
    panel ending under a navy title bar) also borders a second colour, and
    blending both smears it in; there only known pixels near the dominant
    colour of a window around the hole are used, and the object's own shadow
    and border, just outside the hole, do not vote. A small hole (a text
    stroke, an icon) takes its nearest pixels: its window is mostly the
    page around the bar or tab it sits on."""
    from .segment import fill_holes
    H, W = mask.shape
    n, labels, stats, _ = cv2.connectedComponentsWithStats(smooth.astype(np.uint8), connectivity=8)
    for i in range(1, n):
        x, y, w, h = stats[i, :4]
        x0, y0, x1, y1 = max(0, x-pad), max(0, y-pad), min(W, x+w+pad), min(H, y+h+pad)
        hole = mask[y0:y1, x0:x1]
        part = labels[y0:y1, x0:x1] == i
        crop = rgb[y0:y1, x0:x1].astype(np.float32)
        unknown = hole
        if min(w, h) >= large:
            # Islands inside the hole (under native text on the object) belong
            # to the object, not to the surface it lies on.
            part = fill_holes(part)
            unknown = hole | part
            around = ~unknown & ~(cv2.dilate(part.astype(np.uint8), np.ones((11, 11), np.uint8)) > 0)
            if around.sum() >= 8:
                far = np.linalg.norm(crop-np.median(crop[around], 0), axis=2) > outlier
                if (~(unknown | far)).any():
                    unknown = unknown | far
        filled = np.clip(np.rint(pull_push(crop, unknown)), 0, 255).astype(np.uint8)
        result[y0:y1, x0:x1][part] = filled[part]


def inpaint(rgb, mask, model=None, *, margin=48, small_smooth=0):
    """Return (repaired RGB, method). ``mask`` is bool HxW; True means remove.
    Holes no larger than ``small_smooth`` pixels are always filled smoothly."""
    mask = mask.astype(bool)
    if not mask.any():
        return rgb.copy(), 'none'
    result = rgb.copy()
    if model is None:
        filled = cv2.inpaint(rgb, mask.astype(np.uint8)*255, 5, cv2.INPAINT_TELEA)
        result[mask] = filled[mask]
        return result, 'opencv_telea'
    smooth = calm_holes(rgb, mask)
    if small_smooth:
        # Under a lifted badge or icon LaMa redraws a faint copy of it from the
        # copies nearby; a blur of the surroundings shows nothing when dragged.
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        small = np.nonzero(np.maximum(stats[1:, cv2.CC_STAT_WIDTH], stats[1:, cv2.CC_STAT_HEIGHT]) <= small_smooth)[0]+1
        smooth |= np.isin(labels, small)
    if smooth.any():
        smooth_fill(rgb, mask, smooth, result)
        mask = mask & ~smooth
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
