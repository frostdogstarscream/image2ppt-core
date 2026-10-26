"""The same nine source-pixel features for shadow and automatic region decisions."""
from collections import Counter
import cv2
import numpy as np


def features(crop):
    rgb = np.asarray(crop.convert('RGB'))
    height, width = rgb.shape[:2]
    quantized = (rgb // 16) * 16
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    dark = (gray < 170).astype(np.uint8)
    _, _, stats, _ = cv2.connectedComponentsWithStats(dark, 8)
    heights = [int(row[cv2.CC_STAT_HEIGHT]) for row in stats[1:]
               if row[cv2.CC_STAT_AREA] >= 4 and row[cv2.CC_STAT_WIDTH] >= 2
               and row[cv2.CC_STAT_HEIGHT] >= 3]
    dominant = Counter(map(tuple, quantized.reshape(-1, 3))).most_common(1)[0][1]
    return {'area_px': width*height, 'aspect': round(width/height, 4),
            'unique_colors': int(len(np.unique(quantized.reshape(-1, 3), axis=0))),
            'edge_density': round(float((edges > 0).mean()), 5),
            'dark_fraction': round(float(dark.mean()), 5), 'text_components': len(heights),
            'median_text_height_px': round(float(np.median(heights)), 2) if heights else 0.0,
            'microtext_fraction': round(sum(v < 10 for v in heights)/len(heights), 4) if heights else 0.0,
            'dominant_color_fraction': round(dominant/(width*height), 4)}
