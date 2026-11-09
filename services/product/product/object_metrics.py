"""Mask measurements require an explicit reference, never a VLM self-score.

Measurements against provisional references must retain reference_verified=False.
They do not establish semantic acceptance, even when their numerical values are 1.
"""
import numpy as np
import cv2


def skeleton(mask):
    current=mask.astype('uint8');result=np.zeros_like(current);element=cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    while current.any():
        eroded=cv2.erode(current,element,borderType=cv2.BORDER_CONSTANT,borderValue=0)
        opened=cv2.dilate(eroded,element)
        result|=current & ~opened;current=eroded
    return result.astype(bool)


def measure(mask,reference,protected_text,*,reference_verified=False):
    mask=np.asarray(mask,bool);reference=np.asarray(reference,bool);text=np.asarray(protected_text,bool)
    if mask.shape!=reference.shape or text.shape!=mask.shape or mask.ndim!=2:raise ValueError('mask canvas mismatch')
    if not reference.any():raise ValueError('empty reference cannot establish completeness')
    center=skeleton(reference);intersection=int((mask & reference).sum());union=int((mask | reference).sum())
    return {'reference_verified':bool(reference_verified),'semantic_acceptance':False,
        'reference_pixel_recall':intersection/int(reference.sum()),'iou':intersection/max(1,union),
        'thin_line_skeleton_recall':int((mask & center).sum())/max(1,int(center.sum())),
        'text_leak_pixels':int((mask & text).sum()),'extra_pixels':int((mask & ~reference).sum()),
        'missing_pixels':int((reference & ~mask).sum())}
