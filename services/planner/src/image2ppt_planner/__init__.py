"""image2ppt-planner: turn a page image + element inventory into a validated ElementPlan/2.

The planner is the only model step in the reconstruction chain. It never invents
geometry for text: every text decision must key back to an OCR id from the
inventory produced by image2ppt-vision.
"""
__version__ = "0.1.0"

from .config import Settings  # noqa: F401
from .planner import plan_page, PlanResult  # noqa: F401

__all__ = ["Settings", "plan_page", "PlanResult", "__version__"]
