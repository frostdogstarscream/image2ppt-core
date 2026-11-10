"""Re-export the single host lease shared with LayerD and SAM."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from resource_lease import lease,CAPACITY
