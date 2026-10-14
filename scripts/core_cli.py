"""Checkout launcher; installed users can use python -m image2ppt."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from image2ppt.cli import main
raise SystemExit(main())
