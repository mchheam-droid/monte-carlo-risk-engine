import os
import sys

# Ensure Windows finds MinGW/UCRT runtime DLLs if needed
msys_bin = r"C:\msys64\ucrt64\bin"
if os.path.isdir(msys_bin) and hasattr(os, "add_dll_directory"):
    try:
        os.add_dll_directory(msys_bin)
    except Exception:
        pass

from .portfolio import Portfolio
from .risk_engine import RiskMetrics, RiskReport

__all__ = ["Portfolio", "RiskMetrics", "RiskReport"]