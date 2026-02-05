from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class JVSweep:
    substrate: int
    pixel_id: int
    composition_index: int
    position_in_composition: int
    direction: str  # "forward" or "reverse"
    voltage: np.ndarray  # V
    current_A: np.ndarray  # A
    area_cm2: float
    light_mw_cm2: float
    Voc: Optional[float] = None
    Jsc_mAcm2: Optional[float] = None  # positive, mA/cm2
    FF_pct: Optional[float] = None     # 0-100 %
    PCE_pct: Optional[float] = None    # 0-100 %
    Rsc_ohmcm2: Optional[float] = None  # shunt resistance near 0V (ohm*cm^2)
    # Experimental conditions (will be populated from Excel)
    sweep_id: Optional[int] = None
    condition_name: Optional[str] = None
