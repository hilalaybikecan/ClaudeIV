from __future__ import annotations

from typing import Optional

import numpy as np


def _interp_x_at_y_zero(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    if len(x) < 2:
        return None
    s = np.sign(y)
    crossings = np.where(s[:-1] * s[1:] <= 0)[0]
    if len(crossings) == 0:
        return None
    idx = min(crossings, key=lambda k: abs(y[k]))
    x0, x1 = x[idx], x[idx + 1]
    y0, y1 = y[idx], y[idx + 1]
    if x1 == x0 or y1 == y0:
        return float(x0)
    t = -y0 / (y1 - y0)
    return float(x0 + t * (x1 - x0))


def _interp_y_at_x_zero(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    if len(x) < 2:
        return None
    s = np.sign(x)
    crossings = np.where(s[:-1] * s[1:] <= 0)[0]
    if len(crossings) == 0:
        return None
    idx = crossings[0]
    x0, x1 = x[idx], x[idx + 1]
    y0, y1 = y[idx], y[idx + 1]
    if x1 == x0:
        return float(y0)
    t = -x0 / (x1 - x0)
    return float(y0 + t * (y1 - y0))


def compute_metrics(voltage: np.ndarray, current_A: np.ndarray, area_cm2: float, light_mw_cm2: float):
    """
    Return:
      Voc (V),
      Jsc_mAcm2 (positive, mA/cm2),
      FF_pct (0-100),
      PCE_pct (0-100, typical 0-15 for your devices)
    """
    if len(voltage) < 2:
        return None, None, None, None

    order = np.argsort(voltage)
    V = voltage[order]
    I = current_A[order]

    # Current density A/cm2
    J_Acm2 = I / max(area_cm2, 1e-12)

    # Voc at J=0 (A/cm2); Jsc at V=0 (A/cm2)
    Voc = _interp_x_at_y_zero(V, J_Acm2)
    Jsc_Acm2 = _interp_y_at_x_zero(V, J_Acm2)

    # Output power density: device delivers power = -V * J (mW/cm2)
    Pout_mWcm2 = -V * J_Acm2 * 1e3  # note minus sign for quadrant IV
    if len(Pout_mWcm2) == 0:
        Jsc_mAcm2 = None if Jsc_Acm2 is None else abs(Jsc_Acm2) * 1e3
        return Voc, Jsc_mAcm2, None, None

    idx_mpp = int(np.nanargmax(Pout_mWcm2))
    Vmpp = V[idx_mpp]
    Jmpp_Acm2 = J_Acm2[idx_mpp]
    Pmpp_out = Pout_mWcm2[idx_mpp]  # positive

    # FF in percent; use magnitudes
    FF_pct = None
    if Voc is not None and Jsc_Acm2 is not None and abs(Voc) > 1e-12 and abs(Jsc_Acm2) > 1e-12:
        FF = (abs(Vmpp) * abs(Jmpp_Acm2)) / (abs(Voc) * abs(Jsc_Acm2))
        FF_pct = 100.0 * FF

    # PCE (%) = Pout_mpp / Pin * 100 * 1.5 (sun intensity correction)
    PCE_pct = None
    if light_mw_cm2 > 0:
        PCE_pct = (Pmpp_out / light_mw_cm2) * 100.0 * 1.5

    # Report Jsc as positive mA/cm2 * 1.5 (sun intensity correction)
    Jsc_mAcm2 = None if Jsc_Acm2 is None else abs(Jsc_Acm2) * 1e3 * 1.5

    return Voc, Jsc_mAcm2, FF_pct, PCE_pct
