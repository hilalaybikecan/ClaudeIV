from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .metrics import compute_metrics
from .models import JVSweep

DEFAULT_HEADER_REGEX = r"^\s*(\d+)\s*[_-]\s*(\d+)\s*.*$"  # POSITION_COMPOSITION (e.g., "3_10" or "3-10")


def _substrate_from_filename(path: Path) -> Optional[int]:
    name = path.name
    for pat in (r"[Ss]ub[\s_]*(\d+)", r"[Ss]ubstrate[\s_]*(\d+)"):
        m = re.search(pat, name)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                pass
    return None


def _find_data_start(text: str) -> int:
    """Optionally skip instrument preamble until ###DATA###"""
    m = re.search(r"^\s*###DATA###\s*$", text, flags=re.MULTILINE)
    return m.end() if m else 0


def _split_numeric_fields(line: str) -> List[float]:
    parts = re.split(r"[\t,;\s]+", line.strip())
    out: List[float] = []
    for p in parts:
        if not p:
            continue
        try:
            out.append(float(p))
        except Exception:
            return []
    return out


def parse_sections_from_text(text: str, header_pattern: str) -> List[Tuple[int, int, List[List[float]]]]:
    """
    Returns list of (position, composition, rows), where rows are numeric lists:
    [FwdV, FwdI] or [FwdV, FwdI, RevV, RevI]
    Uses a user-supplied regex (header_pattern) with sensible fallbacks.
    """
    start = _find_data_start(text)
    scan = text[start:] if start > 0 else text

    # Compile user regex; fall back to default if it's invalid
    try:
        sec_re = re.compile(header_pattern, re.MULTILINE)
    except re.error:
        sec_re = re.compile(DEFAULT_HEADER_REGEX, re.MULTILINE)

    matches = list(sec_re.finditer(scan))

    # Fallbacks if too few matches
    if len(matches) <= 2:
        # Very permissive fallback: "int _ int" (avoids matching data lines by requiring integers only)
        sec_re_perm = re.compile(r"^\s*(\d{1,2})\s*[_-]\s*(\d{1,2})\s*.*$", re.MULTILINE)
        matches_alt = list(sec_re_perm.finditer(scan))
        if len(matches_alt) > len(matches):
            matches = matches_alt

    if len(matches) <= 2:
        # Accept "int whitespace int" style headers
        sec_re_ws = re.compile(r"^\s*(\d{1,2})\s+(\d{1,2})\s*.*$", re.MULTILINE)
        matches_alt2 = list(sec_re_ws.finditer(scan))
        if len(matches_alt2) > len(matches):
            matches = matches_alt2

    sections: List[Tuple[int, int, List[List[float]]]] = []
    for idx, m in enumerate(matches):
        try:
            pos = int(m.group(1))
            comp = int(m.group(2))
        except Exception:
            continue

        sec_start = m.end()
        sec_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(scan)
        body = scan[sec_start:sec_end].strip()

        rows: List[List[float]] = []
        for ln in body.splitlines():
            nums = _split_numeric_fields(ln)
            if len(nums) >= 2:
                rows.append(nums)

        if rows:
            sections.append((pos, comp, rows))

    return sections


def build_sweeps_from_file(
    path: Path,
    area_cm2: float,
    light_mw_cm2: float,
    header_pattern: str,
) -> Tuple[List[JVSweep], Dict]:
    text = path.read_text(errors="ignore")
    sections = parse_sections_from_text(text, header_pattern)
    sweeps: List[JVSweep] = []
    diag = {"file": str(path), "sections_found": len(sections), "examples": []}

    substrate_id = _substrate_from_filename(path) or -1

    for pos, comp, rows in sections:
        pixel_id = (comp - 1) * 6 + pos
        fV: List[float] = []
        fI: List[float] = []
        rV: List[float] = []
        rI: List[float] = []
        for nums in rows:
            if len(nums) >= 2:
                fV.append(nums[0])
                fI.append(nums[1])
            if len(nums) >= 4:
                rV.append(nums[2])
                rI.append(nums[3])

        if len(fV) >= 2:
            fv = np.asarray(fV, float)
            fi = np.asarray(fI, float)
            Voc, Jsc_mAcm2, FF_pct, PCE_pct = compute_metrics(fv, fi, area_cm2, light_mw_cm2)
            sweeps.append(
                JVSweep(
                    substrate_id,
                    pixel_id,
                    comp,
                    pos,
                    "forward",
                    fv,
                    fi,
                    area_cm2,
                    light_mw_cm2,
                    Voc,
                    Jsc_mAcm2,
                    FF_pct,
                    PCE_pct,
                )
            )
        if len(rV) >= 2:
            rv = np.asarray(rV, float)
            ri = np.asarray(rI, float)
            Voc, Jsc_mAcm2, FF_pct, PCE_pct = compute_metrics(rv, ri, area_cm2, light_mw_cm2)
            sweeps.append(
                JVSweep(
                    substrate_id,
                    pixel_id,
                    comp,
                    pos,
                    "reverse",
                    rv,
                    ri,
                    area_cm2,
                    light_mw_cm2,
                    Voc,
                    Jsc_mAcm2,
                    FF_pct,
                    PCE_pct,
                )
            )

        if len(diag["examples"]) < 8:
            diag["examples"].append({"pos": pos, "comp": comp, "n_fwd": len(fV), "n_rev": len(rV)})

    return sweeps, diag
