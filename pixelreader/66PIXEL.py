# perovskite_jv_gui_with_regex.py
from __future__ import annotations

import re
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from mpl_toolkits.mplot3d import Axes3D

# ---------------------------- Config ----------------------------
DEFAULT_HEADER_REGEX = r"^\s*(\d+)\s*[_-]\s*(\d+)\s*.*$"  # POSITION_COMPOSITION (e.g., "3_10" or "3-10")

# ---------------------------- Data structures ----------------------------
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
    Jsc_mAcm2: Optional[float] = None  # positive, mA/cm²
    FF_pct: Optional[float] = None     # 0–100 %
    PCE_pct: Optional[float] = None    # 0–100 %
    # Experimental conditions (will be populated from Excel)
    sweep_id: Optional[int] = None
    condition_name: Optional[str] = None

# ---------------------------- Parsing helpers ----------------------------
def _substrate_from_filename(path: Path) -> Optional[int]:
    name = path.name
    for pat in (r"[Ss]ub\s*_?\s*(\d+)", r"[Ss]ubstrate\s*_?\s*(\d+)"):
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

def build_sweeps_from_file(path: Path, area_cm2: float, light_mw_cm2: float, header_pattern: str) -> Tuple[List[JVSweep], Dict]:
    text = path.read_text(errors="ignore")
    sections = parse_sections_from_text(text, header_pattern)
    sweeps: List[JVSweep] = []
    diag = {"file": str(path), "sections_found": len(sections), "examples": []}

    substrate_id = _substrate_from_filename(path) or -1

    for pos, comp, rows in sections:
        pixel_id = (comp - 1) * 6 + pos
        fV: List[float] = []; fI: List[float] = []; rV: List[float] = []; rI: List[float] = []
        for nums in rows:
            if len(nums) >= 2:
                fV.append(nums[0]); fI.append(nums[1])
            if len(nums) >= 4:
                rV.append(nums[2]); rI.append(nums[3])

        if len(fV) >= 2:
            fv = np.asarray(fV, float); fi = np.asarray(fI, float)
            Voc, Jsc_mAcm2, FF_pct, PCE_pct = compute_metrics(fv, fi, area_cm2, light_mw_cm2)
            sweeps.append(JVSweep(substrate_id, pixel_id, comp, pos, "forward", fv, fi, area_cm2, light_mw_cm2, Voc, Jsc_mAcm2, FF_pct, PCE_pct))
        if len(rV) >= 2:
            rv = np.asarray(rV, float); ri = np.asarray(rI, float)
            Voc, Jsc_mAcm2, FF_pct, PCE_pct = compute_metrics(rv, ri, area_cm2, light_mw_cm2)
            sweeps.append(JVSweep(substrate_id, pixel_id, comp, pos, "reverse", rv, ri, area_cm2, light_mw_cm2, Voc, Jsc_mAcm2, FF_pct, PCE_pct))

        if len(diag["examples"]) < 8:
            diag["examples"].append({"pos": pos, "comp": comp, "n_fwd": len(fV), "n_rev": len(rV)})

    return sweeps, diag

# ---------------------------- Metrics ----------------------------
def _interp_x_at_y_zero(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    if len(x) < 2: return None
    s = np.sign(y)
    crossings = np.where(s[:-1] * s[1:] <= 0)[0]
    if len(crossings) == 0: return None
    idx = min(crossings, key=lambda k: abs(y[k]))
    x0, x1 = x[idx], x[idx + 1]
    y0, y1 = y[idx], y[idx + 1]
    if x1 == x0 or y1 == y0: return float(x0)
    t = -y0 / (y1 - y0)
    return float(x0 + t * (x1 - x0))

def _interp_y_at_x_zero(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    if len(x) < 2: return None
    s = np.sign(x)
    crossings = np.where(s[:-1] * s[1:] <= 0)[0]
    if len(crossings) == 0: return None
    idx = crossings[0]
    x0, x1 = x[idx], x[idx + 1]
    y0, y1 = y[idx], y[idx + 1]
    if x1 == x0: return float(y0)
    t = -x0 / (x1 - x0)
    return float(y0 + t * (y1 - y0))

def compute_metrics(voltage: np.ndarray, current_A: np.ndarray, area_cm2: float, light_mw_cm2: float):
    """
    Return:
      Voc (V),
      Jsc_mAcm2 (positive, mA/cm²),
      FF_pct (0–100),
      PCE_pct (0–100, typical 0–15 for your devices)
    """
    if len(voltage) < 2:
        return None, None, None, None

    order = np.argsort(voltage)
    V = voltage[order]
    I = current_A[order]

    # Current density A/cm²
    J_Acm2 = I / max(area_cm2, 1e-12)

    # Voc at J=0 (A/cm²); Jsc at V=0 (A/cm²)
    Voc = _interp_x_at_y_zero(V, J_Acm2)
    Jsc_Acm2 = _interp_y_at_x_zero(V, J_Acm2)

    # Output power density: device delivers power = -V * J (mW/cm²)
    Pout_mWcm2 = - V * J_Acm2 * 1e3  # note minus sign → positive in quadrant IV
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

    # PCE (%) = Pout_mpp / Pin × 100
    PCE_pct = None
    if light_mw_cm2 > 0:
        PCE_pct = (Pmpp_out / light_mw_cm2) * 100.0

    # Report Jsc as positive mA/cm²
    Jsc_mAcm2 = None if Jsc_Acm2 is None else abs(Jsc_Acm2) * 1e3

    return Voc, Jsc_mAcm2, FF_pct, PCE_pct

# ---------------------------- Group mapping ----------------------------
def comp_to_group(comp: int) -> int:
    if comp in (1, 2): return 1
    if comp in (10, 11): return 9
    return comp - 1  # 3..9 -> 2..8

# ---------------------------- Experimental conditions ----------------------------
def load_experimental_conditions(excel_path: Optional[str] = None) -> Optional[pd.DataFrame]:
    """Load experimental conditions from Excel file"""
    if excel_path is None:
        # Try to find experiment sheets in the same directory
        excel_path = Path(__file__).parent / "experiment sheets.xlsx"
    
    excel_path = Path(excel_path)
    print(f"Looking for Excel file at: {excel_path}")
    print(f"File exists: {excel_path.exists()}")
    
    if not excel_path.exists():
        print(f"Excel file not found: {excel_path}")
        return None
    
    try:
        # Check available sheets first
        xlsx = pd.ExcelFile(excel_path)
        print(f"Available sheets: {xlsx.sheet_names}")
        
        # Load the ROSIE sheet which contains the actual experimental parameters
        if 'ROSIE' not in xlsx.sheet_names:
            print("'ROSIE' sheet not found in Excel file")
            # Fallback to Runsheet if ROSIE not available
            if 'Runsheet' in xlsx.sheet_names:
                print("Falling back to 'Runsheet' sheet")
                conditions_df = pd.read_excel(excel_path, sheet_name='Runsheet')
                print(f"Successfully loaded {len(conditions_df)} rows from Runsheet")
                return conditions_df
            return None
            
        conditions_df = pd.read_excel(excel_path, sheet_name='ROSIE')
        print(f"Successfully loaded {len(conditions_df)} rows from ROSIE sheet")
        return conditions_df
    except Exception as e:
        print(f"Error loading experimental conditions: {e}")
        import traceback
        traceback.print_exc()
        return None

def analyze_sweep_parameters(conditions_df: pd.DataFrame) -> Dict[int, Dict]:
    """Analyze what parameters vary within each sweep"""
    sweep_analysis = {}
    
    # Check which sheet format we're using by examining columns
    if 'excess PbI2' in conditions_df.columns:
        # ROSIE sheet format - correct experimental parameters
        param_columns = [
            'excess PbI2', 
            'with Thiourea', 
            'with FABF4',
            'FA30Cs70PbI3 DMAI (dmai=high dmpu=low)',
            'FA30Cs70PbI3 DMAI (dmai=low dmpu=low)', 
            'FA30Cs70PbI3 DMAI (dmai=low dmpu=high)', 
            'FA30Cs70PbI3 DMAI (dmai=high dmpu=high)', 
            'FA30CS70PbI3 MAI'
        ]
    else:
        # Runsheet format - fallback
        param_columns = ['PbI2 (M)', 'CsI (M)', 'FAI (M)', 'DMAI (M)', 'DMPU (M)', 'Thiourea (M)', 'FABF4 (M)', 'MAI (M)']
    
    for sweep_id in conditions_df['Sweep'].unique():
        sweep_data = conditions_df[conditions_df['Sweep'] == sweep_id]
        
        varying_params = []
        constant_params = []
        param_info = {}
        
        for col in param_columns:
            if col in sweep_data.columns:
                # Get unique values, filter out zeros, and check for actual variation
                all_vals = sweep_data[col].dropna()
                nonzero_vals = all_vals[all_vals > 0]
                unique_vals = np.unique(nonzero_vals)  # This removes duplicates properly
                
                if len(unique_vals) > 1:
                    varying_params.append(col)
                    param_info[col] = {
                        'type': 'varying',
                        'values': sorted(unique_vals),
                        'range': [min(unique_vals), max(unique_vals)]
                    }
                elif len(unique_vals) == 1:
                    constant_params.append(col)
                    param_info[col] = {
                        'type': 'constant',
                        'value': unique_vals[0]
                    }
        
        # Identify main varying parameters by their range of variation
        # Filter out parameters with very small variations (likely experimental noise)
        main_varying_params = []
        for param in varying_params:
            param_range = param_info[param]['range']
            variation_range = param_range[1] - param_range[0]
            relative_variation = variation_range / param_range[0] if param_range[0] > 0 else 0
            
            # Different thresholds for different parameter types
            if 'excess PbI2' in param or 'with Thiourea' in param or 'with FABF4' in param:
                # These are additive parameters, smaller absolute changes are significant
                if variation_range > 0.005 or relative_variation > 0.1:  # 5mM or 10%
                    main_varying_params.append(param)
            else:
                # DMAI/MAI composition parameters - larger changes expected
                if variation_range > 0.01 or relative_variation > 0.05:  # 10mM or 5%
                    main_varying_params.append(param)
        
        # Helper function to create clean parameter names
        def clean_param_name(param_name):
            if param_name == 'excess PbI2':
                return 'Excess PbI2'
            elif param_name == 'with Thiourea':
                return 'Thiourea'
            elif param_name == 'with FABF4':
                return 'FABF4'
            elif 'DMAI' in param_name and 'high' in param_name and 'low' in param_name:
                return 'DMAI (high/low DMPU)'
            elif 'DMAI' in param_name and 'low' in param_name and 'low' in param_name:
                return 'DMAI (low/low DMPU)'
            elif 'DMAI' in param_name and 'low' in param_name and 'high' in param_name:
                return 'DMAI (low/high DMPU)'
            elif 'DMAI' in param_name and 'high' in param_name and 'high' in param_name:
                return 'DMAI (high/high DMPU)'
            elif 'MAI' in param_name:
                return 'MAI'
            else:
                return param_name.replace(' (M)', '')
        
        # Create descriptive name based on main varying parameters
        if len(main_varying_params) == 1:
            main_param = clean_param_name(main_varying_params[0])
            param_range = param_info[main_varying_params[0]]['range']
            if 'excess' in main_param.lower() or 'thiourea' in main_param.lower() or 'fabf4' in main_param.lower():
                desc = f"{main_param} variation ({param_range[0]:.3f}-{param_range[1]:.3f})"
            else:
                desc = f"{main_param} variation ({param_range[0]:.3f}-{param_range[1]:.3f}M)"
        elif len(main_varying_params) == 2:
            param1 = clean_param_name(main_varying_params[0])
            param2 = clean_param_name(main_varying_params[1])
            desc = f"{param1} vs {param2} (2D)"
        elif len(main_varying_params) > 2:
            param_names = [clean_param_name(p) for p in main_varying_params]
            if len(param_names) <= 3:
                desc = f"{'/'.join(param_names)} (multi-parameter)"
            else:
                desc = f"Multi-parameter ({len(main_varying_params)} variables)"
        elif len(varying_params) > 0:  # Minor variations only
            # Find the parameter with largest variation among the "varying" ones
            max_var_param = max(varying_params, key=lambda p: param_info[p]['range'][1] - param_info[p]['range'][0])
            main_param = clean_param_name(max_var_param)
            desc = f"{main_param} (minor variations)"
        else:
            # Find the main constant parameter
            main_constant = None
            for col in constant_params:
                if param_info[col]['value'] > 0.01:  # Significant concentration
                    main_constant = clean_param_name(col)
                    break
            desc = f"Control ({main_constant or 'base'} composition)"
        
        sweep_analysis[sweep_id] = {
            'description': desc,
            'varying_params': varying_params,
            'constant_params': constant_params,
            'param_info': param_info,
            'substrates': sorted(sweep_data['Substrate'].unique()),
            'n_entries': len(sweep_data)
        }
    
    return sweep_analysis

def map_sweeps_to_conditions(sweeps: List[JVSweep], conditions_df: Optional[pd.DataFrame]) -> List[JVSweep]:
    """Map JV sweeps to experimental conditions based on substrate"""
    if conditions_df is None:
        return sweeps
    
    # Create a mapping from substrate to sweep conditions
    substrate_to_conditions = {}
    for _, row in conditions_df.iterrows():
        substrate = int(row['Substrate'])
        sweep_id = int(row['Sweep'])
        
        # Create a condition name based on the sheet format
        condition_parts = []
        if 'excess PbI2' in conditions_df.columns:
            # ROSIE sheet format
            experimental_params = ['excess PbI2', 'with Thiourea', 'with FABF4']
            for col in experimental_params:
                if col in conditions_df.columns and row[col] > 0:
                    param_name = col.replace('excess ', '').replace('with ', '')
                    concentration = f"{row[col]:.3f}"
                    condition_parts.append(f"{param_name}:{concentration}")
        else:
            # Runsheet format (fallback)
            for col in conditions_df.columns:
                if col.endswith('(M)') and row[col] > 0:
                    chemical = col.replace(' (M)', '')
                    concentration = f"{row[col]:.3f}M"
                    condition_parts.append(f"{chemical}:{concentration}")
        
        condition_name = f"Sweep_{sweep_id}" + (f" ({', '.join(condition_parts)})" if condition_parts else "")
        substrate_to_conditions[substrate] = {'sweep_id': sweep_id, 'condition_name': condition_name}
    
    # Update sweeps with condition information
    for sweep in sweeps:
        if sweep.substrate in substrate_to_conditions:
            conditions = substrate_to_conditions[sweep.substrate]
            sweep.sweep_id = conditions['sweep_id']
            sweep.condition_name = conditions['condition_name']
    
    return sweeps

# ---------------------------- GUI ----------------------------
class JVApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Perovskite JV Analyzer")
        self.geometry("1500x900")

        # Parameters
        self.area_cm2 = tk.DoubleVar(value=0.04)
        self.light_mw_cm2 = tk.DoubleVar(value=100.0)
        self.include_forward = tk.BooleanVar(value=True)
        self.include_reverse = tk.BooleanVar(value=True)
        self.metric_choice = tk.StringVar(value="PCE_pct")
        self.combine_substrates = tk.BooleanVar(value=True)
        self.combine_fr = tk.BooleanVar(value=True)
        self.grouping_mode = tk.StringVar(value="11 compositions")
        self.expand_substrate_axis = tk.BooleanVar(value=True)

        # Filters
        self.min_voc = tk.DoubleVar(value="")
        self.max_voc = tk.DoubleVar(value="")
        self.min_pce = tk.DoubleVar(value="")
        self.max_pce = tk.DoubleVar(value="")
        self.min_jsc = tk.DoubleVar(value="")
        self.max_jsc = tk.DoubleVar(value="")
        self.min_ff  = tk.DoubleVar(value="")
        self.max_ff  = tk.DoubleVar(value="")

        # Header regex (restored)
        self.header_pattern_var = tk.StringVar(value=DEFAULT_HEADER_REGEX)

        # Data
        self.data: List[JVSweep] = []
        self.df: Optional[pd.DataFrame] = None
        self.df_with_flags: Optional[pd.DataFrame] = None
        self._last_paths: List[Path] = []
        self._last_diags: List[Dict] = []
        
        # Table sorting state
        self._sort_column: Optional[str] = None
        self._sort_reverse: bool = False
        
        # Experimental conditions
        self.conditions_df: Optional[pd.DataFrame] = None
        self.excel_path_var = tk.StringVar(value="experiment sheets.xlsx")
        self.sweep_analysis: Dict[int, Dict] = {}
        self.selected_sweep_id = tk.IntVar(value=-1)  # -1 means all sweeps

        # Matplotlib / colorbar tracking
        self.fig = Figure(figsize=(8, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self._cbar = None
        
        # Sweep analysis matplotlib
        self.sweep_fig = Figure(figsize=(8, 5), dpi=100)
        self.sweep_ax = self.sweep_fig.add_subplot(111)
        self.sweep_cbar = None

        self._build_ui()

    # -------------------- UI layout --------------------
    def _build_ui(self):
        self.columnconfigure(0, weight=0); self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self)
        self.notebook.grid(row=0, column=0, columnspan=2, sticky="nsew")
        
        # Tab 1: Original composition analysis
        self.comp_frame = ttk.Frame(self.notebook, padding=8)
        self.notebook.add(self.comp_frame, text="Composition Analysis")
        
        # Tab 2: Sweep-based analysis
        self.sweep_frame = ttk.Frame(self.notebook, padding=8)
        self.notebook.add(self.sweep_frame, text="Sweep Analysis")
        
        self._build_composition_tab()
        self._build_sweep_tab()
        
    def _build_composition_tab(self):
        self.comp_frame.columnconfigure(0, weight=0); self.comp_frame.columnconfigure(1, weight=1)
        self.comp_frame.rowconfigure(0, weight=1)

        side = ttk.Frame(self.comp_frame, padding=8); side.grid(row=0, column=0, sticky="ns")

        ttk.Label(side, text="Data files").grid(row=0, column=0, sticky="w")
        ttk.Button(side, text="Load file", command=self.load_file).grid(row=1, column=0, sticky="ew", pady=2)
        ttk.Button(side, text="Load folder", command=self.load_folder).grid(row=2, column=0, sticky="ew", pady=2)

        ttk.Separator(side).grid(row=3, column=0, sticky="ew", pady=6)
        ttk.Label(side, text="Parameters").grid(row=4, column=0, sticky="w")
        pfrm = ttk.Frame(side); pfrm.grid(row=5, column=0, sticky="ew")
        ttk.Label(pfrm, text="Area (cm²):").grid(row=0, column=0, sticky="w")
        ttk.Entry(pfrm, textvariable=self.area_cm2, width=10).grid(row=0, column=1, sticky="w", padx=4)
        ttk.Label(pfrm, text="Pin (mW/cm²):").grid(row=1, column=0, sticky="w")
        ttk.Entry(pfrm, textvariable=self.light_mw_cm2, width=10).grid(row=1, column=1, sticky="w", padx=4)
        ttk.Button(pfrm, text="Recompute metrics", command=self.recompute_metrics).grid(row=2, column=0, columnspan=2, sticky="ew", pady=4)

        ttk.Separator(side).grid(row=6, column=0, sticky="ew", pady=6)
        ttk.Label(side, text="Include sweeps").grid(row=7, column=0, sticky="w")
        ttk.Checkbutton(side, text="Forward", variable=self.include_forward, command=self.refresh_plots).grid(row=8, column=0, sticky="w")
        ttk.Checkbutton(side, text="Reverse", variable=self.include_reverse, command=self.refresh_plots).grid(row=9, column=0, sticky="w")

        ttk.Separator(side).grid(row=10, column=0, sticky="ew", pady=6)
        ttk.Label(side, text="Metric").grid(row=11, column=0, sticky="w")
        metric_cb = ttk.Combobox(side, textvariable=self.metric_choice, values=["Voc", "Jsc_mAcm2", "FF_pct", "PCE_pct"], state="readonly", width=12)
        metric_cb.grid(row=12, column=0, sticky="ew"); metric_cb.bind("<<ComboboxSelected>>", lambda e: self.refresh_plots())

        ttk.Separator(side).grid(row=13, column=0, sticky="ew", pady=6)
        ttk.Label(side, text="Substrate / Grouping").grid(row=14, column=0, sticky="w")
        self.substrate_cb = ttk.Combobox(side, values=["All"], state="readonly")
        self.substrate_cb.grid(row=15, column=0, sticky="ew", pady=2); self.substrate_cb.bind("<<ComboboxSelected>>", lambda e: self.refresh_plots())
        ttk.Checkbutton(side, text="Combine substrates", variable=self.combine_substrates, command=self.refresh_plots).grid(row=16, column=0, sticky="w")
        ttk.Checkbutton(side, text="Combine F & R", variable=self.combine_fr, command=self.refresh_plots).grid(row=17, column=0, sticky="w")
        ttk.Label(side, text="Grouping").grid(row=18, column=0, sticky="w", pady=(6, 0))
        grp_cb = ttk.Combobox(side, textvariable=self.grouping_mode, values=["11 compositions", "9 groups"], state="readonly", width=18)
        grp_cb.grid(row=19, column=0, sticky="ew", pady=2); grp_cb.bind("<<ComboboxSelected>>", lambda e: self.refresh_plots())
        ttk.Checkbutton(side, text="Expand x-axis by substrate", variable=self.expand_substrate_axis, command=self.refresh_plots).grid(row=20, column=0, sticky="w", pady=(2, 6))

        ttk.Separator(side).grid(row=21, column=0, sticky="ew", pady=6)
        ttk.Label(side, text="Thresholds").grid(row=22, column=0, sticky="w")
        tfrm = ttk.Frame(side); tfrm.grid(row=23, column=0, sticky="ew")
        ttk.Label(tfrm, text="Min Voc (V):").grid(row=0, column=0, sticky="w"); ttk.Entry(tfrm, textvariable=self.min_voc, width=8).grid(row=0, column=1, sticky="w", padx=4)
        ttk.Label(tfrm, text="Max Voc (V):").grid(row=0, column=2, sticky="w"); ttk.Entry(tfrm, textvariable=self.max_voc, width=8).grid(row=0, column=3, sticky="w", padx=4)
        ttk.Label(tfrm, text="Min PCE (%):").grid(row=1, column=0, sticky="w"); ttk.Entry(tfrm, textvariable=self.min_pce, width=8).grid(row=1, column=1, sticky="w", padx=4)
        ttk.Label(tfrm, text="Max PCE (%):").grid(row=1, column=2, sticky="w"); ttk.Entry(tfrm, textvariable=self.max_pce, width=8).grid(row=1, column=3, sticky="w", padx=4)
        ttk.Label(tfrm, text="Min Jsc (mA/cm²):").grid(row=2, column=0, sticky="w"); ttk.Entry(tfrm, textvariable=self.min_jsc, width=8).grid(row=2, column=1, sticky="w", padx=4)
        ttk.Label(tfrm, text="Max Jsc (mA/cm²):").grid(row=2, column=2, sticky="w"); ttk.Entry(tfrm, textvariable=self.max_jsc, width=8).grid(row=2, column=3, sticky="w", padx=4)
        ttk.Label(tfrm, text="Min FF (%):").grid(row=3, column=0, sticky="w"); ttk.Entry(tfrm, textvariable=self.min_ff, width=8).grid(row=3, column=1, sticky="w", padx=4)
        ttk.Label(tfrm, text="Max FF (%):").grid(row=3, column=2, sticky="w"); ttk.Entry(tfrm, textvariable=self.max_ff, width=8).grid(row=3, column=3, sticky="w", padx=4)
        ttk.Button(tfrm, text="Apply thresholds", command=self.apply_thresholds).grid(row=4, column=0, columnspan=2, sticky="ew", pady=4)
        ttk.Button(tfrm, text="Reset filters & include all", command=self.reset_filters_include_all).grid(row=4, column=2, columnspan=2, sticky="ew", pady=4)

        ttk.Separator(side).grid(row=24, column=0, sticky="ew", pady=6)
        ttk.Button(side, text="Remove items…", command=self.open_remove_dialog).grid(row=25, column=0, sticky="ew")

        ttk.Separator(side).grid(row=26, column=0, sticky="ew", pady=6)
        ttk.Label(side, text="Header regex").grid(row=27, column=0, sticky="w")
        ttk.Entry(side, textvariable=self.header_pattern_var, width=38).grid(row=28, column=0, sticky="ew", pady=2)
        ttk.Button(side, text="Reload last folder/file", command=self.reload_last).grid(row=29, column=0, sticky="ew", pady=2)
        ttk.Button(side, text="Parse report", command=self.show_parse_report).grid(row=30, column=0, sticky="ew", pady=2)

        # Right panel
        right = ttk.Frame(self.comp_frame, padding=8); right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(0, weight=1); right.rowconfigure(1, weight=1); right.columnconfigure(0, weight=1)

        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        table_frame = ttk.Frame(right); table_frame.grid(row=1, column=0, sticky="nsew"); right.rowconfigure(1, weight=1)
        columns = ("substrate", "pixel_id", "comp", "group", "pos", "dir", "Voc", "Jsc_mAcm2", "FF_pct", "PCE_pct")
        self.tree = ttk.Treeview(table_frame, columns=columns, show="headings", selectmode="extended")
        for c in columns:
            # All columns are now sortable
            self.tree.heading(c, text=c, command=lambda col=c: self.sort_by_column(col))
            self.tree.column(c, width=90, stretch=True)
        vsb = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        hsb = ttk.Scrollbar(table_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscroll=vsb.set, xscroll=hsb.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns"); hsb.grid(row=1, column=0, sticky="ew")
        table_frame.rowconfigure(0, weight=1); table_frame.columnconfigure(0, weight=1)

        btns = ttk.Frame(table_frame); btns.grid(row=2, column=0, sticky="ew", pady=4)
        ttk.Button(btns, text="Remove Selected", command=self.remove_selected).grid(row=0, column=0, padx=2)
        ttk.Button(btns, text="Export table CSV", command=self.export_table_csv).grid(row=0, column=3, padx=2)

        plot_btns = ttk.Frame(right); plot_btns.grid(row=2, column=0, sticky="ew", pady=4)
        ttk.Button(plot_btns, text="Boxplot by composition/group", command=self.plot_boxplot_groups).grid(row=0, column=0, padx=4)
        ttk.Button(plot_btns, text="Heatmap (substrate × group)", command=self.plot_heatmap).grid(row=0, column=1, padx=4)
        ttk.Button(plot_btns, text="Substrate pixel map", command=self.plot_substrate_pixel_map).grid(row=0, column=2, padx=4)
        ttk.Button(plot_btns, text="Save plot as image", command=self.save_plot_image).grid(row=0, column=3, padx=4)
        
    def _build_sweep_tab(self):
        """Build the sweep-based analysis tab"""
        self.sweep_frame.columnconfigure(0, weight=0); self.sweep_frame.columnconfigure(1, weight=1)
        self.sweep_frame.rowconfigure(0, weight=1)
        
        # Left side controls
        sweep_side = ttk.Frame(self.sweep_frame, padding=8); sweep_side.grid(row=0, column=0, sticky="ns")
        
        # Excel file selection
        ttk.Label(sweep_side, text="Experimental Conditions").grid(row=0, column=0, sticky="w")
        excel_frame = ttk.Frame(sweep_side); excel_frame.grid(row=1, column=0, sticky="ew", pady=2)
        ttk.Entry(excel_frame, textvariable=self.excel_path_var, width=25).grid(row=0, column=0, sticky="ew")
        ttk.Button(excel_frame, text="Browse", command=self.browse_excel_file).grid(row=0, column=1, padx=(2, 0))
        ttk.Button(sweep_side, text="Load conditions", command=self.load_conditions).grid(row=2, column=0, sticky="ew", pady=2)
        
        ttk.Separator(sweep_side).grid(row=3, column=0, sticky="ew", pady=6)
        
        # Sweep selection
        ttk.Label(sweep_side, text="Select Sweep").grid(row=4, column=0, sticky="w")
        self.sweep_selection_var = tk.StringVar(value="All Sweeps")
        self.sweep_selection_cb = ttk.Combobox(sweep_side, textvariable=self.sweep_selection_var, 
                                             state="readonly", width=30)
        self.sweep_selection_cb.grid(row=5, column=0, sticky="ew", pady=2)
        self.sweep_selection_cb.bind("<<ComboboxSelected>>", self.on_sweep_selection_changed)
        
        ttk.Separator(sweep_side).grid(row=6, column=0, sticky="ew", pady=6)
        
        # Sweep analysis options
        ttk.Label(sweep_side, text="Plot Options").grid(row=7, column=0, sticky="w")
        self.sweep_metric = tk.StringVar(value="PCE_pct")
        sweep_metric_cb = ttk.Combobox(sweep_side, textvariable=self.sweep_metric, 
                                     values=["Voc", "Jsc_mAcm2", "FF_pct", "PCE_pct"], 
                                     state="readonly", width=15)
        sweep_metric_cb.grid(row=8, column=0, sticky="ew", pady=2)
        sweep_metric_cb.bind("<<ComboboxSelected>>", lambda e: self.refresh_sweep_plots())
        
        self.sweep_plot_type = tk.StringVar(value="auto")
        ttk.Label(sweep_side, text="Plot Type").grid(row=9, column=0, sticky="w", pady=(6, 0))
        plot_type_cb = ttk.Combobox(sweep_side, textvariable=self.sweep_plot_type,
                                  values=["auto", "1D_line", "2D_scatter", "3D_surface", "boxplot"], 
                                  state="readonly", width=15)
        plot_type_cb.grid(row=10, column=0, sticky="ew", pady=2)
        plot_type_cb.bind("<<ComboboxSelected>>", lambda e: self.refresh_sweep_plots())
        
        self.sweep_combine_directions = tk.BooleanVar(value=True)
        ttk.Checkbutton(sweep_side, text="Combine F & R", 
                       variable=self.sweep_combine_directions, 
                       command=self.refresh_sweep_plots).grid(row=11, column=0, sticky="w", pady=2)
        
        ttk.Separator(sweep_side).grid(row=12, column=0, sticky="ew", pady=6)
        ttk.Button(sweep_side, text="Refresh plot", command=self.refresh_sweep_plots).grid(row=13, column=0, sticky="ew")
        
        # Parameter analysis info
        ttk.Label(sweep_side, text="Selected Sweep Info").grid(row=14, column=0, sticky="w", pady=(10, 2))
        self.sweep_info_text = tk.Text(sweep_side, height=4, width=30, wrap=tk.WORD)
        self.sweep_info_text.grid(row=15, column=0, sticky="ew", pady=2)
        sweep_info_scroll = ttk.Scrollbar(sweep_side, command=self.sweep_info_text.yview)
        self.sweep_info_text.config(yscrollcommand=sweep_info_scroll.set)
        
        # Right panel for sweep analysis
        sweep_right = ttk.Frame(self.sweep_frame, padding=8); sweep_right.grid(row=0, column=1, sticky="nsew")
        sweep_right.rowconfigure(0, weight=1); sweep_right.rowconfigure(1, weight=1); sweep_right.columnconfigure(0, weight=1)
        
        # Plot canvas for sweep analysis
        self.sweep_canvas = FigureCanvasTkAgg(self.sweep_fig, master=sweep_right)
        self.sweep_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        
        # Info table for sweep conditions
        info_frame = ttk.Frame(sweep_right); info_frame.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        info_frame.rowconfigure(0, weight=1); info_frame.columnconfigure(0, weight=1)
        
        ttk.Label(info_frame, text="Sweep Conditions Summary").grid(row=0, column=0, sticky="w")
        
        # Treeview for sweep conditions
        sweep_columns = ("sweep_id", "substrate_count", "condition_name")
        self.sweep_tree = ttk.Treeview(info_frame, columns=sweep_columns, show="headings", height=8)
        # Set custom column headers
        self.sweep_tree.heading("sweep_id", text="Sweep ID")
        self.sweep_tree.heading("substrate_count", text="Substrates")
        self.sweep_tree.heading("condition_name", text="Experimental Conditions")
        
        # Set column widths
        self.sweep_tree.column("sweep_id", width=80, stretch=False)
        self.sweep_tree.column("substrate_count", width=80, stretch=False)
        self.sweep_tree.column("condition_name", width=400, stretch=True)
        
        sweep_vsb = ttk.Scrollbar(info_frame, orient="vertical", command=self.sweep_tree.yview)
        self.sweep_tree.configure(yscroll=sweep_vsb.set)
        self.sweep_tree.grid(row=1, column=0, sticky="nsew", pady=(5, 0))
        sweep_vsb.grid(row=1, column=1, sticky="ns", pady=(5, 0))
        
        # Buttons for sweep analysis
        sweep_btns = ttk.Frame(sweep_right); sweep_btns.grid(row=2, column=0, sticky="ew", pady=4)
        ttk.Button(sweep_btns, text="Export sweep data", command=self.export_sweep_data).grid(row=0, column=0, padx=4)
        ttk.Button(sweep_btns, text="Save sweep plot", command=self.save_sweep_plot).grid(row=0, column=1, padx=4)

    # -------------------- File loading --------------------
    def load_file(self):
        path = filedialog.askopenfilename(title="Select JV file", filetypes=[("Text", "*.txt *.dat *.csv"), ("All", "*.*")])
        if not path: return
        self._load_paths([Path(path)]); self._last_paths = [Path(path)]

    def load_folder(self):
        d = filedialog.askdirectory(title="Select folder with JV files")
        if not d: return
        paths = [p for p in Path(d).glob("**/*") if p.suffix.lower() in (".txt", ".dat", ".csv")]
        if not paths:
            messagebox.showwarning("No files", "No .txt/.dat/.csv files found in that folder."); return
        self._load_paths(paths); self._last_paths = paths

    def reload_last(self):
        if not self._last_paths:
            messagebox.showinfo("Reload", "No previous selection to reload."); return
        self._load_paths(self._last_paths)

    def _load_paths(self, paths: List[Path]):
        area = self.area_cm2.get(); pin = self.light_mw_cm2.get()
        header_pat = self.header_pattern_var.get() or DEFAULT_HEADER_REGEX
        all_sweeps: List[JVSweep] = []
        diags: List[Dict] = []
        for p in paths:
            try:
                sweeps, diag = build_sweeps_from_file(p, area_cm2=area, light_mw_cm2=pin, header_pattern=header_pat)
                all_sweeps.extend(sweeps)
                diags.append(diag)
            except Exception as e:
                diags.append({"file": str(p), "error": str(e), "sections_found": 0})
        self._last_diags = diags

        if not all_sweeps:
            self.show_parse_report()
            messagebox.showerror("No JV sweeps found", "No JV sweeps were parsed. Try adjusting the Header regex, then 'Reload last folder/file'.")
            return

        # Load experimental conditions and map to sweeps
        if self.conditions_df is None:
            print("[AUTO LOAD] Attempting automatic condition loading...")
            # Try automatic loading without showing dialogs
            excel_path = self.excel_path_var.get()
            print(f"[AUTO LOAD] Excel path from var: {excel_path}")
            if excel_path and Path(excel_path).exists():
                print(f"[AUTO LOAD] Loading from specified path: {excel_path}")
                self.conditions_df = load_experimental_conditions(excel_path)
                print(f"[AUTO LOAD] Result from specified path: {self.conditions_df is not None}")
            else:
                # Try default path
                default_path = Path(__file__).parent / "experiment sheets.xlsx"
                print(f"[AUTO LOAD] Trying default path: {default_path}")
                print(f"[AUTO LOAD] Default path exists: {default_path.exists()}")
                if default_path.exists():
                    self.conditions_df = load_experimental_conditions(str(default_path))
                    print(f"[AUTO LOAD] Result from default path: {self.conditions_df is not None}")
                    if self.conditions_df is not None:
                        self.excel_path_var.set(str(default_path))
                        print(f"[AUTO LOAD] Set excel_path_var to: {default_path}")
                else:
                    print("[AUTO LOAD] No valid paths found")
        
        all_sweeps = map_sweeps_to_conditions(all_sweeps, self.conditions_df)
        
        self.data = all_sweeps
        self.df = self._to_dataframe(all_sweeps)
        self.df["group_index"] = self.df["composition_index"].apply(comp_to_group)
        self.df_with_flags = self.df.copy()

        self._populate_substrate_combo()
        self.refresh_table(); self.refresh_plots()
        # Only refresh sweep analysis if we have the methods
        try:
            self.refresh_sweep_info(); self.refresh_sweep_plots()
        except AttributeError:
            pass  # Sweep functions not yet loaded

    def show_parse_report(self):
        top = tk.Toplevel(self); top.title("Parse Report")
        txt = tk.Text(top, width=110, height=32)
        txt.pack(fill="both", expand=True)
        if not self._last_diags:
            txt.insert("end", "No diagnostics yet. Load a file/folder first.\n"); return
        for d in self._last_diags:
            txt.insert("end", f"FILE: {d.get('file')}\n")
            if "error" in d:
                txt.insert("end", f"  ERROR: {d.get('error')}\n\n"); continue
            txt.insert("end", f"  sections_found: {d.get('sections_found')}\n")
            for ex in d.get("examples", [])[:10]:
                txt.insert("end", f"    pos={ex['pos']:>2} comp={ex['comp']:>2}  fwd_pts={ex['n_fwd']:>3} rev_pts={ex['n_rev']:>3}\n")
            txt.insert("end", "\n")

    def _to_dataframe(self, sweeps: List[JVSweep]) -> pd.DataFrame:
        rows = []
        for s in sweeps:
            rows.append({
                "substrate": s.substrate,
                "pixel_id": s.pixel_id,
                "composition_index": s.composition_index,
                "position_in_composition": s.position_in_composition,
                "direction": s.direction,
                "Voc": s.Voc,
                "Jsc_mAcm2": s.Jsc_mAcm2,
                "FF_pct": s.FF_pct,
                "PCE_pct": s.PCE_pct,
                "sweep_id": s.sweep_id,
                "condition_name": s.condition_name
            })
        return pd.DataFrame(rows)

    def _populate_substrate_combo(self):
        if self.df is None or self.df.empty: return
        subs_unique = sorted(pd.unique(self.df["substrate"].dropna().astype(int)))
        values = ["All"] + [str(s) for s in subs_unique]
        self.substrate_cb["values"] = values
        cur = self.substrate_cb.get()
        self.substrate_cb.set(cur if cur in values else "All")

    # -------------------- Filters & removal --------------------
    def apply_thresholds(self):
        if self.df_with_flags is None or self.df_with_flags.empty: return
        df = self.df_with_flags
        include = np.ones(len(df), dtype=bool)

        def _f(x):
            try:
                if x == "" or x is None: return None
                return float(x)
            except Exception:
                return None

        mv, xv = _f(self.min_voc.get()), _f(self.max_voc.get())
        mp, xp = _f(self.min_pce.get()), _f(self.max_pce.get())
        mjs, xjs = _f(self.min_jsc.get()), _f(self.max_jsc.get())
        mff, xff = _f(self.min_ff.get()),  _f(self.max_ff.get())

        if mv  is not None: include &= df["Voc"].fillna(-1e9)      >= mv
        if mp  is not None: include &= df["PCE_pct"].fillna(-1e9)  >= mp
        if mjs is not None: include &= df["Jsc_mAcm2"].fillna(-1e9)>= mjs
        if mff is not None: include &= df["FF_pct"].fillna(-1e9)   >= mff

        if xv  is not None: include &= df["Voc"].fillna(1e9)       <= xv
        if xp  is not None: include &= df["PCE_pct"].fillna(1e9)   <= xp
        if xjs is not None: include &= df["Jsc_mAcm2"].fillna(1e9) <= xjs
        if xff is not None: include &= df["FF_pct"].fillna(1e9)    <= xff

        df["include"] = include
        self.refresh_table(); self.refresh_plots()

    def reset_filters_include_all(self):
        for var in (self.min_voc, self.max_voc, self.min_pce, self.max_pce, self.min_jsc, self.max_jsc, self.min_ff, self.max_ff):
            try: var.set("")
            except Exception: pass
        self.include_forward.set(True); self.include_reverse.set(True)
        self.refresh_table(); self.refresh_plots()

    def open_remove_dialog(self):
        if self.df_with_flags is None or self.df_with_flags.empty:
            messagebox.showinfo("Remove", "No data loaded."); return

        top = tk.Toplevel(self); top.title("Remove items")
        ttk.Label(top, text="Scope:").grid(row=0, column=0, sticky="w", padx=4, pady=4)
        scope_var = tk.StringVar(value="Composition (C#)")
        scope_cb = ttk.Combobox(top, textvariable=scope_var, state="readonly",
                                values=["Composition (C#)", "Group (G#)", "Pixel position (1..6)"], width=22)
        scope_cb.grid(row=0, column=1, sticky="ew", padx=4, pady=4)

        ttk.Label(top, text="Substrate:").grid(row=1, column=0, sticky="w", padx=4, pady=4)
        subs_unique = ["All"] + [str(s) for s in sorted(pd.unique(self.df_with_flags["substrate"].dropna().astype(int)))]
        sub_var = tk.StringVar(value="All")
        sub_cb = ttk.Combobox(top, textvariable=sub_var, values=subs_unique, state="readonly", width=22)
        sub_cb.grid(row=1, column=1, sticky="ew", padx=4, pady=4)

        ttk.Label(top, text="Index:").grid(row=2, column=0, sticky="w", padx=4, pady=4)
        idx_var = tk.StringVar(value="1")
        idx_cb = ttk.Combobox(top, textvariable=idx_var, state="readonly", width=22)
        idx_cb.grid(row=2, column=1, sticky="ew", padx=4, pady=4)

        def _refresh_index_options(*_):
            scope = scope_var.get()
            if scope.startswith("Composition"):
                idx_cb["values"] = [str(i) for i in range(1, 12)]
                idx_cb.set("1")
            elif scope.startswith("Group"):
                idx_cb["values"] = [str(i) for i in range(1, 10)]
                idx_cb.set("1")
            else:
                idx_cb["values"] = [str(i) for i in range(1, 7)]
                idx_cb.set("1")

        scope_cb.bind("<<ComboboxSelected>>", _refresh_index_options)
        _refresh_index_options()

        info = ttk.Label(top, text="", foreground="gray")
        info.grid(row=3, column=0, columnspan=2, sticky="w", padx=4, pady=2)

        def _do_remove():
            scope = scope_var.get()
            sub_sel = sub_var.get()
            idx_str = idx_var.get()
            if not idx_str:
                messagebox.showwarning("Remove", "Choose an index."); return
            try:
                idx = int(idx_str)
            except Exception:
                messagebox.showwarning("Remove", "Index must be a number."); return

            df = self.df_with_flags
            mask = pd.Series([True] * len(df), index=df.index)  # All rows are valid now
            if sub_sel != "All":
                mask &= (df["substrate"] == int(sub_sel))
            if scope.startswith("Composition"):
                mask &= (df["composition_index"] == idx)
            elif scope.startswith("Group"):
                mask &= (df["group_index"] == idx)
            else:  # Pixel position
                mask &= (df["position_in_composition"] == idx)

            removed = int(mask.sum())
            df.loc[mask, "include"] = False
            info.config(text=f"Removed: {removed} rows")
            self.refresh_table(); self.refresh_plots()

        ttk.Button(top, text="Remove", command=_do_remove).grid(row=4, column=0, padx=4, pady=8, sticky="ew")
        ttk.Button(top, text="Close", command=top.destroy).grid(row=4, column=1, padx=4, pady=8, sticky="ew")

    # -------------------- Table utilities --------------------
    def sort_by_column(self, column: str):
        """Sort table by clicked column"""
        if self.df_with_flags is None or self.df_with_flags.empty:
            return
            
        # Toggle sort direction if same column, otherwise ascending
        if self._sort_column == column:
            self._sort_reverse = not self._sort_reverse
        else:
            self._sort_column = column
            self._sort_reverse = False
            
        # Map display column names to dataframe column names
        column_map = {
            "substrate": "substrate",
            "pixel_id": "pixel_id", 
            "comp": "composition_index",
            "group": "group_index",
            "pos": "position_in_composition",
            "dir": "direction",
            "Voc": "Voc",
            "Jsc_mAcm2": "Jsc_mAcm2",
            "FF_pct": "FF_pct",
            "PCE_pct": "PCE_pct"
        }
        
        if column in column_map:
            df_col = column_map[column]
            # Sort the dataframe
            self.df_with_flags = self.df_with_flags.sort_values(
                by=df_col, ascending=not self._sort_reverse, na_position='last'
            ).reset_index(drop=True)
            
            # Update column header to show sort direction
            self.update_column_headers()
            
        self.refresh_table()
        
    def update_column_headers(self):
        """Update column headers to show current sort direction"""
        columns = ("substrate", "pixel_id", "comp", "group", "pos", "dir", "Voc", "Jsc_mAcm2", "FF_pct", "PCE_pct")
        
        for c in columns:
            if c == self._sort_column:
                arrow = " ▼" if self._sort_reverse else " ▲"
                self.tree.heading(c, text=f"{c}{arrow}")
            else:
                self.tree.heading(c, text=c)

    def refresh_table(self):
        self.tree.delete(*self.tree.get_children())
        if self.df_with_flags is None or self.df_with_flags.empty: return
        for idx, r in self.df_with_flags.iterrows():
            vals = (
                int(r["substrate"]) if pd.notna(r["substrate"]) else "",
                int(r["pixel_id"]) if pd.notna(r["pixel_id"]) else "",
                int(r["composition_index"]) if pd.notna(r["composition_index"]) else "",
                int(r["group_index"]) if pd.notna(r["group_index"]) else "",
                int(r["position_in_composition"]) if pd.notna(r["position_in_composition"]) else "",
                r["direction"],
                None if pd.isna(r["Voc"]) else round(float(r["Voc"]), 3),
                None if pd.isna(r["Jsc_mAcm2"]) else round(float(r["Jsc_mAcm2"]), 2),
                None if pd.isna(r["FF_pct"]) else round(float(r["FF_pct"]), 1),
                None if pd.isna(r["PCE_pct"]) else round(float(r["PCE_pct"]), 2),
            )
            self.tree.insert("", "end", iid=str(idx), values=vals)

    def remove_selected(self):
        """Delete selected measurements from the data"""
        if self.df_with_flags is None or self.df_with_flags.empty: 
            return
        sel = self.tree.selection()
        if not sel: 
            return
        
        # Get selected indices and sort in reverse order to delete from end first
        indices_to_remove = sorted([int(iid) for iid in sel], reverse=True)
        
        # Remove rows from dataframe
        self.df_with_flags = self.df_with_flags.drop(indices_to_remove).reset_index(drop=True)
        
        # Also update the underlying df if it exists
        if self.df is not None:
            self.df = self.df.drop(indices_to_remove).reset_index(drop=True)
            
        # Update the original data list as well
        if self.data:
            for idx in indices_to_remove:
                if idx < len(self.data):
                    self.data.pop(idx)
        
        self.refresh_table()
        self.refresh_plots()


    def export_table_csv(self):
        if self.df_with_flags is None or self.df_with_flags.empty:
            messagebox.showinfo("Export", "No data to export."); return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if not path: return
        self.df_with_flags.to_csv(path, index=False); messagebox.showinfo("Export", f"Saved: {path}")

    # -------------------- Metrics recompute --------------------
    def recompute_metrics(self):
        if not self.data:
            messagebox.showwarning("No data", "Load files first."); return
        area = self.area_cm2.get(); pin = self.light_mw_cm2.get()
        for s in self.data:
            s.area_cm2 = area; s.light_mw_cm2 = pin
            Voc, Jsc_mAcm2, FF_pct, PCE_pct = compute_metrics(s.voltage, s.current_A, area_cm2=area, light_mw_cm2=pin)
            s.Voc, s.Jsc_mAcm2, s.FF_pct, s.PCE_pct = Voc, Jsc_mAcm2, FF_pct, PCE_pct
        rows = []
        for s in self.data:
            rows.append({
                "substrate": s.substrate,
                "pixel_id": s.pixel_id,
                "composition_index": s.composition_index,
                "position_in_composition": s.position_in_composition,
                "direction": s.direction,
                "Voc": s.Voc,
                "Jsc_mAcm2": s.Jsc_mAcm2,
                "FF_pct": s.FF_pct,
                "PCE_pct": s.PCE_pct,
                "sweep_id": s.sweep_id,
                "condition_name": s.condition_name
            })
        self.df = pd.DataFrame(rows)
        self.df["group_index"] = self.df["composition_index"].apply(comp_to_group)
        self.df_with_flags = self.df.copy()
        self._populate_substrate_combo()
        self.refresh_table(); self.refresh_plots()

    # -------------------- Plotting helpers --------------------
    def _reset_axes(self):
        """Fresh axes (remove any colorbar) so plots don't stack."""
        if getattr(self, "_cbar", None) is not None:
            try: self._cbar.remove()
            except Exception: pass
            self._cbar = None
        self.fig.clf()
        self.ax = self.fig.add_subplot(111)

    def _filtered_df(self) -> pd.DataFrame:
        if self.df_with_flags is None: return pd.DataFrame()
        df = self.df_with_flags.copy()  # All rows are now valid since we delete instead of mark
        # Handle F/R toggles only when not combining
        if not self.combine_fr.get():
            keep = []
            if self.include_forward.get(): keep.append("forward")
            if self.include_reverse.get(): keep.append("reverse")
            if keep: df = df[df["direction"].isin(keep)]
        return df

    def _clear_ax(self, msg: str):
        self._reset_axes()
        self.ax.text(0.5, 0.5, msg, ha="center", va="center", transform=self.ax.transAxes)
        self.ax.axis("off")
        self.canvas.draw_idle()

    def save_plot_image(self):
        path = filedialog.asksaveasfilename(
            title="Save current plot",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("SVG", "*.svg"), ("PDF", "*.pdf")]
        )
        if not path: return
        try:
            self.fig.savefig(path, dpi=300, bbox_inches="tight")
            messagebox.showinfo("Saved", f"Plot saved to:\n{path}")
        except Exception as e:
            messagebox.showerror("Save failed", str(e))

    def refresh_plots(self):
        self.plot_boxplot_groups()

    # -------------------- Plots --------------------
    def plot_boxplot_groups(self):
        df = self._filtered_df()
        self._reset_axes()
        if df.empty:
            self._clear_ax("No data to plot. Check filters/thresholds."); return
        metric = self.metric_choice.get()

        use_groups = (self.grouping_mode.get() == "9 groups")
        group_col = "group_index" if use_groups else "composition_index"
        expected_groups = list(range(1, 10)) if use_groups else list(range(1, 12))

        data_df = df.copy()
        labels: List[str] = []; series: List[np.ndarray] = []

        if self.expand_substrate_axis.get():
            subs = sorted(data_df["substrate"].dropna().astype(int).unique().tolist())
            for sub in subs:
                sub_df = data_df[data_df["substrate"] == sub]
                vals_by = sub_df.groupby(group_col)[metric].apply(lambda s: s.dropna().values)
                for g in expected_groups:
                    vals = vals_by.get(g, np.array([]))
                    if len(vals) == 0: continue
                    labels.append(f"S{sub}-" + ("G" if use_groups else "C") + str(g))
                    series.append(vals)
        else:
            sel = self.substrate_cb.get()
            if sel and sel != "All" and not self.combine_substrates.get():
                try: data_df = data_df[data_df["substrate"] == int(sel)]
                except Exception: pass
            vals_by = data_df.groupby(group_col)[metric].apply(lambda s: s.dropna().values)
            for g in expected_groups:
                vals = vals_by.get(g, np.array([]))
                if len(vals) == 0: continue
                labels.append(("G" if use_groups else "C") + str(g))
                series.append(vals)

        if not series:
            self._clear_ax("No values for selected metric/filters."); return

        self.ax.boxplot(series, showmeans=True, meanline=False)
        self.ax.set_xticks(range(1, len(labels) + 1))
        self.ax.set_xticklabels(labels, rotation=45, ha="right")
        self.ax.set_ylabel(metric)
        title = f"Boxplot: {metric} by {'Group' if use_groups else 'Composition'}"
        if self.expand_substrate_axis.get(): title += " (expanded by substrate)"
        self.ax.set_title(title)
        self.ax.grid(True, which="both", axis="y")
        self.canvas.draw_idle()

    def plot_heatmap(self):
        df = self._filtered_df()
        self._reset_axes()
        if df.empty:
            self._clear_ax("No data to plot. Check filters/thresholds."); return
        metric = self.metric_choice.get()
        use_groups = (self.grouping_mode.get() == "9 groups")
        group_col = "group_index" if use_groups else "composition_index"
        expected_groups = list(range(1, 10)) if use_groups else list(range(1, 12))

        agg = "mean"  # (can switch to 'max' if you want later)

        data_df = df.copy()
        grouped = (data_df.groupby(["substrate", group_col])[metric].max().reset_index()
                   if agg == "max"
                   else data_df.groupby(["substrate", group_col])[metric].mean().reset_index())

        if grouped.empty:
            self._clear_ax("No data after grouping. Try relaxing filters."); return

        pivot = grouped.pivot(index="substrate", columns=group_col, values=metric)
        for g in expected_groups:
            if g not in pivot.columns: pivot[g] = np.nan
        pivot = pivot[expected_groups].sort_index()

        im = self.ax.imshow(pivot.values, aspect="auto", interpolation="nearest")
        self.ax.set_xlabel("Composition " + ("group (G1..G9)" if use_groups else "index (C1..C11)"))
        self.ax.set_ylabel("Substrate")
        self.ax.set_title(f"Heatmap: {agg} {metric}")
        self.ax.set_xticks(range(pivot.shape[1])); self.ax.set_xticklabels([("G" if use_groups else "C")+str(g) for g in expected_groups])
        self.ax.set_yticks(range(pivot.shape[0])); self.ax.set_yticklabels([str(int(s)) for s in pivot.index])

        vals = pivot.values
        for i in range(vals.shape[0]):
            for j in range(vals.shape[1]):
                v = vals[i, j]
                if isinstance(v, float) and not np.isnan(v):
                    self.ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8)

        self._cbar = self.fig.colorbar(im, ax=self.ax, fraction=0.046, pad=0.04, label=metric)
        self.canvas.draw_idle()

    def plot_substrate_pixel_map(self):
        """
        Single substrate pixel map:
          x = composition 1..11
          y = pixel position 1..6 (1=thick → 6=thin)
          color = chosen metric (Voc, Jsc_mAcm2, FF_pct, PCE_pct)
        """
        df = self._filtered_df()
        self._reset_axes()
        if df.empty:
            self._clear_ax("No data to plot."); return

        sel = self.substrate_cb.get()
        if not sel or sel == "All":
            self._clear_ax("Pick a single substrate in the dropdown first."); return

        try: sub = int(sel)
        except Exception:
            self._clear_ax("Invalid substrate selection."); return

        metric = self.metric_choice.get()
        sdf = df[df["substrate"] == sub].copy()
        if sdf.empty:
            self._clear_ax(f"No data for substrate {sub} after filters."); return

        # Combine F/R or respect toggles
        if self.combine_fr.get():
            grouped = sdf.groupby(["composition_index", "position_in_composition"], as_index=False)[metric].mean()
        else:
            keep = []
            if self.include_forward.get(): keep.append("forward")
            if self.include_reverse.get(): keep.append("reverse")
            if keep: sdf = sdf[sdf["direction"].isin(keep)]
            grouped = sdf.groupby(["composition_index", "position_in_composition"], as_index=False)[metric].mean()

        mat = np.full((6, 11), np.nan, dtype=float)
        for _, r in grouped.iterrows():
            c = int(r["composition_index"]); p = int(r["position_in_composition"])
            if 1 <= c <= 11 and 1 <= p <= 6:
                mat[p-1, c-1] = float(r[metric])

        im = self.ax.imshow(mat, aspect="auto", interpolation="nearest", origin="upper")
        self.ax.set_xlabel("Composition (C1..C11)")
        self.ax.set_ylabel("Pixel position (1=thick → 6=thin)")
        self.ax.set_title(f"Substrate S{sub} — {metric} per composition & pixel position")
        self.ax.set_xticks(range(11)); self.ax.set_xticklabels([f"C{i}" for i in range(1, 12)], rotation=45, ha="right")
        self.ax.set_yticks(range(6));  self.ax.set_yticklabels([str(i) for i in range(1, 7)])
        for i in range(6):
            for j in range(11):
                v = mat[i, j]
                if not np.isnan(v):
                    self.ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7)

        self._cbar = self.fig.colorbar(im, ax=self.ax, fraction=0.046, pad=0.04, label=metric)
        self.canvas.draw_idle()
        
    # -------------------- Experimental conditions methods --------------------
    def browse_excel_file(self):
        """Browse for Excel file containing experimental conditions"""
        path = filedialog.askopenfilename(
            title="Select experiment conditions Excel file",
            filetypes=[("Excel files", "*.xlsx *.xls"), ("All files", "*.*")]
        )
        if path:
            self.excel_path_var.set(path)
            
    def load_conditions(self):
        """Load experimental conditions from Excel file"""
        excel_path = self.excel_path_var.get()
        if not excel_path:
            messagebox.showwarning("No file", "Please specify an Excel file path.")
            return
            
        print(f"[MANUAL LOAD] Attempting to load conditions from: {excel_path}")
        print(f"[MANUAL LOAD] Current working directory: {Path.cwd()}")
        print(f"[MANUAL LOAD] Excel file exists: {Path(excel_path).exists()}")
        
        self.conditions_df = load_experimental_conditions(excel_path)
        print(f"[MANUAL LOAD] Result is None: {self.conditions_df is None}")
        
        if self.conditions_df is None:
            print("[MANUAL LOAD] ERROR: conditions_df is None - showing error dialog")
            messagebox.showerror("Load failed", f"[MANUAL] Could not load conditions from {excel_path}\n\nCheck the console for detailed error information.")
        else:
            has_excess_pbi2 = 'excess PbI2' in self.conditions_df.columns
            sheet_type = "ROSIE" if has_excess_pbi2 else "Runsheet"
            print(f"[MANUAL LOAD] SUCCESS: Loaded {sheet_type} sheet with {len(self.conditions_df)} entries")
            messagebox.showinfo("Success", f"Loaded {len(self.conditions_df)} entries from {sheet_type} sheet")
            
            # Analyze sweep parameters
            self.sweep_analysis = analyze_sweep_parameters(self.conditions_df)
            self.update_sweep_selection_options()
            self.refresh_sweep_info()  # Update the summary table immediately
            
            if self.data:  # Re-map existing data if available
                self.data = map_sweeps_to_conditions(self.data, self.conditions_df)
                self.df = self._to_dataframe(self.data)
                self.df["group_index"] = self.df["composition_index"].apply(comp_to_group)
                self.df_with_flags = self.df.copy()
                self.refresh_table()
                self.refresh_sweep_info()
                self.refresh_sweep_plots()
                
    def refresh_sweep_info(self):
        """Refresh the sweep conditions summary table"""
        self.sweep_tree.delete(*self.sweep_tree.get_children())
        
        # If we have sweep analysis data, use that to show all available conditions
        if self.sweep_analysis:
            for sweep_id in sorted(self.sweep_analysis.keys()):
                analysis = self.sweep_analysis[sweep_id]
                
                # Check if we have JV data for this sweep
                data_available = False
                measured_substrates = 0
                if self.df_with_flags is not None and not self.df_with_flags.empty:
                    sweep_data = self.df_with_flags[self.df_with_flags['sweep_id'] == sweep_id]
                    if not sweep_data.empty:
                        data_available = True
                        measured_substrates = sweep_data['substrate'].nunique()
                
                # Create condition description
                condition_desc = f"{analysis['description']} "
                if data_available:
                    condition_desc += f"[{measured_substrates}/{len(analysis['substrates'])} measured]"
                else:
                    condition_desc += f"[0/{len(analysis['substrates'])} measured]"
                
                values = (
                    sweep_id,
                    len(analysis['substrates']),  # Total substrates in this sweep
                    condition_desc
                )
                self.sweep_tree.insert("", "end", values=values)
            return
        
        # Fallback: use df_with_flags if no sweep analysis available
        if self.df_with_flags is None or self.df_with_flags.empty:
            return
            
        # Group by sweep_id and condition_name
        sweep_summary = self.df_with_flags.groupby(['sweep_id', 'condition_name']).agg({
            'substrate': 'nunique'
        }).reset_index()
        
        sweep_summary = sweep_summary.rename(columns={'substrate': 'substrate_count'})
        
        for _, row in sweep_summary.iterrows():
            if pd.notna(row['sweep_id']):
                values = (
                    int(row['sweep_id']),
                    int(row['substrate_count']),
                    row['condition_name'] or "No conditions"
                )
                self.sweep_tree.insert("", "end", values=values)
                
    def refresh_sweep_plots(self):
        """Generate sweep-based plots"""
        if self.df_with_flags is None or self.df_with_flags.empty:
            self._clear_sweep_ax("No data to plot.")
            return
            
        df = self.df_with_flags.copy()  # All rows are valid since we delete instead of mark
        
        # Filter out entries without sweep information
        df = df.dropna(subset=['sweep_id'])
        
        if df.empty:
            self._clear_sweep_ax("No sweep data available. Load experimental conditions first.")
            return
            
        metric = self.sweep_metric.get()
        plot_type = self.sweep_plot_type.get()
        selected_sweep = self.selected_sweep_id.get()
        
        # Filter by selected sweep if not "All"
        if selected_sweep != -1:
            df = df[df['sweep_id'] == selected_sweep]
            if df.empty:
                self._clear_sweep_ax(f"No data for Sweep {selected_sweep}")
                return
        
        # Combine forward/reverse if selected
        if self.sweep_combine_directions.get():
            df_plot = df.groupby(['sweep_id', 'substrate'])[metric].mean().reset_index()
        else:
            df_plot = df.copy()
            
        self._reset_sweep_axes()
        
        # Auto-select plot type based on sweep analysis
        if plot_type == "auto":
            plot_type = self._determine_optimal_plot_type(selected_sweep)
        
        if plot_type == "boxplot":
            self._plot_sweep_boxplot(df_plot, metric)
        elif plot_type == "1D_line":
            self._plot_parameter_vs_performance(df, metric, selected_sweep)
        elif plot_type == "2D_scatter":
            self._plot_2d_parameter_analysis(df, metric, selected_sweep)
        elif plot_type == "3D_surface":
            self._plot_3d_parameter_analysis(df, metric, selected_sweep)
        else:  # fallback to scatter
            self._plot_sweep_scatter(df_plot, metric)
            
        self.sweep_canvas.draw_idle()
        
    def _reset_sweep_axes(self):
        """Reset sweep plot axes"""
        if getattr(self, "sweep_cbar", None) is not None:
            try: self.sweep_cbar.remove()
            except Exception: pass
            self.sweep_cbar = None
        self.sweep_fig.clf()
        self.sweep_ax = self.sweep_fig.add_subplot(111)
        
    def _clear_sweep_ax(self, msg: str):
        """Clear sweep axes with message"""
        self._reset_sweep_axes()
        self.sweep_ax.text(0.5, 0.5, msg, ha="center", va="center", transform=self.sweep_ax.transAxes)
        self.sweep_ax.axis("off")
        self.sweep_canvas.draw_idle()
        
    def _plot_sweep_boxplot(self, df: pd.DataFrame, metric: str):
        """Create boxplot of metric vs sweep"""
        sweep_ids = sorted(df['sweep_id'].unique())
        data_series = []
        labels = []
        
        for sweep_id in sweep_ids:
            sweep_data = df[df['sweep_id'] == sweep_id][metric].dropna()
            if len(sweep_data) > 0:
                data_series.append(sweep_data.values)
                labels.append(f"Sweep {int(sweep_id)}")
                
        if data_series:
            self.sweep_ax.boxplot(data_series, labels=labels, showmeans=True)
            self.sweep_ax.set_xlabel("Experimental Sweep")
            self.sweep_ax.set_ylabel(metric)
            self.sweep_ax.set_title(f"Performance by Experimental Sweep: {metric}")
            self.sweep_ax.grid(True, alpha=0.3)
            
    def _plot_sweep_scatter(self, df: pd.DataFrame, metric: str):
        """Create scatter plot of metric vs sweep"""
        sweep_ids = df['sweep_id'].values
        metric_values = df[metric].values
        
        # Remove NaN values
        mask = ~(pd.isna(sweep_ids) | pd.isna(metric_values))
        sweep_ids = sweep_ids[mask]
        metric_values = metric_values[mask]
        
        if len(sweep_ids) > 0:
            self.sweep_ax.scatter(sweep_ids, metric_values, alpha=0.6, s=50)
            self.sweep_ax.set_xlabel("Experimental Sweep ID")
            self.sweep_ax.set_ylabel(metric)
            self.sweep_ax.set_title(f"Performance vs Experimental Sweep: {metric}")
            self.sweep_ax.grid(True, alpha=0.3)
            
    def _plot_sweep_bar(self, df: pd.DataFrame, metric: str):
        """Create bar plot of average metric vs sweep"""
        sweep_means = df.groupby('sweep_id')[metric].mean()
        sweep_stds = df.groupby('sweep_id')[metric].std()
        
        if not sweep_means.empty:
            x_pos = range(len(sweep_means))
            self.sweep_ax.bar(x_pos, sweep_means.values, yerr=sweep_stds.values, 
                            capsize=5, alpha=0.7)
            self.sweep_ax.set_xlabel("Experimental Sweep")
            self.sweep_ax.set_ylabel(f"Average {metric}")
            self.sweep_ax.set_title(f"Average Performance by Experimental Sweep: {metric}")
            self.sweep_ax.set_xticks(x_pos)
            self.sweep_ax.set_xticklabels([f"Sweep {int(sid)}" for sid in sweep_means.index])
            self.sweep_ax.grid(True, alpha=0.3)
            
    def _determine_optimal_plot_type(self, sweep_id: int) -> str:
        """Determine the best plot type based on sweep analysis"""
        if sweep_id == -1:  # All sweeps
            return "boxplot"
            
        if sweep_id not in self.sweep_analysis:
            return "boxplot"
            
        analysis = self.sweep_analysis[sweep_id]
        n_varying = len(analysis['varying_params'])
        
        if n_varying == 0:
            return "boxplot"  # No varying parameters, just show distribution
        elif n_varying == 1:
            return "1D_line"   # Single parameter variation
        elif n_varying == 2:
            return "2D_scatter"  # Two parameter variation
        else:
            return "3D_surface"  # Multiple parameters
            
    def _plot_parameter_vs_performance(self, df: pd.DataFrame, metric: str, sweep_id: int):
        """Plot single parameter vs performance"""
        if sweep_id == -1 or sweep_id not in self.sweep_analysis:
            self._plot_sweep_scatter(df, metric)
            return
            
        analysis = self.sweep_analysis[sweep_id]
        if not analysis['varying_params']:
            self._plot_sweep_scatter(df, metric)
            return
            
        # Get the main varying parameter
        param_col = analysis['varying_params'][0]
        param_name = param_col.replace(' (M)', '')
        
        # Get parameter values from conditions_df
        if self.conditions_df is None:
            return
            
        sweep_conditions = self.conditions_df[self.conditions_df['Sweep'] == sweep_id]
        
        # Map measurements to specific experimental conditions using Well ID mapping
        df_matched_rows = []
        
        for _, measurement_row in df.iterrows():
            substrate_id = measurement_row['substrate']
            composition_index = measurement_row['composition_index']
            position_in_composition = measurement_row['position_in_composition']
            
            # Convert pixel position to Well ID
            # position_in_composition (1-6) -> row letter (A-F)
            # composition_index (1-11) -> column number (1-11)
            if 1 <= position_in_composition <= 6 and 1 <= composition_index <= 11:
                row_letter = chr(ord('A') + position_in_composition - 1)  # 1->A, 2->B, etc.
                well_id = f"{row_letter}{composition_index}"
                
                # Find the experimental condition for this specific Well ID
                condition_match = sweep_conditions[
                    (sweep_conditions['Substrate'] == substrate_id) & 
                    (sweep_conditions['Well'] == well_id)
                ]
                
                if not condition_match.empty:
                    condition_row = condition_match.iloc[0]
                    param_value = condition_row[param_col]
                    if pd.notna(param_value):  # Allow zero values
                        # Create a new row with matched experimental condition
                        new_row = measurement_row.copy()
                        new_row['param_value'] = param_value
                        new_row['well_id'] = well_id
                        df_matched_rows.append(new_row)
        
        if not df_matched_rows:
            df_plot = pd.DataFrame()
        else:
            df_plot = pd.DataFrame(df_matched_rows)
        
        if df_plot.empty:
            return
            
        # Group by parameter value and calculate statistics
        grouped = df_plot.groupby('param_value')[metric].agg(['mean', 'std', 'count']).reset_index()
        grouped = grouped.sort_values('param_value')
        
        # Plot line with error bars
        self.sweep_ax.errorbar(grouped['param_value'], grouped['mean'], 
                             yerr=grouped['std'], marker='o', capsize=5, capthick=2)
        
        # Add individual points
        for param_val in grouped['param_value']:
            subset = df_plot[df_plot['param_value'] == param_val]
            self.sweep_ax.scatter([param_val] * len(subset), subset[metric], 
                                alpha=0.3, s=20)
        
        self.sweep_ax.set_xlabel(f"{param_name} Concentration (M)")
        self.sweep_ax.set_ylabel(metric)
        self.sweep_ax.set_title(f"Sweep {sweep_id}: {metric} vs {param_name}")
        self.sweep_ax.grid(True, alpha=0.3)
        
    def _plot_2d_parameter_analysis(self, df: pd.DataFrame, metric: str, sweep_id: int):
        """Plot 2D parameter analysis with color-coded performance"""
        if sweep_id == -1 or sweep_id not in self.sweep_analysis:
            self._plot_sweep_scatter(df, metric)
            return
            
        analysis = self.sweep_analysis[sweep_id]
        if len(analysis['varying_params']) < 2:
            self._plot_parameter_vs_performance(df, metric, sweep_id)
            return
            
        # Get the two main varying parameters
        param1_col = analysis['varying_params'][0]
        param2_col = analysis['varying_params'][1]
        param1_name = param1_col.replace(' (M)', '')
        param2_name = param2_col.replace(' (M)', '')
        
        # Get parameter values from conditions_df
        if self.conditions_df is None:
            return
            
        sweep_conditions = self.conditions_df[self.conditions_df['Sweep'] == sweep_id]
        
        # Map measurements to specific experimental conditions using Well ID mapping
        df_matched_rows = []
        
        for _, measurement_row in df.iterrows():
            substrate_id = measurement_row['substrate']
            composition_index = measurement_row['composition_index']
            position_in_composition = measurement_row['position_in_composition']
            
            # Convert pixel position to Well ID
            if 1 <= position_in_composition <= 6 and 1 <= composition_index <= 11:
                row_letter = chr(ord('A') + position_in_composition - 1)  # 1->A, 2->B, etc.
                well_id = f"{row_letter}{composition_index}"
                
                # Find the experimental condition for this specific Well ID
                condition_match = sweep_conditions[
                    (sweep_conditions['Substrate'] == substrate_id) & 
                    (sweep_conditions['Well'] == well_id)
                ]
                
                if not condition_match.empty:
                    condition_row = condition_match.iloc[0]
                    param1_value = condition_row[param1_col]
                    param2_value = condition_row[param2_col]
                    
                    if pd.notna(param1_value) and pd.notna(param2_value):
                        # Create a new row with matched experimental condition
                        new_row = measurement_row.copy()
                        new_row['param1_value'] = param1_value
                        new_row['param2_value'] = param2_value
                        new_row['well_id'] = well_id
                        df_matched_rows.append(new_row)
        
        if not df_matched_rows:
            df_plot = pd.DataFrame()
        else:
            df_plot = pd.DataFrame(df_matched_rows)
        
        if df_plot.empty:
            return
            
        # Create scatter plot with color coding
        scatter = self.sweep_ax.scatter(df_plot['param1_value'], df_plot['param2_value'], 
                                      c=df_plot[metric], s=100, alpha=0.7, 
                                      cmap='viridis')
        
        self.sweep_ax.set_xlabel(f"{param1_name} Concentration (M)")
        self.sweep_ax.set_ylabel(f"{param2_name} Concentration (M)")
        self.sweep_ax.set_title(f"Sweep {sweep_id}: {param1_name} vs {param2_name} (Color = {metric})")
        
        # Add colorbar
        try:
            self.sweep_cbar = self.sweep_fig.colorbar(scatter, ax=self.sweep_ax, 
                                                    fraction=0.046, pad=0.04, 
                                                    label=metric)
        except:
            pass  # Colorbar might fail in some cases
            
    def _plot_3d_parameter_analysis(self, df: pd.DataFrame, metric: str, sweep_id: int):
        """Create 3D surface plot for multi-parameter analysis"""
        if sweep_id == -1 or sweep_id not in self.sweep_analysis:
            self._plot_2d_parameter_analysis(df, metric, sweep_id)
            return
            
        analysis = self.sweep_analysis[sweep_id]
        if len(analysis['varying_params']) < 2:
            self._plot_parameter_vs_performance(df, metric, sweep_id)
            return
            
        # For 3D plot, use first two varying parameters as X,Y and metric as Z
        param1_col = analysis['varying_params'][0]
        param2_col = analysis['varying_params'][1]
        param1_name = param1_col.replace(' (M)', '')
        param2_name = param2_col.replace(' (M)', '')
        
        # Get parameter values
        if self.conditions_df is None:
            return
            
        sweep_conditions = self.conditions_df[self.conditions_df['Sweep'] == sweep_id]
        
        # Map measurements to specific experimental conditions using Well ID mapping
        df_matched_rows = []
        
        for _, measurement_row in df.iterrows():
            substrate_id = measurement_row['substrate']
            composition_index = measurement_row['composition_index']
            position_in_composition = measurement_row['position_in_composition']
            
            # Convert pixel position to Well ID
            if 1 <= position_in_composition <= 6 and 1 <= composition_index <= 11:
                row_letter = chr(ord('A') + position_in_composition - 1)  # 1->A, 2->B, etc.
                well_id = f"{row_letter}{composition_index}"
                
                # Find the experimental condition for this specific Well ID
                condition_match = sweep_conditions[
                    (sweep_conditions['Substrate'] == substrate_id) & 
                    (sweep_conditions['Well'] == well_id)
                ]
                
                if not condition_match.empty:
                    condition_row = condition_match.iloc[0]
                    param1_value = condition_row[param1_col]
                    param2_value = condition_row[param2_col]
                    
                    if pd.notna(param1_value) and pd.notna(param2_value):
                        # Create a new row with matched experimental condition
                        new_row = measurement_row.copy()
                        new_row['param1_value'] = param1_value
                        new_row['param2_value'] = param2_value
                        new_row['well_id'] = well_id
                        df_matched_rows.append(new_row)
        
        if not df_matched_rows:
            df_plot = pd.DataFrame()
        else:
            df_plot = pd.DataFrame(df_matched_rows)
        
        if df_plot.empty:
            return
            
        # Create 3D plot
        self.sweep_fig.clf()
        self.sweep_ax = self.sweep_fig.add_subplot(111, projection='3d')
        
        # Group by parameter combinations and get mean performance
        grouped = df_plot.groupby(['param1_value', 'param2_value'])[metric].mean().reset_index()
        
        # Create 3D scatter plot
        scatter = self.sweep_ax.scatter(grouped['param1_value'], grouped['param2_value'], 
                                      grouped[metric], s=100, alpha=0.7, 
                                      c=grouped[metric], cmap='viridis')
        
        self.sweep_ax.set_xlabel(f"{param1_name} (M)")
        self.sweep_ax.set_ylabel(f"{param2_name} (M)")
        self.sweep_ax.set_zlabel(metric)
        self.sweep_ax.set_title(f"Sweep {sweep_id}: 3D Parameter Analysis")
        
        # Try to add colorbar for 3D plot
        try:
            self.sweep_cbar = self.sweep_fig.colorbar(scatter, ax=self.sweep_ax, 
                                                    shrink=0.5, aspect=20, 
                                                    label=metric)
        except:
            pass
            
    def export_sweep_data(self):
        """Export sweep analysis data to CSV"""
        if self.df_with_flags is None or self.df_with_flags.empty:
            messagebox.showinfo("Export", "No data to export.")
            return
            
        path = filedialog.asksaveasfilename(
            title="Save sweep analysis data",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not path:
            return
            
        # Export data with sweep information
        export_df = self.df_with_flags.copy()
        export_df.to_csv(path, index=False)
        messagebox.showinfo("Export", f"Sweep data exported to {path}")
        
    def save_sweep_plot(self):
        """Save current sweep plot"""
        path = filedialog.asksaveasfilename(
            title="Save sweep plot",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("SVG", "*.svg"), ("PDF", "*.pdf")]
        )
        if not path:
            return
            
        try:
            self.sweep_fig.savefig(path, dpi=300, bbox_inches="tight")
            messagebox.showinfo("Saved", f"Sweep plot saved to {path}")
        except Exception as e:
            messagebox.showerror("Save failed", str(e))
            
    def update_sweep_selection_options(self):
        """Update the sweep selection dropdown with descriptive names"""
        if not self.sweep_analysis:
            self.sweep_selection_cb["values"] = ["All Sweeps"]
            return
            
        options = ["All Sweeps"]
        for sweep_id in sorted(self.sweep_analysis.keys()):
            analysis = self.sweep_analysis[sweep_id]
            option = f"Sweep {sweep_id}: {analysis['description']}"
            options.append(option)
            
        self.sweep_selection_cb["values"] = options
        self.sweep_selection_cb.set("All Sweeps")
        
    def on_sweep_selection_changed(self, event=None):
        """Handle sweep selection change"""
        selection = self.sweep_selection_var.get()
        if selection == "All Sweeps":
            self.selected_sweep_id.set(-1)
            self.sweep_info_text.delete(1.0, tk.END)
            self.sweep_info_text.insert(tk.END, "All sweeps selected\nShowing overview comparison")
        else:
            # Extract sweep ID from selection
            try:
                sweep_id = int(selection.split(":")[0].replace("Sweep ", ""))
                self.selected_sweep_id.set(sweep_id)
                self.update_sweep_info_display(sweep_id)
            except ValueError:
                self.selected_sweep_id.set(-1)
                
        self.refresh_sweep_plots()
        
    def update_sweep_info_display(self, sweep_id: int):
        """Update the sweep info text display"""
        if sweep_id not in self.sweep_analysis:
            return
            
        analysis = self.sweep_analysis[sweep_id]
        info_text = f"Sweep {sweep_id}: {analysis['description']}\n\n"
        
        if analysis['varying_params']:
            info_text += "Varying parameters:\n"
            for param in analysis['varying_params']:
                param_info = analysis['param_info'][param]
                param_name = param.replace(' (M)', '')
                if param_info['type'] == 'varying':
                    range_info = param_info['range']
                    info_text += f"• {param_name}: {range_info[0]:.3f} - {range_info[1]:.3f} M\n"
        
        if analysis['constant_params']:
            info_text += "\nConstant parameters:\n"
            for param in analysis['constant_params']:
                param_info = analysis['param_info'][param]
                param_name = param.replace(' (M)', '')
                if param_info['type'] == 'constant':
                    info_text += f"• {param_name}: {param_info['value']:.3f} M\n"
                    
        info_text += f"\nSubstrates: {analysis['substrates']}\n"
        info_text += f"Data points: {analysis['n_entries']}"
        
        self.sweep_info_text.delete(1.0, tk.END)
        self.sweep_info_text.insert(tk.END, info_text)

if __name__ == "__main__":
    app = JVApp()
    app.mainloop()
