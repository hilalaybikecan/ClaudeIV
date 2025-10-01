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
import matplotlib.pyplot as plt

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

    # PCE (%) = Pout_mpp / Pin × 100 × 1.3 (sun intensity correction)
    PCE_pct = None
    if light_mw_cm2 > 0:
        PCE_pct = (Pmpp_out / light_mw_cm2) * 100.0 * 1.3

    # Report Jsc as positive mA/cm² × 1.3 (sun intensity correction)
    Jsc_mAcm2 = None if Jsc_Acm2 is None else abs(Jsc_Acm2) * 1e3 * 1.3

    return Voc, Jsc_mAcm2, FF_pct, PCE_pct

# ---------------------------- Group mapping ----------------------------
def comp_to_group(comp: int) -> int:
    if comp in (1, 2): return 1
    if comp in (10, 11): return 9
    return comp - 1  # 3..9 -> 2..8

# ---------------------------- Experimental conditions ----------------------------
def load_experimental_conditions(excel_path: Optional[str] = None) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Load experimental conditions from Excel file
    
    Returns:
        tuple: (conditions_df, runsheet_df) where conditions_df is from ROSIE sheet
               and runsheet_df is from Runsheet sheet
    """
    if excel_path is None:
        # Try to find experiment sheets in the same directory
        excel_path = Path(__file__).parent / "experiment sheets.xlsx"
    
    excel_path = Path(excel_path)
    print(f"Looking for Excel file at: {excel_path}")
    print(f"File exists: {excel_path.exists()}")
    
    if not excel_path.exists():
        print(f"Excel file not found: {excel_path}")
        return None, None
    
    try:
        # Check available sheets first
        xlsx = pd.ExcelFile(excel_path)
        print(f"Available sheets: {xlsx.sheet_names}")
        
        conditions_df = None
        runsheet_df = None
        
        # Load ROSIE sheet if available
        if 'ROSIE' in xlsx.sheet_names:
            conditions_df = pd.read_excel(excel_path, sheet_name='ROSIE')
            print(f"Successfully loaded {len(conditions_df)} rows from ROSIE sheet")
        else:
            print("'ROSIE' sheet not found in Excel file")
        
        # Load Runsheet if available
        if 'Runsheet' in xlsx.sheet_names:
            runsheet_df = pd.read_excel(excel_path, sheet_name='Runsheet')
            print(f"Successfully loaded {len(runsheet_df)} rows from Runsheet")
        else:
            print("'Runsheet' sheet not found in Excel file")
        
        # If neither sheet is found, return None
        if conditions_df is None and runsheet_df is None:
            print("Neither 'ROSIE' nor 'Runsheet' sheets found")
            return None, None
            
        # If only Runsheet is available, use it as conditions_df for backwards compatibility
        if conditions_df is None and runsheet_df is not None:
            print("Using Runsheet as conditions_df for backwards compatibility")
            conditions_df = runsheet_df
            
        return conditions_df, runsheet_df
        
    except Exception as e:
        print(f"Error loading experimental conditions: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def detect_varying_columns(sweep_data: pd.DataFrame, threshold: float = 0.01) -> List[str]:
    """Detect which columns have meaningful variation within a sweep
    
    Args:
        sweep_data: DataFrame containing data for a single sweep
        threshold: Minimum relative variation to consider significant
    
    Returns:
        List of column names that show meaningful variation
    """
    varying_columns = []
    
    # Skip non-numeric columns and known identifier columns
    skip_columns = {'Substrate', 'Sweep', 'Sample', 'Position', 'Pixel'}
    
    for col in sweep_data.columns:
        if col in skip_columns:
            continue
            
        # Only check numeric columns
        if not pd.api.types.is_numeric_dtype(sweep_data[col]):
            continue
            
        # Get non-null, non-zero values
        values = sweep_data[col].dropna()
        nonzero_values = values[values > 0]
        
        if len(nonzero_values) < 2:
            continue
            
        # Calculate relative variation
        min_val, max_val = nonzero_values.min(), nonzero_values.max()
        if min_val > 0:
            relative_variation = (max_val - min_val) / min_val
            if relative_variation > threshold:
                varying_columns.append(col)
        else:
            # For values that can be zero, use absolute variation
            if (max_val - min_val) > threshold:
                varying_columns.append(col)
                
    return varying_columns


def generate_param_name(column_name: str) -> str:
    """Generate clean parameter name from column header
    
    Args:
        column_name: Original column name from Excel
        
    Returns:
        Clean, formatted parameter name
    """
    # Remove common units and formatting
    clean_name = column_name.strip()
    
    # Remove concentration units like (M), (mM), etc.
    clean_name = re.sub(r'\s*\([mM]*[Mm]*\)\s*$', '', clean_name)
    
    # Handle common chemical names and abbreviations
    replacements = {
        'PbI2': 'PbI₂',
        'excess PbI2': 'Excess PbI₂', 
        'with Thiourea': 'Thiourea',
        'with FABF4': 'FABF₄',
        'FABF4': 'FABF₄',
        'CsI': 'CsI',
        'FAI': 'FAI', 
        'DMAI': 'DMAI',
        'DMPU': 'DMPU',
        'MAI': 'MAI'
    }
    
    for old, new in replacements.items():
        if old in clean_name:
            clean_name = clean_name.replace(old, new)
            break
    
    return clean_name


def classify_sweep_type(varying_params: List[str], param_info: Dict) -> str:
    """Classify sweep type and generate description
    
    Args:
        varying_params: List of varying parameter names
        param_info: Dictionary with parameter details
        
    Returns:
        Descriptive string for the sweep type
    """
    n_params = len(varying_params)
    
    if n_params == 0:
        return "Control (no variation)"
    elif n_params == 1:
        param_name = generate_param_name(varying_params[0])
        param_range = param_info[varying_params[0]]['range']
        return f"{param_name} ({param_range[0]:.3f}–{param_range[1]:.3f})"
    elif n_params == 2:
        param1 = generate_param_name(varying_params[0])
        param2 = generate_param_name(varying_params[1])
        return f"{param1} × {param2} (2D)"
    else:
        param_names = [generate_param_name(p) for p in varying_params[:3]]
        if n_params > 3:
            return f"{' × '.join(param_names)} + {n_params-3} more ({n_params}D)"
        else:
            return f"{' × '.join(param_names)} ({n_params}D)"


def analyze_sweep_parameters(conditions_df: pd.DataFrame, runsheet_df: Optional[pd.DataFrame] = None) -> Dict[int, Dict]:
    """Analyze what parameters vary within each sweep using Runsheet data for robustness
    
    Args:
        conditions_df: DataFrame from ROSIE sheet (for backwards compatibility)  
        runsheet_df: DataFrame from Runsheet sheet (preferred for analysis)
    
    Returns:
        Dictionary mapping sweep IDs to analysis results
    """
    sweep_analysis = {}
    
    # Use runsheet_df if available, otherwise fall back to conditions_df
    analysis_df = runsheet_df if runsheet_df is not None else conditions_df
    
    print(f"[SWEEP ANALYSIS] Using {'Runsheet' if runsheet_df is not None else 'conditions'} data for analysis")
    print(f"[SWEEP ANALYSIS] Columns available: {list(analysis_df.columns)}")
    
    # Get unique sweeps
    if 'Sweep' not in analysis_df.columns:
        print("[SWEEP ANALYSIS] ERROR: No 'Sweep' column found in analysis data")
        return {}
        
    unique_sweeps = analysis_df['Sweep'].unique()
    print(f"[SWEEP ANALYSIS] Found {len(unique_sweeps)} unique sweeps: {sorted(unique_sweeps)}")
    
    for sweep_id in unique_sweeps:
        print(f"\n[SWEEP ANALYSIS] Analyzing Sweep {sweep_id}")
        sweep_data = analysis_df[analysis_df['Sweep'] == sweep_id]
        print(f"[SWEEP ANALYSIS] Sweep {sweep_id} has {len(sweep_data)} entries")
        
        # Detect varying columns using our new helper function
        varying_params = detect_varying_columns(sweep_data, threshold=0.01)
        print(f"[SWEEP ANALYSIS] Sweep {sweep_id} varying parameters: {varying_params}")
        
        # Build parameter info for varying parameters
        param_info = {}
        for col in varying_params:
            values = sweep_data[col].dropna()
            nonzero_values = values[values > 0]
            if len(nonzero_values) > 0:
                param_info[col] = {
                    'type': 'varying',
                    'values': sorted(nonzero_values.unique()),
                    'range': [nonzero_values.min(), nonzero_values.max()]
                }
        
        # Also track constant parameters with significant values
        constant_params = []
        skip_columns = {'Substrate', 'Sweep', 'Sample', 'Position', 'Pixel'}
        for col in analysis_df.columns:
            if col in skip_columns or col in varying_params:
                continue
            if not pd.api.types.is_numeric_dtype(analysis_df[col]):
                continue
                
            values = sweep_data[col].dropna()
            nonzero_values = values[values > 0]
            if len(nonzero_values) > 0 and nonzero_values.nunique() == 1:
                constant_params.append(col)
                param_info[col] = {
                    'type': 'constant',
                    'value': nonzero_values.iloc[0]
                }
        
        # Generate description using our new helper function
        desc = classify_sweep_type(varying_params, param_info)
        print(f"[SWEEP ANALYSIS] Sweep {sweep_id} description: {desc}")
        
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
        self.aggregation_method = tk.StringVar(value="max")
        self.combine_substrates = tk.BooleanVar(value=True)
        self.combine_fr = tk.BooleanVar(value=True)
        self.grouping_mode = tk.StringVar(value="11 compositions")
        self.expand_substrate_axis = tk.BooleanVar(value=True)

        # Filters
        self.min_voc = tk.StringVar(value="")
        self.max_voc = tk.StringVar(value="")
        self.min_pce = tk.StringVar(value="")
        self.max_pce = tk.StringVar(value="")
        self.min_jsc = tk.StringVar(value="")
        self.max_jsc = tk.StringVar(value="")
        self.min_ff  = tk.StringVar(value="")
        self.max_ff  = tk.StringVar(value="")

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
        self.runsheet_df: Optional[pd.DataFrame] = None
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

        # Performance analysis matplotlib

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
        
        # Tab 3: JV Curve visualization
        self.jv_frame = ttk.Frame(self.notebook, padding=8)
        self.notebook.add(self.jv_frame, text="JV Curves")

        self._build_composition_tab()
        self._build_sweep_tab()
        self._build_jv_tab()
        
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
        
        ttk.Label(side, text="Aggregation").grid(row=13, column=0, sticky="w", pady=(4, 0))
        agg_cb = ttk.Combobox(side, textvariable=self.aggregation_method, values=["mean", "max"], state="readonly", width=12)
        agg_cb.grid(row=14, column=0, sticky="ew"); agg_cb.bind("<<ComboboxSelected>>", lambda e: self.refresh_plots())

        ttk.Separator(side).grid(row=15, column=0, sticky="ew", pady=6)
        ttk.Label(side, text="Substrate / Grouping").grid(row=16, column=0, sticky="w")
        self.substrate_cb = ttk.Combobox(side, values=["All"], state="readonly")
        self.substrate_cb.grid(row=17, column=0, sticky="ew", pady=2); self.substrate_cb.bind("<<ComboboxSelected>>", lambda e: self.refresh_plots())
        ttk.Checkbutton(side, text="Combine substrates", variable=self.combine_substrates, command=self.refresh_plots).grid(row=18, column=0, sticky="w")
        ttk.Checkbutton(side, text="Combine F & R", variable=self.combine_fr, command=self.refresh_plots).grid(row=19, column=0, sticky="w")
        ttk.Label(side, text="Grouping").grid(row=20, column=0, sticky="w", pady=(6, 0))
        grp_cb = ttk.Combobox(side, textvariable=self.grouping_mode, values=["11 compositions", "9 groups"], state="readonly", width=18)
        grp_cb.grid(row=21, column=0, sticky="ew", pady=2); grp_cb.bind("<<ComboboxSelected>>", lambda e: self.refresh_plots())
        ttk.Checkbutton(side, text="Expand x-axis by substrate", variable=self.expand_substrate_axis, command=self.refresh_plots).grid(row=22, column=0, sticky="w", pady=(2, 6))

        ttk.Separator(side).grid(row=23, column=0, sticky="ew", pady=6)
        ttk.Label(side, text="Thresholds").grid(row=24, column=0, sticky="w")
        tfrm = ttk.Frame(side); tfrm.grid(row=25, column=0, sticky="ew")
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
        """Build the parameter vs performance analysis tab"""
        self.sweep_frame.columnconfigure(0, weight=0); self.sweep_frame.columnconfigure(1, weight=1)
        self.sweep_frame.rowconfigure(0, weight=1)

        # Left side controls
        sweep_side = ttk.Frame(self.sweep_frame, padding=8); sweep_side.grid(row=0, column=0, sticky="ns")

        # Excel file selection
        ttk.Label(sweep_side, text="Parameter Data", font=("Arial", 10, "bold")).grid(row=0, column=0, sticky="w")
        excel_frame = ttk.Frame(sweep_side); excel_frame.grid(row=1, column=0, sticky="ew", pady=2)
        self.param_excel_path = tk.StringVar(value="")
        ttk.Entry(excel_frame, textvariable=self.param_excel_path, width=25).grid(row=0, column=0, sticky="ew")
        ttk.Button(excel_frame, text="Browse", command=self.browse_parameter_file).grid(row=0, column=1, padx=(2, 0))
        ttk.Button(sweep_side, text="Load Excel Data", command=self.load_parameter_data).grid(row=2, column=0, sticky="ew", pady=2)

        ttk.Separator(sweep_side).grid(row=3, column=0, sticky="ew", pady=6)

        # Parameter analysis controls
        ttk.Label(sweep_side, text="Analysis Options", font=("Arial", 10, "bold")).grid(row=4, column=0, sticky="w")

        # X-axis parameter selection
        ttk.Label(sweep_side, text="X-axis Parameter:").grid(row=5, column=0, sticky="w", pady=(5, 0))
        self.x_param_var = tk.StringVar(value="")
        self.x_param_cb = ttk.Combobox(sweep_side, textvariable=self.x_param_var,
                                       state="readonly", width=30)
        self.x_param_cb.grid(row=6, column=0, sticky="ew", pady=2)
        self.x_param_cb.bind("<<ComboboxSelected>>", self.update_parameter_plot)

        # Y-axis parameter selection (flexible - can be any parameter or performance metric)
        ttk.Label(sweep_side, text="Y-axis Parameter:").grid(row=7, column=0, sticky="w", pady=(5, 0))
        self.y_param_var = tk.StringVar(value="PCE_pct")
        self.y_param_cb = ttk.Combobox(sweep_side, textvariable=self.y_param_var,
                                       state="readonly", width=30)
        self.y_param_cb.grid(row=8, column=0, sticky="ew", pady=2)
        self.y_param_cb.bind("<<ComboboxSelected>>", self.update_parameter_plot)

        # Optional second parameter for 2D plots
        ttk.Label(sweep_side, text="Color Parameter (optional):").grid(row=9, column=0, sticky="w", pady=(5, 0))
        self.color_param_var = tk.StringVar(value="None")
        self.color_param_cb = ttk.Combobox(sweep_side, textvariable=self.color_param_var,
                                           state="readonly", width=30)
        self.color_param_cb.grid(row=10, column=0, sticky="ew", pady=2)
        self.color_param_cb.bind("<<ComboboxSelected>>", self.update_parameter_plot)

        # Plot type selection (enhanced with performance plotting options)
        ttk.Label(sweep_side, text="Plot Type:").grid(row=11, column=0, sticky="w", pady=(5, 0))
        self.plot_type_var = tk.StringVar(value="scatter")
        plot_type_cb = ttk.Combobox(sweep_side, textvariable=self.plot_type_var,
                                    values=["scatter", "bubble", "line", "heatmap", "surface", "violin", "box", "parallel_coords"],
                                    state="readonly", width=20)
        plot_type_cb.grid(row=12, column=0, sticky="ew", pady=2)
        plot_type_cb.bind("<<ComboboxSelected>>", self.update_parameter_plot)

        ttk.Separator(sweep_side).grid(row=13, column=0, sticky="ew", pady=6)

        # Plot control buttons
        ttk.Button(sweep_side, text="Update Plot", command=self.update_parameter_plot).grid(row=14, column=0, sticky="ew", pady=2)
        ttk.Button(sweep_side, text="Save Plot", command=self.save_parameter_plot).grid(row=15, column=0, sticky="ew", pady=2)

        # Data info
        ttk.Label(sweep_side, text="Data Info", font=("Arial", 10, "bold")).grid(row=16, column=0, sticky="w", pady=(10, 2))
        self.param_info_text = tk.Text(sweep_side, height=6, width=30, wrap=tk.WORD)
        self.param_info_text.grid(row=17, column=0, sticky="ew", pady=2)
        param_info_scroll = ttk.Scrollbar(sweep_side, command=self.param_info_text.yview)
        self.param_info_text.config(yscrollcommand=param_info_scroll.set)

        # Right panel for parameter analysis
        sweep_right = ttk.Frame(self.sweep_frame, padding=8); sweep_right.grid(row=0, column=1, sticky="nsew")
        sweep_right.rowconfigure(0, weight=1); sweep_right.columnconfigure(0, weight=1)

        # Plot canvas for parameter analysis
        self.sweep_canvas = FigureCanvasTkAgg(self.sweep_fig, master=sweep_right)
        self.sweep_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        # Initialize parameter data storage
        self.parameter_data = None
        self.available_columns = []

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
                self.conditions_df, self.runsheet_df = load_experimental_conditions(excel_path)
                print(f"[AUTO LOAD] Result from specified path: conditions={self.conditions_df is not None}, runsheet={self.runsheet_df is not None}")
            else:
                # Try default path
                default_path = Path(__file__).parent / "experiment sheets.xlsx"
                print(f"[AUTO LOAD] Trying default path: {default_path}")
                print(f"[AUTO LOAD] Default path exists: {default_path.exists()}")
                if default_path.exists():
                    self.conditions_df, self.runsheet_df = load_experimental_conditions(str(default_path))
                    print(f"[AUTO LOAD] Result from default path: conditions={self.conditions_df is not None}, runsheet={self.runsheet_df is not None}")
                    if self.conditions_df is not None:
                        self.excel_path_var.set(str(default_path))
                        print(f"[AUTO LOAD] Set excel_path_var to: {default_path}")
                else:
                    print("[AUTO LOAD] No valid paths found")
        
        all_sweeps = map_sweeps_to_conditions(all_sweeps, self.conditions_df)
        
        # Analyze sweep parameters if conditions were loaded
        if self.conditions_df is not None:
            self.sweep_analysis = analyze_sweep_parameters(self.conditions_df, self.runsheet_df)
            print(f"[AUTO LOAD] Sweep analysis completed for {len(self.sweep_analysis)} sweeps")
        
        self.data = all_sweeps
        self.df = self._to_dataframe(all_sweeps)
        self.df["group_index"] = self.df["composition_index"].apply(comp_to_group)
        self.df_with_flags = self.df.copy()
        self.df_with_flags["include"] = True  # Initialize all rows as included

        self._populate_substrate_combo()
        self.refresh_table(); self.refresh_plots()
        # Clear any previous parameter plot
        try:
            self._clear_sweep_ax("Load parameter data to begin analysis.")
        except AttributeError:
            pass  # Parameter functions not yet loaded

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

        # Aggregate data by substrate and composition_index
        df_to_export = self.df_with_flags.copy()

        # Group by substrate and composition_index, then calculate means for numerical columns
        numerical_cols = ["Voc", "Jsc_mAcm2", "FF_pct", "PCE_pct"]
        grouping_cols = ["substrate", "composition_index"]

        # Calculate aggregated metrics - both mean and max
        aggregated_mean = df_to_export.groupby(grouping_cols)[numerical_cols].mean().reset_index()
        aggregated_max = df_to_export.groupby(grouping_cols)[numerical_cols].max().reset_index()

        # Merge mean and max data
        aggregated = aggregated_mean.copy()

        # Add max columns with "_max" suffix
        for col in numerical_cols:
            aggregated[f"{col}_max"] = aggregated_max[col]

        # Rename mean columns with "_mean" suffix for clarity
        for col in numerical_cols:
            aggregated[f"{col}_mean"] = aggregated[col]
            aggregated.drop(columns=[col], inplace=True)

        # Also include group_index for consistency
        aggregated["group_index"] = aggregated["composition_index"].apply(comp_to_group)

        # Add Well column mapping substrate to letter and composition to number
        aggregated["Well"] = aggregated.apply(lambda row: f"{chr(ord('A') + int(row['substrate']) - 1)}{int(row['composition_index'])}", axis=1)

        # Add count of aggregated measurements for reference
        counts = df_to_export.groupby(grouping_cols).size().reset_index(name='n_measurements')
        aggregated = aggregated.merge(counts, on=grouping_cols, how='left')

        # Create column order with mean and max columns
        mean_cols = [f"{col}_mean" for col in numerical_cols]
        max_cols = [f"{col}_max" for col in numerical_cols]
        column_order = ["substrate", "composition_index", "group_index", "Well", "n_measurements"] + mean_cols + max_cols
        aggregated = aggregated[column_order]

        aggregated.to_csv(path, index=False)
        messagebox.showinfo("Export", f"Aggregated data saved: {path}\n{len(aggregated)} composition groups exported")

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
        self.df_with_flags["include"] = True  # Initialize all rows as included
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
        df = self.df_with_flags.copy()
        
        # Apply threshold filtering using "include" column
        if "include" in df.columns:
            df = df[df["include"]]
        
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
        # Also refresh JV selection table when plots are refreshed
        try:
            self.refresh_jv_selection_table()
        except AttributeError:
            # JV tab not yet initialized
            pass

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
        agg_method = self.aggregation_method.get()

        if self.expand_substrate_axis.get():
            subs = sorted(data_df["substrate"].dropna().astype(int).unique().tolist())
            for sub in subs:
                sub_df = data_df[data_df["substrate"] == sub]
                if agg_method == "max":
                    # For max: group by substrate+composition, take max, then collect those max values
                    vals_by = sub_df.groupby([group_col, "substrate"])[metric].max().reset_index().groupby(group_col)[metric].apply(lambda s: s.dropna().values)
                else:
                    # For mean: show distribution of all individual measurements (current behavior)
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
            if agg_method == "max":
                # For max: group by substrate+composition, take max, then collect those max values
                vals_by = data_df.groupby([group_col, "substrate"])[metric].max().reset_index().groupby(group_col)[metric].apply(lambda s: s.dropna().values)
            else:
                # For mean: show distribution of all individual measurements (current behavior)
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
        title = f"Boxplot ({agg_method}): {metric} by {'Group' if use_groups else 'Composition'}"
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

        agg = self.aggregation_method.get()

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

        # Fix substrate axis labels - ensure they correspond to actual substrate numbers in sorted order
        substrate_labels = [str(int(s)) for s in sorted(pivot.index)]
        self.ax.set_yticks(range(pivot.shape[0])); self.ax.set_yticklabels(substrate_labels)

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
        
    # -------------------- Parameter analysis methods --------------------
    def browse_parameter_file(self):
        """Browse for Excel file containing parameter data"""
        from tkinter import filedialog
        path = filedialog.askopenfilename(
            title="Select parameter data Excel file",
            filetypes=[("Excel files", "*.xlsx *.xls"), ("All files", "*.*")]
        )
        if path:
            self.param_excel_path.set(path)
            
    def load_parameter_data(self):
        """Load parameter data from Excel file"""
        excel_path = self.param_excel_path.get()
        if not excel_path:
            messagebox.showwarning("No file", "Please specify an Excel file path.")
            return

        try:
            # Try to load the Excel file
            self.parameter_data = pd.read_excel(excel_path)

            # Detect available columns for parameter selection
            numeric_columns = self.parameter_data.select_dtypes(include=[np.number]).columns.tolist()
            all_columns = self.parameter_data.columns.tolist()

            # Performance metrics that are always available from JV data
            performance_metrics = ["Voc", "Jsc_mAcm2", "FF_pct", "PCE_pct"]

            # Combine parameter columns with performance metrics for Y-axis
            y_axis_options = performance_metrics + all_columns

            # Update dropdowns with available columns
            self.available_columns = all_columns
            self.x_param_cb['values'] = all_columns
            self.y_param_cb['values'] = y_axis_options
            self.color_param_cb['values'] = ['None'] + y_axis_options

            # Set default parameters
            if numeric_columns:
                self.x_param_var.set(numeric_columns[0])
            # Keep PCE as default Y parameter
            self.y_param_var.set("PCE_pct")

            # Update info display
            self.update_parameter_info()

            messagebox.showinfo("Success", f"Loaded Excel file with {len(self.parameter_data)} rows and {len(all_columns)} columns.")

        except Exception as e:
            messagebox.showerror("Load failed", f"Could not load parameter data from {excel_path}\n\nError: {str(e)}")

    def update_parameter_info(self):
        """Update the parameter info text display"""
        self.param_info_text.delete(1.0, tk.END)

        if self.parameter_data is None:
            self.param_info_text.insert(tk.END, "No data loaded.")
            return

        info_text = f"Loaded data:\n"
        info_text += f"Rows: {len(self.parameter_data)}\n"
        info_text += f"Columns: {len(self.parameter_data.columns)}\n\n"
        info_text += "Available columns:\n"

        for col in self.parameter_data.columns:
            dtype = str(self.parameter_data[col].dtype)
            non_null = self.parameter_data[col].notna().sum()
            info_text += f"• {col} ({dtype}): {non_null}/{len(self.parameter_data)} values\n"

        self.param_info_text.insert(tk.END, info_text)
                
    def update_parameter_plot(self, event=None):
        """Generate parameter vs performance plots"""
        if self.parameter_data is None:
            self._clear_sweep_ax("No parameter data loaded.")
            return

        if self.df_with_flags is None or self.df_with_flags.empty:
            self._clear_sweep_ax("No JV data available.")
            return

        x_param = self.x_param_var.get()
        y_param = self.y_param_var.get()
        color_param = self.color_param_var.get()
        plot_type = self.plot_type_var.get()

        if not x_param:
            self._clear_sweep_ax("Please select an X-axis parameter.")
            return

        # Try to match parameter data with JV data
        try:
            combined_data = self._merge_parameter_and_jv_data()
            if combined_data is None or combined_data.empty:
                self._clear_sweep_ax("Could not match parameter data with JV data.")
                return
        except Exception as e:
            self._clear_sweep_ax(f"Error combining data: {str(e)}")
            return

        self._reset_sweep_axes()

        # Generate plot based on type
        if plot_type == "scatter":
            self._plot_parameter_scatter(combined_data, x_param, y_param, color_param)
        elif plot_type == "bubble":
            self._plot_parameter_bubble(combined_data, x_param, y_param, color_param)
        elif plot_type == "line":
            self._plot_parameter_line(combined_data, x_param, y_param, color_param)
        elif plot_type == "heatmap":
            self._plot_parameter_heatmap(combined_data, x_param, y_param, color_param)
        elif plot_type == "surface":
            self._plot_parameter_surface(combined_data, x_param, y_param, color_param)
        elif plot_type == "violin":
            self._plot_parameter_violin(combined_data, x_param, y_param, color_param)
        elif plot_type == "box":
            self._plot_parameter_box(combined_data, x_param, y_param, color_param)
        elif plot_type == "parallel_coords":
            self._plot_parameter_parallel_coords(combined_data, x_param, y_param, color_param)

        self.sweep_canvas.draw_idle()
                
    def _merge_parameter_and_jv_data(self):
        """Merge parameter data with JV data"""
        if self.parameter_data is None or self.df_with_flags is None:
            return None

        # Try to find common columns for merging (substrate, sample, position, etc.)
        jv_data = self.df_with_flags.copy()
        param_data = self.parameter_data.copy()

        # Common merge strategies
        merge_columns = []

        # Strategy 1: Try 'substrate' column
        if 'substrate' in jv_data.columns and 'substrate' in param_data.columns:
            merge_columns = ['substrate']
        elif 'Substrate' in param_data.columns and 'substrate' in jv_data.columns:
            param_data = param_data.rename(columns={'Substrate': 'substrate'})
            merge_columns = ['substrate']

        # Strategy 2: Try other common identifiers
        if not merge_columns:
            for col_jv, col_param in [('substrate', 'Sample'), ('substrate', 'Position'),
                                     ('pixel_id', 'Pixel'), ('composition_index', 'Composition')]:
                if col_jv in jv_data.columns and col_param in param_data.columns:
                    param_data = param_data.rename(columns={col_param: col_jv})
                    merge_columns = [col_jv]
                    break

        if not merge_columns:
            # Fallback: try to merge by index if sizes match
            if len(jv_data) == len(param_data):
                # Reset indices and merge by position
                jv_data = jv_data.reset_index(drop=True)
                param_data = param_data.reset_index(drop=True)
                combined = pd.concat([jv_data, param_data], axis=1)
                return combined
            else:
                return None

        # Perform the merge
        try:
            combined = jv_data.merge(param_data, on=merge_columns, how='inner')
            return combined
        except Exception:
            return None

    def save_parameter_plot(self):
        """Save the current parameter plot"""
        if not hasattr(self.sweep_ax, 'figure') or self.sweep_ax.figure is None:
            messagebox.showwarning("No plot", "No plot to save. Generate a plot first.")
            return

        try:
            from tkinter import filedialog
            path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("SVG files", "*.svg"), ("PDF files", "*.pdf")]
            )
            if path:
                self.sweep_fig.savefig(path, dpi=300, bbox_inches='tight')
                messagebox.showinfo("Success", f"Plot saved to {path}")
        except Exception as e:
            messagebox.showerror("Save failed", f"Could not save plot: {str(e)}")

    def _plot_parameter_scatter(self, data, x_param, y_param, color_param):
        """Create scatter plot of parameter vs performance"""
        x_values = data[x_param].dropna()
        y_values = data[y_param].dropna()

        # Align x and y values
        valid_indices = data[[x_param, y_param]].dropna().index
        x_values = data.loc[valid_indices, x_param]
        y_values = data.loc[valid_indices, y_param]

        if len(x_values) == 0:
            self._clear_sweep_ax("No valid data points for selected parameters.")
            return

        if color_param and color_param != 'None' and color_param in data.columns:
            color_values = data.loc[valid_indices, color_param]
            scatter = self.sweep_ax.scatter(x_values, y_values, c=color_values, cmap='viridis', alpha=0.7)
            self.sweep_fig.colorbar(scatter, ax=self.sweep_ax, label=color_param)
        else:
            self.sweep_ax.scatter(x_values, y_values, alpha=0.7)

        self.sweep_ax.set_xlabel(x_param)
        self.sweep_ax.set_ylabel(y_param)
        self.sweep_ax.set_title(f'{y_param} vs {x_param}')
        self.sweep_ax.grid(True, alpha=0.3)

    def _plot_parameter_line(self, data, x_param, y_param, color_param):
        """Create line plot of parameter vs performance"""
        # Group by x_param and calculate mean y_param
        grouped = data.groupby(x_param)[y_param].agg(['mean', 'std']).reset_index()

        if grouped.empty:
            self._clear_sweep_ax("No valid data for line plot.")
            return

        x_values = grouped[x_param]
        y_means = grouped['mean']
        y_stds = grouped['std'].fillna(0)

        self.sweep_ax.plot(x_values, y_means, 'o-', linewidth=2, markersize=6)
        self.sweep_ax.fill_between(x_values, y_means - y_stds, y_means + y_stds, alpha=0.3)

        self.sweep_ax.set_xlabel(x_param)
        self.sweep_ax.set_ylabel(y_param)
        self.sweep_ax.set_title(f'{y_param} vs {x_param} (with std)')
        self.sweep_ax.grid(True, alpha=0.3)

    def _plot_parameter_heatmap(self, data, x_param, y_param, color_param):
        """Create heatmap of parameter vs performance"""
        if color_param == 'None' or not color_param:
            # Use y_param as the color parameter
            color_param = y_param

        # Create pivot table for heatmap
        try:
            # For heatmap, we need to discretize continuous variables
            x_bins = pd.cut(data[x_param], bins=10, duplicates='drop')
            if color_param != y_param and color_param in data.columns:
                # y_param goes to y-axis, color_param values are plotted
                y_bins = pd.cut(data[y_param], bins=10, duplicates='drop')
                # Note: unstack() makes the last groupby level into columns, so we need y_bins as the first argument
                pivot_table = data.groupby([y_bins, x_bins])[color_param].mean().unstack()
                ylabel = y_param
                color_label = color_param
            else:
                # Use substrate or another categorical variable for y-axis
                if 'substrate' in data.columns:
                    pivot_table = data.groupby([x_bins, 'substrate'])[y_param].mean().unstack()
                    ylabel = 'Substrate'
                    color_label = y_param
                else:
                    self._clear_sweep_ax("Cannot create heatmap: need categorical variable for y-axis.")
                    return

            if pivot_table.empty:
                self._clear_sweep_ax("No data for heatmap.")
                return

            im = self.sweep_ax.imshow(pivot_table.values, cmap='viridis', aspect='auto', interpolation='nearest')
            self.sweep_fig.colorbar(im, ax=self.sweep_ax, label=color_label)

            # Set ticks and labels
            # For heatmap: rows (y-axis) = index, columns (x-axis) = columns
            self.sweep_ax.set_xticks(range(len(pivot_table.columns)))
            # Format x-axis labels (these are x_param values)
            x_labels = []
            for col in pivot_table.columns:
                if hasattr(col, 'mid'):  # pd.Interval object (x_bins)
                    x_labels.append(f"{col.mid:.2f}")
                else:
                    x_labels.append(str(col))
            self.sweep_ax.set_xticklabels(x_labels, rotation=45)

            self.sweep_ax.set_yticks(range(len(pivot_table.index)))
            # Format y-axis labels (these are y_param values)
            y_labels = []
            for idx in pivot_table.index:
                if hasattr(idx, 'mid'):  # pd.Interval object (y_bins)
                    y_labels.append(f"{idx.mid:.2f}")
                else:
                    y_labels.append(str(idx))
            self.sweep_ax.set_yticklabels(y_labels)

            self.sweep_ax.set_xlabel(x_param)
            self.sweep_ax.set_ylabel(ylabel)
            self.sweep_ax.set_title(f'Heatmap: {color_label} vs {x_param} and {ylabel}')

        except Exception as e:
            self._clear_sweep_ax(f"Error creating heatmap: {str(e)}")

    def _plot_parameter_surface(self, data, x_param, y_param, color_param):
        """Create 3D surface plot of parameter vs performance"""
        if color_param == 'None' or not color_param or color_param not in data.columns:
            self._clear_sweep_ax("Surface plot requires a color parameter.")
            return

        try:
            from mpl_toolkits.mplot3d import Axes3D

            # Clear and create 3D subplot
            self.sweep_fig.clf()
            ax = self.sweep_fig.add_subplot(111, projection='3d')

            # Get the data
            valid_data = data[[x_param, color_param, y_param]].dropna()
            if len(valid_data) < 4:
                self._clear_sweep_ax("Not enough data points for surface plot.")
                return

            x_vals = valid_data[x_param]
            y_vals = valid_data[color_param]
            z_vals = valid_data[y_param]

            # Create a scatter plot in 3D space
            scatter = ax.scatter(x_vals, y_vals, z_vals, c=z_vals, cmap='viridis')

            ax.set_xlabel(x_param)
            ax.set_ylabel(color_param)
            ax.set_zlabel(y_param)
            ax.set_title(f'3D Plot: {y_param} vs {x_param} and {color_param}')

            self.sweep_fig.colorbar(scatter, ax=ax, label=y_param, shrink=0.8)

        except ImportError:
            self._clear_sweep_ax("3D plotting not available.")
        except Exception as e:
            self._clear_sweep_ax(f"Error creating surface plot: {str(e)}")

    def _plot_parameter_bubble(self, data, x_param, y_param, color_param):
        """Create bubble plot with size based on color parameter"""
        valid_indices = data[[x_param, y_param]].dropna().index
        x_values = data.loc[valid_indices, x_param]
        y_values = data.loc[valid_indices, y_param]

        if len(x_values) == 0:
            self._clear_sweep_ax("No valid data points for selected parameters.")
            return

        if color_param and color_param != 'None' and color_param in data.columns:
            size_values = data.loc[valid_indices, color_param]
            # Normalize sizes to reasonable range (20-200 pixels)
            size_norm = (size_values - size_values.min()) / (size_values.max() - size_values.min()) if size_values.max() != size_values.min() else 0.5
            sizes = 20 + 180 * size_norm
            scatter = self.sweep_ax.scatter(x_values, y_values, s=sizes, c=size_values, cmap='viridis', alpha=0.6)
            self.sweep_cbar = self.sweep_fig.colorbar(scatter, ax=self.sweep_ax, label=f'{color_param} (size & color)')
        else:
            self.sweep_ax.scatter(x_values, y_values, alpha=0.7, s=100)

        self.sweep_ax.set_xlabel(x_param, fontsize=12)
        self.sweep_ax.set_ylabel(y_param, fontsize=12)
        self.sweep_ax.set_title(f'Bubble Plot: {y_param} vs {x_param}', fontsize=14)
        self.sweep_ax.grid(True, alpha=0.3)

    def _plot_parameter_violin(self, data, x_param, y_param, color_param):
        """Create violin plot"""
        try:
            import seaborn as sns

            # Clear and recreate axes for seaborn
            self._reset_sweep_axes()

            # Create violin plot
            if color_param and color_param != 'None' and color_param in data.columns:
                sns.violinplot(data=data, x=x_param, y=y_param, hue=color_param, ax=self.sweep_ax)
            else:
                sns.violinplot(data=data, x=x_param, y=y_param, ax=self.sweep_ax)

            self.sweep_ax.set_xlabel(x_param, fontsize=12)
            self.sweep_ax.set_ylabel(y_param, fontsize=12)
            self.sweep_ax.set_title(f'Violin Plot: {y_param} vs {x_param}', fontsize=14)
            self.sweep_ax.grid(True, alpha=0.3)

        except ImportError:
            self._clear_sweep_ax("Seaborn not available for violin plots. Install with: pip install seaborn")
        except Exception as e:
            self._clear_sweep_ax(f"Error creating violin plot: {str(e)}")

    def _plot_parameter_box(self, data, x_param, y_param, color_param):
        """Create box plot"""
        try:
            # Group data for box plot
            grouped_data = []
            labels = []

            for group_val in data[x_param].unique():
                group_data = data[data[x_param] == group_val][y_param].dropna()
                if len(group_data) > 0:
                    grouped_data.append(group_data)
                    labels.append(str(group_val))

            if not grouped_data:
                self._clear_sweep_ax("No valid data for box plot.")
                return

            box_plot = self.sweep_ax.boxplot(grouped_data, labels=labels, patch_artist=True)

            # Color boxes if color parameter is specified
            if color_param and color_param != 'None' and color_param in data.columns:
                colors = plt.cm.viridis(np.linspace(0, 1, len(box_plot['boxes'])))
                for patch, color in zip(box_plot['boxes'], colors):
                    patch.set_facecolor(color)

            self.sweep_ax.set_xlabel(x_param, fontsize=12)
            self.sweep_ax.set_ylabel(y_param, fontsize=12)
            self.sweep_ax.set_title(f'Box Plot: {y_param} vs {x_param}', fontsize=14)
            self.sweep_ax.grid(True, alpha=0.3)

        except Exception as e:
            self._clear_sweep_ax(f"Error creating box plot: {str(e)}")

    def _plot_parameter_parallel_coords(self, data, x_param, y_param, color_param):
        """Create parallel coordinates plot"""
        try:
            import pandas as pd
            from pandas.plotting import parallel_coordinates

            # Select numeric columns for parallel coordinates
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

            # Include the parameters and performance metrics
            important_cols = [x_param, y_param]
            if color_param and color_param != 'None' and color_param in data.columns:
                important_cols.append(color_param)

            # Add performance metrics if available
            perf_metrics = ['Voc', 'Jsc_mAcm2', 'FF_pct', 'PCE_pct']
            for metric in perf_metrics:
                if metric in numeric_cols and metric not in important_cols:
                    important_cols.append(metric)

            # Add other numeric columns (limit to reasonable number)
            other_cols = [col for col in numeric_cols if col not in important_cols][:5]
            plot_cols = important_cols + other_cols

            # Remove duplicates and ensure columns exist
            plot_cols = [col for col in plot_cols if col in data.columns]
            plot_cols = list(dict.fromkeys(plot_cols))  # Remove duplicates while preserving order

            if len(plot_cols) < 2:
                self._clear_sweep_ax("Need at least 2 numeric columns for parallel coordinates plot.")
                return

            # Prepare data for parallel coordinates
            plot_data = data[plot_cols].dropna()

            if len(plot_data) == 0:
                self._clear_sweep_ax("No valid data for parallel coordinates plot.")
                return

            # Use color parameter for grouping, or create groups based on y_param quartiles
            if color_param and color_param != 'None' and color_param in data.columns:
                if data[color_param].dtype in ['object', 'category'] or data[color_param].nunique() <= 10:
                    class_col = color_param
                    plot_data[class_col] = data[color_param]
                else:
                    # Create quartile groups for continuous color parameter
                    quartiles = pd.qcut(data[color_param], q=4, duplicates='drop', labels=['Q1', 'Q2', 'Q3', 'Q4'])
                    plot_data['quartile_group'] = quartiles
                    class_col = 'quartile_group'
            else:
                # Create quartile groups based on y_param
                quartiles = pd.qcut(data[y_param], q=4, duplicates='drop', labels=['Low', 'Mid-Low', 'Mid-High', 'High'])
                plot_data['performance_group'] = quartiles
                class_col = 'performance_group'

            # Remove rows with NaN in class column
            plot_data = plot_data.dropna(subset=[class_col])

            if len(plot_data) == 0:
                self._clear_sweep_ax("No valid data after grouping for parallel coordinates plot.")
                return

            # Create the parallel coordinates plot
            self._reset_sweep_axes()
            parallel_coordinates(plot_data, class_col, ax=self.sweep_ax, alpha=0.7)

            self.sweep_ax.set_title(f'Parallel Coordinates Plot (colored by {class_col})', fontsize=14)
            self.sweep_ax.grid(True, alpha=0.3)

            # Rotate x-axis labels for better readability
            self.sweep_ax.tick_params(axis='x', rotation=45)

            # Move legend outside plot area
            self.sweep_ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        except ImportError as e:
            self._clear_sweep_ax("Pandas plotting not available for parallel coordinates. Error: " + str(e))
        except Exception as e:
            self._clear_sweep_ax(f"Error creating parallel coordinates plot: {str(e)}")

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
        
        # Get parameter values from the appropriate sheet
        # Use runsheet_df if available since that's where our analysis comes from
        source_df = self.runsheet_df if self.runsheet_df is not None else self.conditions_df
        if source_df is None:
            return
            
        sweep_conditions = source_df[source_df['Sweep'] == sweep_id]
        
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
                # Handle case where Well column might not exist (e.g., in Runsheet)
                if 'Well' in sweep_conditions.columns:
                    condition_match = sweep_conditions[
                        (sweep_conditions['Substrate'] == substrate_id) & 
                        (sweep_conditions['Well'] == well_id)
                    ]
                else:
                    # For sheets without Well column, just match by substrate
                    condition_match = sweep_conditions[
                        sweep_conditions['Substrate'] == substrate_id
                    ]
                
                if not condition_match.empty:
                    condition_row = condition_match.iloc[0]
                    param_value = condition_row[param_col] if param_col in condition_row.index else None
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
        
        # Get parameter values from the appropriate sheet
        # Use runsheet_df if available since that's where our analysis comes from
        source_df = self.runsheet_df if self.runsheet_df is not None else self.conditions_df
        if source_df is None:
            return
            
        sweep_conditions = source_df[source_df['Sweep'] == sweep_id]
        
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
                # Handle case where Well column might not exist (e.g., in Runsheet)
                if 'Well' in sweep_conditions.columns:
                    condition_match = sweep_conditions[
                        (sweep_conditions['Substrate'] == substrate_id) & 
                        (sweep_conditions['Well'] == well_id)
                    ]
                else:
                    # For sheets without Well column, just match by substrate
                    condition_match = sweep_conditions[
                        sweep_conditions['Substrate'] == substrate_id
                    ]
                
                if not condition_match.empty:
                    condition_row = condition_match.iloc[0]
                    param1_value = condition_row[param1_col] if param1_col in condition_row.index else None
                    param2_value = condition_row[param2_col] if param2_col in condition_row.index else None
                    
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
        
        # Get parameter values from the appropriate sheet
        # Use runsheet_df if available since that's where our analysis comes from
        source_df = self.runsheet_df if self.runsheet_df is not None else self.conditions_df
        if source_df is None:
            return
            
        sweep_conditions = source_df[source_df['Sweep'] == sweep_id]
        
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
                # Handle case where Well column might not exist (e.g., in Runsheet)
                if 'Well' in sweep_conditions.columns:
                    condition_match = sweep_conditions[
                        (sweep_conditions['Substrate'] == substrate_id) & 
                        (sweep_conditions['Well'] == well_id)
                    ]
                else:
                    # For sheets without Well column, just match by substrate
                    condition_match = sweep_conditions[
                        sweep_conditions['Substrate'] == substrate_id
                    ]
                
                if not condition_match.empty:
                    condition_row = condition_match.iloc[0]
                    param1_value = condition_row[param1_col] if param1_col in condition_row.index else None
                    param2_value = condition_row[param2_col] if param2_col in condition_row.index else None
                    
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


    def _build_jv_tab(self):
        """Build the JV curve visualization tab"""
        self.jv_frame.columnconfigure(0, weight=0); self.jv_frame.columnconfigure(1, weight=1)
        self.jv_frame.rowconfigure(0, weight=1)

        # Left panel: Sample selection and controls
        jv_left = ttk.Frame(self.jv_frame, padding=8); jv_left.grid(row=0, column=0, sticky="ns")
        
        # Sample selection from main table
        ttk.Label(jv_left, text="Sample Selection").grid(row=0, column=0, sticky="w")
        ttk.Label(jv_left, text="Select samples from the table below:").grid(row=1, column=0, sticky="w", pady=(0, 4))
        
        # Mini table for sample selection
        selection_frame = ttk.Frame(jv_left); selection_frame.grid(row=2, column=0, sticky="ew")
        jv_left.columnconfigure(0, weight=1)
        
        # Table showing filtered data for selection (same format as main table)
        columns = ("substrate", "pixel_id", "comp", "group", "pos", "dir", "Voc", "Jsc_mAcm2", "FF_pct", "PCE_pct")
        self.jv_selection_tree = ttk.Treeview(selection_frame, columns=columns, show="headings", 
                                             selectmode="extended", height=8)
        # Initialize sorting state
        self._jv_sort_column = None
        self._jv_sort_reverse = False
        
        for c in columns:
            self.jv_selection_tree.heading(c, text=c, command=lambda col=c: self.sort_jv_by_column(col))
            self.jv_selection_tree.column(c, width=90, stretch=True)
        
        # Scrollbars for selection table
        jv_vsb = ttk.Scrollbar(selection_frame, orient="vertical", command=self.jv_selection_tree.yview)
        jv_hsb = ttk.Scrollbar(selection_frame, orient="horizontal", command=self.jv_selection_tree.xview)
        self.jv_selection_tree.configure(yscrollcommand=jv_vsb.set, xscrollcommand=jv_hsb.set)
        
        self.jv_selection_tree.grid(row=0, column=0, sticky="nsew")
        jv_vsb.grid(row=0, column=1, sticky="ns")
        jv_hsb.grid(row=1, column=0, sticky="ew")
        selection_frame.rowconfigure(0, weight=1)
        selection_frame.columnconfigure(0, weight=1)
        
        # Plot controls
        ttk.Separator(jv_left).grid(row=3, column=0, sticky="ew", pady=8)
        ttk.Label(jv_left, text="Plot Controls").grid(row=4, column=0, sticky="w")
        
        # Grouping option
        self.jv_group_by_direction = tk.BooleanVar(value=True)
        ttk.Checkbutton(jv_left, text="Group by F/R direction", 
                       variable=self.jv_group_by_direction,
                       command=self.plot_jv_curves).grid(row=5, column=0, sticky="w", pady=2)
        
        # Axis adjustment controls
        ttk.Separator(jv_left).grid(row=6, column=0, sticky="ew", pady=8)
        ttk.Label(jv_left, text="Plot Axes:").grid(row=7, column=0, sticky="w")
        
        # X-axis controls
        x_frame = ttk.Frame(jv_left); x_frame.grid(row=8, column=0, sticky="ew", pady=2)
        ttk.Label(x_frame, text="X-axis:").grid(row=0, column=0, sticky="w")
        self.jv_x_auto = tk.BooleanVar(value=True)
        ttk.Checkbutton(x_frame, text="Auto", variable=self.jv_x_auto).grid(row=0, column=1, sticky="w")
        
        x_limit_frame = ttk.Frame(jv_left); x_limit_frame.grid(row=9, column=0, sticky="ew")
        ttk.Label(x_limit_frame, text="Min:").grid(row=0, column=0, sticky="w")
        self.jv_x_min = tk.StringVar(value="")
        ttk.Entry(x_limit_frame, textvariable=self.jv_x_min, width=8).grid(row=0, column=1, sticky="w", padx=2)
        ttk.Label(x_limit_frame, text="Max:").grid(row=0, column=2, sticky="w")
        self.jv_x_max = tk.StringVar(value="")
        ttk.Entry(x_limit_frame, textvariable=self.jv_x_max, width=8).grid(row=0, column=3, sticky="w", padx=2)
        
        # Y-axis controls
        y_frame = ttk.Frame(jv_left); y_frame.grid(row=10, column=0, sticky="ew", pady=2)
        ttk.Label(y_frame, text="Y-axis:").grid(row=0, column=0, sticky="w")
        self.jv_y_auto = tk.BooleanVar(value=True)
        ttk.Checkbutton(y_frame, text="Auto", variable=self.jv_y_auto).grid(row=0, column=1, sticky="w")
        
        y_limit_frame = ttk.Frame(jv_left); y_limit_frame.grid(row=11, column=0, sticky="ew")
        ttk.Label(y_limit_frame, text="Min:").grid(row=0, column=0, sticky="w")
        self.jv_y_min = tk.StringVar(value="")
        ttk.Entry(y_limit_frame, textvariable=self.jv_y_min, width=8).grid(row=0, column=1, sticky="w", padx=2)
        ttk.Label(y_limit_frame, text="Max:").grid(row=0, column=2, sticky="w")
        self.jv_y_max = tk.StringVar(value="")
        ttk.Entry(y_limit_frame, textvariable=self.jv_y_max, width=8).grid(row=0, column=3, sticky="w", padx=2)
        
        # Apply axes button
        ttk.Button(jv_left, text="Apply Axes", 
                  command=self.update_jv_axes).grid(row=12, column=0, sticky="ew", pady=2)
        
        # Plot button
        ttk.Button(jv_left, text="Plot Selected JV Curves", 
                  command=self.plot_jv_curves).grid(row=13, column=0, sticky="ew", pady=8)
        
        # Clear button
        ttk.Button(jv_left, text="Clear Plot",
                  command=self.clear_jv_plot).grid(row=14, column=0, sticky="ew")

        # Save plot button
        ttk.Button(jv_left, text="Save Plot",
                  command=self.save_jv_plot).grid(row=15, column=0, sticky="ew", pady=2)
        
        # Right panel: JV curve plot
        jv_right = ttk.Frame(self.jv_frame, padding=8); jv_right.grid(row=0, column=1, sticky="nsew")
        jv_right.rowconfigure(0, weight=1); jv_right.columnconfigure(0, weight=1)
        
        # JV plot figure and canvas
        self.jv_fig = Figure(figsize=(8, 6), dpi=100)
        self.jv_ax = self.jv_fig.add_subplot(111)
        self.jv_canvas = FigureCanvasTkAgg(self.jv_fig, master=jv_right)
        self.jv_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        
        # Initialize plot
        self.clear_jv_plot()
        
    def refresh_jv_selection_table(self):
        """Refresh the JV selection table with current filtered data (same format as main table)"""
        # Clear existing items
        for item in self.jv_selection_tree.get_children():
            self.jv_selection_tree.delete(item)
        
        if self.df_with_flags is None or self.df_with_flags.empty:
            return
            
        # Get filtered data
        filtered_df = self._filtered_df()
        if filtered_df.empty:
            return
        
        # Add rows to selection table (exact same format as main table)
        for idx, r in filtered_df.iterrows():
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
            # Use dataframe index as item identifier for lookup
            self.jv_selection_tree.insert("", "end", iid=str(idx), values=vals)
    
    def sort_jv_by_column(self, column: str):
        """Sort JV selection table by clicked column, keeping F/R pairs together"""
        if self.df_with_flags is None or self.df_with_flags.empty:
            return
            
        # Toggle sort direction if same column, otherwise ascending
        if self._jv_sort_column == column:
            self._jv_sort_reverse = not self._jv_sort_reverse
        else:
            self._jv_sort_column = column
            self._jv_sort_reverse = False
            
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
            # Get filtered data to sort
            filtered_df = self._filtered_df().copy()
            if filtered_df.empty:
                return
                
            df_col = column_map[column]
            
            # Sort by main column first, then by grouping fields to keep F/R pairs together
            sort_columns = [df_col, "substrate", "pixel_id", "composition_index", "position_in_composition", "direction"]
            sort_ascending = [not self._jv_sort_reverse, True, True, True, True, True]  # forward before reverse
            
            try:
                sorted_df = filtered_df.sort_values(by=sort_columns, ascending=sort_ascending, na_position='last')
                
                # Clear and repopulate JV table with sorted data
                for item in self.jv_selection_tree.get_children():
                    self.jv_selection_tree.delete(item)
                
                for idx, r in sorted_df.iterrows():
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
                    self.jv_selection_tree.insert("", "end", iid=str(idx), values=vals)
                
                # Update column headers to show sort direction
                for col in ["substrate", "pixel_id", "comp", "group", "pos", "dir", "Voc", "Jsc_mAcm2", "FF_pct", "PCE_pct"]:
                    if col == column:
                        arrow = " ↓" if self._jv_sort_reverse else " ↑"
                        self.jv_selection_tree.heading(col, text=f"{col}{arrow}")
                    else:
                        self.jv_selection_tree.heading(col, text=col)
                        
            except Exception as e:
                print(f"Sorting error: {e}")
                # Fall back to refreshing the table normally
                self.refresh_jv_selection_table()
    
    def plot_jv_curves(self):
        """Plot JV curves for selected samples"""
        selected_items = self.jv_selection_tree.selection()
        if not selected_items:
            self.clear_jv_plot()
            return
        
        # Clear the plot
        self.jv_ax.clear()
        
        # Get the original filtered dataframe
        filtered_df = self._filtered_df()
        if filtered_df.empty:
            self.clear_jv_plot()
            return
        
        # Collect selected sweeps
        selected_sweeps = []
        for item in selected_items:
            values = self.jv_selection_tree.item(item)["values"]
            if len(values) >= 6:
                # Find matching sweep in data (updated for new table format)
                substrate = int(values[0])
                pixel_id = int(values[1])
                comp = int(values[2])
                # group = int(values[3])  # Skip group column
                pos = int(values[4])
                direction = values[5]
                
                # Find the corresponding JVSweep object
                for sweep in self.data:
                    if (sweep.substrate == substrate and 
                        sweep.pixel_id == pixel_id and
                        sweep.composition_index == comp and 
                        sweep.position_in_composition == pos and
                        sweep.direction == direction):
                        selected_sweeps.append(sweep)
                        break
        
        if not selected_sweeps:
            self.clear_jv_plot()
            return
        
        # Color schemes
        colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(selected_sweeps))))
        direction_colors = {"forward": "blue", "reverse": "red"}
        
        # Group by direction if requested
        if self.jv_group_by_direction.get():
            forward_sweeps = [s for s in selected_sweeps if s.direction == "forward"]
            reverse_sweeps = [s for s in selected_sweeps if s.direction == "reverse"]
            
            # Plot F/R pairs together with same colors
            unique_samples = {}
            # First, group by sample identity
            for sweep in selected_sweeps:
                sample_key = (sweep.substrate, sweep.composition_index, sweep.position_in_composition)
                if sample_key not in unique_samples:
                    unique_samples[sample_key] = {'forward': None, 'reverse': None}
                unique_samples[sample_key][sweep.direction] = sweep
            
            # Plot each sample's F/R pair with same color
            color_idx = 0
            for sample_key, pair in unique_samples.items():
                substrate, comp, pos = sample_key
                base_color = colors[color_idx % len(colors)]
                
                # Plot forward if available
                if pair['forward'] is not None:
                    sweep = pair['forward']
                    if len(sweep.voltage) > 0 and len(sweep.current_A) > 0:
                        current_density = sweep.current_A / sweep.area_cm2 * 1000 * 1.3  # mA/cm² (corrected)
                        label = f"S{substrate}-C{comp}P{pos} (F)"
                        self.jv_ax.plot(sweep.voltage, current_density, color=base_color, label=label, linewidth=2, alpha=1.0)
                
                # Plot reverse if available (same color, dashed, alpha 0.6)
                if pair['reverse'] is not None:
                    sweep = pair['reverse']
                    if len(sweep.voltage) > 0 and len(sweep.current_A) > 0:
                        current_density = sweep.current_A / sweep.area_cm2 * 1000 * 1.3  # mA/cm² (corrected)
                        label = f"S{substrate}-C{comp}P{pos} (R)"
                        self.jv_ax.plot(sweep.voltage, current_density, color=base_color, label=label, linewidth=2, linestyle="--", alpha=0.6)
                
                color_idx += 1
        else:
            # Plot all curves individually
            for i, sweep in enumerate(selected_sweeps):
                if len(sweep.voltage) > 0 and len(sweep.current_A) > 0:
                    current_density = sweep.current_A / sweep.area_cm2 * 1000 * 1.5  # mA/cm² (corrected)
                    color = colors[i % len(colors)]
                    linestyle = "--" if sweep.direction == "reverse" else "-"
                    alpha = 0.6 if sweep.direction == "reverse" else 1.0
                    label = f"S{sweep.substrate}-C{sweep.composition_index}P{sweep.position_in_composition} ({sweep.direction[0].upper()})"
                    self.jv_ax.plot(sweep.voltage, current_density, color=color, label=label, 
                                   linewidth=2, linestyle=linestyle, alpha=alpha)
        
        # Formatting
        self.jv_ax.set_xlabel("Voltage (V)")
        self.jv_ax.set_ylabel("Current Density (mA/cm²)")
        self.jv_ax.set_title("JV Curves")
        self.jv_ax.grid(True, alpha=0.3)
        self.jv_ax.legend(loc='best')
        
        self.jv_fig.tight_layout()
        self.jv_canvas.draw()
        
        # Apply custom axis limits if not in auto mode
        self.update_jv_axes()
    
    def clear_jv_plot(self):
        """Clear the JV plot"""
        self.jv_ax.clear()
        self.jv_ax.set_xlabel("Voltage (V)")
        self.jv_ax.set_ylabel("Current Density (mA/cm²)")
        self.jv_ax.set_title("JV Curves - Select samples to plot")
        self.jv_ax.grid(True, alpha=0.3)
        self.jv_canvas.draw()

    def save_jv_plot(self):
        """Save the JV curves plot"""
        path = filedialog.asksaveasfilename(
            title="Save JV curves plot",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("SVG", "*.svg"), ("PDF", "*.pdf")]
        )
        if not path: return
        try:
            self.jv_fig.savefig(path, dpi=300, bbox_inches="tight")
            messagebox.showinfo("Saved", f"JV curves plot saved to:\n{path}")
        except Exception as e:
            messagebox.showerror("Save failed", f"Could not save JV plot: {str(e)}")

    def update_jv_axes(self):
        """Update JV plot axes based on user controls"""
        if not hasattr(self, 'jv_ax'):
            return
            
        # Helper function to parse axis limits
        def parse_limit(value_str):
            try:
                if value_str and value_str.strip():
                    return float(value_str.strip())
                return None
            except (ValueError, AttributeError):
                return None
        
        try:
            # Update X-axis
            if not self.jv_x_auto.get():
                x_min = parse_limit(self.jv_x_min.get())
                x_max = parse_limit(self.jv_x_max.get())
                if x_min is not None or x_max is not None:
                    current_xlim = self.jv_ax.get_xlim()
                    new_xlim = (x_min if x_min is not None else current_xlim[0],
                               x_max if x_max is not None else current_xlim[1])
                    self.jv_ax.set_xlim(new_xlim)
            
            # Update Y-axis
            if not self.jv_y_auto.get():
                y_min = parse_limit(self.jv_y_min.get())
                y_max = parse_limit(self.jv_y_max.get())
                if y_min is not None or y_max is not None:
                    current_ylim = self.jv_ax.get_ylim()
                    new_ylim = (y_min if y_min is not None else current_ylim[0],
                               y_max if y_max is not None else current_ylim[1])
                    self.jv_ax.set_ylim(new_ylim)
            
            self.jv_canvas.draw()
        except Exception as e:
            print(f"Error updating axes: {e}")


if __name__ == "__main__":
    app = JVApp()
    app.mainloop()
