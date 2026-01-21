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

    # PCE (%) = Pout_mpp / Pin × 100 × 1.5 (sun intensity correction)
    PCE_pct = None
    if light_mw_cm2 > 0:
        PCE_pct = (Pmpp_out / light_mw_cm2) * 100.0 * 1.5

    # Report Jsc as positive mA/cm² × 1.5 (sun intensity correction)
    Jsc_mAcm2 = None if Jsc_Acm2 is None else abs(Jsc_Acm2) * 1e3 * 1.5

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

# ---------------------------- Well ID helpers ----------------------------
def pixel_id_to_well(pixel_id: int) -> str:
    """Convert numeric pixel_id to well notation (e.g., 7 -> A1, 13 -> B1)"""
    # pixel_id = (comp - 1) * 6 + pos
    # pos is 1-6, comp is 1-11
    # So pixel_id ranges from 1 to 66
    if pixel_id < 1 or pixel_id > 66:
        return f"#{pixel_id}"

    # Calculate composition (1-11) and position (1-6)
    comp = (pixel_id - 1) // 6 + 1
    pos = (pixel_id - 1) % 6 + 1

    # Map to well format: rows A-F (position 1-6), columns 1-11 (composition 1-11)
    row_letter = chr(ord('A') + pos - 1)
    col_number = comp

    return f"{row_letter}{col_number}"

def well_to_pixel_id(well: str) -> Optional[int]:
    """Convert well notation to pixel_id (e.g., 'A1' -> 7, 'B1' -> 13)"""
    well = well.strip().upper()
    if len(well) < 2:
        return None

    # Extract row letter and column number
    row_letter = well[0]
    try:
        col_number = int(well[1:])
    except ValueError:
        return None

    # Validate
    if row_letter < 'A' or row_letter > 'F':
        return None
    if col_number < 1 or col_number > 11:
        return None

    # Convert to position and composition
    pos = ord(row_letter) - ord('A') + 1  # 1-6
    comp = col_number  # 1-11

    # Calculate pixel_id
    pixel_id = (comp - 1) * 6 + pos

    return pixel_id

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
        self.selected_sweep_filter = tk.StringVar(value="All")  # Sweep filter for Tab 1
        self.metric_choice = tk.StringVar(value="PCE_pct")
        self.aggregation_method = tk.StringVar(value="max")
        self.combine_substrates = tk.BooleanVar(value=True)
        self.combine_fr = tk.BooleanVar(value=True)
        self.grouping_mode = tk.StringVar(value="11 compositions")
        self.expand_substrate_axis = tk.BooleanVar(value=True)

        # Plotting options for tab1
        self.colormap_choice = tk.StringVar(value="viridis")
        self.x_min_var = tk.StringVar(value="")
        self.x_max_var = tk.StringVar(value="")
        self.y_min_var = tk.StringVar(value="")
        self.y_max_var = tk.StringVar(value="")

        # Figure size and font controls for tab1
        self.comp_fig_width = tk.StringVar(value="8")
        self.comp_fig_height = tk.StringVar(value="5")
        self.comp_title_fontsize = tk.StringVar(value="12")
        self.comp_axis_fontsize = tk.StringVar(value="10")

        # Filters

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
        
        # Tab 1: Original composition analysis (data table only)
        self.comp_frame = ttk.Frame(self.notebook, padding=8)
        self.notebook.add(self.comp_frame, text="Data Table")

        # Tab 2: Plot Settings and Visualization
        self.plot_frame = ttk.Frame(self.notebook, padding=8)
        self.notebook.add(self.plot_frame, text="Plot Settings")

        # Tab 3: Sweep-based analysis
        self.sweep_frame = ttk.Frame(self.notebook, padding=8)
        self.notebook.add(self.sweep_frame, text="Sweep Analysis")

        # Tab 4: JV Curve visualization
        self.jv_frame = ttk.Frame(self.notebook, padding=8)
        self.notebook.add(self.jv_frame, text="JV Curves")

        self._build_composition_tab()
        self._build_plot_tab()
        self._build_sweep_tab()
        self._build_jv_tab()
        
    def _build_composition_tab(self):
        """Build the Data Table tab - only data loading and table viewing"""
        self.comp_frame.columnconfigure(0, weight=1)
        self.comp_frame.rowconfigure(0, weight=1)

        # Left sidebar for data loading and parameters
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
        ttk.Button(side, text="Remove items…", command=self.open_remove_dialog).grid(row=7, column=0, sticky="ew")

        ttk.Separator(side).grid(row=8, column=0, sticky="ew", pady=6)
        ttk.Label(side, text="Header regex").grid(row=9, column=0, sticky="w")
        ttk.Entry(side, textvariable=self.header_pattern_var, width=38).grid(row=10, column=0, sticky="ew", pady=2)
        ttk.Button(side, text="Reload last folder/file", command=self.reload_last).grid(row=11, column=0, sticky="ew", pady=2)
        ttk.Button(side, text="Parse report", command=self.show_parse_report).grid(row=12, column=0, sticky="ew", pady=2)

        # Right panel - Just the data table
        right = ttk.Frame(self.comp_frame, padding=8); right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(0, weight=1); right.columnconfigure(0, weight=1)

        table_frame = ttk.Frame(right); table_frame.grid(row=0, column=0, sticky="nsew")
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
        ttk.Button(btns, text="Export table CSV", command=self.export_table_csv).grid(row=0, column=1, padx=2)

    def _build_plot_tab(self):
        """Build the Plot Settings tab with all plotting controls and figure"""
        self.plot_frame.columnconfigure(0, weight=0); self.plot_frame.columnconfigure(1, weight=1)
        self.plot_frame.rowconfigure(0, weight=1)

        # Left panel - All plotting controls
        plot_left = ttk.Frame(self.plot_frame, padding=8); plot_left.grid(row=0, column=0, sticky="ns")

        ttk.Label(plot_left, text="Filtering").grid(row=0, column=0, sticky="w")
        ttk.Label(plot_left, text="Sweep filter").grid(row=1, column=0, sticky="w", pady=(4, 0))
        self.sweep_filter_cb = ttk.Combobox(plot_left, textvariable=self.selected_sweep_filter, values=["All"], state="readonly", width=12)
        self.sweep_filter_cb.grid(row=2, column=0, sticky="ew", pady=2)
        self.sweep_filter_cb.bind("<<ComboboxSelected>>", lambda e: self.refresh_plots())

        ttk.Label(plot_left, text="Include sweeps").grid(row=3, column=0, sticky="w", pady=(6, 0))
        self.fwd_cb = ttk.Checkbutton(plot_left, text="Forward", variable=self.include_forward, command=self.refresh_plots)
        self.fwd_cb.grid(row=4, column=0, sticky="w")
        self.rev_cb = ttk.Checkbutton(plot_left, text="Reverse", variable=self.include_reverse, command=self.refresh_plots)
        self.rev_cb.grid(row=5, column=0, sticky="w")

        ttk.Separator(plot_left).grid(row=6, column=0, sticky="ew", pady=6)
        ttk.Label(plot_left, text="Plot Parameters").grid(row=7, column=0, sticky="w")
        ttk.Label(plot_left, text="Metric:").grid(row=8, column=0, sticky="w", pady=(4, 0))
        metric_cb = ttk.Combobox(plot_left, textvariable=self.metric_choice, values=["Voc", "Jsc_mAcm2", "FF_pct", "PCE_pct"], state="readonly", width=12)
        metric_cb.grid(row=9, column=0, sticky="ew"); metric_cb.bind("<<ComboboxSelected>>", lambda e: self.refresh_plots())

        ttk.Label(plot_left, text="Aggregation:").grid(row=10, column=0, sticky="w", pady=(4, 0))
        agg_cb = ttk.Combobox(plot_left, textvariable=self.aggregation_method, values=["mean", "max"], state="readonly", width=12)
        agg_cb.grid(row=11, column=0, sticky="ew"); agg_cb.bind("<<ComboboxSelected>>", lambda e: self.refresh_plots())

        ttk.Separator(plot_left).grid(row=12, column=0, sticky="ew", pady=6)
        ttk.Label(plot_left, text="Substrate / Grouping").grid(row=13, column=0, sticky="w")
        ttk.Label(plot_left, text="Substrate:").grid(row=14, column=0, sticky="w", pady=(4, 0))
        self.substrate_cb = ttk.Combobox(plot_left, values=["All"], state="readonly")
        self.substrate_cb.grid(row=15, column=0, sticky="ew", pady=2); self.substrate_cb.bind("<<ComboboxSelected>>", lambda e: self.refresh_plots())
        ttk.Checkbutton(plot_left, text="Combine substrates", variable=self.combine_substrates, command=self.refresh_plots).grid(row=16, column=0, sticky="w")
        ttk.Checkbutton(plot_left, text="Combine F & R", variable=self.combine_fr, command=self.on_combine_fr_changed).grid(row=17, column=0, sticky="w")
        ttk.Label(plot_left, text="Grouping:").grid(row=18, column=0, sticky="w", pady=(6, 0))
        grp_cb = ttk.Combobox(plot_left, textvariable=self.grouping_mode, values=["11 compositions", "9 groups"], state="readonly", width=18)
        grp_cb.grid(row=19, column=0, sticky="ew", pady=2); grp_cb.bind("<<ComboboxSelected>>", lambda e: self.refresh_plots())
        ttk.Checkbutton(plot_left, text="Expand x-axis by substrate", variable=self.expand_substrate_axis, command=self.refresh_plots).grid(row=20, column=0, sticky="w", pady=(2, 6))

        ttk.Separator(plot_left).grid(row=21, column=0, sticky="ew", pady=6)
        ttk.Label(plot_left, text="Plot Styling").grid(row=22, column=0, sticky="w")
        ttk.Label(plot_left, text="Colormap:").grid(row=23, column=0, sticky="w", pady=(2, 0))
        colormap_cb = ttk.Combobox(plot_left, textvariable=self.colormap_choice,
                                    values=["viridis", "plasma", "inferno", "magma", "cividis", "coolwarm", "RdYlGn", "RdYlBu"],
                                    state="readonly", width=12)
        colormap_cb.grid(row=24, column=0, sticky="ew")
        colormap_cb.bind("<<ComboboxSelected>>", lambda e: self.refresh_plots())

        ttk.Label(plot_left, text="Axis Limits (optional):").grid(row=25, column=0, sticky="w", pady=(4, 0))
        axis_frame = ttk.Frame(plot_left)
        axis_frame.grid(row=26, column=0, sticky="ew")
        ttk.Label(axis_frame, text="X min:").grid(row=0, column=0, sticky="w")
        ttk.Entry(axis_frame, textvariable=self.x_min_var, width=6).grid(row=0, column=1, sticky="w", padx=2)
        ttk.Label(axis_frame, text="max:").grid(row=0, column=2, sticky="w")
        ttk.Entry(axis_frame, textvariable=self.x_max_var, width=6).grid(row=0, column=3, sticky="w", padx=2)
        ttk.Label(axis_frame, text="Y min:").grid(row=1, column=0, sticky="w")
        ttk.Entry(axis_frame, textvariable=self.y_min_var, width=6).grid(row=1, column=1, sticky="w", padx=2)
        ttk.Label(axis_frame, text="max:").grid(row=1, column=2, sticky="w")
        ttk.Entry(axis_frame, textvariable=self.y_max_var, width=6).grid(row=1, column=3, sticky="w", padx=2)
        ttk.Button(plot_left, text="Apply limits", command=self.refresh_plots).grid(row=27, column=0, sticky="ew", pady=2)
        ttk.Button(plot_left, text="Clear limits", command=self.clear_axis_limits).grid(row=28, column=0, sticky="ew", pady=2)

        ttk.Separator(plot_left).grid(row=29, column=0, sticky="ew", pady=6)
        ttk.Label(plot_left, text="Figure Size & Fonts").grid(row=30, column=0, sticky="w")
        figsize_frame = ttk.Frame(plot_left)
        figsize_frame.grid(row=31, column=0, sticky="ew")
        ttk.Label(figsize_frame, text="Width:").grid(row=0, column=0, sticky="w")
        ttk.Entry(figsize_frame, textvariable=self.comp_fig_width, width=6).grid(row=0, column=1, sticky="w", padx=2)
        ttk.Label(figsize_frame, text="Height:").grid(row=0, column=2, sticky="w")
        ttk.Entry(figsize_frame, textvariable=self.comp_fig_height, width=6).grid(row=0, column=3, sticky="w", padx=2)

        font_frame = ttk.Frame(plot_left)
        font_frame.grid(row=32, column=0, sticky="ew")
        ttk.Label(font_frame, text="Title:").grid(row=0, column=0, sticky="w")
        ttk.Entry(font_frame, textvariable=self.comp_title_fontsize, width=6).grid(row=0, column=1, sticky="w", padx=2)
        ttk.Label(font_frame, text="Axis:").grid(row=0, column=2, sticky="w")
        ttk.Entry(font_frame, textvariable=self.comp_axis_fontsize, width=6).grid(row=0, column=3, sticky="w", padx=2)

        ttk.Button(plot_left, text="Apply Figure Settings", command=self.update_comp_figsize).grid(row=33, column=0, sticky="ew", pady=2)

        ttk.Separator(plot_left).grid(row=34, column=0, sticky="ew", pady=6)
        ttk.Label(plot_left, text="Substrate-Composition").grid(row=35, column=0, sticky="w")
        ttk.Label(plot_left, text="Pairs (e.g., 5-3, 2-8):").grid(row=36, column=0, sticky="w", pady=(2, 0))
        self.sub_comp_selection_var = tk.StringVar(value="")
        ttk.Entry(plot_left, textvariable=self.sub_comp_selection_var, width=20).grid(row=37, column=0, sticky="ew", pady=2)
        ttk.Button(plot_left, text="Manual Pixel Plot", command=self.plot_substrate_composition_boxplot).grid(row=38, column=0, sticky="ew")

        # Right panel - Plot display with scrollbars
        plot_right = ttk.Frame(self.plot_frame, padding=8); plot_right.grid(row=0, column=1, sticky="nsew")
        plot_right.rowconfigure(0, weight=1); plot_right.columnconfigure(0, weight=1)

        # Create a container frame for the plot with scrollbars
        plot_container = ttk.Frame(plot_right)
        plot_container.grid(row=0, column=0, sticky="nsew")
        plot_container.rowconfigure(0, weight=1)
        plot_container.columnconfigure(0, weight=1)

        # Create canvas with scrollbars
        plot_scroll_canvas = tk.Canvas(plot_container, bg='white')
        plot_scroll_canvas.grid(row=0, column=0, sticky="nsew")

        plot_vsb = ttk.Scrollbar(plot_container, orient="vertical", command=plot_scroll_canvas.yview)
        plot_vsb.grid(row=0, column=1, sticky="ns")
        plot_hsb = ttk.Scrollbar(plot_container, orient="horizontal", command=plot_scroll_canvas.xview)
        plot_hsb.grid(row=1, column=0, sticky="ew")

        plot_scroll_canvas.configure(yscrollcommand=plot_vsb.set, xscrollcommand=plot_hsb.set)

        # Create a frame inside the canvas to hold the matplotlib figure
        self.plot_inner_frame = ttk.Frame(plot_scroll_canvas)
        plot_scroll_canvas.create_window((0, 0), window=self.plot_inner_frame, anchor="nw")

        # Create the matplotlib canvas inside the inner frame
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_inner_frame)
        self.canvas.get_tk_widget().pack()

        # Store the scroll canvas for later use
        self.plot_scroll_canvas = plot_scroll_canvas

        # Update scroll region when the inner frame changes size
        def configure_scroll_region(event=None):
            plot_scroll_canvas.configure(scrollregion=plot_scroll_canvas.bbox("all"))

        self.plot_inner_frame.bind("<Configure>", configure_scroll_region)

        # Plot buttons below the figure
        plot_btns = ttk.Frame(plot_right); plot_btns.grid(row=1, column=0, sticky="ew", pady=4)
        ttk.Button(plot_btns, text="Boxplot by composition/group", command=self.plot_boxplot_groups).grid(row=0, column=0, padx=4)
        ttk.Button(plot_btns, text="Heatmap (substrate × group)", command=self.plot_heatmap).grid(row=0, column=1, padx=4)
        ttk.Button(plot_btns, text="Substrate pixel map", command=self.plot_substrate_pixel_map).grid(row=0, column=2, padx=4)
        ttk.Button(plot_btns, text="Save plot as image", command=self.save_plot_image).grid(row=0, column=3, padx=4)

        # Initialize the state of forward/reverse checkboxes based on combine_fr
        self.on_combine_fr_changed()

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

        ttk.Separator(sweep_side).grid(row=18, column=0, sticky="ew", pady=6)

        # Box plot controls
        ttk.Label(sweep_side, text="Box Plot Analysis", font=("Arial", 10, "bold")).grid(row=19, column=0, sticky="w", pady=(5, 2))

        # Multi-column selection for box plot grouping
        ttk.Label(sweep_side, text="Select Columns (Ctrl+Click):").grid(row=20, column=0, sticky="w", pady=(5, 0))

        # Frame for listbox and scrollbar
        listbox_frame = ttk.Frame(sweep_side)
        listbox_frame.grid(row=21, column=0, sticky="ew", pady=2)

        # Listbox for multi-selection
        self.boxplot_columns_listbox = tk.Listbox(listbox_frame, height=6, selectmode=tk.EXTENDED, exportselection=False)
        self.boxplot_columns_listbox.grid(row=0, column=0, sticky="ew")

        # Scrollbar for listbox
        listbox_scroll = ttk.Scrollbar(listbox_frame, orient="vertical", command=self.boxplot_columns_listbox.yview)
        listbox_scroll.grid(row=0, column=1, sticky="ns")
        self.boxplot_columns_listbox.config(yscrollcommand=listbox_scroll.set)

        listbox_frame.columnconfigure(0, weight=1)

        # Performance metric selection for box plot
        ttk.Label(sweep_side, text="Performance Metric:").grid(row=22, column=0, sticky="w", pady=(5, 0))
        self.boxplot_metric_var = tk.StringVar(value="PCE_pct")
        boxplot_metric_cb = ttk.Combobox(sweep_side, textvariable=self.boxplot_metric_var,
                                         values=["Voc", "Jsc_mAcm2", "FF_pct", "PCE_pct"],
                                         state="readonly", width=30)
        boxplot_metric_cb.grid(row=23, column=0, sticky="ew", pady=2)

        # Control filtering checkbox
        self.include_controls_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(sweep_side, text="Include controls (all zeros)",
                       variable=self.include_controls_var).grid(row=24, column=0, sticky="w", pady=2)

        # Show mean checkbox
        self.show_mean_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(sweep_side, text="Show mean values",
                       variable=self.show_mean_var).grid(row=25, column=0, sticky="w", pady=2)

        ttk.Button(sweep_side, text="Generate Box Plot", command=self.generate_box_plot).grid(row=26, column=0, sticky="ew", pady=2)

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
        self._populate_sweep_filter_combo()
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

    def _populate_sweep_filter_combo(self):
        """Populate sweep filter dropdown with available sweep IDs"""
        if self.df is None or self.df.empty:
            return
        if "sweep_id" not in self.df.columns:
            return

        # Get unique sweep IDs, excluding NaN values
        sweep_ids = self.df["sweep_id"].dropna().unique()

        # If no valid sweep IDs, just show "All"
        if len(sweep_ids) == 0:
            self.sweep_filter_cb["values"] = ["All"]
            self.selected_sweep_filter.set("All")
            return

        # Sort and convert to strings
        sweep_ids = sorted([int(s) for s in sweep_ids])
        values = ["All"] + [str(s) for s in sweep_ids]
        self.sweep_filter_cb["values"] = values

        # Preserve current selection if valid
        cur = self.selected_sweep_filter.get()
        self.selected_sweep_filter.set(cur if cur in values else "All")

        print(f"[SWEEP FILTER] Populated with sweep IDs: {sweep_ids}")

    # -------------------- Filters & removal --------------------

    def on_combine_fr_changed(self):
        """Update forward/reverse checkboxes state based on combine_fr setting"""
        if self.combine_fr.get():
            # When combining F&R, disable the individual checkboxes
            self.fwd_cb.config(state="disabled")
            self.rev_cb.config(state="disabled")
        else:
            # When not combining, enable the individual checkboxes
            self.fwd_cb.config(state="normal")
            self.rev_cb.config(state="normal")
        self.refresh_plots()

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

        # Use filtered data (respects sweep filter, F/R toggles, and threshold filters)
        df_to_export = self._filtered_df()

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
        self._populate_sweep_filter_combo()
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

        # Apply sweep filter
        sweep_filter = self.selected_sweep_filter.get()
        if sweep_filter != "All" and "sweep_id" in df.columns:
            try:
                sweep_num = int(sweep_filter)
                df = df[df["sweep_id"] == sweep_num]
            except ValueError:
                pass  # If conversion fails, ignore filter

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

    def update_comp_figsize(self):
        """Update composition plot figure size based on user input"""
        try:
            width = float(self.comp_fig_width.get())
            height = float(self.comp_fig_height.get())

            if width <= 0 or height <= 0:
                messagebox.showerror("Invalid Size", "Width and height must be positive numbers")
                return

            # Destroy old canvas widget
            self.canvas.get_tk_widget().destroy()

            # Update figure size
            self.fig.set_size_inches(width, height)

            # Create new canvas with updated figure
            self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_inner_frame)
            self.canvas.get_tk_widget().pack()

            # Update scroll region to accommodate new figure size
            self.plot_inner_frame.update_idletasks()
            self.plot_scroll_canvas.configure(scrollregion=self.plot_scroll_canvas.bbox("all"))

            # Redraw with current plot
            self.refresh_plots()
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter valid numbers for width and height")

    def refresh_plots(self):
        self.plot_boxplot_groups()
        # Also refresh JV selection table when plots are refreshed
        try:
            self.refresh_jv_selection_table()
        except AttributeError:
            # JV tab not yet initialized
            pass

        # Update scroll region after plot refresh
        try:
            self.plot_inner_frame.update_idletasks()
            self.plot_scroll_canvas.configure(scrollregion=self.plot_scroll_canvas.bbox("all"))
        except AttributeError:
            # Scroll canvas not yet initialized
            pass

    def clear_axis_limits(self):
        """Clear all axis limit entries"""
        self.x_min_var.set("")
        self.x_max_var.set("")
        self.y_min_var.set("")
        self.y_max_var.set("")
        self.refresh_plots()

    def _apply_axis_limits(self):
        """Apply axis limits if specified"""
        try:
            if self.x_min_var.get():
                x_min = float(self.x_min_var.get())
                self.ax.set_xlim(left=x_min)
            if self.x_max_var.get():
                x_max = float(self.x_max_var.get())
                self.ax.set_xlim(right=x_max)
            if self.y_min_var.get():
                y_min = float(self.y_min_var.get())
                self.ax.set_ylim(bottom=y_min)
            if self.y_max_var.get():
                y_max = float(self.y_max_var.get())
                self.ax.set_ylim(top=y_max)
        except ValueError:
            # Invalid number format, silently ignore
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
        substrate_ids: List[int] = []  # Track which substrate each box belongs to
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
                    substrate_ids.append(sub)
        else:
            sel = self.substrate_cb.get()
            if sel and sel != "All" and not self.combine_substrates.get():
                try: data_df = data_df[data_df["substrate"] == int(sel)]
                except Exception: pass
            if agg_method == "max":
                # For max: group by substrate+composition, take max, then collect those max values
                # Keep track of substrate for each max value
                grouped_max = data_df.groupby([group_col, "substrate"])[metric].max().reset_index()
                for g in expected_groups:
                    g_data = grouped_max[grouped_max[group_col] == g]
                    if len(g_data) == 0: continue
                    labels.append(("G" if use_groups else "C") + str(g))
                    series.append(g_data[metric].dropna().values)
                    # Store substrate IDs for this group (for coloring)
                    substrate_ids.append(g_data["substrate"].values.tolist())
            else:
                # For mean: show distribution of all individual measurements (current behavior)
                vals_by = data_df.groupby(group_col)[metric].apply(lambda s: s.dropna().values)
                for g in expected_groups:
                    vals = vals_by.get(g, np.array([]))
                    if len(vals) == 0: continue
                    labels.append(("G" if use_groups else "C") + str(g))
                    series.append(vals)
                    # For non-expanded mode, track substrates from the data
                    substrate_ids.append(-1)  # Placeholder, will be filled below

        if not series:
            self._clear_ax("No values for selected metric/filters."); return

        bp = self.ax.boxplot(series, showmeans=True, meanline=False, patch_artist=True)

        # Color boxes by substrate
        unique_subs = sorted(data_df["substrate"].dropna().astype(int).unique().tolist())
        colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(unique_subs))))
        substrate_color_map = {sub: colors[i % len(colors)] for i, sub in enumerate(unique_subs)}

        if self.expand_substrate_axis.get():
            # Apply colors to each box (substrate_ids contains single substrate per box)
            for patch, sub_id in zip(bp['boxes'], substrate_ids):
                patch.set_facecolor(substrate_color_map[sub_id])
                patch.set_alpha(0.7)

            # For max mode: also plot scatter points with substrate colors (since boxes show as single points)
            if agg_method == "max":
                for i, (vals, sub_id) in enumerate(zip(series, substrate_ids)):
                    x_positions = [i + 1] * len(vals)  # boxplot positions are 1-indexed
                    self.ax.scatter(x_positions, vals, color=substrate_color_map[sub_id],
                                   s=100, alpha=0.8, zorder=3, edgecolors='black', linewidths=0.5)
        else:
            # When not expanded, color by substrate if combining substrates
            if self.combine_substrates.get() or (sel == "All"):
                # For each box, determine substrate color
                for box_idx, (patch, sub_info) in enumerate(zip(bp['boxes'], substrate_ids)):
                    if isinstance(sub_info, list):
                        # For max mode: sub_info is a list of substrate IDs
                        if len(sub_info) > 0:
                            # Color by the most common substrate, or first if tie
                            from collections import Counter
                            most_common_sub = Counter(sub_info).most_common(1)[0][0]
                            patch.set_facecolor(substrate_color_map[most_common_sub])
                            patch.set_alpha(0.7)
                    else:
                        # For mean mode: need to look up substrate from data
                        g = expected_groups[box_idx] if box_idx < len(expected_groups) else None
                        if g is not None:
                            vals = data_df[data_df[group_col] == g]["substrate"].dropna().astype(int).unique()
                            if len(vals) > 0:
                                sub_id = int(vals[0])
                                patch.set_facecolor(substrate_color_map[sub_id])
                                patch.set_alpha(0.7)

                # For max mode: also plot scatter points with substrate colors
                if agg_method == "max":
                    for i, (vals, sub_info) in enumerate(zip(series, substrate_ids)):
                        if isinstance(sub_info, list) and len(sub_info) > 0:
                            # Color each point by its substrate
                            x_positions = [i + 1] * len(vals)
                            point_colors = [substrate_color_map[sub_id] for sub_id in sub_info]
                            self.ax.scatter(x_positions, vals, c=point_colors,
                                          s=100, alpha=0.8, zorder=3, edgecolors='black', linewidths=0.5)
            else:
                # Single substrate selected - use default coloring
                for patch in bp['boxes']:
                    patch.set_facecolor('lightblue')
                    patch.set_alpha(0.7)

        # Get font sizes
        try:
            title_fontsize = int(self.comp_title_fontsize.get())
            axis_fontsize = int(self.comp_axis_fontsize.get())
        except ValueError:
            title_fontsize = 12
            axis_fontsize = 10

        self.ax.set_xticks(range(1, len(labels) + 1))
        self.ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=axis_fontsize)
        self.ax.set_ylabel(metric, fontsize=axis_fontsize)
        self.ax.tick_params(axis='y', labelsize=axis_fontsize)
        title = f"Boxplot ({agg_method}): {metric} by {'Group' if use_groups else 'Composition'}"
        if self.expand_substrate_axis.get(): title += " (expanded by substrate)"
        self.ax.set_title(title, fontsize=title_fontsize)
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

        # Get font sizes
        try:
            title_fontsize = int(self.comp_title_fontsize.get())
            axis_fontsize = int(self.comp_axis_fontsize.get())
        except ValueError:
            title_fontsize = 12
            axis_fontsize = 10

        cmap = self.colormap_choice.get()
        im = self.ax.imshow(pivot.values, aspect="auto", interpolation="nearest", cmap=cmap)
        self.ax.set_xlabel("Composition " + ("group (G1..G9)" if use_groups else "index (C1..C11)"), fontsize=axis_fontsize)
        self.ax.set_ylabel("Substrate", fontsize=axis_fontsize)
        self.ax.set_title(f"Heatmap: {agg} {metric}", fontsize=title_fontsize)
        self.ax.set_xticks(range(pivot.shape[1])); self.ax.set_xticklabels([("G" if use_groups else "C")+str(g) for g in expected_groups], fontsize=axis_fontsize)

        # Fix substrate axis labels - ensure they correspond to actual substrate numbers in sorted order
        substrate_labels = [str(int(s)) for s in sorted(pivot.index)]
        self.ax.set_yticks(range(pivot.shape[0])); self.ax.set_yticklabels(substrate_labels, fontsize=axis_fontsize)

        vals = pivot.values
        for i in range(vals.shape[0]):
            for j in range(vals.shape[1]):
                v = vals[i, j]
                if isinstance(v, float) and not np.isnan(v):
                    self.ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8)

        self._cbar = self.fig.colorbar(im, ax=self.ax, fraction=0.046, pad=0.04, label=metric)
        self._apply_axis_limits()
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

        # Get font sizes
        try:
            title_fontsize = int(self.comp_title_fontsize.get())
            axis_fontsize = int(self.comp_axis_fontsize.get())
        except ValueError:
            title_fontsize = 12
            axis_fontsize = 10

        cmap = self.colormap_choice.get()
        im = self.ax.imshow(mat, aspect="auto", interpolation="nearest", origin="upper", cmap=cmap)
        self.ax.set_xlabel("Composition (C1..C11)", fontsize=axis_fontsize)
        self.ax.set_ylabel("Pixel position (1=thick → 6=thin)", fontsize=axis_fontsize)
        self.ax.set_title(f"Substrate S{sub} — {metric} per composition & pixel position", fontsize=title_fontsize)
        self.ax.set_xticks(range(11)); self.ax.set_xticklabels([f"C{i}" for i in range(1, 12)], rotation=45, ha="right", fontsize=axis_fontsize)
        self.ax.set_yticks(range(6));  self.ax.set_yticklabels([str(i) for i in range(1, 7)], fontsize=axis_fontsize)
        for i in range(6):
            for j in range(11):
                v = mat[i, j]
                if not np.isnan(v):
                    self.ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7)

        self._cbar = self.fig.colorbar(im, ax=self.ax, fraction=0.046, pad=0.04, label=metric)
        self._apply_axis_limits()
        self.canvas.draw_idle()

    def plot_substrate_composition_boxplot(self):
        """
        Plot box plots comparing specified substrate-composition combinations.
        Shows distribution of all pixel measurements (not just mean/max) for each combination.
        Input format: substrate-composition pairs separated by comma, e.g., "5-3, 2-8, 1-11"
        """
        # Parse input
        input_str = self.sub_comp_selection_var.get().strip()
        if not input_str:
            messagebox.showwarning("No selection", "Please enter substrate-composition pairs (e.g., 5-3, 2-8)")
            return

        # Parse substrate-composition pairs
        pairs = []
        invalid_pairs = []
        for pair_str in input_str.split(","):
            pair_str = pair_str.strip()
            if not pair_str:
                continue
            # Try to parse as "substrate-composition"
            parts = pair_str.replace(" ", "").split("-")
            if len(parts) == 2:
                try:
                    substrate = int(parts[0])
                    composition = int(parts[1])
                    pairs.append((substrate, composition))
                except ValueError:
                    invalid_pairs.append(pair_str)
            else:
                invalid_pairs.append(pair_str)

        if invalid_pairs:
            messagebox.showwarning("Invalid format",
                f"Invalid pairs: {', '.join(invalid_pairs)}\nExpected format: substrate-composition (e.g., 5-3, 2-8)")
            return

        if not pairs:
            messagebox.showwarning("No valid pairs", "No valid substrate-composition pairs found")
            return

        # Get the data
        df = self._filtered_df()
        if df.empty:
            messagebox.showwarning("No data", "No data available. Load JV data first.")
            return

        # Get the selected metric
        metric = self.metric_choice.get()

        # Collect data for each pair
        labels = []
        series = []
        for substrate, composition in pairs:
            # Filter data for this substrate-composition pair
            mask = (df["substrate"] == substrate) & (df["composition_index"] == composition)
            subset = df[mask]

            if subset.empty:
                print(f"Warning: No data found for Substrate {substrate}, Composition {composition}")
                continue

            # Get all pixel values (not aggregated)
            values = subset[metric].dropna().values
            if len(values) > 0:
                labels.append(f"S{substrate}-C{composition}")
                series.append(values)

        if not series:
            messagebox.showwarning("No data",
                f"No data found for the specified substrate-composition pairs.\n" +
                f"Pairs requested: {', '.join([f'S{s}-C{c}' for s, c in pairs])}")
            return

        # Plot box plots
        self._reset_axes()
        bp = self.ax.boxplot(series, showmeans=True, meanline=False, patch_artist=True)

        # Color the boxes using selected colormap
        cmap = plt.get_cmap(self.colormap_choice.get())
        colors = cmap(np.linspace(0, 1, len(series)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

        # Get font sizes
        try:
            title_fontsize = int(self.comp_title_fontsize.get())
            axis_fontsize = int(self.comp_axis_fontsize.get())
        except ValueError:
            title_fontsize = 12
            axis_fontsize = 10

        self.ax.set_xticks(range(1, len(labels) + 1))
        self.ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=axis_fontsize)
        self.ax.set_ylabel(metric, fontsize=axis_fontsize)
        self.ax.tick_params(axis='y', labelsize=axis_fontsize)
        self.ax.set_title(f"Substrate-Composition Comparison: {metric}", fontsize=title_fontsize)

        self.ax.grid(True, which="both", axis="y", alpha=0.3)

        

        self._apply_axis_limits()
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

            # Populate box plot listbox with all columns
            self.boxplot_columns_listbox.delete(0, tk.END)
            for col in all_columns:
                self.boxplot_columns_listbox.insert(tk.END, col)

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

        # Apply forward/reverse filtering
        if not self.combine_fr.get():
            keep = []
            if self.include_forward.get(): keep.append("forward")
            if self.include_reverse.get(): keep.append("reverse")
            if keep: jv_data = jv_data[jv_data["direction"].isin(keep)]

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

    def generate_box_plot(self):
        """Generate box plots for multiple selected columns showing performance metrics"""
        if self.parameter_data is None:
            messagebox.showwarning("No data", "Please load Excel data first.")
            return

        if self.df_with_flags is None or self.df_with_flags.empty:
            messagebox.showwarning("No JV data", "Please load JV data first.")
            return

        # Get selected columns from listbox
        selected_indices = self.boxplot_columns_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("No columns selected", "Please select at least one column (use Ctrl+Click for multiple).")
            return

        selected_columns = [self.boxplot_columns_listbox.get(i) for i in selected_indices]

        metric = self.boxplot_metric_var.get()
        include_controls = self.include_controls_var.get()
        show_mean = self.show_mean_var.get()

        try:
            # Merge parameter and JV data
            combined_data = self._merge_parameter_and_jv_data()
            if combined_data is None or combined_data.empty:
                messagebox.showerror("Error", "Could not merge parameter and JV data.")
                return

            # Check if selected columns exist
            missing_cols = [col for col in selected_columns if col not in combined_data.columns]
            if missing_cols:
                messagebox.showerror("Error", f"Columns not found in data: {', '.join(missing_cols)}")
                return

            # Filter controls if needed
            if not include_controls:
                # Identify control rows where all additive columns are zero
                additive_cols = [col for col in combined_data.columns
                               if col.endswith('(M)') or col.endswith('(mg/mL)') or
                               col in ['KI (mg/mL)', 'RbI (M)', 'GSCN (M)']]

                if additive_cols:
                    # Keep only rows where at least one additive column is non-zero
                    is_control = combined_data[additive_cols].fillna(0).sum(axis=1) == 0
                    combined_data = combined_data[~is_control]

            # Prepare data for box plot - collect data for all columns
            all_plot_data = []
            all_labels = []
            all_positions = []
            all_colors = []

            # Use different colors for different columns
            column_colors = plt.cm.Set3(np.linspace(0, 1, len(selected_columns)))

            current_position = 1
            spacing_between_columns = 0.5  # Gap between different column groups

            for col_idx, group_column in enumerate(selected_columns):
                # Get unique values for the grouping column (excluding zeros unless include_controls is True)
                if not include_controls:
                    group_values = sorted([v for v in combined_data[group_column].unique() if v != 0 and not pd.isna(v)])
                else:
                    group_values = sorted([v for v in combined_data[group_column].unique() if not pd.isna(v)])

                if len(group_values) == 0:
                    continue

                # Clean column name for label
                col_label = group_column.replace(' (M)', '').replace(' (mg/mL)', '')

                for val in group_values:
                    group_data = combined_data[combined_data[group_column] == val][metric].dropna()
                    if len(group_data) > 0:
                        all_plot_data.append(group_data.values)
                        # Format label with column name and value
                        val_str = f"{val:.3g}" if isinstance(val, (int, float)) else str(val)
                        all_labels.append(f"{col_label}\n{val_str}")
                        all_positions.append(current_position)
                        all_colors.append(column_colors[col_idx])
                        current_position += 1

                # Add spacing between different columns
                current_position += spacing_between_columns

            if len(all_plot_data) == 0:
                messagebox.showwarning("No data", "No valid data for the selected columns and metric.")
                return

            # Create the plot
            self._reset_sweep_axes()

            # Create box plot with custom positions
            bp = self.sweep_ax.boxplot(all_plot_data, positions=all_positions,
                                      labels=all_labels, patch_artist=True,
                                      showmeans=show_mean, widths=0.6,
                                      meanprops=dict(marker='D', markerfacecolor='red',
                                                   markeredgecolor='darkred', markersize=6))

            # Color the boxes according to their column
            for patch, color in zip(bp['boxes'], all_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
                patch.set_edgecolor('black')
                patch.set_linewidth(1.5)

            # Add mean values as text if requested
            if show_mean:
                for i, data in enumerate(all_plot_data):
                    mean_val = np.mean(data)
                    self.sweep_ax.text(all_positions[i], mean_val, f'{mean_val:.2f}',
                                      ha='center', va='bottom', fontsize=8,
                                      bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.6))

            # Labels and title
            metric_labels = {
                'Voc': 'Open Circuit Voltage (V)',
                'Jsc_mAcm2': 'Short Circuit Current Density (mA/cm²)',
                'FF_pct': 'Fill Factor (%)',
                'PCE_pct': 'Power Conversion Efficiency (%)'
            }

            self.sweep_ax.set_xlabel('Parameter Groups', fontsize=12, fontweight='bold')
            self.sweep_ax.set_ylabel(metric_labels.get(metric, metric), fontsize=12, fontweight='bold')

            # Create title with selected columns
            col_names_short = [c.replace(' (M)', '').replace(' (mg/mL)', '') for c in selected_columns]
            title = f'Box Plot Comparison: {metric_labels.get(metric, metric)}\n'
            title += f"Parameters: {', '.join(col_names_short)}"
            if not include_controls:
                title += ' (Controls excluded)'
            self.sweep_ax.set_title(title, fontsize=13, fontweight='bold')

            self.sweep_ax.grid(True, alpha=0.3, linestyle='--', axis='y')
            self.sweep_ax.tick_params(axis='x', rotation=45, labelsize=9)
            self.sweep_ax.tick_params(axis='y', labelsize=10)

            # Add legend to identify columns by color
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor=column_colors[i], edgecolor='black',
                                    label=col.replace(' (M)', '').replace(' (mg/mL)', ''), alpha=0.7)
                             for i, col in enumerate(selected_columns)]
            self.sweep_ax.legend(handles=legend_elements, loc='best', fontsize=9, framealpha=0.9)

            # Adjust layout
            self.sweep_fig.tight_layout()
            self.sweep_canvas.draw_idle()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate box plot:\n{str(e)}")
            import traceback
            traceback.print_exc()

    def plot_well_comparison(self):
        """Plot comparison of selected wells with chosen performance metric"""
        # Parse well IDs from the input
        well_input = self.well_selection_var.get().strip()
        if not well_input:
            messagebox.showwarning("No wells selected", "Please enter well IDs (e.g., A5, B6, C3).")
            return

        # Split by comma and clean up
        well_ids = [w.strip().upper() for w in well_input.replace(" ", "").split(",")]

        # Convert well IDs to pixel_ids
        pixel_ids = []
        invalid_wells = []
        for well in well_ids:
            pixel_id = well_to_pixel_id(well)
            if pixel_id is not None:
                pixel_ids.append(pixel_id)
            else:
                invalid_wells.append(well)

        if invalid_wells:
            messagebox.showwarning("Invalid wells", f"Invalid well IDs: {', '.join(invalid_wells)}\nValid format: A1-F11")
            return

        if not pixel_ids:
            messagebox.showwarning("No valid wells", "No valid well IDs found.")
            return

        # Get the selected parameter/metric from y-axis parameter
        metric = self.y_param_var.get()

        # Build list of valid options: performance metrics + parameter columns
        valid_metrics = ['Voc', 'Jsc_mAcm2', 'FF_pct', 'PCE_pct']
        if self.parameter_data is not None:
            valid_metrics.extend(self.parameter_data.columns.tolist())

        if not metric or metric not in valid_metrics:
            messagebox.showwarning("Invalid metric", f"Please select a valid parameter from the Y-axis Parameter dropdown.\nAvailable: {', '.join(valid_metrics[:5])}...")
            return

        # Determine which dataframe to use based on the metric
        if metric in ['Voc', 'Jsc_mAcm2', 'FF_pct', 'PCE_pct']:
            # Performance metric from JV data - requires JV data to be loaded
            if self.df is None or self.df.empty:
                messagebox.showwarning("No JV data", "Please load JV data first to plot performance metrics.")
                return
            data_source = self.df
        else:
            # Parameter column from Excel sheet - can work with just parameter_data
            if self.parameter_data is None:
                messagebox.showwarning("No parameter data", "Please load Excel parameter sheet first.")
                return

            # Check if we need to identify which rows correspond to the selected wells
            # If parameter_data has a 'Well ID' or 'pixel_id' column, use it directly
            param_df = self.parameter_data.copy()

            # Try to find a well/pixel ID column in parameter data
            well_col_candidates = ['Well ID', 'well_id', 'Well', 'well', 'WellID', 'Pixel ID', 'pixel_id']
            well_col = None
            for col in well_col_candidates:
                if col in param_df.columns:
                    well_col = col
                    break

            if well_col:
                # Filter parameter data by well IDs
                filtered_df = param_df[param_df[well_col].isin(well_ids + [w.lower() for w in well_ids])].copy()
                if filtered_df.empty:
                    messagebox.showwarning("No data", f"No data found in parameter sheet for wells: {', '.join(well_ids)}")
                    return
                # Ensure we have a normalized well_id column
                filtered_df['well_id'] = filtered_df[well_col].astype(str).str.upper()
            else:
                # No well ID column found - use pixel_id directly if available
                if 'pixel_id' not in param_df.columns:
                    messagebox.showwarning("Column not found", "Parameter data does not have a 'Well ID' or 'pixel_id' column.\nCannot identify which rows correspond to selected wells.")
                    return
                filtered_df = param_df[param_df['pixel_id'].isin(pixel_ids)].copy()
                if filtered_df.empty:
                    messagebox.showwarning("No data", f"No data found in parameter sheet for wells: {', '.join(well_ids)}")
                    return
                filtered_df['well_id'] = filtered_df['pixel_id'].apply(pixel_id_to_well)

            data_source = filtered_df

        # For JV data source, filter by pixel_ids
        if metric in ['Voc', 'Jsc_mAcm2', 'FF_pct', 'PCE_pct']:
            filtered_df = data_source[data_source['pixel_id'].isin(pixel_ids)].copy()
            if filtered_df.empty:
                messagebox.showwarning("No data", f"No data found for wells: {', '.join(well_ids)}")
                return
            # Add well_id column for plotting
            filtered_df['well_id'] = filtered_df['pixel_id'].apply(pixel_id_to_well)
        else:
            # Already filtered above for parameter data
            pass

        # Check if metric column exists in the filtered data
        if metric not in filtered_df.columns:
            messagebox.showwarning("Column not found", f"Column '{metric}' not found in the data.\nAvailable columns: {', '.join(filtered_df.columns[:10])}...")
            return

        # Get plot type from the UI
        plot_type = self.plot_type_var.get()

        # Clear the current plot
        self.sweep_ax.clear()

        # Create the plot based on plot type
        try:
            if plot_type == "bar":
                self._plot_well_bar(filtered_df, metric, well_ids)
            elif plot_type == "scatter":
                self._plot_well_scatter(filtered_df, metric, well_ids)
            elif plot_type == "line":
                self._plot_well_line(filtered_df, metric, well_ids)
            elif plot_type == "box":
                self._plot_well_box(filtered_df, metric, well_ids)
            else:
                # Default to bar plot
                self._plot_well_bar(filtered_df, metric, well_ids)

            self.sweep_canvas.draw()

        except Exception as e:
            messagebox.showerror("Plot failed", f"Could not create plot: {str(e)}")
            import traceback
            traceback.print_exc()

    def _plot_well_bar(self, df, metric, well_ids):
        """Create bar plot comparing wells"""
        # Group by well_id and aggregate (mean)
        grouped = df.groupby('well_id')[metric].mean().reindex([w.upper() for w in well_ids])

        x_pos = range(len(grouped))
        self.sweep_ax.bar(x_pos, grouped.values, alpha=0.7, color='steelblue')
        self.sweep_ax.set_xticks(x_pos)
        self.sweep_ax.set_xticklabels(grouped.index, rotation=45, ha='right')
        self.sweep_ax.set_xlabel('Well ID', fontsize=12)
        self.sweep_ax.set_ylabel(metric, fontsize=12)
        self.sweep_ax.set_title(f'{metric} Comparison by Well', fontsize=14, fontweight='bold')
        self.sweep_ax.grid(axis='y', alpha=0.3)
        self.sweep_fig.tight_layout()

    def _plot_well_scatter(self, df, metric, well_ids):
        """Create scatter plot comparing wells"""
        for well_id in [w.upper() for w in well_ids]:
            well_data = df[df['well_id'] == well_id]
            if not well_data.empty:
                x_vals = [well_id] * len(well_data)
                y_vals = well_data[metric].values
                self.sweep_ax.scatter(x_vals, y_vals, alpha=0.6, s=100, label=well_id)

        self.sweep_ax.set_xlabel('Well ID', fontsize=12)
        self.sweep_ax.set_ylabel(metric, fontsize=12)
        self.sweep_ax.set_title(f'{metric} Comparison by Well', fontsize=14, fontweight='bold')
        self.sweep_ax.legend()
        self.sweep_ax.grid(axis='y', alpha=0.3)
        self.sweep_fig.tight_layout()

    def _plot_well_line(self, df, metric, well_ids):
        """Create line plot comparing wells"""
        grouped = df.groupby('well_id')[metric].mean().reindex([w.upper() for w in well_ids])

        self.sweep_ax.plot(range(len(grouped)), grouped.values, marker='o', linewidth=2, markersize=8)
        self.sweep_ax.set_xticks(range(len(grouped)))
        self.sweep_ax.set_xticklabels(grouped.index, rotation=45, ha='right')
        self.sweep_ax.set_xlabel('Well ID', fontsize=12)
        self.sweep_ax.set_ylabel(metric, fontsize=12)
        self.sweep_ax.set_title(f'{metric} Comparison by Well', fontsize=14, fontweight='bold')
        self.sweep_ax.grid(alpha=0.3)
        self.sweep_fig.tight_layout()

    def _plot_well_box(self, df, metric, well_ids):
        """Create box plot comparing wells"""
        data_to_plot = []
        labels = []
        for well_id in [w.upper() for w in well_ids]:
            well_data = df[df['well_id'] == well_id][metric].dropna()
            if not well_data.empty:
                data_to_plot.append(well_data.values)
                labels.append(well_id)

        if data_to_plot:
            self.sweep_ax.boxplot(data_to_plot, labels=labels)
            self.sweep_ax.set_xlabel('Well ID', fontsize=12)
            self.sweep_ax.set_ylabel(metric, fontsize=12)
            self.sweep_ax.set_title(f'{metric} Distribution by Well', fontsize=14, fontweight='bold')
            self.sweep_ax.grid(axis='y', alpha=0.3)
            self.sweep_fig.tight_layout()

    def _plot_parameter_scatter(self, data, x_param, y_param, color_param):
        """Create scatter plot of parameter vs performance"""
        # Filter data to only include substrate-matched controls for 0 values
        # Control wells are those with NO additives (all additive columns are 0 or NaN)

        if 'substrate' in data.columns:
            # Identify additive columns (columns ending with '(M)' or starting with 'excess '/'with ')
            additive_cols = [col for col in data.columns
                           if col.endswith('(M)') or col.startswith('excess ') or col.startswith('with ')]

            # Get substrates that have the swept parameter (non-zero x_param)
            substrates_with_param = data[data[x_param] > 0]['substrate'].unique()

            # Identify control wells: all additive columns are 0 or NaN
            if additive_cols:
                is_control = data[additive_cols].fillna(0).sum(axis=1) == 0
            else:
                # If no additive columns found, fall back to checking only x_param == 0
                is_control = data[x_param] == 0

            # Filter data to include:
            # 1. All samples with non-zero x_param values
            # 2. Only true control wells (no additives) from substrates that have this parameter
            filtered_data = data[
                (data[x_param] > 0) |  # Include all samples with the parameter
                ((data[x_param] == 0) & is_control & (data['substrate'].isin(substrates_with_param)))  # Only substrate-matched pure controls
            ].copy()
        else:
            filtered_data = data

        x_values = filtered_data[x_param].dropna()
        y_values = filtered_data[y_param].dropna()

        # Align x and y values
        valid_indices = filtered_data[[x_param, y_param]].dropna().index
        x_values = filtered_data.loc[valid_indices, x_param]
        y_values = filtered_data.loc[valid_indices, y_param]

        if len(x_values) == 0:
            self._clear_sweep_ax("No valid data points for selected parameters.")
            return

        if color_param and color_param != 'None' and color_param in filtered_data.columns:
            color_values = filtered_data.loc[valid_indices, color_param]
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
        # Filter data to only include substrate-matched controls for 0 values
        # Control wells are those with NO additives (all additive columns are 0 or NaN)

        if 'substrate' not in data.columns:
            # Fallback to original behavior if substrate info is missing
            grouped = data.groupby(x_param)[y_param].agg(['mean', 'std']).reset_index()
        else:
            # Identify additive columns (columns ending with '(M)' or starting with 'excess '/'with ')
            additive_cols = [col for col in data.columns
                           if col.endswith('(M)') or col.startswith('excess ') or col.startswith('with ')]

            # Get substrates that have the swept parameter (non-zero x_param)
            substrates_with_param = data[data[x_param] > 0]['substrate'].unique()

            # Identify control wells: all additive columns are 0 or NaN
            if additive_cols:
                is_control = data[additive_cols].fillna(0).sum(axis=1) == 0
            else:
                # If no additive columns found, fall back to checking only x_param == 0
                is_control = data[x_param] == 0

            # Filter data to include:
            # 1. All samples with non-zero x_param values
            # 2. Only true control wells (no additives) from substrates that have this parameter
            filtered_data = data[
                (data[x_param] > 0) |  # Include all samples with the parameter
                ((data[x_param] == 0) & is_control & (data['substrate'].isin(substrates_with_param)))  # Only substrate-matched pure controls
            ].copy()

            # Group by x_param and calculate mean y_param
            grouped = filtered_data.groupby(x_param)[y_param].agg(['mean', 'std']).reset_index()

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
        # Filter data to only include substrate-matched controls
        if 'substrate' in data.columns:
            additive_cols = [col for col in data.columns
                           if col.endswith('(M)') or col.startswith('excess ') or col.startswith('with ')]
            substrates_with_param = data[data[x_param] > 0]['substrate'].unique()
            if additive_cols:
                is_control = data[additive_cols].fillna(0).sum(axis=1) == 0
            else:
                is_control = data[x_param] == 0
            filtered_data = data[
                (data[x_param] > 0) |
                ((data[x_param] == 0) & is_control & (data['substrate'].isin(substrates_with_param)))
            ].copy()
        else:
            filtered_data = data

        valid_indices = filtered_data[[x_param, y_param]].dropna().index
        x_values = filtered_data.loc[valid_indices, x_param]
        y_values = filtered_data.loc[valid_indices, y_param]

        if len(x_values) == 0:
            self._clear_sweep_ax("No valid data points for selected parameters.")
            return

        if color_param and color_param != 'None' and color_param in filtered_data.columns:
            size_values = filtered_data.loc[valid_indices, color_param]
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

            # Filter data to only include substrate-matched controls
            if 'substrate' in data.columns:
                additive_cols = [col for col in data.columns
                               if col.endswith('(M)') or col.startswith('excess ') or col.startswith('with ')]
                substrates_with_param = data[data[x_param] > 0]['substrate'].unique()
                if additive_cols:
                    is_control = data[additive_cols].fillna(0).sum(axis=1) == 0
                else:
                    is_control = data[x_param] == 0
                filtered_data = data[
                    (data[x_param] > 0) |
                    ((data[x_param] == 0) & is_control & (data['substrate'].isin(substrates_with_param)))
                ].copy()
            else:
                filtered_data = data

            # Clear and recreate axes for seaborn
            self._reset_sweep_axes()

            # Create violin plot
            if color_param and color_param != 'None' and color_param in filtered_data.columns:
                sns.violinplot(data=filtered_data, x=x_param, y=y_param, hue=color_param, ax=self.sweep_ax)
            else:
                sns.violinplot(data=filtered_data, x=x_param, y=y_param, ax=self.sweep_ax)

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
            # Filter data to only include substrate-matched controls
            if 'substrate' in data.columns:
                additive_cols = [col for col in data.columns
                               if col.endswith('(M)') or col.startswith('excess ') or col.startswith('with ')]
                substrates_with_param = data[data[x_param] > 0]['substrate'].unique()
                if additive_cols:
                    is_control = data[additive_cols].fillna(0).sum(axis=1) == 0
                else:
                    is_control = data[x_param] == 0
                filtered_data = data[
                    (data[x_param] > 0) |
                    ((data[x_param] == 0) & is_control & (data['substrate'].isin(substrates_with_param)))
                ].copy()
            else:
                filtered_data = data

            # Group data for box plot
            grouped_data = []
            labels = []

            for group_val in filtered_data[x_param].unique():
                group_data = filtered_data[filtered_data[x_param] == group_val][y_param].dropna()
                if len(group_data) > 0:
                    grouped_data.append(group_data)
                    labels.append(str(group_val))

            if not grouped_data:
                self._clear_sweep_ax("No valid data for box plot.")
                return

            box_plot = self.sweep_ax.boxplot(grouped_data, labels=labels, patch_artist=True)

            # Color boxes if color parameter is specified
            if color_param and color_param != 'None' and color_param in filtered_data.columns:
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

        self.update_parameter_plot()
        
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
        self.jv_frame.rowconfigure(0, weight=1)
        self.jv_frame.columnconfigure(0, weight=1)

        # Use PanedWindow for resizable layout
        jv_paned = ttk.PanedWindow(self.jv_frame, orient=tk.HORIZONTAL)
        jv_paned.grid(row=0, column=0, sticky="nsew")

        # Left panel: Sample selection and controls
        jv_left = ttk.Frame(jv_paned, padding=8)
        jv_paned.add(jv_left, weight=0)
        
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

        # Figure size controls
        ttk.Separator(jv_left).grid(row=13, column=0, sticky="ew", pady=8)
        ttk.Label(jv_left, text="Figure Size:").grid(row=14, column=0, sticky="w")

        figsize_frame = ttk.Frame(jv_left); figsize_frame.grid(row=15, column=0, sticky="ew")
        ttk.Label(figsize_frame, text="Width:").grid(row=0, column=0, sticky="w")
        self.jv_fig_width = tk.StringVar(value="8")
        ttk.Entry(figsize_frame, textvariable=self.jv_fig_width, width=6).grid(row=0, column=1, sticky="w", padx=2)
        ttk.Label(figsize_frame, text="Height:").grid(row=0, column=2, sticky="w")
        self.jv_fig_height = tk.StringVar(value="8")
        ttk.Entry(figsize_frame, textvariable=self.jv_fig_height, width=6).grid(row=0, column=3, sticky="w", padx=2)

        ttk.Button(jv_left, text="Apply Figure Size",
                  command=self.update_jv_figsize).grid(row=16, column=0, sticky="ew", pady=2)

        # Plot button
        ttk.Button(jv_left, text="Plot Selected JV Curves",
                  command=self.plot_jv_curves).grid(row=17, column=0, sticky="ew", pady=8)
        
        # Clear button
        ttk.Button(jv_left, text="Clear Plot",
                  command=self.clear_jv_plot).grid(row=18, column=0, sticky="ew")

        # Save plot button
        ttk.Button(jv_left, text="Save Plot",
                  command=self.save_jv_plot).grid(row=19, column=0, sticky="ew", pady=2)
        
        # Right panel: JV curve plot
        jv_right = ttk.Frame(jv_paned, padding=8)
        jv_paned.add(jv_right, weight=1)
        jv_right.rowconfigure(0, weight=1); jv_right.columnconfigure(0, weight=1)

        # JV plot figure and canvas (square aspect ratio)
        self.jv_fig = Figure(figsize=(8, 8), dpi=100)
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

        # Collect selected sweeps
        selected_sweeps = []

        for item in selected_items:
            # Use the iid (dataframe index) to look up the row directly from original df
            try:
                df_idx = int(item)
                if df_idx not in self.df_with_flags.index:
                    continue

                row = self.df_with_flags.loc[df_idx]
                substrate = int(row["substrate"])
                pixel_id = int(row["pixel_id"])
                comp = int(row["composition_index"])
                pos = int(row["position_in_composition"])
                direction = row["direction"]

                # Find the corresponding JVSweep object
                for sweep in self.data:
                    if (sweep.substrate == substrate and
                        sweep.pixel_id == pixel_id and
                        sweep.composition_index == comp and
                        sweep.position_in_composition == pos and
                        sweep.direction == direction):
                        selected_sweeps.append(sweep)
                        break
            except (ValueError, KeyError):
                continue
        
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
        self.jv_ax.set_xlabel("Voltage (V)", fontsize=16)
        self.jv_ax.set_ylabel("Current Density (mA/cm²)", fontsize=16)
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
        self.jv_ax.set_xlabel("Voltage (V)", fontsize=16)
        self.jv_ax.set_ylabel("Current Density (mA/cm²)", fontsize=16)
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

    def update_jv_figsize(self):
        """Update JV plot figure size based on user input"""
        if not hasattr(self, 'jv_fig') or not hasattr(self, 'jv_canvas'):
            return

        try:
            width = float(self.jv_fig_width.get())
            height = float(self.jv_fig_height.get())

            if width <= 0 or height <= 0:
                messagebox.showerror("Invalid Size", "Width and height must be positive numbers")
                return

            # Store the parent widget
            parent = self.jv_canvas.get_tk_widget().master

            # Destroy old canvas
            self.jv_canvas.get_tk_widget().destroy()

            # Update figure size
            self.jv_fig.set_size_inches(width, height)

            # Create new canvas with updated figure
            self.jv_canvas = FigureCanvasTkAgg(self.jv_fig, master=parent)
            self.jv_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

            self.jv_fig.tight_layout()
            self.jv_canvas.draw()
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter valid numbers for width and height")


if __name__ == "__main__":
    app = JVApp()
    app.mainloop()
