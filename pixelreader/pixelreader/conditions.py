from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .models import JVSweep


def load_experimental_conditions(
    excel_path: Optional[str] = None,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Load experimental conditions from Excel file.

    Returns:
        tuple: (conditions_df, runsheet_df) where conditions_df is from ROSIE sheet
               and runsheet_df is from Runsheet sheet
    """
    if excel_path is None:
        # Try to find experiment sheets next to the main app entrypoint
        excel_path = Path(__file__).resolve().parents[1] / "experiment sheets.xlsx"

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
        if "ROSIE" in xlsx.sheet_names:
            conditions_df = pd.read_excel(excel_path, sheet_name="ROSIE")
            print(f"Successfully loaded {len(conditions_df)} rows from ROSIE sheet")
        else:
            print("'ROSIE' sheet not found in Excel file")

        # Load Runsheet if available
        if "Runsheet" in xlsx.sheet_names:
            runsheet_df = pd.read_excel(excel_path, sheet_name="Runsheet")
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
    """Detect which columns have meaningful variation within a sweep.

    Args:
        sweep_data: DataFrame containing data for a single sweep
        threshold: Minimum relative variation to consider significant

    Returns:
        List of column names that show meaningful variation
    """
    varying_columns = []

    # Skip non-numeric columns and known identifier columns
    skip_columns = {"Substrate", "Sweep", "Sample", "Position", "Pixel"}

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
    """Generate clean parameter name from column header.

    Args:
        column_name: Original column name from Excel

    Returns:
        Clean, formatted parameter name
    """
    # Remove common units and formatting
    clean_name = column_name.strip()

    # Remove concentration units like (M), (mM), etc.
    clean_name = re.sub(r"\s*\([mM]*[Mm]*\)\s*$", "", clean_name)

    # Handle common chemical names and abbreviations
    replacements = {
        "PbI2": "PbI₂",
        "excess PbI2": "Excess PbI₂",
        "with Thiourea": "Thiourea",
        "with FABF4": "FABF₄",
        "FABF4": "FABF₄",
        "CsI": "CsI",
        "FAI": "FAI",
        "DMAI": "DMAI",
        "DMPU": "DMPU",
        "MAI": "MAI",
    }

    for old, new in replacements.items():
        if old in clean_name:
            clean_name = clean_name.replace(old, new)
            break

    return clean_name


def classify_sweep_type(varying_params: List[str], param_info: Dict) -> str:
    """Classify sweep type and generate description.

    Args:
        varying_params: List of varying parameter names
        param_info: Dictionary with parameter details

    Returns:
        Descriptive string for the sweep type
    """
    n_params = len(varying_params)

    if n_params == 0:
        return "Control (no variation)"
    if n_params == 1:
        param_name = generate_param_name(varying_params[0])
        param_range = param_info[varying_params[0]]["range"]
        return f"{param_name} ({param_range[0]:.3f}–{param_range[1]:.3f})"
    if n_params == 2:
        param1 = generate_param_name(varying_params[0])
        param2 = generate_param_name(varying_params[1])
        return f"{param1} A- {param2} (2D)"

    param_names = [generate_param_name(p) for p in varying_params[:3]]
    if n_params > 3:
        return f"{' A- '.join(param_names)} + {n_params-3} more ({n_params}D)"
    return f"{' A- '.join(param_names)} ({n_params}D)"


def analyze_sweep_parameters(
    conditions_df: pd.DataFrame,
    runsheet_df: Optional[pd.DataFrame] = None,
) -> Dict[int, Dict]:
    """Analyze what parameters vary within each sweep using Runsheet data for robustness.

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
    if "Sweep" not in analysis_df.columns:
        print("[SWEEP ANALYSIS] ERROR: No 'Sweep' column found in analysis data")
        return {}

    unique_sweeps = analysis_df["Sweep"].unique()
    print(f"[SWEEP ANALYSIS] Found {len(unique_sweeps)} unique sweeps: {sorted(unique_sweeps)}")

    for sweep_id in unique_sweeps:
        print(f"\n[SWEEP ANALYSIS] Analyzing Sweep {sweep_id}")
        sweep_data = analysis_df[analysis_df["Sweep"] == sweep_id]
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
                    "type": "varying",
                    "values": sorted(nonzero_values.unique()),
                    "range": [nonzero_values.min(), nonzero_values.max()],
                }

        # Also track constant parameters with significant values
        constant_params = []
        skip_columns = {"Substrate", "Sweep", "Sample", "Position", "Pixel"}
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
                    "type": "constant",
                    "value": nonzero_values.iloc[0],
                }

        # Generate description using our new helper function
        desc = classify_sweep_type(varying_params, param_info)
        print(f"[SWEEP ANALYSIS] Sweep {sweep_id} description: {desc}")

        sweep_analysis[sweep_id] = {
            "description": desc,
            "varying_params": varying_params,
            "constant_params": constant_params,
            "param_info": param_info,
            "substrates": sorted(sweep_data["Substrate"].unique()),
            "n_entries": len(sweep_data),
        }

    return sweep_analysis


def map_sweeps_to_conditions(sweeps: List[JVSweep], conditions_df: Optional[pd.DataFrame]) -> List[JVSweep]:
    """Map JV sweeps to experimental conditions based on substrate"""
    if conditions_df is None:
        return sweeps

    # Create a mapping from substrate to sweep conditions
    substrate_to_conditions = {}
    for _, row in conditions_df.iterrows():
        substrate = int(row["Substrate"])
        sweep_id = int(row["Sweep"])

        # Create a condition name based on the sheet format
        condition_parts = []
        if "excess PbI2" in conditions_df.columns:
            # ROSIE sheet format
            experimental_params = ["excess PbI2", "with Thiourea", "with FABF4"]
            for col in experimental_params:
                if col in conditions_df.columns and row[col] > 0:
                    param_name = col.replace("excess ", "").replace("with ", "")
                    concentration = f"{row[col]:.3f}"
                    condition_parts.append(f"{param_name}:{concentration}")
        else:
            # Runsheet format (fallback)
            for col in conditions_df.columns:
                if col.endswith("(M)") and row[col] > 0:
                    chemical = col.replace(" (M)", "")
                    concentration = f"{row[col]:.3f}M"
                    condition_parts.append(f"{chemical}:{concentration}")

        condition_name = f"Sweep_{sweep_id}" + (f" ({', '.join(condition_parts)})" if condition_parts else "")
        substrate_to_conditions[substrate] = {"sweep_id": sweep_id, "condition_name": condition_name}

    # Update sweeps with condition information
    for sweep in sweeps:
        if sweep.substrate in substrate_to_conditions:
            conditions = substrate_to_conditions[sweep.substrate]
            sweep.sweep_id = conditions["sweep_id"]
            sweep.condition_name = conditions["condition_name"]

    return sweeps
