from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import tkinter as tk
from tkinter import ttk

from matplotlib.figure import Figure

from pixelreader.conditions import analyze_sweep_parameters, load_experimental_conditions, map_sweeps_to_conditions
from pixelreader.grouping import comp_to_group
from pixelreader.metrics import compute_metrics
from pixelreader.models import JVSweep
from pixelreader.parsing import DEFAULT_HEADER_REGEX, build_sweeps_from_file
from pixelreader.ui.composition_tab import CompositionTabMixin
from pixelreader.ui.jv_tab import JVTabMixin
from pixelreader.ui.plot_tab import PlotTabMixin
from pixelreader.ui.sweep_tab import SweepTabMixin


class JVApp(tk.Tk, CompositionTabMixin, PlotTabMixin, SweepTabMixin, JVTabMixin):
    def __init__(self):
        super().__init__()
        self.title("Perovskite JV Analyzer")
        self.geometry("1500x900")

        # Parameters
        self.area_cm2 = tk.DoubleVar(value=0.04)
        self.light_mw_cm2 = tk.DoubleVar(value=75.0)
        self.include_forward = tk.BooleanVar(value=True)
        self.include_reverse = tk.BooleanVar(value=True)
        self.selected_sweep_filter = tk.StringVar(value="All")  # Sweep filter for Tab 1
        self.metric_choice = tk.StringVar(value="PCE_pct")
        self.aggregation_method = tk.StringVar(value="max")
        self.combine_substrates = tk.BooleanVar(value=True)
        self.combine_fr = tk.BooleanVar(value=True)
        self.grouping_mode = tk.StringVar(value="11 compositions")
        self.expand_substrate_axis = tk.BooleanVar(value=True)
        self.discard_edge_rows = tk.BooleanVar(value=False)
        self.load_recursive = tk.BooleanVar(value=True)

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

        # Tab 2: Substrate Plots and Visualization
        self.plot_frame = ttk.Frame(self.notebook, padding=8)
        self.notebook.add(self.plot_frame, text="Substrate Plots")

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
        
