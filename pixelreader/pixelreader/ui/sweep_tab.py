from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from mpl_toolkits.mplot3d import Axes3D


class SweepTabMixin:
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

        # Multi-box plot controls
        ttk.Label(sweep_side, text="Multi-Box Plots", font=("Arial", 10, "bold")).grid(row=19, column=0, sticky="w", pady=(5, 2))

        # Multi-column selection for box plot grouping
        ttk.Label(sweep_side, text="Select Columns (Ctrl+Click):").grid(row=20, column=0, sticky="w", pady=(5, 0))

        # Frame for listbox and scrollbar
        listbox_frame = ttk.Frame(sweep_side)
        listbox_frame.grid(row=21, column=0, sticky="ew", pady=2)

        # Listbox for multi-selection
        self.boxplot_columns_listbox = tk.Listbox(listbox_frame, height=6, selectmode=tk.EXTENDED, exportselection=False)
        self.boxplot_columns_listbox.grid(row=0, column=0, sticky="ew")
        self.boxplot_columns_listbox.bind("<<ListboxSelect>>", lambda e: self.generate_box_plot())

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
        boxplot_metric_cb.bind("<<ComboboxSelected>>", lambda e: self.generate_box_plot())

        # Control filtering checkbox
        self.include_controls_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(sweep_side, text="Include controls (all zeros)",
                       variable=self.include_controls_var,
                       command=self.generate_box_plot).grid(row=24, column=0, sticky="w", pady=2)

        # Show mean checkbox
        self.show_mean_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(sweep_side, text="Show mean values",
                       variable=self.show_mean_var,
                       command=self.generate_box_plot).grid(row=25, column=0, sticky="w", pady=2)

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


