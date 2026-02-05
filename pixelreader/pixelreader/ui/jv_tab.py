from __future__ import annotations

import numpy as np
import pandas as pd

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class JVTabMixin:
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
        columns = ("substrate", "pixel_id", "comp", "group", "pos", "dir", "Voc", "Jsc_mAcm2", "FF_pct", "PCE_pct", "Rsc_ohmcm2", "avPCE_pct")
        self.jv_selection_tree = ttk.Treeview(selection_frame, columns=columns, show="headings",
                                             selectmode="extended", height=8)
        # Initialize sorting state
        self._jv_sort_column = None
        self._jv_sort_reverse = False
        
        for c in columns:
            self.jv_selection_tree.heading(c, text=c, command=lambda col=c: self.sort_jv_by_column(col))
            self.jv_selection_tree.column(c, width=90, stretch=True, anchor="center")
        
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
                None if pd.isna(r.get("Rsc_ohmcm2")) else round(float(r["Rsc_ohmcm2"]), 2),
                None if pd.isna(r.get("avPCE_pct")) else round(float(r["avPCE_pct"]), 2),
            )
            # Use sweep uid when available for stable lookups
            sweep_uid = r.get("sweep_uid")
            if pd.notna(sweep_uid):
                item_key = str(int(sweep_uid))
            else:
                # Fallback to key based on identifying attributes
                substrate = int(r["substrate"]) if pd.notna(r["substrate"]) else 0
                pixel_id = int(r["pixel_id"]) if pd.notna(r["pixel_id"]) else 0
                comp = int(r["composition_index"]) if pd.notna(r["composition_index"]) else 0
                pos = int(r["position_in_composition"]) if pd.notna(r["position_in_composition"]) else 0
                direction = r["direction"]
                sweep_id = int(r["sweep_id"]) if pd.notna(r.get("sweep_id")) else "na"
                item_key = f"{substrate}_{pixel_id}_{comp}_{pos}_{direction}_{sweep_id}"
            self.jv_selection_tree.insert("", "end", iid=item_key, values=vals)
    

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
            "PCE_pct": "PCE_pct",
            "avPCE_pct": "avPCE_pct",
            "Rsc_ohmcm2": "Rsc_ohmcm2"
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
                        None if pd.isna(r.get("Rsc_ohmcm2")) else round(float(r["Rsc_ohmcm2"]), 2),
                        None if pd.isna(r.get("avPCE_pct")) else round(float(r["avPCE_pct"]), 2),
                    )
                    # Use a unique key based on identifying attributes (not DataFrame index)
                    substrate = int(r["substrate"]) if pd.notna(r["substrate"]) else 0
                    pixel_id = int(r["pixel_id"]) if pd.notna(r["pixel_id"]) else 0
                    comp = int(r["composition_index"]) if pd.notna(r["composition_index"]) else 0
                    pos = int(r["position_in_composition"]) if pd.notna(r["position_in_composition"]) else 0
                    direction = r["direction"]
                    sweep_uid = r.get("sweep_uid")
                    if pd.notna(sweep_uid):
                        item_key = str(int(sweep_uid))
                    else:
                        sweep_id = int(r["sweep_id"]) if pd.notna(r.get("sweep_id")) else "na"
                        item_key = f"{substrate}_{pixel_id}_{comp}_{pos}_{direction}_{sweep_id}"
                    self.jv_selection_tree.insert("", "end", iid=item_key, values=vals)

                # Update column headers to show sort direction
                for col in ["substrate", "pixel_id", "comp", "group", "pos", "dir", "Voc", "Jsc_mAcm2", "FF_pct", "PCE_pct", "Rsc_ohmcm2", "avPCE_pct"]:
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
            # Prefer direct lookup by sweep uid if available
            try:
                uid = int(item)
                sweep = getattr(self, "_sweep_by_uid", {}).get(uid)
                if sweep is not None:
                    selected_sweeps.append(sweep)
                    continue
            except ValueError:
                pass
            # Parse the iid key: substrate_pixelid_comp_pos_direction
            try:
                parts = item.split("_")
                if len(parts) < 5:
                    continue
                substrate = int(parts[0])
                pixel_id = int(parts[1])
                comp = int(parts[2])
                pos = int(parts[3])
                direction = parts[4]
                sweep_id = None
                if len(parts) >= 6 and parts[5] != "na":
                    sweep_id = int(parts[5])

                # Find the corresponding JVSweep object
                for sweep in self.data:
                    if (sweep.substrate == substrate and
                        sweep.pixel_id == pixel_id and
                        sweep.composition_index == comp and
                        sweep.position_in_composition == pos and
                        sweep.direction == direction and
                        (sweep_id is None or getattr(sweep, "sweep_id", None) == sweep_id)):
                        selected_sweeps.append(sweep)
                        break
            except (ValueError, KeyError, IndexError):
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
