from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

from pixelreader.grouping import comp_to_group
from pixelreader.wellmap import pixel_id_to_well, well_to_pixel_id


class PlotTabMixin:
    def _build_plot_tab(self):
        """Build the Substrate Plots tab with all plotting controls and figure"""
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
        ttk.Checkbutton(plot_left, text="Discard edges", variable=self.discard_edge_rows, command=self.refresh_plots).grid(row=20, column=0, sticky="w")
        ttk.Checkbutton(plot_left, text="Expand x-axis by substrate", variable=self.expand_substrate_axis, command=self.refresh_plots).grid(row=21, column=0, sticky="w", pady=(2, 6))

        ttk.Separator(plot_left).grid(row=22, column=0, sticky="ew", pady=6)
        ttk.Label(plot_left, text="Plot Styling").grid(row=23, column=0, sticky="w")
        ttk.Label(plot_left, text="Colormap:").grid(row=24, column=0, sticky="w", pady=(2, 0))
        colormap_cb = ttk.Combobox(plot_left, textvariable=self.colormap_choice,
                                    values=["viridis", "plasma", "inferno", "magma", "cividis", "coolwarm", "RdYlGn", "RdYlBu"],
                                    state="readonly", width=12)
        colormap_cb.grid(row=25, column=0, sticky="ew")
        colormap_cb.bind("<<ComboboxSelected>>", lambda e: self.refresh_plots())

        ttk.Label(plot_left, text="Axis Limits (optional):").grid(row=26, column=0, sticky="w", pady=(4, 0))
        axis_frame = ttk.Frame(plot_left)
        axis_frame.grid(row=27, column=0, sticky="ew")
        ttk.Label(axis_frame, text="X min:").grid(row=0, column=0, sticky="w")
        ttk.Entry(axis_frame, textvariable=self.x_min_var, width=6).grid(row=0, column=1, sticky="w", padx=2)
        ttk.Label(axis_frame, text="max:").grid(row=0, column=2, sticky="w")
        ttk.Entry(axis_frame, textvariable=self.x_max_var, width=6).grid(row=0, column=3, sticky="w", padx=2)
        ttk.Label(axis_frame, text="Y min:").grid(row=1, column=0, sticky="w")
        ttk.Entry(axis_frame, textvariable=self.y_min_var, width=6).grid(row=1, column=1, sticky="w", padx=2)
        ttk.Label(axis_frame, text="max:").grid(row=1, column=2, sticky="w")
        ttk.Entry(axis_frame, textvariable=self.y_max_var, width=6).grid(row=1, column=3, sticky="w", padx=2)
        ttk.Button(plot_left, text="Apply limits", command=self.refresh_plots).grid(row=28, column=0, sticky="ew", pady=2)
        ttk.Button(plot_left, text="Clear limits", command=self.clear_axis_limits).grid(row=29, column=0, sticky="ew", pady=2)

        ttk.Separator(plot_left).grid(row=30, column=0, sticky="ew", pady=6)
        ttk.Label(plot_left, text="Figure Size & Fonts").grid(row=31, column=0, sticky="w")
        figsize_frame = ttk.Frame(plot_left)
        figsize_frame.grid(row=32, column=0, sticky="ew")
        ttk.Label(figsize_frame, text="Width:").grid(row=0, column=0, sticky="w")
        ttk.Entry(figsize_frame, textvariable=self.comp_fig_width, width=6).grid(row=0, column=1, sticky="w", padx=2)
        ttk.Label(figsize_frame, text="Height:").grid(row=0, column=2, sticky="w")
        ttk.Entry(figsize_frame, textvariable=self.comp_fig_height, width=6).grid(row=0, column=3, sticky="w", padx=2)

        font_frame = ttk.Frame(plot_left)
        font_frame.grid(row=33, column=0, sticky="ew")
        ttk.Label(font_frame, text="Title:").grid(row=0, column=0, sticky="w")
        ttk.Entry(font_frame, textvariable=self.comp_title_fontsize, width=6).grid(row=0, column=1, sticky="w", padx=2)
        ttk.Label(font_frame, text="Axis:").grid(row=0, column=2, sticky="w")
        ttk.Entry(font_frame, textvariable=self.comp_axis_fontsize, width=6).grid(row=0, column=3, sticky="w", padx=2)

        ttk.Button(plot_left, text="Apply Figure Settings", command=self.update_comp_figsize).grid(row=34, column=0, sticky="ew", pady=2)

        ttk.Separator(plot_left).grid(row=35, column=0, sticky="ew", pady=6)
        ttk.Label(plot_left, text="Substrate-Composition").grid(row=36, column=0, sticky="w")
        ttk.Label(plot_left, text="Pairs (e.g., 5-3, 2-8):").grid(row=37, column=0, sticky="w", pady=(2, 0))
        self.sub_comp_selection_var = tk.StringVar(value="")
        ttk.Entry(plot_left, textvariable=self.sub_comp_selection_var, width=20).grid(row=38, column=0, sticky="ew", pady=2)
        ttk.Button(plot_left, text="Manual Pixel Plot", command=self.plot_substrate_composition_boxplot).grid(row=39, column=0, sticky="ew")

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

        # Apply sweep filter
        sweep_filter = self.selected_sweep_filter.get()
        if sweep_filter != "All" and "sweep_id" in df.columns:
            try:
                sweep_num = int(sweep_filter)
                df = df[df["sweep_id"] == sweep_num]
            except ValueError:
                pass  # If conversion fails, ignore filter

        # Optionally drop edge compositions (1 and 11)
        if self.discard_edge_rows.get() and "composition_index" in df.columns:
            df = df[~df["composition_index"].isin([1, 11])]

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
        """Merge parameter data with JV data.

        The runsheet format has 11 rows per substrate, where the row index within
        each substrate group (0-10) corresponds to composition_index (1-11).
        This function creates a composition_index column in the parameter data
        and merges on both substrate and composition_index.
        """
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

        # Normalize substrate column name
        if 'Substrate' in param_data.columns and 'substrate' not in param_data.columns:
            param_data = param_data.rename(columns={'Substrate': 'substrate'})

        # Check if we have substrate column for the primary merge strategy
        if 'substrate' in jv_data.columns and 'substrate' in param_data.columns:
            # Primary strategy: merge on substrate + composition_index
            # The runsheet has 11 rows per substrate, where row index within each
            # substrate group corresponds to composition_index - 1

            # Create composition_index in param_data based on row position within each substrate
            if 'composition_index' not in param_data.columns:
                # Calculate composition_index as the 1-based row index within each substrate group
                param_data = param_data.copy()
                param_data['composition_index'] = param_data.groupby('substrate').cumcount() + 1

            # Merge on both substrate and composition_index
            if 'composition_index' in jv_data.columns:
                try:
                    combined = jv_data.merge(
                        param_data,
                        on=['substrate', 'composition_index'],
                        how='inner'
                    )
                    if not combined.empty:
                        return combined
                except Exception:
                    pass  # Fall through to other strategies

            # Fallback: merge on substrate only (less accurate but may work for some cases)
            try:
                combined = jv_data.merge(param_data, on=['substrate'], how='inner')
                if not combined.empty:
                    return combined
            except Exception:
                pass

        # Strategy 2: Try other common identifiers
        for col_jv, col_param in [('substrate', 'Sample'), ('substrate', 'Position'),
                                 ('pixel_id', 'Pixel'), ('composition_index', 'Composition')]:
            if col_jv in jv_data.columns and col_param in param_data.columns:
                param_data_renamed = param_data.rename(columns={col_param: col_jv})
                try:
                    combined = jv_data.merge(param_data_renamed, on=[col_jv], how='inner')
                    if not combined.empty:
                        return combined
                except Exception:
                    continue

        # Fallback: try to merge by index if sizes match
        if len(jv_data) == len(param_data):
            # Reset indices and merge by position
            jv_data = jv_data.reset_index(drop=True)
            param_data = param_data.reset_index(drop=True)
            combined = pd.concat([jv_data, param_data], axis=1)
            return combined

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
        # Silently return if data isn't ready (allows automatic updates without error dialogs)
        if self.parameter_data is None:
            return

        if self.df_with_flags is None or self.df_with_flags.empty:
            return

        # Get selected columns from listbox
        selected_indices = self.boxplot_columns_listbox.curselection()
        if not selected_indices:
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

            # Round numeric columns to avoid floating point precision issues
            # (e.g., 0.8 vs 0.8000000000000002 being treated as different values)
            for col in selected_columns:
                if col in combined_data.columns and pd.api.types.is_numeric_dtype(combined_data[col]):
                    combined_data[col] = combined_data[col].round(6)

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
                # Get unique values for the grouping column (always include zeros)
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
            # Performance metric from JV data - respect current filtering/removals
            data_source = self._filtered_df()
            if data_source.empty:
                messagebox.showwarning(
                    "No JV data",
                    "No JV data available after filters/removals. Load JV data or adjust filters."
                )
                return
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

            # Include all data points (including zero values)
            filtered_data = data.copy()
        else:
            filtered_data = data

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
        # Include all data points (including zero values)
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
        # Include all data points (including zero values)
        filtered_data = data.copy()

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

            # Include all data points (including zero values)
            filtered_data = data.copy()

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
        """Create box plot with proper grouping by color parameter when selected"""
        try:
            from matplotlib.patches import Patch

            filtered_data = data.copy()

            # Round numeric columns to avoid floating point precision issues
            # (e.g., 0.8 vs 0.8000000000000002 being treated as different values)
            if x_param in filtered_data.columns and pd.api.types.is_numeric_dtype(filtered_data[x_param]):
                filtered_data[x_param] = filtered_data[x_param].round(6)
            if color_param and color_param != 'None' and color_param in filtered_data.columns:
                if pd.api.types.is_numeric_dtype(filtered_data[color_param]):
                    filtered_data[color_param] = filtered_data[color_param].round(6)

            # Metric labels for nice display
            metric_labels = {
                'Voc': 'Open Circuit Voltage (V)',
                'Jsc_mAcm2': 'Short Circuit Current Density (mA/cm²)',
                'FF_pct': 'Fill Factor (%)',
                'PCE_pct': 'Power Conversion Efficiency (%)'
            }

            x_label = x_param.replace(' (M)', '').replace(' (mg/mL)', '')
            y_label = metric_labels.get(y_param, y_param)

            # Check if we should group by color parameter
            use_color_grouping = (color_param and color_param != 'None' and
                                  color_param in filtered_data.columns)

            if use_color_grouping:
                # Group by BOTH x_param AND color_param
                unique_x_vals = sorted([v for v in filtered_data[x_param].unique() if not pd.isna(v)])
                unique_color_vals = sorted([v for v in filtered_data[color_param].unique() if not pd.isna(v)])

                if len(unique_x_vals) == 0 or len(unique_color_vals) == 0:
                    self._clear_sweep_ax("No valid data for box plot.")
                    return

                # Create color map for color parameter values
                color_map = plt.cm.Set2(np.linspace(0, 1, len(unique_color_vals)))
                color_dict = {val: color_map[i] for i, val in enumerate(unique_color_vals)}

                grouped_data = []
                labels = []
                positions = []
                box_colors = []

                n_color_groups = len(unique_color_vals)
                box_width = 0.8 / n_color_groups  # Width of each box within a group

                for x_idx, x_val in enumerate(unique_x_vals):
                    group_center = x_idx + 1

                    for c_idx, color_val in enumerate(unique_color_vals):
                        # Filter for this x_val AND color_val combination
                        mask = (filtered_data[x_param] == x_val) & (filtered_data[color_param] == color_val)
                        group_data = filtered_data.loc[mask, y_param].dropna()

                        if len(group_data) > 0:
                            grouped_data.append(group_data)
                            # Position boxes side by side within each x-value group
                            offset = (c_idx - (n_color_groups - 1) / 2) * box_width
                            positions.append(group_center + offset)
                            box_colors.append(color_dict[color_val])
                            # Label only shows x value (color is shown in legend)
                            val_str = f"{x_val:.3g}" if isinstance(x_val, (int, float)) else str(x_val)
                            labels.append(val_str)

                if not grouped_data:
                    self._clear_sweep_ax("No valid data for box plot.")
                    return

                # Create box plot
                box_plot = self.sweep_ax.boxplot(grouped_data, positions=positions,
                                                 patch_artist=True, showmeans=True,
                                                 widths=box_width * 0.9,
                                                 meanprops=dict(marker='D', markerfacecolor='red',
                                                              markeredgecolor='darkred', markersize=5))

                # Color boxes according to color parameter
                for patch, color in zip(box_plot['boxes'], box_colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                    patch.set_edgecolor('black')
                    patch.set_linewidth(1.5)

                # Add mean values as text labels
                for i, gdata in enumerate(grouped_data):
                    mean_val = np.mean(gdata)
                    self.sweep_ax.text(positions[i], mean_val, f'{mean_val:.2f}',
                                      ha='center', va='bottom', fontsize=7,
                                      bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.6))

                # Set x-axis ticks at group centers with x-value labels
                self.sweep_ax.set_xticks(range(1, len(unique_x_vals) + 1))
                x_tick_labels = [f"{v:.3g}" if isinstance(v, (int, float)) else str(v) for v in unique_x_vals]
                self.sweep_ax.set_xticklabels(x_tick_labels)

                # Add legend for color parameter
                color_label = color_param.replace(' (M)', '').replace(' (mg/mL)', '')
                legend_elements = [Patch(facecolor=color_dict[val], edgecolor='black', alpha=0.7,
                                        label=f"{color_label}={val:.3g}" if isinstance(val, (int, float)) else f"{color_label}={val}")
                                  for val in unique_color_vals]
                self.sweep_ax.legend(handles=legend_elements, loc='best', fontsize=8, framealpha=0.9)

                self.sweep_ax.set_title(f'Box Plot: {y_label} vs {x_label} (grouped by {color_label})',
                                       fontsize=13, fontweight='bold')
            else:
                # Original behavior: group only by x_param
                grouped_data = []
                labels = []
                positions = []

                unique_vals = sorted([v for v in filtered_data[x_param].unique() if not pd.isna(v)])
                for i, group_val in enumerate(unique_vals):
                    group_data = filtered_data[filtered_data[x_param] == group_val][y_param].dropna()
                    if len(group_data) > 0:
                        grouped_data.append(group_data)
                        val_str = f"{group_val:.3g}" if isinstance(group_val, (int, float)) else str(group_val)
                        labels.append(val_str)
                        positions.append(i + 1)

                if not grouped_data:
                    self._clear_sweep_ax("No valid data for box plot.")
                    return

                # Create box plot with enhanced styling
                box_plot = self.sweep_ax.boxplot(grouped_data, positions=positions, labels=labels,
                                                 patch_artist=True, showmeans=True, widths=0.6,
                                                 meanprops=dict(marker='D', markerfacecolor='red',
                                                              markeredgecolor='darkred', markersize=6))

                # Color boxes with sequential colors
                colors = plt.cm.Set3(np.linspace(0, 1, len(box_plot['boxes'])))
                for patch, color in zip(box_plot['boxes'], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                    patch.set_edgecolor('black')
                    patch.set_linewidth(1.5)

                # Add mean values as text labels
                for i, gdata in enumerate(grouped_data):
                    mean_val = np.mean(gdata)
                    self.sweep_ax.text(positions[i], mean_val, f'{mean_val:.2f}',
                                      ha='center', va='bottom', fontsize=8,
                                      bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.6))

                self.sweep_ax.set_title(f'Box Plot: {y_label} vs {x_label}', fontsize=13, fontweight='bold')

            self.sweep_ax.set_xlabel(x_label, fontsize=12, fontweight='bold')
            self.sweep_ax.set_ylabel(y_label, fontsize=12, fontweight='bold')
            self.sweep_ax.grid(True, alpha=0.3, linestyle='--', axis='y')
            self.sweep_ax.tick_params(axis='x', rotation=45, labelsize=9)
            self.sweep_ax.tick_params(axis='y', labelsize=10)

            self.sweep_fig.tight_layout()

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

