from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from pixelreader.conditions import analyze_sweep_parameters, load_experimental_conditions, map_sweeps_to_conditions
from pixelreader.grouping import comp_to_group
from pixelreader.metrics import compute_metrics
from pixelreader.models import JVSweep
from pixelreader.parsing import build_sweeps_from_file


class CompositionTabMixin:
    def _build_composition_tab(self):
        """Build the Data Table tab - only data loading and table viewing"""
        self.comp_frame.columnconfigure(0, weight=1)
        self.comp_frame.rowconfigure(0, weight=1)

        # Left sidebar for data loading and parameters
        side = ttk.Frame(self.comp_frame, padding=8); side.grid(row=0, column=0, sticky="ns")

        ttk.Label(side, text="Data files").grid(row=0, column=0, sticky="w")
        ttk.Button(side, text="Load file", command=self.load_file).grid(row=1, column=0, sticky="ew", pady=2)
        ttk.Button(side, text="Load folder", command=self.load_folder).grid(row=2, column=0, sticky="ew", pady=2)
        ttk.Checkbutton(side, text="Load subfolders", variable=self.load_recursive).grid(row=3, column=0, sticky="w", pady=2)

        ttk.Separator(side).grid(row=4, column=0, sticky="ew", pady=6)
        ttk.Label(side, text="Parameters").grid(row=5, column=0, sticky="w")
        pfrm = ttk.Frame(side); pfrm.grid(row=6, column=0, sticky="ew")
        ttk.Label(pfrm, text="Area (cmA?):").grid(row=0, column=0, sticky="w")
        ttk.Entry(pfrm, textvariable=self.area_cm2, width=10).grid(row=0, column=1, sticky="w", padx=4)
        ttk.Label(pfrm, text="Pin (mW/cmA?):").grid(row=1, column=0, sticky="w")
        ttk.Entry(pfrm, textvariable=self.light_mw_cm2, width=10).grid(row=1, column=1, sticky="w", padx=4)
        ttk.Button(pfrm, text="Recompute metrics", command=self.recompute_metrics).grid(row=2, column=0, columnspan=2, sticky="ew", pady=4)

        ttk.Separator(side).grid(row=7, column=0, sticky="ew", pady=6)
        ttk.Button(side, text="Remove items???", command=self.open_remove_dialog).grid(row=8, column=0, sticky="ew")

        ttk.Separator(side).grid(row=9, column=0, sticky="ew", pady=6)
        ttk.Label(side, text="Header regex").grid(row=10, column=0, sticky="w")
        ttk.Entry(side, textvariable=self.header_pattern_var, width=38).grid(row=11, column=0, sticky="ew", pady=2)
        ttk.Button(side, text="Reload last folder/file", command=self.reload_last).grid(row=12, column=0, sticky="ew", pady=2)
        ttk.Button(side, text="Parse report", command=self.show_parse_report).grid(row=13, column=0, sticky="ew", pady=2)

        # Right panel - Just the data table
        right = ttk.Frame(self.comp_frame, padding=8); right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(0, weight=1); right.columnconfigure(0, weight=1)

        table_frame = ttk.Frame(right); table_frame.grid(row=0, column=0, sticky="nsew")
        columns = ("substrate", "pixel_id", "comp", "group", "pos", "dir", "Voc", "Jsc_mAcm2", "FF_pct", "PCE_pct", "avPCE_pct")
        self.tree = ttk.Treeview(table_frame, columns=columns, show="headings", selectmode="extended")
        for c in columns:
            # All columns are now sortable
            self.tree.heading(c, text=c, command=lambda col=c: self.sort_by_column(col))
            self.tree.column(c, width=90, stretch=True, anchor="center")
        vsb = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        hsb = ttk.Scrollbar(table_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscroll=vsb.set, xscroll=hsb.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns"); hsb.grid(row=1, column=0, sticky="ew")
        table_frame.rowconfigure(0, weight=1); table_frame.columnconfigure(0, weight=1)

        btns = ttk.Frame(table_frame); btns.grid(row=2, column=0, sticky="ew", pady=4)
        ttk.Button(btns, text="Remove Selected", command=self.remove_selected).grid(row=0, column=0, padx=2)
        ttk.Button(btns, text="Export table CSV", command=self.export_table_csv).grid(row=0, column=1, padx=2)
        ttk.Button(btns, text="Clear all", command=self.clear_all_data).grid(row=0, column=2, padx=2)


    def load_file(self):
        path = filedialog.askopenfilename(title="Select JV file", filetypes=[("Text", "*.txt *.dat *.csv"), ("All", "*.*")])
        if not path: return
        self._load_paths([Path(path)]); self._last_paths = [Path(path)]


    def load_folder(self):
        d = filedialog.askdirectory(title="Select folder with JV files")
        if not d: return
        recursive = self.load_recursive.get()
        pattern = "**/*" if recursive else "*"
        paths = [
            p for p in Path(d).glob(pattern)
            if p.is_file() and p.suffix.lower() in (".txt", ".dat", ".csv")
        ]
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
        self._rebuild_sweep_uid_map()

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
                "sweep_uid": id(s),
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


    def _rebuild_sweep_uid_map(self):
        """Build a lookup from sweep uid to sweep object."""
        self._sweep_by_uid = {id(s): s for s in self.data} if self.data else {}


    def _populate_substrate_combo(self):
        if self.df is None or self.df.empty: return
        subs_unique = sorted(pd.unique(self.df["substrate"].dropna().astype(int)))
        values = ["All"] + [str(s) for s in subs_unique]
        self.substrate_cb["values"] = values
        cur = self.substrate_cb.get()
        # Default to first substrate instead of "All" to enable pixel map view
        default = str(subs_unique[0]) if subs_unique else "All"
        self.substrate_cb.set(cur if cur in values else default)


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
            mask = pd.Series([True] * len(df), index=df.index)
            if sub_sel != "All":
                mask &= (df["substrate"] == int(sub_sel))
            if scope.startswith("Composition"):
                mask &= (df["composition_index"] == idx)
            elif scope.startswith("Group"):
                mask &= (df["group_index"] == idx)
            else:  # Pixel position
                mask &= (df["position_in_composition"] == idx)

            indices_to_remove = df.index[mask].tolist()
            removed = len(indices_to_remove)
            if removed == 0:
                info.config(text="No matching rows found")
                return

            # Physical deletion from all data structures
            self.df_with_flags = self.df_with_flags.drop(indices_to_remove).reset_index(drop=True)
            if self.df is not None:
                self.df = self.df.drop(indices_to_remove).reset_index(drop=True)
            # Remove from self.data in reverse order to maintain index validity
            for i in sorted(indices_to_remove, reverse=True):
                if i < len(self.data):
                    self.data.pop(i)

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
            "PCE_pct": "PCE_pct",
            "avPCE_pct": "avPCE_pct"
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
        columns = ("substrate", "pixel_id", "comp", "group", "pos", "dir", "Voc", "Jsc_mAcm2", "FF_pct", "PCE_pct", "avPCE_pct")

        for c in columns:
            if c == self._sort_column:
                arrow = " ▼" if self._sort_reverse else " ▲"
                self.tree.heading(c, text=f"{c}{arrow}")
            else:
                self.tree.heading(c, text=c)


    def refresh_table(self):
        self.tree.delete(*self.tree.get_children())
        if self.df_with_flags is None or self.df_with_flags.empty: return

        # Compute average PCE for each substrate+pixel_id combination
        self._compute_average_pce()

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
                None if pd.isna(r.get("avPCE_pct")) else round(float(r["avPCE_pct"]), 2),
            )
            self.tree.insert("", "end", iid=str(idx), values=vals)

    def _compute_average_pce(self):
        """Compute average PCE for each substrate+pixel_id combination (typically F+R average)"""
        if self.df_with_flags is None or self.df_with_flags.empty:
            return

        # Group by substrate and pixel_id to get average PCE
        avg_pce = self.df_with_flags.groupby(["substrate", "pixel_id"])["PCE_pct"].mean()

        # Map back to the dataframe
        self.df_with_flags["avPCE_pct"] = self.df_with_flags.apply(
            lambda row: avg_pce.get((row["substrate"], row["pixel_id"]), np.nan),
            axis=1
        )


    def _remove_rows_by_indices(self, indices_to_remove, confirm_message: Optional[str] = None, refresh_plots: bool = True):
        """Remove rows by dataframe indices and update all dependent data structures."""
        if self.df_with_flags is None or self.df_with_flags.empty:
            return 0
        if not indices_to_remove:
            return 0

        # Normalize and validate indices
        cleaned = []
        for i in indices_to_remove:
            try:
                cleaned.append(int(i))
            except (TypeError, ValueError):
                continue
        indices_to_remove = sorted(set(cleaned), reverse=True)
        indices_to_remove = [i for i in indices_to_remove if 0 <= i < len(self.df_with_flags)]
        if not indices_to_remove:
            return 0

        if confirm_message and not messagebox.askyesno("Delete data", confirm_message):
            return 0

        # Capture selected rows before mutating dataframes
        selected_rows = self.df_with_flags.loc[indices_to_remove].copy()

        # Remove rows from dataframe
        self.df_with_flags = self.df_with_flags.drop(indices_to_remove).reset_index(drop=True)

        # Also update the underlying df if it exists
        if self.df is not None:
            self.df = self.df.drop(indices_to_remove).reset_index(drop=True)

        # Update the original data list as well (match by sweep uid when available)
        if self.data:
            sweep_uids = set()
            if "sweep_uid" in selected_rows.columns:
                sweep_uids = set(int(u) for u in selected_rows["sweep_uid"].dropna().tolist())

            if sweep_uids:
                self.data = [s for s in self.data if id(s) not in sweep_uids]
            else:
                to_remove_counts = {}
                for _, r in selected_rows.iterrows():
                    sweep_id = None if pd.isna(r.get("sweep_id")) else int(r.get("sweep_id"))
                    key = (
                        int(r["substrate"]) if pd.notna(r["substrate"]) else None,
                        int(r["pixel_id"]) if pd.notna(r["pixel_id"]) else None,
                        int(r["composition_index"]) if pd.notna(r["composition_index"]) else None,
                        int(r["position_in_composition"]) if pd.notna(r["position_in_composition"]) else None,
                        r["direction"],
                        sweep_id,
                    )
                    to_remove_counts[key] = to_remove_counts.get(key, 0) + 1

                new_data = []
                for sweep in self.data:
                    key = (
                        getattr(sweep, "substrate", None),
                        getattr(sweep, "pixel_id", None),
                        getattr(sweep, "composition_index", None),
                        getattr(sweep, "position_in_composition", None),
                        getattr(sweep, "direction", None),
                        getattr(sweep, "sweep_id", None),
                    )
                    if to_remove_counts.get(key, 0) > 0:
                        to_remove_counts[key] -= 1
                        continue
                    new_data.append(sweep)
                self.data = new_data
            self._rebuild_sweep_uid_map()

        self.refresh_table()
        if refresh_plots:
            self.refresh_plots()
        return len(selected_rows)


    def remove_selected(self):
        """Delete selected measurements from the data"""
        if self.df_with_flags is None or self.df_with_flags.empty:
            return
        sel = self.tree.selection()
        if not sel:
            return

        indices_to_remove = [int(iid) for iid in sel]
        self._remove_rows_by_indices(indices_to_remove, refresh_plots=True)

    def clear_all_data(self):
        """Clear all loaded data and reset tables/plots."""
        has_data = bool(self.data) or (self.df_with_flags is not None and not self.df_with_flags.empty)
        if not has_data:
            messagebox.showinfo("Clear all", "No data to clear.")
            return

        if not messagebox.askyesno("Clear all", "Clear all loaded data from the tables and plots?"):
            return

        self.data = []
        self.df = None
        self.df_with_flags = None
        self._sweep_by_uid = {}
        self._sort_column = None
        self._sort_reverse = False

        try:
            self.substrate_cb["values"] = ["All"]
            self.substrate_cb.set("All")
        except Exception:
            pass

        try:
            self.sweep_filter_cb["values"] = ["All"]
            self.selected_sweep_filter.set("All")
        except Exception:
            pass

        self.refresh_table()
        self.refresh_plots()

        try:
            self.clear_jv_plot()
        except AttributeError:
            pass

        try:
            self._clear_sweep_ax("No parameter data loaded.")
        except AttributeError:
            pass



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
                "sweep_uid": id(s),
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
        self._rebuild_sweep_uid_map()
        self._populate_substrate_combo()
        self._populate_sweep_filter_combo()
        self.refresh_table(); self.refresh_plots()

    # -------------------- Plotting helpers --------------------
