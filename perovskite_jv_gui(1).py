
"""
Perovskite JV GUI (supports 4-column F/R data blocks)
-----------------------------------------------------
- Skips preamble until a line containing "###DATA###"
- Section headers like "3_10" (also supports "3-10"), one per pixel
- Rows have 4 numeric columns: Forward V, Forward I, Reverse V, Reverse I
- Creates TWO sweeps per section directly from those columns.
"""

from __future__ import annotations

import re
import math
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

SECTION_RE = re.compile(r"^\\s*(\\d+)\\s*[_-]\\s*(\\d+)\\s*$", re.MULTILINE)

def split_numeric_fields(line: str) -> List[float]:
    parts = re.split(r"[\\t,;\\s]+", line.strip())
    out = []
    for p in parts:
        if p == "":
            continue
        try:
            out.append(float(p))
        except Exception:
            return []
    return out

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
    # Derived
    Voc: Optional[float] = None
    Jsc_mAcm2: Optional[float] = None
    FF: Optional[float] = None
    PCE_pct: Optional[float] = None

def interpolate_x_at_y_zero(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    s = np.sign(y)
    crossings = np.where(s[:-1] * s[1:] <= 0)[0]
    if len(crossings) == 0:
        return None
    # choose crossing with smallest |y| at left point
    idx = min(crossings, key=lambda k: abs(y[k]))
    x0, x1 = x[idx], x[idx+1]
    y0, y1 = y[idx], y[idx+1]
    if y1 == y0:
        return float(x0)
    t = -y0 / (y1 - y0)
    return float(x0 + t * (x1 - x0))

def interpolate_y_at_x_zero(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    s = np.sign(x)
    crossings = np.where(s[:-1] * s[1:] <= 0)[0]
    if len(crossings) == 0:
        return None
    idx = crossings[0]
    x0, x1 = x[idx], x[idx+1]
    y0, y1 = y[idx], y[idx+1]
    if x1 == x0:
        return float(y0)
    t = -x0 / (x1 - x0)
    return float(y0 + t * (y1 - y0))

def compute_metrics(voltage: np.ndarray, current_A: np.ndarray, area_cm2: float, light_mw_cm2: float) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    if len(voltage) < 2:
        return None, None, None, None
    order = np.argsort(voltage)
    V = voltage[order]
    I = current_A[order]
    J_Acm2 = I / max(area_cm2, 1e-12)
    J_mAcm2 = J_Acm2 * 1e3

    Voc = interpolate_x_at_y_zero(V, J_Acm2)
    Jsc_Acm2 = interpolate_y_at_x_zero(V, J_Acm2)
    Jsc_mAcm2 = None if Jsc_Acm2 is None else (Jsc_Acm2 * 1e3)

    P_mWcm2 = V * J_Acm2 * 1e3
    if len(P_mWcm2) == 0:
        return Voc, Jsc_mAcm2, None, None
    idx_mpp = int(np.nanargmax(P_mWcm2))
    Vmpp = V[idx_mpp]
    Jmpp_mAcm2 = J_mAcm2[idx_mpp]
    Pmpp = P_mWcm2[idx_mpp]

    FF = None
    PCE_pct = None
    if Voc is not None and Jsc_mAcm2 is not None and (abs(Voc) > 1e-9) and (abs(Jsc_mAcm2) > 1e-9):
        denom = (abs(Voc) * abs(Jsc_mAcm2))
        FF = (abs(Vmpp) * abs(Jmpp_mAcm2)) / denom
    if light_mw_cm2 > 0:
        PCE_pct = (Pmpp / light_mw_cm2) * 100.0

    return Voc, Jsc_mAcm2, FF, PCE_pct

def pixel_to_comp_and_pos(pixel_id: int) -> Tuple[int, int]:
    comp = ((pixel_id - 1) // 6) + 1
    pos = ((pixel_id - 1) % 6) + 1
    return comp, pos

def find_data_start(text: str) -> int:
    m = re.search(r"^\\s*###DATA###\\s*$", text, flags=re.MULTILINE)
    return m.end() if m else 0

def parse_sections_from_text(text: str) -> List[Tuple[int, int, List[List[float]]]]:
    """
    Returns a list of (substrate, pixel_id, rows) where rows is a list of numeric lists per line.
    Assumes 4-column lines: FV, FI, RV, RI (but will accept 2+ columns).
    """
    start = find_data_start(text)
    scan = text[start:] if start > 0 else text
    out = []
    # Iterate over headers
    matches = list(SECTION_RE.finditer(scan))
    for idx, m in enumerate(matches):
        substrate = int(m.group(1))
        pixel_id = int(m.group(2))
        sec_start = m.end()
        sec_end = matches[idx+1].start() if idx+1 < len(matches) else len(scan)
        body = scan[sec_start:sec_end].strip()
        rows = []
        for ln in body.splitlines():
            nums = split_numeric_fields(ln)
            if len(nums) >= 2:
                rows.append(nums)
        if rows:
            out.append((substrate, pixel_id, rows))
    return out

def build_sweeps_from_file(path: Path, area_cm2: float, light_mw_cm2: float) -> Tuple[List[JVSweep], dict]:
    text = path.read_text(errors="ignore")
    sec = parse_sections_from_text(text)
    sweeps: List[JVSweep] = []
    diag = {"file": str(path), "headers_found": len(sec), "sections": []}
    for substrate, pixel_id, rows in sec:
        comp, pos = pixel_to_comp_and_pos(pixel_id)
        # Build forward arrays from col0/col1; reverse arrays from col2/col3 if present
        fV, fI, rV, rI = [], [], [], []
        for nums in rows:
            if len(nums) >= 2:
                fV.append(nums[0]); fI.append(nums[1])
            if len(nums) >= 4:
                rV.append(nums[2]); rI.append(nums[3])
        nF = len(fV); nR = len(rV)
        diag["sections"].append({"substrate": substrate, "pixel_id": pixel_id, "rows": len(rows), "f_points": nF, "r_points": nR})
        if nF >= 2:
            fV = np.asarray(fV, dtype=float); fI = np.asarray(fI, dtype=float)
            Voc, Jsc_mAcm2, FF, PCE_pct = compute_metrics(fV, fI, area_cm2=area_cm2, light_mw_cm2=light_mw_cm2)
            sweeps.append(JVSweep(substrate, pixel_id, comp, pos, "forward", fV, fI, area_cm2, light_mw_cm2, Voc, Jsc_mAcm2, FF, PCE_pct))
        if nR >= 2:
            rV = np.asarray(rV, dtype=float); rI = np.asarray(rI, dtype=float)
            Voc, Jsc_mAcm2, FF, PCE_pct = compute_metrics(rV, rI, area_cm2=area_cm2, light_mw_cm2=light_mw_cm2)
            sweeps.append(JVSweep(substrate, pixel_id, comp, pos, "reverse", rV, rI, area_cm2, light_mw_cm2, Voc, Jsc_mAcm2, FF, PCE_pct))
    return sweeps, diag

class JVApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Perovskite JV Analyzer")
        self.geometry("1200x800")

        self.area_cm2 = tk.DoubleVar(value=0.04)
        self.light_mw_cm2 = tk.DoubleVar(value=100.0)
        self.include_forward = tk.BooleanVar(value=True)
        self.include_reverse = tk.BooleanVar(value=True)

        self.metric_choice = tk.StringVar(value="PCE_pct")

        self.data: List[JVSweep] = []
        self.df: Optional[pd.DataFrame] = None
        self.last_diagnostics: List[dict] = []

        self._build_ui()

    def _build_ui(self):
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        side = ttk.Frame(self, padding=10)
        side.grid(row=0, column=0, sticky="ns")

        ttk.Label(side, text="Data files").grid(row=0, column=0, sticky="w")
        ttk.Button(side, text="Load file", command=self.load_file).grid(row=1, column=0, sticky="ew", pady=2)
        ttk.Button(side, text="Load folder", command=self.load_folder).grid(row=2, column=0, sticky="ew", pady=2)

        ttk.Separator(side).grid(row=3, column=0, sticky="ew", pady=6)
        ttk.Label(side, text="Parameters").grid(row=4, column=0, sticky="w")
        pfrm = ttk.Frame(side)
        pfrm.grid(row=5, column=0, sticky="ew")
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
        metric_cb = ttk.Combobox(side, textvariable=self.metric_choice, values=["Voc", "Jsc_mAcm2", "FF", "PCE_pct"], state="readonly", width=12)
        metric_cb.grid(row=12, column=0, sticky="ew"); metric_cb.bind("<<ComboboxSelected>>", lambda e: self.refresh_plots())

        ttk.Separator(side).grid(row=13, column=0, sticky="ew", pady=6)
        ttk.Label(side, text="Filters").grid(row=14, column=0, sticky="w")
        self.substrate_cb = ttk.Combobox(side, values=[], state="readonly")
        self.substrate_cb.grid(row=15, column=0, sticky="ew", pady=2)
        self.substrate_cb.bind("<<ComboboxSelected>>", lambda e: self.update_comp_options())

        self.composition_cb = ttk.Combobox(side, values=[], state="readonly")
        self.composition_cb.grid(row=16, column=0, sticky="ew", pady=2)
        self.composition_cb.bind("<<ComboboxSelected>>", lambda e: self.refresh_plots())

        ttk.Separator(side).grid(row=17, column=0, sticky="ew", pady=6)
        ttk.Label(side, text="Plot actions").grid(row=18, column=0, sticky="w")
        ttk.Button(side, text="Boxplot (all groups)", command=self.plot_boxplot).grid(row=19, column=0, sticky="ew", pady=2)
        ttk.Button(side, text="Metric vs. position", command=self.plot_metric_vs_position).grid(row=20, column=0, sticky="ew", pady=2)

        ttk.Separator(side).grid(row=21, column=0, sticky="ew", pady=6)
        ttk.Label(side, text="JV Curves").grid(row=22, column=0, sticky="w")
        ttk.Button(side, text="(Placeholder) Plot JV for selection", command=self.placeholder_jv).grid(row=23, column=0, sticky="ew", pady=2)

        ttk.Separator(side).grid(row=24, column=0, sticky="ew", pady=6)
        ttk.Button(side, text="Show diagnostics", command=self.show_diagnostics).grid(row=25, column=0, sticky="ew", pady=2)

        main = ttk.Frame(self, padding=10)
        main.grid(row=0, column=1, sticky="nsew")
        main.rowconfigure(0, weight=1)
        main.columnconfigure(0, weight=1)
        self.fig = Figure(figsize=(8,6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=main)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

    def load_file(self):
        path = filedialog.askopenfilename(title="Select JV file", filetypes=[("Text", "*.txt *.dat *.csv"), ("All", "*.*")])
        if not path:
            return
        self._load_paths([Path(path)])

    def load_folder(self):
        d = filedialog.askdirectory(title="Select folder with JV files")
        if not d:
            return
        paths = [p for p in Path(d).glob("**/*") if p.suffix.lower() in (".txt", ".dat", ".csv")]
        if not paths:
            messagebox.showwarning("No files", "No .txt/.dat/.csv files found in that folder.")
            return
        self._load_paths(paths)

    def _load_paths(self, paths: List[Path]):
        area = self.area_cm2.get()
        pin = self.light_mw_cm2.get()
        all_sweeps: List[JVSweep] = []
        diagnostics: List[dict] = []
        for p in paths:
            try:
                sweeps, diag = build_sweeps_from_file(p, area_cm2=area, light_mw_cm2=pin)
                diagnostics.append(diag)
                all_sweeps.extend(sweeps)
            except Exception as e:
                diagnostics.append({"file": str(p), "error": str(e)})
        self.last_diagnostics = diagnostics
        if not all_sweeps:
            self.show_diagnostics()
            messagebox.showerror("No JV sweeps found", "No JV sweeps were parsed. See 'Show diagnostics' for header/row stats.\nIf headers look different than '3_10' lines, send me a short snippet.")
            return
        self.data = all_sweeps
        self.df = self._to_dataframe(all_sweeps)
        self._populate_filters()
        self.refresh_plots()

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
                "FF": s.FF,
                "PCE_pct": s.PCE_pct
            })
        return pd.DataFrame(rows)

    def _populate_filters(self):
        if self.df is None or self.df.empty:
            return
        subs = sorted(self.df["substrate"].dropna().unique().tolist())
        self.substrate_cb["values"] = subs
        if subs:
            self.substrate_cb.set(str(subs[0]))
            self.update_comp_options()

    def update_comp_options(self):
        if self.df is None or self.df.empty:
            return
        try:
            sub = int(self.substrate_cb.get())
        except Exception:
            return
        comps = sorted(self.df.loc[self.df["substrate"]==sub, "composition_index"].dropna().unique().tolist())
        self.composition_cb["values"] = comps
        if comps:
            self.composition_cb.set(str(comps[0]))
        self.refresh_plots()

    def recompute_metrics(self):
        if not self.data:
            messagebox.showwarning("No data", "Load files first.")
            return
        area = self.area_cm2.get()
        pin = self.light_mw_cm2.get()
        for s in self.data:
            s.area_cm2 = area
            s.light_mw_cm2 = pin
            Voc, Jsc_mAcm2, FF, PCE_pct = compute_metrics(s.voltage, s.current_A, area_cm2=area, light_mw_cm2=pin)
            s.Voc, s.Jsc_mAcm2, s.FF, s.PCE_pct = Voc, Jsc_mAcm2, FF, PCE_pct
        self.df = self._to_dataframe(self.data)
        self.refresh_plots()

    def _filtered_df(self) -> pd.DataFrame:
        if self.df is None:
            return pd.DataFrame()
        df = self.df.copy()
        keep_dirs = []
        if self.include_forward.get():
            keep_dirs.append("forward")
        if self.include_reverse.get():
            keep_dirs.append("reverse")
        if keep_dirs:
            df = df[df["direction"].isin(keep_dirs)]
        return df

    def refresh_plots(self):
        self.plot_boxplot()

    def plot_boxplot(self):
        df = self._filtered_df()
        if df.empty:
            self._clear_ax("No data. Load files first.")
            return
        metric = self.metric_choice.get()
        self.ax.clear()
        groups = []
        labels = []
        for (sub, comp, direction), g in df.groupby(["substrate", "composition_index", "direction"]):
            vals = g[metric].dropna().values
            if len(vals)==0:
                continue
            groups.append(vals)
            labels.append(f"S{sub}-C{comp}-{direction[0].upper()}")
        if not groups:
            self._clear_ax("No values for selected metric / filters.")
            return
        self.ax.boxplot(groups, showmeans=True, meanline=False)
        self.ax.set_xticks(range(1, len(labels)+1))
        self.ax.set_xticklabels(labels, rotation=45, ha="right")
        self.ax.set_ylabel(metric)
        self.ax.set_title(f"Boxplot: {metric} by Substrate/Composition (F/R filtered)")
        self.ax.grid(True, which="both", axis="y")
        self.canvas.draw_idle()

    def plot_metric_vs_position(self):
        df = self._filtered_df()
        if df.empty:
            self._clear_ax("No data. Load files first.")
            return
        try:
            sub = int(self.substrate_cb.get())
            comp = int(self.composition_cb.get())
        except Exception:
            self._clear_ax("Select a substrate and a composition.")
            return
        metric = self.metric_choice.get()
        sdf = df[(df["substrate"]==sub) & (df["composition_index"]==comp)]
        if sdf.empty:
            self._clear_ax("No data for selection.")
            return
        self.ax.clear()
        for direction, g in sdf.groupby("direction"):
            pos_means = g.groupby("position_in_composition")[metric].mean()
            pos_stds = g.groupby("position_in_composition")[metric].std()
            xs = sorted(pos_means.index.tolist())
            ys = [pos_means.loc[x] for x in xs]
            yerr = [pos_stds.loc[x] if not math.isnan(pos_stds.loc[x]) else 0.0 for x in xs]
            self.ax.errorbar(xs, ys, yerr=yerr, marker='o', linestyle='-', label=direction.capitalize())
        self.ax.set_xlabel("Position in composition (1 = thickest, near blade start)")
        self.ax.set_ylabel(metric)
        self.ax.set_title(f"{metric} vs. position | Substrate {sub}, Composition {comp}")
        self.ax.grid(True, which="both")
        self.ax.legend()
        self.canvas.draw_idle()

    def placeholder_jv(self):
        messagebox.showinfo("JV Curves", "JV curve plotting will be added next. Forward/Reverse toggles already exist.")

    def show_diagnostics(self):
        top = tk.Toplevel(self)
        top.title("Parsing diagnostics")
        txt = tk.Text(top, width=100, height=30)
        txt.pack(fill="both", expand=True)
        if not self.last_diagnostics:
            txt.insert("end", "No diagnostics captured yet. Load a file or folder first.")
            return
        for d in self.last_diagnostics:
            if "error" in d:
                txt.insert("end", f"FILE: {d.get('file')}\n  ERROR: {d.get('error')}\n\n")
                continue
            txt.insert("end", f"FILE: {d.get('file')}\n  headers_found: {d.get('headers_found')}\n")
            for s in d.get("sections", [])[:10]:
                txt.insert("end", f"    section S{s['substrate']}_{s['pixel_id']}: rows={s['rows']}, forward_points={s['f_points']}, reverse_points={s['r_points']}\n")
            if d.get("sections") and len(d["sections"]) > 10:
                txt.insert("end", f"    ... ({len(d['sections'])-10} more)\n")
            txt.insert("end", "\n")

    def _clear_ax(self, msg: str):
        self.ax.clear()
        self.ax.text(0.5, 0.5, msg, ha="center", va="center", transform=self.ax.transAxes)
        self.ax.axis("off")
        self.canvas.draw_idle()

if __name__ == "__main__":
    app = JVApp()
    app.mainloop()
