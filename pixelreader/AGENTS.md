# AGENTS.md

This file provides guidance to AI agents when working with code in this directory.

## Project Overview

This is a specialized Python-based JV (Current-Voltage) data analyzer application for perovskite solar cell measurements. The `IVapp_66PIXEL.py` application is designed to handle multi-section data files containing measurements from multiple pixel positions on a single substrate while pairing each sweep with experimental metadata.

## Architecture

### Main Application Structure
- **JVApp class**: Main Tkinter GUI application managing the full interface, data flow, and visualization tabs.
- **JVSweep dataclass**: Core data structure representing individual J-V measurements and their experimental context.
- **Parsing system**: Regex-based section parsing with fallbacks for flexible data format handling.
- **Experimental conditions pipeline**: Optional Excel ingestion layer that enriches sweeps with sweep IDs and condition descriptions for the Sweep Analysis tab.

### Core Data Storage
- `data`: List of JVSweep objects containing all measurement data.
- `df`: Pandas DataFrame derived from sweep data for analysis and plotting.
- `df_with_flags`: Extended DataFrame with filtering flags, sweep metadata, and display columns.
- `sweep_analysis`: Cached per-sweep summary produced from experimental conditions to drive advanced plots.
- Uses Treeview widgets for tabular data display with interactive selection.

### Key Data Structure (JVSweep)
```python
@dataclass
class JVSweep:
    substrate: int              # Substrate identifier
    pixel_id: int               # Unique pixel identifier (derived from composition/position)
    composition_index: int      # Composition number (1-11)
    position_in_composition: int  # Position within composition (1-6)
    direction: str              # "forward" or "reverse" sweep direction
    voltage: np.ndarray         # Voltage measurements
    current_A: np.ndarray       # Current measurements in amperes
    area_cm2: float             # Device area in cm^2
    light_mw_cm2: float         # Illumination power density in mW/cm^2
    Voc: Optional[float] = None        # Open circuit voltage
    Jsc_mAcm2: Optional[float] = None  # Short circuit current density (mA/cm^2)
    FF_pct: Optional[float] = None     # Fill factor (percentage)
    PCE_pct: Optional[float] = None    # Power conversion efficiency (percentage)
    sweep_id: Optional[int] = None            # Linked experimental sweep identifier
    condition_name: Optional[str] = None      # Human-readable experimental condition label
```

### Key Dependencies
- `tkinter` + `ttk`: GUI framework.
- `pandas`: Data manipulation, Excel ingestion, and aggregations.
- `numpy`: Numerical computations and array handling.
- `matplotlib`: Plotting and visualization (2D and 3D sweeps).
- `re`: Regular expression parsing for flexible data format handling.
- `pathlib`: Modern file path handling.
- `openpyxl`: Required by pandas to read `.xlsx` sheets (install if Excel loading fails).

## File Format Support

### Data Structure
- **Multi-section format**: Each file contains multiple measurement sections.
- **Header pattern matching**: Configurable regex pattern for section identification.
- **Default pattern**: `r"^\s*(\d+)\s*[_-]\s*(\d+)\s*.*$"` matching "POSITION_COMPOSITION" headers.
- **Flexible parsing**: Multiple fallback patterns for robust data extraction.

### Section Format
- Each section represents one pixel measurement.
- Position and composition indices extracted from section headers.
- Data rows contain numerical measurements (voltage/current pairs).
- Support for both forward-only and forward+reverse sweep data.

### Data Columns
- Forward sweep: `[FwdV, FwdI]`.
- Forward + Reverse: `[FwdV, FwdI, RevV, RevI]`.
- Scientific notation supported.
- Automatic sweep direction detection.

## Key Functions and Locations

### File Parsing
- `parse_sections_from_text()` at `pixelreader/IVapp_66PIXEL.py:72` – main parsing function.
- `build_sweeps_from_file()` at `pixelreader/IVapp_66PIXEL.py:127` – converts parsed sections into JVSweep objects.
- `_substrate_from_filename()` at `pixelreader/IVapp_66PIXEL.py:44` – extracts substrate ID from filenames.

### Experimental Conditions & Sweep Metadata
- `load_experimental_conditions()` at `pixelreader/IVapp_66PIXEL.py:239` – loads ROSIE/Runsheet Excel data.
- `analyze_sweep_parameters()` at `pixelreader/IVapp_66PIXEL.py:409` – inspects experimental parameters to classify sweeps.
- `map_sweeps_to_conditions()` at `pixelreader/IVapp_66PIXEL.py:489` – attaches sweep IDs and condition labels to JVSweep objects.

### Metrics Calculation
- `_interp_x_at_y_zero()` at `pixelreader/IVapp_66PIXEL.py:159` – Voc calculation helper.
- `_interp_y_at_x_zero()` at `pixelreader/IVapp_66PIXEL.py:171` – Jsc calculation helper.
- `compute_metrics()` at `pixelreader/IVapp_66PIXEL.py:183` – primary metrics computation.

### Data Management
- `_to_dataframe()` at `pixelreader/IVapp_66PIXEL.py:899` – converts sweeps to pandas DataFrame.
- `apply_thresholds()` at `pixelreader/IVapp_66PIXEL.py:926` – threshold-based filtering handler.
- `_filtered_df()` at `pixelreader/IVapp_66PIXEL.py:1177` – applies filters and toggles before plotting/export.

### Visualization
- `plot_boxplot_groups()` at `pixelreader/IVapp_66PIXEL.py:1216` – statistical box plots for compositions/groups.
- `plot_heatmap()` at `pixelreader/IVapp_66PIXEL.py:1265` – substrate vs. composition heatmaps.
- `plot_substrate_pixel_map()` at `pixelreader/IVapp_66PIXEL.py:1307` – per-substrate pixel maps.
- Sweep plots are generated via `refresh_sweep_plots()` at `pixelreader/IVapp_66PIXEL.py:1466` with helpers `_plot_sweep_boxplot()`, `_plot_sweep_scatter()`, `_plot_sweep_bar()`, and parameter analysis methods.

## Advanced Features

### Flexible Data Parsing
- Configurable regex patterns for header matching.
- Automatic fallback strategies during section parsing.
- Parse diagnostics stored per-file for troubleshooting.

### Composition Grouping
- **11 compositions mode**: Direct composition indices (1-11).
- **9 groups mode**: Grouped via `comp_to_group()` at `pixelreader/IVapp_66PIXEL.py:233`.
- Pixel mapping formula: `pixel_id = (comp - 1) * 6 + pos`.

### Interactive Filtering
- Threshold filters for Voc, Jsc, FF, PCE.
- Sweep direction toggles (forward/reverse inclusion).
- Substrate selection plus combined substrate view.
- Manual exclusion dialog for removing sweeps from analysis.

### Visualization Modes
- Composition tab: box plots, heatmaps, pixel maps, and export functions.
- Sweep tab: box, scatter, bar, 2D parameter, and 3D parameter plots with automatic selection of plot type per sweep.
- Plot export to PNG/SVG/PDF and sweep data export to CSV.

### Experimental Conditions Integration
- Optional Excel ingestion enriches sweeps with `sweep_id` and human-readable `condition_name`.
- Supports both ROSIE sheets (columns such as `Substrate`, `Sweep`, `excess PbI2`, `with Thiourea`, `with FABF4`) and Runsheet-style sheets (columns ending with `(M)` for molarity).
- Provides console diagnostics when files or sheets are missing.

## Experimental Conditions Workflow
1. Place `experiment sheets.xlsx` alongside `IVapp_66PIXEL.py` or browse to an alternate path via the GUI.
2. Ensure the workbook contains either a `ROSIE` sheet (preferred) or a `Runsheet` sheet with `Substrate` and `Sweep` columns plus parameter columns described above.
3. When loaded, `load_experimental_conditions()` prints available sheets, row counts, and errors to the console for quick debugging.
4. `analyze_sweep_parameters()` inspects the resulting data to classify sweeps, detect varying parameters, and populate `sweep_analysis` for the Sweep tab.
5. `map_sweeps_to_conditions()` injects sweep IDs and descriptive labels into each JVSweep, enabling filtering, summaries, and exports.
6. If no Excel file is found or parsing fails, the application continues with core composition analysis while disabling sweep-specific tooling.

## Sweep Analysis Tab
- Accessible via the second tab in the notebook (`self.sweep_frame`).
- Left panel controls: sweep selection, metric choice, direction combination toggle, plot refresh, and metadata display (`update_sweep_info_display()`).
- Right panel: Matplotlib canvas for sweep plots plus a Treeview summarizing sweeps by ID, substrate count, and condition text.
- Plot helpers automatically choose box/scatter/bar views per sweep and can pivot into parameter-vs-performance (2D/3D) analyses when experimental variables are available.
- Export buttons call `export_sweep_data()` and `save_sweep_plot()` to persist filtered sweep tables and figures.

## Configuration Parameters

### Measurement Settings
- `area_cm2`: Device active area (default: 0.04 cm^2).
- `light_mw_cm2`: Illumination intensity (default: 100 mW/cm^2).

### Analysis Options
- `combine_substrates`: Merge data across substrates.
- `combine_fr`: Combine forward/reverse sweep data.
- `expand_substrate_axis`: Separate substrate data in plots.
- `grouping_mode`: Toggle between composition and group modes.

## Common Development Tasks

### Running the Application
```bash
python pixelreader/IVapp_66PIXEL.py
```

### Custom Header Patterns
The application supports custom regex patterns for section identification. Examples:
- Default: `r"^\s*(\d+)\s*[_-]\s*(\d+)\s*.*$"`
- Whitespace separated: `r"^\s*(\d{1,2})\s+(\d{1,2})\s*.*$"`
- Permissive underscores/dashes: `r"^\s*(\d{1,2})\s*[_-]\s*(\d{1,2})\s*.*$"`

### Substrate Identification
Substrates are identified from filenames using patterns:
- `[Ss]ub\s*_?\s*(\d+)`
- `[Ss]ubstrate\s*_?\s*(\d+)`

## File Structure Notes

This is a single-file application optimized for perovskite solar cell array analysis. The flexible parsing system accommodates various data formats from different measurement systems, while the Sweep Analysis tab layers in experimental metadata to correlate processing parameters with device performance.
