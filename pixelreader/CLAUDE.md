# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a specialized Python-based JV (Current-Voltage) Data Analyzer application for perovskite solar cell measurements from a different measurement system format than the main IVapp_v5.py application. The 66PIXEL.py application is designed to handle multi-section data files containing measurements from multiple pixel positions and compositions.

## Architecture

### Main Application Structure
- **JVApp class**: Main Tkinter GUI application managing the entire interface and data flow
- **JVSweep dataclass**: Core data structure representing individual J-V measurements
- **Parsing system**: Regex-based section parsing for flexible data format handling

### Core Data Storage
- `data`: List of JVSweep objects containing all measurement data
- `df`: Pandas DataFrame derived from sweep data for analysis
- `df_with_flags`: Extended DataFrame with filtering flags and metadata
- Uses Treeview widget for tabular data display with interactive selection

### Key Data Structure (JVSweep)
```python
@dataclass
class JVSweep:
    substrate: int              # Substrate identifier
    pixel_id: int              # Unique pixel identifier (derived from composition/position)
    composition_index: int      # Composition number (1-11)
    position_in_composition: int # Position within composition (1-6)
    direction: str             # "forward" or "reverse" sweep direction
    voltage: np.ndarray        # Voltage measurements
    current_A: np.ndarray      # Current measurements in Amperes
    area_cm2: float           # Device area in cm²
    light_mw_cm2: float       # Illumination power density
    # Calculated metrics:
    Voc: Optional[float]       # Open circuit voltage
    Jsc_mAcm2: Optional[float] # Short circuit current density (mA/cm²)
    FF_pct: Optional[float]    # Fill factor (percentage)
    PCE_pct: Optional[float]   # Power conversion efficiency (percentage)
```

### Key Dependencies
- `tkinter` + `ttk`: GUI framework
- `pandas`: Data manipulation and analysis
- `numpy`: Numerical computations and array handling
- `matplotlib`: Plotting and visualization
- `re`: Regular expression parsing for flexible data format handling
- `pathlib`: Modern file path handling

## File Format Support

### Data Structure
- **Multi-section format**: Each file contains multiple measurement sections
- **Header pattern matching**: Configurable regex pattern for section identification
- **Default pattern**: `r"^\s*(\d+)\s*[_-]\s*(\d+)\s*.*$"` matching "POSITION_COMPOSITION" format
- **Flexible parsing**: Multiple fallback patterns for robust data extraction

### Section Format
- Each section represents one pixel measurement
- Position and composition indices extracted from section headers
- Data rows contain numerical measurements (voltage/current pairs)
- Support for both forward-only and forward+reverse sweep data

### Data Columns
- Forward sweep: `[FwdV, FwdI]`
- Forward+Reverse: `[FwdV, FwdI, RevV, RevI]`
- Scientific notation support
- Automatic sweep direction detection

## Key Functions and Locations

### File Parsing
- `parse_sections_from_text()`: Main parsing function at 66PIXEL.py:67
- `build_sweeps_from_file()`: File processing at 66PIXEL.py:122
- `_substrate_from_filename()`: Substrate ID extraction at 66PIXEL.py:39

### Metrics Calculation  
- `compute_metrics()`: Primary metrics computation at 66PIXEL.py:178
- `_interp_x_at_y_zero()`: Voc calculation helper at 66PIXEL.py:154
- `_interp_y_at_x_zero()`: Jsc calculation helper at 66PIXEL.py:166

### Data Management
- `_to_dataframe()`: Convert sweeps to pandas DataFrame at 66PIXEL.py:436
- `_filtered_df()`: Apply filters and toggles at 66PIXEL.py:653
- `apply_thresholds()`: Threshold-based filtering at 66PIXEL.py:461

### Visualization
- `plot_boxplot_groups()`: Statistical box plots at 66PIXEL.py:687
- `plot_heatmap()`: Substrate × composition heatmaps at 66PIXEL.py:736
- `plot_substrate_pixel_map()`: Individual substrate pixel maps at 66PIXEL.py:778

## Advanced Features

### Flexible Data Parsing
- **Configurable regex patterns**: User-adjustable header matching
- **Automatic fallbacks**: Multiple parsing strategies for robustness
- **Parse diagnostics**: Detailed parsing reports for troubleshooting

### Composition Grouping
- **11 compositions mode**: Direct composition indices (1-11)
- **9 groups mode**: Mapped grouping using `comp_to_group()` function at 66PIXEL.py:228
- **Pixel mapping**: `pixel_id = (comp - 1) * 6 + pos` formula

### Interactive Filtering
- **Threshold filters**: Min/max values for Voc, Jsc, FF, PCE
- **Sweep direction toggles**: Include/exclude forward/reverse sweeps
- **Substrate selection**: Individual or combined substrate analysis
- **Manual exclusion**: Interactive row-by-row data exclusion

### Visualization Modes
- **Box plots**: Statistical distribution analysis with customizable grouping
- **Heatmaps**: 2D visualization of substrate × composition performance
- **Pixel maps**: Detailed per-pixel analysis for individual substrates
- **Export functionality**: Save plots as PNG, SVG, or PDF

### Data Export
- **CSV export**: Full dataset with filtering flags
- **Plot export**: High-resolution image output
- **Parse reports**: Detailed diagnostics for troubleshooting

## Configuration Parameters

### Measurement Settings
- `area_cm2`: Device active area (default: 0.04 cm²)
- `light_mw_cm2`: Illumination intensity (default: 100 mW/cm²)

### Analysis Options
- `combine_substrates`: Merge data across substrates
- `combine_fr`: Combine forward/reverse sweep data
- `expand_substrate_axis`: Separate substrate data in plots
- `grouping_mode`: Toggle between composition and group modes

## Common Development Tasks

### Running the Application
```bash
python pixelreader/66PIXEL.py
```

### Custom Header Patterns
The application supports custom regex patterns for section identification. Examples:
- Default: `r"^\s*(\d+)\s*[_-]\s*(\d+)\s*.*$"`
- Whitespace: `r"^\s*(\d{1,2})\s+(\d{1,2})\s*.*$"`
- Permissive: `r"^\s*(\d{1,2})\s*[_-]\s*(\d{1,2})\s*.*$"`

### Substrate Identification
Substrates are identified from filenames using patterns:
- `[Ss]ub\s*_?\s*(\d+)`
- `[Ss]ubstrate\s*_?\s*(\d+)`

## File Structure Notes

This is a single-file application optimized for perovskite solar cell array analysis. Unlike the main IV analyzer, this application handles multi-pixel, multi-composition datasets with sophisticated grouping and visualization capabilities. The flexible parsing system accommodates various data formats from different measurement systems.