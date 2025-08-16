# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python-based IV (Current-Voltage) Data Analyzer application built with Tkinter for analyzing photovoltaic cell measurements. The application provides a GUI for loading, managing, and visualizing IV measurement data from .iv files.

## Architecture

### Main Application Structure
- **IVDataAnalyzer class**: Main application class that manages the entire GUI and data flow
- **Three-tab interface**:
  - Data Management: File loading, measurement data display, and condition assignment
  - Plotting: Box plots and strip plots for parameter visualization
  - IV Plot: Individual IV curve visualization

### Core Data Storage
- `measurements_data`: Pandas DataFrame storing parsed IV measurement data
- `conditions_data`: Pandas DataFrame mapping substrate IDs to experimental conditions
- Uses Treeview widgets for tabular data display with sorting capabilities

### Key Dependencies
- `tkinter` + `ttk`: GUI framework
- `pandas`: Data manipulation and storage
- `matplotlib`: Plotting and visualization
- `seaborn`: Statistical plotting
- `re`: Regular expression parsing for .iv files
- `json`: Condition data persistence

## Common Development Tasks

### Running the Application
```bash
python IVapp_v5.py
```

### File Format Support
The application processes `.iv` files with specific format requirements:

#### File Structure
- **Header sections**: Measurement Information, Measurement options, Config sourcemeter, Config voltmeter
- **Key identifiers**: 
  - `Deposition ID:` field for substrate identification (line 5 in example)
  - Filename pattern `*_c1.iv`, `*_c2.iv`, `*_c3.iv` for pixel identification (A/B/C respectively)
- **Analysis outputs section**: `% ANALYSIS OUTPUT` containing extracted parameters
- **Data section**: `% MEASURED IV FRLOOP DATA` with tab-separated voltage/current measurements

#### Required Fields
- `Deposition ID:` - Used for substrate identification and grouping
- Analysis parameters with scientific notation:
  - `Voc [V]:` - Open circuit voltage
  - `Jsc [A/m2]:` - Short circuit current density  
  - `FF [.]:` - Fill factor
  - `Efficiency [.]` - Power conversion efficiency
  - `Pmpp [W/m2]:` - Maximum power density
  - `Vmpp [V]:` - Voltage at maximum power
  - `Jmpp [A]:` - Current density at maximum power
  - `Roc [Ohm.m2]:` - Open circuit resistance
  - `Rsc [Ohm.m2]:` - Short circuit resistance

#### Data Format
- Tab-separated values in data section
- Four columns: `V (measured) [V]`, `I (measured) [A]`, `V (corrected) [V]`, `J (corrected) [A/m2]`
- Scientific notation format (e.g., `1.691674E-1`)
- Voltage sweep typically from negative to positive values

### Key Functions and Locations
- File parsing: `parse_iv_file()` method at IVapp_v5.py:325
- Data extraction: `extract_value()` helper at IVapp_v5.py:388
- Plot generation: `generate_plot()` at IVapp_v5.py:562
- IV curve plotting: `plot_iv_curve()` at IVapp_v5.py:710
- Condition management: `assign_condition()` at IVapp_v5.py:445

### Data Management Features
- Automatic substrate ID extraction from file content
- Pixel identification from filename patterns
- Scan direction detection (forward/reverse) from voltage progression
- Condition assignment with display order management
- Save/load conditions as JSON files

### Visualization Capabilities
- Box plots with strip plots overlay for statistical analysis
- Multiple color palette support (viridis, magma, plasma, etc.)
- Customizable y-axis limits
- Individual IV curve plotting
- Plot export functionality (PNG, JPEG, PDF, SVG)

## File Structure Notes

This is a single-file application with no external configuration files or dependencies beyond standard Python packages. The application stores temporary state in memory and provides JSON export/import for condition data persistence.