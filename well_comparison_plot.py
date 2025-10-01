#!/usr/bin/env python3
"""
Simple Excel Comparison Script for Well Data

Compares parameters (Jsc, FF, PCE/Efficiency, Voc) between Malibu and CellTester data
by averaging values for each well and plotting them.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def load_excel_files():
    """Load the Excel files from example data folder"""

    script_dir = os.path.dirname(os.path.abspath(__file__))
    example_data_path = os.path.join(script_dir, "pixelreader", "example data files")

    malibu_path = os.path.join(example_data_path, "malibu.xlsx")
    celltester_path = os.path.join(example_data_path, "celltester.csv")

    # Load files
    malibu_data = pd.read_excel(malibu_path)
    celltester_data = pd.read_csv(celltester_path)

    print(f"Loaded Malibu data: {len(malibu_data)} rows")
    print(f"Loaded CellTester data: {len(celltester_data)} rows")

    print(f"\nMalibu columns: {list(malibu_data.columns)}")
    print(f"CellTester columns: {list(celltester_data.columns)}")

    return malibu_data, celltester_data

def find_well_column(data, data_name):
    """Find the well column in the dataset"""

    possible_names = ['well', 'Well', 'WELL', 'Well ID', 'well_id', 'Condition', 'condition']

    for col in data.columns:
        if col in possible_names:
            print(f"Found well column '{col}' in {data_name}")
            return col

    # If exact match not found, look for columns containing 'well' or 'substrate'
    for col in data.columns:
        if 'well' in col.lower() or 'substrate' in col.lower():
            print(f"Found well-like column '{col}' in {data_name}")
            return col

    print(f"Available columns in {data_name}: {list(data.columns)}")
    raise ValueError(f"Could not find well column in {data_name}")

def find_parameter_columns(data, data_name):
    """Find parameter columns (Jsc, FF, PCE/Efficiency, Voc) in the dataset"""

    parameters = {}

    # Define parameter mappings (case insensitive)
    param_mappings = {
        'Jsc': ['jsc', 'j_sc', 'current_density', 'short_circuit_current', 'jsc_macm2'],
        'FF': ['ff', 'fill_factor', 'fillfactor', 'ff_pct'],
        'PCE': ['pce', 'efficiency', 'eff', 'power_conversion_efficiency', 'pce_pct'],
        'Voc': ['voc', 'v_oc', 'open_circuit_voltage', 'voltage']
    }

    for param_name, possible_names in param_mappings.items():
        found = False
        for col in data.columns:
            col_lower = col.lower().replace(' ', '_').replace('[', '').replace(']', '')
            for possible in possible_names:
                if possible in col_lower:
                    parameters[param_name] = col
                    print(f"Found {param_name} as '{col}' in {data_name}")
                    found = True
                    break
            if found:
                break

        if not found:
            print(f"Warning: {param_name} not found in {data_name}")

    return parameters

def average_by_well(data, well_col, param_cols, data_name):
    """Average parameter values by well"""

    # Only keep rows with valid well names
    data_clean = data.dropna(subset=[well_col])

    # Select only well column and parameter columns that exist
    cols_to_keep = [well_col] + [col for col in param_cols.values() if col in data.columns]
    data_subset = data_clean[cols_to_keep]

    # Group by well and calculate mean
    averaged_data = data_subset.groupby(well_col).mean().reset_index()

    print(f"{data_name} - Wells found: {list(averaged_data[well_col])}")
    print(f"{data_name} - Averaged data shape: {averaged_data.shape}")

    return averaged_data

def create_comparison_plots(malibu_avg, celltester_avg, malibu_well_col, celltester_well_col,
                          malibu_params, celltester_params):
    """Create comparison plots for each parameter"""

    # Find common wells
    common_wells = set(malibu_avg[malibu_well_col]) & set(celltester_avg[celltester_well_col])
    common_wells = sorted(list(common_wells))

    print(f"Common wells: {common_wells}")

    if not common_wells:
        print("No common wells found!")
        print(f"Malibu wells: {sorted(list(malibu_avg[malibu_well_col]))}")
        print(f"CellTester wells: {sorted(list(celltester_avg[celltester_well_col]))}")
        return

    # Filter data to common wells only
    malibu_common = malibu_avg[malibu_avg[malibu_well_col].isin(common_wells)]
    celltester_common = celltester_avg[celltester_avg[celltester_well_col].isin(common_wells)]

    # Find common parameters
    common_params = set(malibu_params.keys()) & set(celltester_params.keys())
    common_params = sorted(list(common_params))

    print(f"Common parameters: {common_params}")

    if not common_params:
        print("No common parameters found!")
        return

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Malibu vs CellTester Comparison by Well', fontsize=16, fontweight='bold')

    axes = axes.flatten()

    for i, param in enumerate(common_params[:4]):  # Limit to 4 parameters
        ax = axes[i]

        # Get parameter column names
        malibu_col = malibu_params[param]
        celltester_col = celltester_params[param]

        # Prepare data for plotting
        wells_for_plot = []
        malibu_values = []
        celltester_values = []

        for well in common_wells:
            malibu_row = malibu_common[malibu_common[malibu_well_col] == well]
            celltester_row = celltester_common[celltester_common[celltester_well_col] == well]

            if not malibu_row.empty and not celltester_row.empty:
                wells_for_plot.append(well)
                malibu_values.append(malibu_row[malibu_col].iloc[0])
                celltester_values.append(celltester_row[celltester_col].iloc[0])

        if wells_for_plot:
            x_pos = np.arange(len(wells_for_plot))

            # Create bar plot
            width = 0.35
            ax.bar(x_pos - width/2, malibu_values, width, label='Malibu', alpha=0.8, color='blue')
            ax.bar(x_pos + width/2, celltester_values, width, label='CellTester', alpha=0.8, color='red')

            ax.set_xlabel('Well')
            ax.set_ylabel(param)
            ax.set_title(f'{param} Comparison')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(wells_for_plot, rotation=45)
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, f'No data for {param}', transform=ax.transAxes, ha='center', va='center')

    # Hide any unused subplots
    for i in range(len(common_params), 4):
        axes[i].set_visible(False)

    plt.tight_layout()

    # Save plot
    output_filename = "well_comparison_plot.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved as: {output_filename}")

    plt.show()

    # Print summary statistics
    print("\n=== Summary Statistics ===")
    for param in common_params:
        malibu_col = malibu_params[param]
        celltester_col = celltester_params[param]

        malibu_vals = [malibu_common[malibu_common[malibu_well_col] == well][malibu_col].iloc[0]
                      for well in common_wells
                      if not malibu_common[malibu_common[malibu_well_col] == well].empty]

        celltester_vals = [celltester_common[celltester_common[celltester_well_col] == well][celltester_col].iloc[0]
                          for well in common_wells
                          if not celltester_common[celltester_common[celltester_well_col] == well].empty]

        if malibu_vals and celltester_vals:
            print(f"\n{param}:")
            print(f"  Malibu   - Mean: {np.mean(malibu_vals):.3f}, Std: {np.std(malibu_vals):.3f}")
            print(f"  CellTester - Mean: {np.mean(celltester_vals):.3f}, Std: {np.std(celltester_vals):.3f}")

def main():
    """Main function"""

    try:
        # Load Excel files
        print("Loading Excel files...")
        malibu_data, celltester_data = load_excel_files()

        # Find well columns
        malibu_well_col = find_well_column(malibu_data, "Malibu")
        celltester_well_col = find_well_column(celltester_data, "CellTester")

        # Find parameter columns
        malibu_params = find_parameter_columns(malibu_data, "Malibu")
        celltester_params = find_parameter_columns(celltester_data, "CellTester")

        # Average data by well
        print("\nAveraging data by well...")
        malibu_avg = average_by_well(malibu_data, malibu_well_col, malibu_params, "Malibu")
        celltester_avg = average_by_well(celltester_data, celltester_well_col, celltester_params, "CellTester")

        # Create comparison plots
        print("\nCreating comparison plots...")
        create_comparison_plots(malibu_avg, celltester_avg, malibu_well_col, celltester_well_col,
                              malibu_params, celltester_params)

        print("\nAnalysis complete!")

    except Exception as e:
        print(f"Error: {str(e)}")
        print("Please check that the Excel files exist and contain the expected columns.")

if __name__ == "__main__":
    main()