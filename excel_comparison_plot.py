#!/usr/bin/env python3
"""
Excel Comparison Plotting Script

This script compares parameters between two Excel files (Malibu and CellTester data)
by well number/substrate and creates comparison plots.

Usage:
    python excel_comparison_plot.py

The script will automatically look for:
- malibu.xlsx (performance data)
- experiment sheets.xlsx (experimental conditions data)

In the pixelreader/example data files/ directory.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_data():
    """Load the Excel data files"""

    # Define paths to example files
    script_dir = os.path.dirname(os.path.abspath(__file__))
    example_data_path = os.path.join(script_dir, "pixelreader", "example data files")

    malibu_path = os.path.join(example_data_path, "malibu.xlsx")
    experiment_path = os.path.join(example_data_path, "experiment sheets.xlsx")

    # Load Malibu data
    if not os.path.exists(malibu_path):
        raise FileNotFoundError(f"Malibu file not found at: {malibu_path}")

    malibu_data = pd.read_excel(malibu_path)
    print(f"Loaded Malibu data: {len(malibu_data)} rows")
    print(f"Malibu columns: {list(malibu_data.columns)}")

    # Load Experiment Sheets data (try ROSIE sheet first)
    if not os.path.exists(experiment_path):
        raise FileNotFoundError(f"Experiment sheets file not found at: {experiment_path}")

    try:
        # Try to load the ROSIE sheet specifically
        celltester_data = pd.read_excel(experiment_path, sheet_name='ROSIE')
        print(f"Loaded CellTester data from ROSIE sheet: {len(celltester_data)} rows")
    except:
        # If ROSIE doesn't exist, try first sheet
        celltester_data = pd.read_excel(experiment_path)
        print(f"Loaded CellTester data from first sheet: {len(celltester_data)} rows")

    print(f"CellTester columns: {list(celltester_data.columns)}")

    return malibu_data, celltester_data

def process_malibu_data(malibu_data, parameter):
    """Process Malibu data to extract means by substrate"""

    if parameter not in malibu_data.columns:
        raise ValueError(f"Parameter '{parameter}' not found in Malibu data")

    # Map substrate IDs to well numbers
    substrate_well_map = {
        'Sub4': 'Substrate 4',
        'Sub5': 'Substrate 5',
        'Sub6': 'Substrate 6'
    }

    # Create a copy and map substrate IDs
    malibu_copy = malibu_data.copy()
    malibu_copy['Well'] = malibu_copy['Substrate ID'].map(substrate_well_map)
    malibu_copy = malibu_copy.dropna(subset=['Well'])  # Remove unmapped substrates

    # Calculate means by well
    malibu_means = malibu_copy.groupby('Well')[parameter].agg(['mean', 'std', 'count']).reset_index()
    malibu_means.columns = ['Well', f'Malibu_{parameter}_mean', f'Malibu_{parameter}_std', f'Malibu_{parameter}_count']

    return malibu_means

def process_celltester_data(celltester_data, parameter):
    """Process CellTester data to extract values by substrate"""

    # Look for parameter in celltester data - try exact match first
    if parameter in celltester_data.columns:
        param_col = parameter
    else:
        # Try to find similar parameter names
        possible_matches = [col for col in celltester_data.columns
                          if parameter.lower().replace(' ', '').replace('[', '').replace(']', '') in
                          col.lower().replace(' ', '').replace('[', '').replace(']', '')]

        if not possible_matches:
            raise ValueError(f"Parameter '{parameter}' not found in CellTester data")

        param_col = possible_matches[0]
        print(f"Using '{param_col}' from CellTester data to match '{parameter}'")

    # Map substrate numbers to well names
    substrate_well_map = {
        4: 'Substrate 4',
        5: 'Substrate 5',
        6: 'Substrate 6'
    }

    # Find substrate column
    substrate_col = None
    for col in celltester_data.columns:
        if 'substrate' in col.lower():
            substrate_col = col
            break

    if substrate_col is None:
        raise ValueError("Could not find substrate column in CellTester data")

    # Create a copy and map substrate numbers
    celltester_copy = celltester_data.copy()
    celltester_copy['Well'] = celltester_copy[substrate_col].map(substrate_well_map)
    celltester_copy = celltester_copy.dropna(subset=['Well'])  # Remove unmapped substrates

    # Calculate means by well
    celltester_means = celltester_copy.groupby('Well')[param_col].agg(['mean', 'std', 'count']).reset_index()
    celltester_means.columns = ['Well', f'CellTester_{parameter}_mean', f'CellTester_{parameter}_std', f'CellTester_{parameter}_count']

    return celltester_means

def prepare_comparison_data(malibu_data, celltester_data, parameter):
    """Prepare data for comparison by finding common wells and calculating means"""

    # Process both datasets
    malibu_processed = process_malibu_data(malibu_data, parameter)
    celltester_processed = process_celltester_data(celltester_data, parameter)

    # Merge on well identifiers
    plot_data = pd.merge(malibu_processed, celltester_processed, on='Well', how='inner')

    return plot_data

def create_comparison_plots(plot_data, parameter):
    """Create comparison plots between Malibu and CellTester data"""

    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Malibu vs CellTester Comparison: {parameter}', fontsize=16, fontweight='bold')

    # Prepare data for plotting
    malibu_col = f'Malibu_{parameter}_mean'
    celltester_col = f'CellTester_{parameter}_mean'

    # 1. Scatter plot with 1:1 line
    ax1 = axes[0, 0]
    ax1.scatter(plot_data[malibu_col], plot_data[celltester_col],
               alpha=0.7, s=100, edgecolors='black', linewidth=1)

    # Add well labels
    for idx, row in plot_data.iterrows():
        ax1.annotate(row['Well'], (row[malibu_col], row[celltester_col]),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)

    # Add 1:1 line
    min_val = min(plot_data[malibu_col].min(), plot_data[celltester_col].min())
    max_val = max(plot_data[malibu_col].max(), plot_data[celltester_col].max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='1:1 line')

    ax1.set_xlabel(f'Malibu {parameter}', fontsize=12)
    ax1.set_ylabel(f'CellTester {parameter}', fontsize=12)
    ax1.set_title('Scatter Plot Comparison', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 2. Bar plot comparison
    ax2 = axes[0, 1]
    wells = plot_data['Well'].tolist()
    x_pos = np.arange(len(wells))

    width = 0.35
    ax2.bar(x_pos - width/2, plot_data[malibu_col], width,
           label='Malibu', alpha=0.8, edgecolor='black')
    ax2.bar(x_pos + width/2, plot_data[celltester_col], width,
           label='CellTester', alpha=0.8, edgecolor='black')

    ax2.set_xlabel('Well Number', fontsize=12)
    ax2.set_ylabel(parameter, fontsize=12)
    ax2.set_title('Bar Plot Comparison', fontsize=12, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(wells, rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # 3. Line plot comparison
    ax3 = axes[1, 0]
    x_pos = range(len(wells))

    ax3.plot(x_pos, plot_data[malibu_col], 'o-', label='Malibu',
            linewidth=2, markersize=8, markeredgecolor='black')
    ax3.plot(x_pos, plot_data[celltester_col], 's-', label='CellTester',
            linewidth=2, markersize=8, markeredgecolor='black')

    ax3.set_xlabel('Well Number', fontsize=12)
    ax3.set_ylabel(parameter, fontsize=12)
    ax3.set_title('Line Plot Comparison', fontsize=12, fontweight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(wells, rotation=45)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Difference plot
    ax4 = axes[1, 1]
    differences = plot_data[malibu_col] - plot_data[celltester_col]

    bars = ax4.bar(x_pos, differences, alpha=0.7, edgecolor='black')

    # Color bars based on positive/negative differences
    for i, bar in enumerate(bars):
        if differences.iloc[i] >= 0:
            bar.set_color('green')
        else:
            bar.set_color('red')

    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.8)
    ax4.set_xlabel('Well Number', fontsize=12)
    ax4.set_ylabel(f'Difference (Malibu - CellTester)', fontsize=12)
    ax4.set_title('Difference Plot', fontsize=12, fontweight='bold')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(wells, rotation=45)
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    return fig, plot_data

def print_summary_statistics(plot_data, parameter):
    """Print summary statistics for the comparison"""

    malibu_col = f'Malibu_{parameter}_mean'
    celltester_col = f'CellTester_{parameter}_mean'

    print(f"\n=== Summary Statistics for {parameter} ===")
    print(f"Number of common wells: {len(plot_data)}")
    print(f"\nMalibu {parameter}:")
    print(f"  Mean: {plot_data[malibu_col].mean():.3f}")
    print(f"  Std:  {plot_data[malibu_col].std():.3f}")
    print(f"  Min:  {plot_data[malibu_col].min():.3f}")
    print(f"  Max:  {plot_data[malibu_col].max():.3f}")

    print(f"\nCellTester {parameter}:")
    print(f"  Mean: {plot_data[celltester_col].mean():.3f}")
    print(f"  Std:  {plot_data[celltester_col].std():.3f}")
    print(f"  Min:  {plot_data[celltester_col].min():.3f}")
    print(f"  Max:  {plot_data[celltester_col].max():.3f}")

    # Calculate correlation
    correlation = plot_data[malibu_col].corr(plot_data[celltester_col])
    print(f"\nCorrelation between Malibu and CellTester: {correlation:.3f}")

    # Calculate differences
    differences = plot_data[malibu_col] - plot_data[celltester_col]
    print(f"\nDifferences (Malibu - CellTester):")
    print(f"  Mean difference: {differences.mean():.3f}")
    print(f"  Std difference:  {differences.std():.3f}")
    print(f"  Max difference:  {differences.max():.3f}")
    print(f"  Min difference:  {differences.min():.3f}")

def main():
    """Main function to run the comparison analysis"""

    try:
        # Load data
        print("Loading Excel data files...")
        malibu_data, celltester_data = load_data()

        # Find common parameters between datasets
        malibu_numeric = malibu_data.select_dtypes(include=[np.number]).columns.tolist()
        print(f"\nMalibu numeric parameters: {malibu_numeric}")

        # Common performance parameters to try
        performance_params = ['Voc [V]', 'Jsc [mA/cm2]', 'FF [.]', 'Efficiency [.]']

        # Find which parameters are available
        available_params = [param for param in performance_params if param in malibu_numeric]

        if not available_params:
            print("No common performance parameters found. Available Malibu parameters:")
            print(malibu_numeric)
            return

        print(f"\nAvailable parameters for comparison: {available_params}")

        # Create plots for each available parameter
        for parameter in available_params:
            try:
                print(f"\n{'='*50}")
                print(f"Processing parameter: {parameter}")
                print(f"{'='*50}")

                # Prepare comparison data
                plot_data = prepare_comparison_data(malibu_data, celltester_data, parameter)

                if plot_data.empty:
                    print(f"No common wells found for parameter: {parameter}")
                    continue

                # Create plots
                fig, plot_data = create_comparison_plots(plot_data, parameter)

                # Print statistics
                print_summary_statistics(plot_data, parameter)

                # Save plot
                output_filename = f"excel_comparison_{parameter.replace(' ', '_').replace('[', '').replace(']', '')}.png"
                fig.savefig(output_filename, dpi=300, bbox_inches='tight')
                print(f"\nPlot saved as: {output_filename}")

                # Show plot
                plt.show()

                # Save data to CSV
                csv_filename = f"excel_comparison_data_{parameter.replace(' ', '_').replace('[', '').replace(']', '')}.csv"
                plot_data.to_csv(csv_filename, index=False)
                print(f"Data saved as: {csv_filename}")

            except Exception as e:
                print(f"Error processing parameter {parameter}: {str(e)}")
                continue

        print(f"\n{'='*50}")
        print("Analysis complete!")
        print(f"{'='*50}")

    except Exception as e:
        print(f"Error: {str(e)}")
        return

if __name__ == "__main__":
    main()