#!/usr/bin/env python3
"""
Simple Well Comparison Script

Creates scatter plots comparing Malibu and CellTester data by well.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def main():
    # Load data files
    script_dir = os.path.dirname(os.path.abspath(__file__))
    example_data_path = os.path.join(script_dir, "pixelreader", "example data files")

    malibu_path = os.path.join(example_data_path, "malibu.xlsx")
    celltester_path = os.path.join(example_data_path, "celltester.csv")

    print("Loading data files...")
    malibu_data = pd.read_excel(malibu_path)
    celltester_data = pd.read_csv(celltester_path)

    print(f"Malibu data: {len(malibu_data)} rows")
    print(f"CellTester data: {len(celltester_data)} rows")

    print(f"\nMalibu columns: {list(malibu_data.columns)}")
    print(f"CellTester columns: {list(celltester_data.columns)}")

    # Show first few rows to understand structure
    print("\nMalibu data sample:")
    print(malibu_data[['Deposition ID', 'Voc [V]', 'Jsc [mA/cm2]', 'FF [.]', 'Efficiency [.]']].head())

    print("\nCellTester data sample:")
    print(celltester_data[['Well', 'Voc_mean', 'Jsc_mAcm2_mean', 'FF_pct_mean', 'PCE_pct_mean']].head())

    # Determine which column contains well information
    well_col = None
    for col in ['Condition', 'Well', 'Deposition ID']:
        if col in malibu_data.columns:
            well_col = col
            break

    if well_col is None:
        print("Error: Could not find well column in Malibu data")
        return

    print(f"\nUsing '{well_col}' as well column in Malibu data")

    # Check unique wells
    print(f"Malibu wells ({well_col}): {sorted(malibu_data[well_col].unique())}")
    print(f"CellTester wells (Well): {sorted(celltester_data['Well'].unique())}")

    # Average Malibu data by well column
    malibu_avg = malibu_data.groupby(well_col).agg({
        'Voc [V]': 'mean',
        'Jsc [mA/cm2]': 'mean',
        'FF [.]': 'mean',
        'Efficiency [.]': 'mean'
    }).reset_index()

    # CellTester data is already averaged by well, so just use it directly
    celltester_avg = celltester_data[['Well', 'Voc_mean', 'Jsc_mAcm2_mean', 'FF_pct_mean', 'PCE_pct_mean']].copy()

    # Find common wells
    common_wells = set(malibu_avg[well_col]) & set(celltester_avg['Well'])
    common_wells = sorted(list(common_wells))

    print(f"\nCommon wells: {common_wells}")

    if not common_wells:
        print("No exact matches found. Will show all data for comparison.")
        # Create side-by-side comparison without exact matching
        create_side_by_side_plots(malibu_avg, celltester_avg)
    else:
        # Create matched comparison plots
        create_matched_plots(malibu_avg, celltester_avg, common_wells, well_col)

def create_side_by_side_plots(malibu_avg, celltester_avg):
    """Create side-by-side scatter plots when no exact well matches exist"""

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Malibu vs CellTester Data Comparison (Side by Side)', fontsize=16, fontweight='bold')

    parameters = [
        ('Voc [V]', 'Voc_mean', 'Voc (V)'),
        ('Jsc [mA/cm2]', 'Jsc_mAcm2_mean', 'Jsc (mA/cm²)'),
        ('FF [.]', 'FF_pct_mean', 'FF (%)'),
        ('Efficiency [.]', 'PCE_pct_mean', 'Efficiency (%)')
    ]

    axes = axes.flatten()

    for i, (malibu_col, celltester_col, ylabel) in enumerate(parameters):
        ax = axes[i]

        # Plot Malibu data
        x_malibu = range(len(malibu_avg))
        y_malibu = malibu_avg[malibu_col]
        ax.scatter(x_malibu, y_malibu, alpha=0.7, s=80, color='blue', label='Malibu', edgecolors='black')

        # Plot CellTester data (offset x position)
        x_celltester = [x + 0.3 for x in range(len(celltester_avg))]
        y_celltester = celltester_avg[celltester_col]
        ax.scatter(x_celltester, y_celltester, alpha=0.7, s=80, color='red', label='CellTester', edgecolors='black', marker='s')

        ax.set_xlabel('Sample Index')
        ax.set_ylabel(ylabel)
        ax.set_title(f'{ylabel} Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("side_by_side_comparison.png", dpi=300, bbox_inches='tight')
    print("Side-by-side plot saved as: side_by_side_comparison.png")
    plt.show()

def create_matched_plots(malibu_avg, celltester_avg, common_wells, well_col):
    """Create scatter plots for wells that match between datasets"""

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Malibu vs CellTester Data Comparison (Matched Wells)', fontsize=16, fontweight='bold')

    parameters = [
        ('Voc [V]', 'Voc_mean', 'Voc (V)'),
        ('Jsc [mA/cm2]', 'Jsc_mAcm2_mean', 'Jsc (mA/cm²)'),
        ('FF [.]', 'FF_pct_mean', 'FF (%)'),
        ('Efficiency [.]', 'PCE_pct_mean', 'Efficiency (%)')
    ]

    axes = axes.flatten()

    for i, (malibu_col, celltester_col, ylabel) in enumerate(parameters):
        ax = axes[i]

        # Prepare matched data
        malibu_values = []
        celltester_values = []
        well_labels = []

        for well in common_wells:
            malibu_row = malibu_avg[malibu_avg[well_col] == well]
            celltester_row = celltester_avg[celltester_avg['Well'] == well]

            if not malibu_row.empty and not celltester_row.empty:
                malibu_values.append(malibu_row[malibu_col].iloc[0])
                celltester_values.append(celltester_row[celltester_col].iloc[0])
                well_labels.append(well)

        if malibu_values and celltester_values:
            # Create scatter plot
            ax.scatter(malibu_values, celltester_values, alpha=0.7, s=100,
                      color='purple', edgecolors='black', linewidth=1)

            # Add well labels
            for j, well in enumerate(well_labels):
                ax.annotate(well, (malibu_values[j], celltester_values[j]),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)

            # Add 1:1 line
            min_val = min(min(malibu_values), min(celltester_values))
            max_val = max(max(malibu_values), max(celltester_values))
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='1:1 line')

            ax.set_xlabel(f'Malibu {ylabel}')
            ax.set_ylabel(f'CellTester {ylabel}')
            ax.set_title(f'{ylabel} Comparison')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Calculate and display correlation
            correlation = np.corrcoef(malibu_values, celltester_values)[0, 1]
            ax.text(0.05, 0.95, f'R = {correlation:.3f}', transform=ax.transAxes,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        else:
            ax.text(0.5, 0.5, f'No data for {ylabel}', transform=ax.transAxes,
                   ha='center', va='center')

    plt.tight_layout()
    plt.savefig("matched_well_comparison.png", dpi=300, bbox_inches='tight')
    print("Matched wells plot saved as: matched_well_comparison.png")
    plt.show()

    # Print summary statistics
    print("\n=== Summary Statistics ===")
    for malibu_col, celltester_col, ylabel in parameters:
        malibu_vals = []
        celltester_vals = []

        for well in common_wells:
            malibu_row = malibu_avg[malibu_avg[well_col] == well]
            celltester_row = celltester_avg[celltester_avg['Well'] == well]

            if not malibu_row.empty and not celltester_row.empty:
                malibu_vals.append(malibu_row[malibu_col].iloc[0])
                celltester_vals.append(celltester_row[celltester_col].iloc[0])

        if malibu_vals and celltester_vals:
            correlation = np.corrcoef(malibu_vals, celltester_vals)[0, 1]
            print(f"\n{ylabel}:")
            print(f"  Malibu   - Mean: {np.mean(malibu_vals):.3f}, Std: {np.std(malibu_vals):.3f}")
            print(f"  CellTester - Mean: {np.mean(celltester_vals):.3f}, Std: {np.std(celltester_vals):.3f}")
            print(f"  Correlation: {correlation:.3f}")

if __name__ == "__main__":
    main()