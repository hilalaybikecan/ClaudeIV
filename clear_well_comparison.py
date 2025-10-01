#!/usr/bin/env python3
"""
Clear Well Comparison Script

Compares averaged Malibu data with CellTester mean data by well.
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

    # Show what's in the data
    print(f"\nMalibu columns: {list(malibu_data.columns)}")
    print(f"CellTester columns: {list(celltester_data.columns)}")

    # Find well column in Malibu data
    well_column = None
    for col in ['well', 'Well', 'Condition']:
        if col in malibu_data.columns:
            well_column = col
            break

    if well_column is None:
        print("Error: Cannot find well column in Malibu data")
        return

    print(f"\nUsing '{well_column}' as well identifier in Malibu data")

    # Average Malibu data by well
    malibu_averaged = malibu_data.groupby(well_column).agg({
        'Voc [V]': 'mean',
        'Jsc [mA/cm2]': 'mean',
        'FF [.]': 'mean',
        'Efficiency [.]': 'mean'
    }).reset_index()

    print(f"\nMalibu wells: {sorted(malibu_averaged[well_column].tolist())}")
    print(f"CellTester wells: {sorted(celltester_data['Well'].tolist())}")

    # Find common wells
    malibu_wells = set(malibu_averaged[well_column])
    celltester_wells = set(celltester_data['Well'])
    common_wells = malibu_wells & celltester_wells

    print(f"\nCommon wells: {sorted(list(common_wells))}")

    if len(common_wells) == 0:
        print("No common wells found! Plotting all data separately.")
        plot_separate_data(malibu_averaged, celltester_data, well_column)
    else:
        print(f"Found {len(common_wells)} common wells. Creating comparison plots.")
        plot_comparison(malibu_averaged, celltester_data, common_wells, well_column)

def plot_separate_data(malibu_avg, celltester_data, well_column):
    """Plot data separately when no common wells exist"""

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Malibu vs CellTester Data (Separate)', fontsize=16, fontweight='bold')

    # Parameters to plot
    params = [
        ('Voc [V]', 'Voc_mean', 'Voc (V)'),
        ('Jsc [mA/cm2]', 'Jsc_mAcm2_mean', 'Jsc (mA/cm²)'),
        ('FF [.]', 'FF_pct_mean', 'FF (%)'),
        ('Efficiency [.]', 'PCE_pct_mean', 'Efficiency (%)')
    ]

    axes = axes.flatten()

    for i, (malibu_col, celltester_col, title) in enumerate(params):
        ax = axes[i]

        # Plot both datasets
        ax.scatter(range(len(malibu_avg)), malibu_avg[malibu_col],
                  color='blue', alpha=0.7, s=60, label='Malibu', edgecolors='black')
        ax.scatter(range(len(celltester_data)), celltester_data[celltester_col],
                  color='red', alpha=0.7, s=60, label='CellTester', marker='s', edgecolors='black')

        ax.set_xlabel('Sample Index')
        ax.set_ylabel(title)
        ax.set_title(f'{title} Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("separate_comparison.png", dpi=300, bbox_inches='tight')
    print("Plot saved as: separate_comparison.png")
    plt.show()

def plot_comparison(malibu_avg, celltester_data, common_wells, well_column):
    """Create scatter plots for matched wells"""

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Malibu vs CellTester Comparison (Matched Wells)', fontsize=16, fontweight='bold')

    # Parameters to plot
    params = [
        ('Voc [V]', 'Voc_mean', 'Voc (V)'),
        ('Jsc [mA/cm2]', 'Jsc_mAcm2_mean', 'Jsc (mA/cm²)'),
        ('FF [.]', 'FF_pct_mean', 'FF (%)'),
        ('Efficiency [.]', 'PCE_pct_mean', 'Efficiency (%)')
    ]

    axes = axes.flatten()

    for i, (malibu_col, celltester_col, title) in enumerate(params):
        ax = axes[i]

        # Collect matched data
        malibu_values = []
        celltester_values = []
        labels = []

        for well in common_wells:
            # Get Malibu value
            malibu_row = malibu_avg[malibu_avg[well_column] == well]
            if not malibu_row.empty:
                malibu_val = malibu_row[malibu_col].iloc[0]

                # Get CellTester value
                celltester_row = celltester_data[celltester_data['Well'] == well]
                if not celltester_row.empty:
                    celltester_val = celltester_row[celltester_col].iloc[0]

                    malibu_values.append(malibu_val)
                    celltester_values.append(celltester_val)
                    labels.append(well)

        if malibu_values and celltester_values:
            # Create scatter plot
            ax.scatter(malibu_values, celltester_values, alpha=0.7, s=100,
                      color='purple', edgecolors='black', linewidth=1)

            # Add well labels
            for j, label in enumerate(labels):
                ax.annotate(label, (malibu_values[j], celltester_values[j]),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)

            # Add 1:1 line
            min_val = min(min(malibu_values), min(celltester_values))
            max_val = max(max(malibu_values), max(celltester_values))
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='1:1 line')

            ax.set_xlabel(f'Malibu {title}')
            ax.set_ylabel(f'CellTester {title}')
            ax.set_title(f'{title} Comparison')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Calculate correlation
            if len(malibu_values) > 1:
                correlation = np.corrcoef(malibu_values, celltester_values)[0, 1]
                ax.text(0.05, 0.95, f'R = {correlation:.3f}', transform=ax.transAxes,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        else:
            ax.text(0.5, 0.5, f'No data for {title}', transform=ax.transAxes,
                   ha='center', va='center')

    plt.tight_layout()
    plt.savefig("matched_well_comparison.png", dpi=300, bbox_inches='tight')
    print("Plot saved as: matched_well_comparison.png")
    plt.show()

    # Print summary
    print("\n=== Summary Statistics ===")
    for malibu_col, celltester_col, title in params:
        malibu_vals = []
        celltester_vals = []

        for well in common_wells:
            malibu_row = malibu_avg[malibu_avg[well_column] == well]
            celltester_row = celltester_data[celltester_data['Well'] == well]

            if not malibu_row.empty and not celltester_row.empty:
                malibu_vals.append(malibu_row[malibu_col].iloc[0])
                celltester_vals.append(celltester_row[celltester_col].iloc[0])

        if malibu_vals and celltester_vals:
            correlation = np.corrcoef(malibu_vals, celltester_vals)[0, 1] if len(malibu_vals) > 1 else 0
            print(f"\n{title}:")
            print(f"  Malibu     - Mean: {np.mean(malibu_vals):.3f}, Std: {np.std(malibu_vals):.3f}")
            print(f"  CellTester - Mean: {np.mean(celltester_vals):.3f}, Std: {np.std(celltester_vals):.3f}")
            print(f"  Correlation: {correlation:.3f}")

if __name__ == "__main__":
    main()