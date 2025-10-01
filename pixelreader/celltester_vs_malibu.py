#!/usr/bin/env python3
"""
Simple Well Comparison Script - No GUI display
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import os

def main():
    # Load data files
    script_dir = os.path.dirname(os.path.abspath(__file__))
    example_data_path = os.path.join(script_dir, "pixelreader", "example data files")

    malibu_path = os.path.join(example_data_path, "malibu.xlsx")
    celltester_path = os.path.join(example_data_path, "celltester.csv")

    print("Loading data...")
    malibu_data = pd.read_excel(malibu_path)
    celltester_data = pd.read_csv(celltester_path)

    print(f"Malibu: {len(malibu_data)} rows")
    print(f"CellTester: {len(celltester_data)} rows")

    # Average Malibu data by well
    malibu_avg = malibu_data.groupby('well').agg({
        'Voc [V]': 'mean',
        'Jsc [mA/cm2]': 'mean',
        'FF [.]': 'mean',
        'Efficiency [.]': 'mean'
    }).reset_index()

    print(f"Malibu averaged: {len(malibu_avg)} wells")

    # Find common wells
    malibu_wells = set(malibu_avg['well'])
    celltester_wells = set(celltester_data['Well'])
    common_wells = sorted(list(malibu_wells & celltester_wells))

    print(f"Common wells ({len(common_wells)}): {common_wells}")

    if len(common_wells) == 0:
        print("No common wells found!")
        return

    # Create the plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Parameter Comparison: Malibu vs CellTester by Well', fontsize=16, fontweight='bold')

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

        # Prepare data
        well_positions = {well: idx for idx, well in enumerate(common_wells)}

        malibu_x = []
        malibu_y = []
        celltester_x = []
        celltester_y = []

        # Get data points
        for well in common_wells:
            # Malibu data
            malibu_row = malibu_avg[malibu_avg['well'] == well]
            if not malibu_row.empty:
                malibu_x.append(well_positions[well])
                malibu_y.append(malibu_row[malibu_col].iloc[0])

            # CellTester data
            celltester_row = celltester_data[celltester_data['Well'] == well]
            if not celltester_row.empty:
                celltester_x.append(well_positions[well])
                celltester_y.append(celltester_row[celltester_col].iloc[0])

        # Plot
        ax.scatter(malibu_x, malibu_y, color='blue', alpha=0.7, s=80,
                  label='Malibu', edgecolors='black', linewidth=1)
        ax.scatter(celltester_x, celltester_y, color='red', alpha=0.7, s=80,
                  label='CellTester', marker='s', edgecolors='black', linewidth=1)

        ax.set_xlabel('Well ID')
        ax.set_ylabel(title)
        ax.set_title(f'{title} by Well')
        ax.set_xticks(range(len(common_wells)))
        ax.set_xticklabels(common_wells, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("well_comparison_plot.png", dpi=300, bbox_inches='tight')
    print("Plot saved as: well_comparison_plot.png")

    # Print summary
    print("\n=== Summary ===")
    for malibu_col, celltester_col, title in params:
        malibu_vals = []
        celltester_vals = []

        for well in common_wells:
            malibu_row = malibu_avg[malibu_avg['well'] == well]
            celltester_row = celltester_data[celltester_data['Well'] == well]

            if not malibu_row.empty and not celltester_row.empty:
                malibu_vals.append(malibu_row[malibu_col].iloc[0])
                celltester_vals.append(celltester_row[celltester_col].iloc[0])

        if malibu_vals and celltester_vals:
            print(f"{title}: Malibu={np.mean(malibu_vals):.3f}, CellTester={np.mean(celltester_vals):.3f}")

    print("Done!")

if __name__ == "__main__":
    main()