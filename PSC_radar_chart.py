"""
Perovskite Solar Cells Passport
Multi-Parameter Photovoltaic Aging Visualization Suite (Radar Mapping)
======================================================================
Author: Laboratory of New Materials for Solar Energetics (NMSE)
Year: 2026
Institution: Faculty of Material Science, Lomonosov Moscow State University

License: MIT License
Copyright (c) 2026 Laboratory of New Materials for Solar Energetics, Lomonosov Moscow State University

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 20,              
    "mathtext.fontset": "stix"
})

def process_all_devices_in_file(file_path, base_output_dir):
    """
    Parses an excel spreadsheet row by row, automatically mitigating missing 
    baseline metrics and plotting percentage labels strictly adjacent to the final 
    measurement line contour.
    """
    file_name = os.path.basename(file_path)
    file_base, _ = os.path.splitext(file_name)
    print(f"Parsing data matrix: {file_name}")
    
    file_specific_dir = os.path.join(base_output_dir, f"Radars_{file_base}")
    if not os.path.exists(file_specific_dir):
        os.makedirs(file_specific_dir)

    df = pd.read_excel(file_path).dropna(subset=['Sample']).reset_index(drop=True)
    params = ['PCE', 'Voc', 'Jsc', 'FF', 'Rsh', 'Rs', 'HI']
    
    # Dynamically extract all time-step suffixes from the Excel columns
    time_steps = sorted(list(set([int(c.split('_')[-1]) for c in df.columns if c.startswith('PCE_')])))
    total_steps = len(time_steps)
    
    # Robust normalization block with fallback for missing baseline steps
    df_rel = df.copy()
    for idx, row in df.iterrows():
        for p in params:
            p_cols = sorted([c for c in df.columns if c.startswith(f"{p}_")], key=lambda x: int(x.split('_')[-1]))
            
            valid_baseline_val = None
            for col in p_cols:
                if not pd.isna(row[col]):
                    valid_baseline_val = row[col]
                    break
            
            if valid_baseline_val is None or valid_baseline_val == 0:
                valid_baseline_val = 1e-9
                
            for col in p_cols:
                vt = row[col]
                if pd.isna(vt):
                    df_rel.loc[idx, col] = np.nan
                    continue
                    
                if p == 'HI':
                    base_val = row[f"{p}_0"] if f"{p}_0" in row and not pd.isna(row[f"{p}_0"]) else valid_baseline_val
                    df_rel.loc[idx, col] = np.clip(1.0 - (vt - base_val), 0.05, 1.1)
                elif p == 'Rs':
                    base_val = row[f"{p}_0"] if f"{p}_0" in row and not pd.isna(row[f"{p}_0"]) else valid_baseline_val
                    df_rel.loc[idx, col] = np.clip(2 / (vt / (base_val + 1e-9) + 1), 0.05, 1.1)
                else:
                    df_rel.loc[idx, col] = np.clip(vt / valid_baseline_val, 0, 1.1)

    fancy_labels = [r'$PCE$', r'$V_{oc}$', r'$J_{sc}$', r'$FF$', r'$R_{sh}$', r'$R_s$', r'$HI$']
    num_vars = len(params)

    # Loop through every single row (device) in the cleaned relative dataframe
    for idx, row in df_rel.iterrows():
        sample_name = str(df.loc[idx, 'Sample']).strip()
        excel_row_num = idx + 2 
        
        valid_steps_for_device = []
        for t in time_steps:
            if f"PCE_{t}" in row and not pd.isna(row[f"PCE_{t}"]):
                valid_steps_for_device.append(t)
                
        if not valid_steps_for_device:
            print(f"  Warning: Skipping row {excel_row_num} ({sample_name}) due to completely missing data profiles.")
            continue
            
        device_last_valid_t = valid_steps_for_device[-1]
        
        fig, ax = plt.subplots(figsize=(7.5, 7.5), subplot_kw=dict(polar=True))
        
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        
        # Concentric grid scaling
        for r in [0.25, 0.5, 0.75, 1.0]:
            ax.plot(angles, [r]*(num_vars+1), color='#D3D3D3', lw=1.2, ls='--')
        ax.plot(angles, [1.0]*(num_vars+1), color='#777777', lw=2.0, ls='-')

        # Array to explicitly cache the final plotted iteration values for precise labeling
        final_plotted_values = None
        final_plotted_color = None

        # Sequential rendering pipeline mapping active intervals
        for i, t in enumerate(time_steps):
            if f"PCE_{t}" not in row or pd.isna(row[f"PCE_{t}"]):
                continue
                
            values = [row[f"{p}_{t}"] for p in params] + [row[f"{params[0]}_{t}"]]
            t_color = plt.cm.Blues(0.4 + (i / max(1, total_steps - 1)) * 0.45) if total_steps > 1 else plt.cm.Blues(0.7)
            label_text = f"Interval: {t}"
            
            is_final_valid_step = (int(t) == int(device_last_valid_t))
            
            ax.plot(angles, values, color=t_color, lw=5 if is_final_valid_step else 3, label=label_text)
            ax.fill(angles, values, color=t_color, alpha=0.05)
            
            # Cache metadata strictly if this step represents the device boundary
            if is_final_valid_step:
                final_plotted_values = values[:-1]
                final_plotted_color = t_color

        # TEXT ANNOTATION SEQUENCE: Guaranteed fallback execution outside the rendering loops
        if final_plotted_values is not None:
            for angle, val in zip(angles[:-1], final_plotted_values):
                if pd.isna(val) or np.isnan(val):
                    val = 0.0
                    
                # Precise contour matching
                if val > 0.92:
                    display_radius = val - 0.06
                    text_color = "white"
                    bg_color = final_plotted_color
                    bg_alpha = 0.95
                else:
                    display_radius = val + 0.08
                    text_color = final_plotted_color
                    bg_color = "white"
                    bg_alpha = 0.85
                    
                ax.text(angle, display_radius, f'{val:.0%}', size=12, fontweight='bold', color=text_color, 
                        ha='center', va='center',
                        bbox=dict(boxstyle="round,pad=0.12", fc=bg_color, ec="none", alpha=bg_alpha))

        # Polar configuration and tick cleanup
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_ylim(0, 1.50) 
        ax.spines['polar'].set_visible(False)

        # Precise position mapping for parameters labels
        text_radius = 1.32  
        for angle, label in zip(angles[:-1], fancy_labels):
            eff_angle = np.pi / 2 - angle
            if np.cos(eff_angle) > 0.1: ha = 'left'
            elif np.cos(eff_angle) < -0.1: ha = 'right'
            else: ha = 'center'
                
            ax.text(angle, text_radius, label, size=24, fontweight='bold',
                    ha=ha, va='center', transform=ax.transData)

        # TITLE DESIGN
        title_text = f"Device: {sample_name}\nSource: {file_name} (Row {excel_row_num})"
        ax.set_title(title_text, size=20, pad=45, fontweight='bold')

        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=total_steps, fontsize=18, frameon=False)

        output_filename = f"DEVICE_{sample_name}_row_{excel_row_num}"
        full_output_path = os.path.join(file_specific_dir, output_filename)
        
        plt.savefig(full_output_path + ".png", dpi=300, bbox_inches='tight')
        plt.close()

def batch_execute_pipeline():
    """
    Scans the execution directory for spreadsheet logs and maps them sequentially 
    to the visualization engine.
    """
    target_folder = "Output_Radars"
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    excel_files = glob.glob("*.xlsx")
    if not excel_files:
        print("Execution halted: No valid spreadsheet profiles (.xlsx) discovered.")
        return
        
    print(f"Discovered {len(excel_files)} file matrices. Starting serial processing...")
    
    for file_path in sorted(excel_files):
        try:
            process_all_devices_in_file(file_path, target_folder)
        except Exception as error_context:
            print(f"Array execution skipped for {file_path}. Error: {error_context}")
            
    print(f"Pipeline complete. Datasets compiled in directory: ./{target_folder}")

if __name__ == "__main__":
    batch_execute_pipeline()


