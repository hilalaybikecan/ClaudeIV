import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import re
import json
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class IVDataAnalyzer:
    def __init__(self, root):
        self.root = root
        self.root.title("IV Data Analyzer")
        self.root.geometry("1200x800")
        
        # Main data storage
        self.measurements_data = pd.DataFrame(columns=[
            'Filename', 'Substrate ID', 'Pixel', 'Scan Direction', 
            'Voc [V]', 'Jsc [mA/cm2]', 'FF [.]', 'Efficiency [.]', 
            'Pmpp [W/m2]', 'Vmpp [V]', 'Jmpp [mA/cm2]', 'Roc [Ohm.m2]', 'Rsc [Ohm.m2]', 'Scan Speed [V/s]', 'Filepath'
        ])
        
        # Persistent copy for display sorting (never modified directly)
        self.measurements_data_original = pd.DataFrame(columns=[
            'Filename', 'Substrate ID', 'Pixel', 'Scan Direction', 
            'Voc [V]', 'Jsc [mA/cm2]', 'FF [.]', 'Efficiency [.]', 
            'Pmpp [W/m2]', 'Vmpp [V]', 'Jmpp [mA/cm2]', 'Roc [Ohm.m2]', 'Rsc [Ohm.m2]'
        ])
        
        # Conditions data storage
        self.conditions_data = pd.DataFrame(columns=['Substrate ID', 'Condition', 'Display Order'])
        
        # Store the current figure for saving
        self.current_figure = None
        self.current_plot_data = None
        
        # Track sort state for measurements tree
        self.last_sorted_column = None
        self.last_sort_descending = False
        
        # Available color palettes
        self.color_palettes = {
            "Default": None,
            "viridis": "viridis",
            "magma": "magma",
            "plasma": "plasma",
            "inferno": "inferno",
            "Set1": "Set1",
            "Set2": "Set2",
            "Set3": "Set3",
            "tab10": "tab10",
            "pastel": "pastel"
        }
        
        # Auto-save custom order to temp file
        self.auto_save_file = "temp_condition_order.json"
        
        # Create UI elements
        self.create_ui()
    
    def create_ui(self):
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True, padx=10, pady=10)
    
        # Existing tabs
        data_frame = ttk.Frame(notebook)
        notebook.add(data_frame, text="Data Management")
    
        plot_frame = ttk.Frame(notebook)
        notebook.add(plot_frame, text="Condition Plots")
    
        # New IV Plot tab
        iv_plot_frame = ttk.Frame(notebook)
        notebook.add(iv_plot_frame, text="IV Plots")
    
        # Setup UI components
        self.setup_data_management(data_frame)
        self.setup_plotting(plot_frame)
        self.setup_iv_plot(iv_plot_frame)  # Add this line

    
    def setup_data_management(self, parent):
        # File management frame
        file_frame = ttk.LabelFrame(parent, text="File Management")
        file_frame.pack(fill="x", expand=False, padx=10, pady=5)
        
        # Browse button
        browse_button = ttk.Button(file_frame, text="Browse for Files", command=self.browse_files)
        browse_button.pack(side="left", padx=5, pady=5)
        
        # Remove button
        remove_button = ttk.Button(file_frame, text="Remove Selected", command=self.remove_selected)
        remove_button.pack(side="left", padx=5, pady=5)
        
        # Measurements table frame
        measurements_frame = ttk.LabelFrame(parent, text="Measurements")
        measurements_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Create treeview for measurements data
        columns = ('Filename', 'Substrate ID', 'Pixel', 'Scan Direction', 
                  'Voc [V]', 'Jsc [mA/cm2]', 'FF [.]', 'Efficiency [.]', 
                  'Pmpp [W/m2]', 'Vmpp [V]', 'Jmpp [mA/cm2]', 'Roc [Ohm.m2]', 'Rsc [Ohm.m2]', 'Scan Speed [V/s]')
        self.display_columns = columns
        
        self.measurements_tree = ttk.Treeview(measurements_frame, columns=self.display_columns, show='headings')
        
        # Set column headings and widths
        for col in self.display_columns:
            self.measurements_tree.heading(col, text=col, command=lambda c=col: self.sort_measurements_by_column(c))
            width = 80 if col != 'Filename' else 200  # Set wider for filename
            self.measurements_tree.column(col, width=width, anchor='center')
        
        # Add scrollbars
        vsb = ttk.Scrollbar(measurements_frame, orient="vertical", command=self.measurements_tree.yview)
        hsb = ttk.Scrollbar(measurements_frame, orient="horizontal", command=self.measurements_tree.xview)
        self.measurements_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        
        # Pack scrollbars and treeview
        vsb.pack(side="right", fill="y")
        hsb.pack(side="bottom", fill="x")
        self.measurements_tree.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Conditions frame
        conditions_frame = ttk.LabelFrame(parent, text="Conditions")
        conditions_frame.pack(fill="both", expand=False, padx=10, pady=5)
        
        # Conditions management
        conditions_input_frame = ttk.Frame(conditions_frame)
        conditions_input_frame.pack(fill="x", expand=False, padx=5, pady=5)
        
        ttk.Label(conditions_input_frame, text="New Condition:").pack(side="left", padx=5)
        self.condition_entry = ttk.Entry(conditions_input_frame, width=30)
        self.condition_entry.pack(side="left", padx=5)
        
        add_condition_button = ttk.Button(conditions_input_frame, text="Add Condition", command=self.add_condition)
        add_condition_button.pack(side="left", padx=5)
        
        # Conditions assignment frame
        assignment_frame = ttk.Frame(conditions_frame)
        assignment_frame.pack(fill="x", expand=False, padx=5, pady=5)
        
        ttk.Label(assignment_frame, text="Assign selected measurements to:").pack(side="left", padx=5)
        self.condition_combobox = ttk.Combobox(assignment_frame, width=30, state="readonly")
        self.condition_combobox.pack(side="left", padx=5)
        
        assign_button = ttk.Button(assignment_frame, text="Assign", command=self.assign_condition)
        assign_button.pack(side="left", padx=5)
        
        # Save/Load conditions frame
        save_load_frame = ttk.Frame(conditions_frame)
        save_load_frame.pack(fill="x", expand=False, padx=5, pady=5)
        
        save_conditions_button = ttk.Button(save_load_frame, text="Save Conditions", command=self.save_conditions)
        save_conditions_button.pack(side="left", padx=5)
        
        load_conditions_button = ttk.Button(save_load_frame, text="Load Conditions", command=self.load_conditions)
        load_conditions_button.pack(side="left", padx=5)
        
        # Conditions table frame
        conditions_table_frame = ttk.Frame(conditions_frame)
        conditions_table_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Create treeview for conditions with order column
        self.conditions_tree = ttk.Treeview(
            conditions_table_frame, 
            columns=('Substrate ID', 'Condition', 'Display Order'), 
            show='headings'
        )
        self.conditions_tree.heading('Substrate ID', text='Substrate ID')
        self.conditions_tree.heading('Condition', text='Condition')
        self.conditions_tree.heading('Display Order', text='Display Order')
        self.conditions_tree.column('Substrate ID', width=100, anchor='center')
        self.conditions_tree.column('Condition', width=150, anchor='center')
        self.conditions_tree.column('Display Order', width=80, anchor='center')
        
        # Add summary frame for unique conditions
        summary_frame = ttk.LabelFrame(conditions_frame, text="Condition Summary")
        summary_frame.pack(fill="x", expand=False, padx=10, pady=(0, 5))
        
        self.summary_label = ttk.Label(summary_frame, text="No conditions assigned yet")
        self.summary_label.pack(padx=5, pady=5)
        
        # Add scrollbar
        conditions_vsb = ttk.Scrollbar(conditions_table_frame, orient="vertical", command=self.conditions_tree.yview)
        self.conditions_tree.configure(yscrollcommand=conditions_vsb.set)
        
        # Pack scrollbar and treeview
        conditions_vsb.pack(side="right", fill="y")
        self.conditions_tree.pack(fill="both", expand=True)
        
        # Buttons for reordering conditions
        conditions_order_frame = ttk.Frame(conditions_frame)
        conditions_order_frame.pack(fill="x", expand=False, padx=5, pady=5)
        
        move_up_button = ttk.Button(conditions_order_frame, text="Move Up", command=lambda: self.reorder_condition(-1))
        move_up_button.pack(side="left", padx=5)
        
        move_down_button = ttk.Button(conditions_order_frame, text="Move Down", command=lambda: self.reorder_condition(1))
        move_down_button.pack(side="left", padx=5)
    
    def setup_plotting(self, parent):
        # Plot controls frame
        plot_control_frame = ttk.LabelFrame(parent, text="Plot Controls")
        plot_control_frame.pack(fill="x", expand=False, padx=10, pady=5)
        
        # Parameter selection
        param_frame = ttk.Frame(plot_control_frame)
        param_frame.pack(fill="x", expand=False, padx=5, pady=5)
        
        ttk.Label(param_frame, text="Select parameter to plot:").pack(side="left", padx=5)
        
        # Parameters for plotting
        plot_parameters = ['Voc [V]', 'Jsc [mA/cm2]', 'FF [.]', 'Efficiency [.]', 'Pmpp [W/m2]', 'Vmpp [V]', 'Jmpp [mA/cm2]', 'Roc [Ohm.m2]', 'Rsc [Ohm.m2]', 'Scan Speed [V/s]']
        self.plot_param_combobox = ttk.Combobox(param_frame, values=plot_parameters, width=30, state="readonly")
        self.plot_param_combobox.pack(side="left", padx=5)
        self.plot_param_combobox.current(0)  # Select first parameter by default
        
        # Y-axis limits frame
        ylim_frame = ttk.Frame(plot_control_frame)
        ylim_frame.pack(fill="x", expand=False, padx=5, pady=5)
        
        ttk.Label(ylim_frame, text="Y-axis Limits:").pack(side="left", padx=5)
        
        ttk.Label(ylim_frame, text="Min:").pack(side="left", padx=(20, 5))
        self.ymin_entry = ttk.Entry(ylim_frame, width=10)
        self.ymin_entry.pack(side="left", padx=5)
        
        ttk.Label(ylim_frame, text="Max:").pack(side="left", padx=(20, 5))
        self.ymax_entry = ttk.Entry(ylim_frame, width=10)
        self.ymax_entry.pack(side="left", padx=5)
        
        ttk.Label(ylim_frame, text="(Leave empty for auto)").pack(side="left", padx=5)
        
        # Color palette selection
        color_frame = ttk.Frame(plot_control_frame)
        color_frame.pack(fill="x", expand=False, padx=5, pady=5)
        
        ttk.Label(color_frame, text="Color Palette:").pack(side="left", padx=5)
        
        self.color_palette_combobox = ttk.Combobox(
            color_frame, 
            values=list(self.color_palettes.keys()), 
            width=20, 
            state="readonly"
        )
        self.color_palette_combobox.pack(side="left", padx=5)
        self.color_palette_combobox.current(0)  # Select default palette
        
        # X-axis ordering selection
        xorder_frame = ttk.Frame(plot_control_frame)
        xorder_frame.pack(fill="x", expand=False, padx=5, pady=5)
        
        ttk.Label(xorder_frame, text="X-axis Order:").pack(side="left", padx=5)
        
        self.xorder_combobox = ttk.Combobox(
            xorder_frame,
            values=["Display Order (Default)", "Alphabetical", "Custom"],
            width=25,
            state="readonly"
        )
        self.xorder_combobox.pack(side="left", padx=5)
        self.xorder_combobox.current(0)  # Select default ordering
        self.xorder_combobox.bind("<<ComboboxSelected>>", self.on_xorder_change)
        
        # Custom order interface (initially hidden)
        self.custom_order_frame = ttk.Frame(plot_control_frame)
        
        # Create a labeled frame for the custom ordering
        custom_order_labelframe = ttk.LabelFrame(self.custom_order_frame, text="Custom Condition Order")
        custom_order_labelframe.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Create frame for listbox and buttons
        listbox_frame = ttk.Frame(custom_order_labelframe)
        listbox_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Instructions label
        instructions_label = ttk.Label(listbox_frame, text="Reorder conditions: Use buttons, Ctrl+Up/Down, or double-click to move to top:")
        instructions_label.pack(anchor="w", pady=(0, 5))
        
        # Frame for listbox and scrollbar
        list_scroll_frame = ttk.Frame(listbox_frame)
        list_scroll_frame.pack(fill="both", expand=True)
        
        # Listbox for conditions
        self.conditions_listbox = tk.Listbox(list_scroll_frame, height=6, selectmode=tk.SINGLE)
        self.conditions_listbox.pack(side="left", fill="both", expand=True)
        
        # Add keyboard shortcuts for easier reordering
        self.conditions_listbox.bind('<Control-Up>', lambda e: self.move_condition_up())
        self.conditions_listbox.bind('<Control-Down>', lambda e: self.move_condition_down())
        self.conditions_listbox.bind('<Double-Button-1>', self.on_condition_double_click)
        
        # Scrollbar for listbox
        listbox_scrollbar = ttk.Scrollbar(list_scroll_frame, orient="vertical", command=self.conditions_listbox.yview)
        listbox_scrollbar.pack(side="right", fill="y")
        self.conditions_listbox.configure(yscrollcommand=listbox_scrollbar.set)
        
        # Buttons frame
        buttons_frame = ttk.Frame(listbox_frame)
        buttons_frame.pack(fill="x", pady=(5, 0))
        
        # Move up button
        self.move_up_btn = ttk.Button(buttons_frame, text="↑ Move Up", command=self.move_condition_up)
        self.move_up_btn.pack(side="left", padx=(0, 5))
        
        # Move down button
        self.move_down_btn = ttk.Button(buttons_frame, text="↓ Move Down", command=self.move_condition_down)
        self.move_down_btn.pack(side="left", padx=5)
        
        # Reset button
        self.reset_order_btn = ttk.Button(buttons_frame, text="Reset to Default", command=self.reset_condition_order)
        self.reset_order_btn.pack(side="left", padx=5)
        
        # Plot buttons
        button_frame = ttk.Frame(plot_control_frame)
        button_frame.pack(fill="x", expand=False, padx=5, pady=5)
        
        plot_button = ttk.Button(button_frame, text="Generate Plot", command=self.generate_plot)
        plot_button.pack(side="left", padx=5)
        
        save_plot_button = ttk.Button(button_frame, text="Save Plot", command=self.save_plot)
        save_plot_button.pack(side="left", padx=5)

        export_data_button = ttk.Button(button_frame, text="Export Data to Excel", command=self.export_condition_data)
        export_data_button.pack(side="left", padx=5)
        
        # Frame for the condition plot
        self.condition_plot_frame = ttk.Frame(parent)
        self.condition_plot_frame.pack(fill="both", expand=True, padx=10, pady=10)
    
    def save_conditions(self):
        """Save conditions data to a JSON file"""
        if self.conditions_data.empty:
            messagebox.showwarning("Warning", "No conditions to save.")
            return
        
        # Get conditions to save
        custom_order = [self.conditions_listbox.get(i) for i in range(self.conditions_listbox.size())] if hasattr(self, 'conditions_listbox') else []
        conditions_to_save = {
            "conditions_list": list(self.condition_combobox['values']),
            "conditions_data": self.conditions_data.to_dict(orient='records'),
            "custom_order": custom_order
        }
        
        # Ask for save location
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Save Conditions As"
        )
        
        if not file_path:
            return  # User cancelled
        
        try:
            with open(file_path, 'w') as f:
                json.dump(conditions_to_save, f, indent=4)
            messagebox.showinfo("Success", f"Conditions saved successfully to:\n{file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save conditions: {str(e)}")
    
    def load_conditions(self):
        """Load conditions data from a JSON file"""
        # Ask for file to load
        file_path = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Load Conditions"
        )
        
        if not file_path:
            return  # User cancelled
        
        try:
            with open(file_path, 'r') as f:
                saved_data = json.load(f)
            
            # Load conditions list for dropdown
            self.condition_combobox['values'] = saved_data.get("conditions_list", [])
            
            # Load conditions data
            conditions_data = saved_data.get("conditions_data", [])
            self.conditions_data = pd.DataFrame(conditions_data)
            
            # Update conditions tree
            self.update_conditions_tree()
            
            # Load custom order if available
            custom_order = saved_data.get("custom_order", [])
            if custom_order and hasattr(self, 'conditions_listbox'):
                self.populate_conditions_listbox(custom_order)
                
            messagebox.showinfo("Success", f"Conditions loaded successfully from:\n{file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load conditions: {str(e)}")
    
    def browse_files(self):
        filepaths = filedialog.askopenfilenames(
            title="Select IV Measurement Files",
            filetypes=(("IV Files", "*.iv"), ("All Files", "*.*"))
        )
        
        if not filepaths:
            return
        
        for filepath in filepaths:
            # Extract data from file
            try:
                data = self.parse_iv_file(filepath)
                
                # Add to both dataframes
                # Align columns in case of new fields
                for col in data.keys():
                    if col not in self.measurements_data.columns:
                        self.measurements_data[col] = None
                self.measurements_data = pd.concat([self.measurements_data, pd.DataFrame([data])], ignore_index=True)
                self.measurements_data_original = self.measurements_data.copy()
                
                # Add to treeview
                values = [data[col] for col in self.measurements_data.columns]
                self.measurements_tree.insert('', 'end', values=values)
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to parse file {os.path.basename(filepath)}: {str(e)}")
        
        self.update_measurements_display()
        self.refresh_iv_selection()

    def parse_iv_file(self, filepath):
        filename = os.path.basename(filepath)
        
        # Extract pixel information from filename

        pixel_match = re.search(r'c(\d)', filename)
        pixel = {'1': 'A', '2': 'B', '3': 'C'}.get(pixel_match.group(1), 'Unknown') if pixel_match else 'Unknown'
        
        with open(filepath, 'r') as f:
            content = f.read()

        # Extract sweep metadata to compute scan speed (V/s)
        vstart = self.extract_value(content, r'Vstart:\s*([-\d\.]+E[+-]\d+|[-\d\.]+)')
        vend = self.extract_value(content, r'Vend:\s*([-\d\.]+E[+-]\d+|[-\d\.]+)')
        npoints = None
        m_np = re.search(r'Number of points:\s*([-\d\.]+E[+-]\d+|[-\d\.]+)', content)
        if m_np:
            try:
                npoints = int(float(m_np.group(1)))
            except:
                npoints = None
        delay_s = self.extract_value(content, r'Delay \[s\]:\s*([-\d\.]+E[+-]\d+|[-\d\.]+)')
        it_s = self.extract_value(content, r'Integration time \[s\]:\s*([-\d\.]+E[+-]\d+|[-\d\.]+)')
        scan_speed = None
        try:
            if vstart is not None and vend is not None and npoints and npoints > 1 and (delay_s is not None) and (it_s is not None):
                step_time = (delay_s + it_s)
                total_time = (npoints - 1) * step_time
                if total_time > 0:
                    scan_speed = abs(vend - vstart) / total_time
        except Exception:
            scan_speed = None

        
        # Extract substrate ID
        substrate_id_match = re.search(r'Deposition ID:\s*([A-Za-z0-9_]+)', content)
        substrate_id = substrate_id_match.group(1) if substrate_id_match else 'Unknown'
        
        # Extract measurement data
        data_section_match = re.search(r'% MEASURED IV FRLOOP DATA.*?\nV \(measured\) \[V\].*?\n(.*?)$', content, re.DOTALL)
        if not data_section_match:
            raise ValueError("Could not find measurement data section in the file.")
        
        data_lines = data_section_match.group(1).strip().split('\n')
        voltages = []
        
        for line in data_lines:
            if line:
                values = line.split('\t')
                if len(values) >= 2:
                    try:
                        voltages.append(float(values[0]))
                    except:
                        pass
        
        # Determine scan direction
        if len(voltages) > 1:
            scan_direction = "fwd" if voltages[1] > voltages[0] else "rev"
        else:
            scan_direction = "unknown"
        
        # Extract analysis outputs
        analysis_outputs = {
            'Voc [V]': self.extract_value(content, r'Voc \[V\]:\s*(\d+\.\d+E[+-]\d+)'),
            'Jsc [mA/cm2]': self.extract_value(content, r'Jsc \[A/m2\]:\s*(\d+\.\d+E[+-]\d+)', convert_to_ma_cm2=True),
            'FF [.]': self.extract_value(content, r'FF \[.\]:\s*(\d+\.\d+E[+-]\d+)'),
            'Efficiency [.]': self.extract_value(content, r'Efficiency \[.\]:\s*(\d+\.\d+E[+-]\d+)'),
            'Pmpp [W/m2]': self.extract_value(content, r'Pmpp \[W/m2\]:\s*(\d+\.\d+E[+-]\d+)'),
            'Vmpp [V]': self.extract_value(content, r'Vmpp \[V\]:\s*(\d+\.\d+E[+-]\d+)'),
            'Jmpp [mA/cm2]': self.extract_value(content, r'Jmpp \[A\]:\s*(\d+\.\d+E[+-]\d+)', convert_to_ma_cm2=True),
            'Roc [Ohm.m2]': self.extract_value(content, r'Roc \[Ohm\.m2\]:\s*(\d+\.\d+E[+-]\d+)'),
            'Rsc [Ohm.m2]': self.extract_value(content, r'Rsc \[Ohm\.m2\]:\s*(\d+\.\d+E[+-]\d+)')
        }
        
        # Create data dictionary
        data = {
            'Filename': filename,
            'Substrate ID': substrate_id,
            'Pixel': pixel,
            'Scan Direction': scan_direction,
            'Scan Speed [V/s]': scan_speed,
            'Filepath': filepath,
            **analysis_outputs
        }
        
        return data
    
    def extract_value(self, content, pattern, convert_to_ma_cm2=False):
        match = re.search(pattern, content)
        if match:
            try:
                value = float(match.group(1))
                if "Efficiency" in pattern or "FF" in pattern:
                    return 100 * value
                elif convert_to_ma_cm2:
                    return value * 0.1  # Convert A/m2 to mA/cm2
                else:
                    return value
            except:
                return None
        return None
    
    def update_measurements_display(self):
        # Clear treeview
        for item in self.measurements_tree.get_children():
            self.measurements_tree.delete(item)
        
        # Repopulate with sorted data
        for _, row in self.measurements_data.iterrows():
            values = [row.get(col, '') for col in self.display_columns]
            self.measurements_tree.insert('', 'end', values=values)
    
    def remove_selected(self):
        selected_items = self.measurements_tree.selection()
        if not selected_items:
            return
        
        for item in selected_items:
            item_values = self.measurements_tree.item(item, 'values')
            filename = item_values[0]
            
            # Remove from both dataframes
            self.measurements_data = self.measurements_data[self.measurements_data['Filename'] != filename]
            self.measurements_data_original = self.measurements_data.copy()
            
            # Remove from treeview
            self.measurements_tree.delete(item)
    
    def add_condition(self):
        condition = self.condition_entry.get().strip()
        if not condition:
            messagebox.showwarning("Warning", "Please enter a condition name.")
            return
        
        # Check if condition already exists
        existing_conditions = self.get_conditions()
        if condition in existing_conditions:
            messagebox.showwarning("Warning", f"Condition '{condition}' already exists.")
            return
        
        # Add condition to combobox
        existing_conditions.append(condition)
        self.condition_combobox['values'] = existing_conditions
        self.condition_entry.delete(0, tk.END)
    
    def get_conditions(self):
        return list(self.condition_combobox['values']) if self.condition_combobox['values'] else []
    
    def assign_condition(self):
        selected_items = self.measurements_tree.selection()
        if not selected_items:
            messagebox.showwarning("Warning", "Please select one or more measurements.")
            return
        
        condition = self.condition_combobox.get()
        if not condition:
            messagebox.showwarning("Warning", "Please select a condition.")
            return
        
        # Get unique substrate IDs from selected items
        substrate_ids = set()
        for item in selected_items:
            item_values = self.measurements_tree.item(item, 'values')
            substrate_id = item_values[1]  # Substrate ID is the second column
            substrate_ids.add(substrate_id)
        
        # Determine next display order value
        next_order = 1
        if not self.conditions_data.empty:
            next_order = self.conditions_data['Display Order'].max() + 1 if not self.conditions_data.empty else 1
        
        # Update conditions data
        for substrate_id in substrate_ids:
            # Remove existing condition for this substrate
            self.conditions_data = self.conditions_data[self.conditions_data['Substrate ID'] != substrate_id]
            
            # Add new condition
            new_condition = pd.DataFrame({
                'Substrate ID': [substrate_id],
                'Condition': [condition],
                'Display Order': [next_order]
            })
            self.conditions_data = pd.concat([self.conditions_data, new_condition], ignore_index=True)
        
        # Update conditions tree
        self.update_conditions_tree()
    
    def update_conditions_tree(self):
        # Clear tree
        for item in self.conditions_tree.get_children():
            self.conditions_tree.delete(item)
        
        # Sort by display order
        sorted_conditions = self.conditions_data.sort_values(by='Display Order')
        
        # Add conditions to tree
        for _, row in sorted_conditions.iterrows():
            self.conditions_tree.insert('', 'end', values=(row['Substrate ID'], row['Condition'], row['Display Order']))
        
        # Update summary
        self.update_condition_summary()
    
    def update_condition_summary(self):
        """Update the condition summary display"""
        if self.conditions_data.empty:
            self.summary_label.config(text="No conditions assigned yet")
        else:
            # Get unique conditions and their counts
            condition_counts = self.conditions_data['Condition'].value_counts().sort_index()
            summary_text = "Unique conditions: " + ", ".join([f"{cond} ({count})" for cond, count in condition_counts.items()])
            self.summary_label.config(text=summary_text)
    
    def reorder_condition(self, direction):
        selected_items = self.conditions_tree.selection()
        if not selected_items or len(selected_items) != 1:
            messagebox.showwarning("Warning", "Please select exactly one condition to reorder.")
            return
        
        item = selected_items[0]
        item_values = self.conditions_tree.item(item, 'values')
        substrate_id = item_values[0]
        condition = item_values[1]
        current_order = int(item_values[2])
        
        # Find the condition to swap with
        target_order = current_order + direction
        
        # Check if target order is valid
        if target_order < 1 or target_order > len(self.conditions_tree.get_children()):
            return  # Nothing to swap with
        
        # Find the item to swap with
        target_item = None
        for tree_item in self.conditions_tree.get_children():
            tree_item_values = self.conditions_tree.item(tree_item, 'values')
            if int(tree_item_values[2]) == target_order:
                target_item = tree_item
                target_substrate_id = tree_item_values[0]
                target_condition = tree_item_values[1]
                break
        
        if target_item:
            # Swap orders in the dataframe
            self.conditions_data.loc[
                self.conditions_data['Substrate ID'] == substrate_id, 'Display Order'
            ] = target_order
            
            self.conditions_data.loc[
                self.conditions_data['Substrate ID'] == target_substrate_id, 'Display Order'
            ] = current_order
            
            # Update the treeview
            self.update_conditions_tree()
    
    def sort_measurements_by_column(self, col):
        # Determine sort direction
        if self.last_sorted_column == col:
            # Same column clicked - toggle direction
            descending = not self.last_sort_descending
        else:
            # New column clicked - start with ascending
            descending = False
        
        # Update tracking variables
        self.last_sorted_column = col
        self.last_sort_descending = descending
        
        # Create a sorted copy of the original data for display
        try:
            # For numerical columns, ensure proper numeric sorting
            if col not in ['Filename', 'Substrate ID', 'Pixel', 'Scan Direction']:
                # Convert column to numeric, handling any non-numeric values
                self.measurements_data[col] = pd.to_numeric(self.measurements_data[col], errors='coerce')
            
            # Sort the data using pandas (creates a copy, doesn't modify original)
            sorted_data = self.measurements_data_original.sort_values(by=col, ascending=not descending, na_position='last')
            
            # Update the display data (this is what gets shown)
            self.measurements_data = sorted_data.copy()
            
            # Update the display
            self.update_measurements_display()

        except Exception as e:
            # Fallback to string sorting if numeric sorting fails
            sorted_data = self.measurements_data_original.sort_values(by=col, ascending=not descending, na_position='last', key=lambda x: x.astype(str))
            self.measurements_data = sorted_data.copy()
            self.update_measurements_display()

        # Update column headings to show sort direction
        for column in self.measurements_tree['columns']:
            current_text = self.measurements_tree.heading(column)['text']
            clean_text = current_text.replace(" ▲", "").replace(" ▼", "")
            if column == col:
                sort_indicator = " ▼" if descending else " ▲"
                self.measurements_tree.heading(column, text=clean_text + sort_indicator)
            else:
                self.measurements_tree.heading(column, text=clean_text)
    
    def on_xorder_change(self, event=None):
        """Handle x-axis ordering combobox selection change"""
        selected_order = self.xorder_combobox.get()
        if selected_order == "Custom":
            # Show custom order interface and populate it
            self.populate_conditions_listbox()
            self.custom_order_frame.pack(fill="both", expand=True, pady=(10, 0))
        else:
            # Hide custom order interface
            self.custom_order_frame.pack_forget()
    
    def populate_conditions_listbox(self, custom_order=None):
        """Populate the conditions listbox with current conditions"""
        self.conditions_listbox.delete(0, tk.END)
        
        # Get unique conditions from the data
        if not self.measurements_data.empty and not self.conditions_data.empty:
            # Get conditions that have associated measurements
            plot_data = self.measurements_data.merge(
                self.conditions_data, 
                on='Substrate ID', 
                how='inner'
            )
            
            if not plot_data.empty:
                unique_conditions = plot_data['Condition'].unique()
                
                if custom_order:
                    # Use provided custom order, then add any missing conditions
                    for condition in custom_order:
                        if condition in unique_conditions:
                            self.conditions_listbox.insert(tk.END, condition)
                    # Add any remaining conditions not in custom order
                    for condition in unique_conditions:
                        if condition not in custom_order:
                            self.conditions_listbox.insert(tk.END, condition)
                else:
                    # Try to auto-load saved custom order
                    auto_order = self.auto_load_custom_order()
                    if auto_order:
                        # Use auto-loaded order
                        for condition in auto_order:
                            if condition in unique_conditions:
                                self.conditions_listbox.insert(tk.END, condition)
                        # Add any remaining conditions not in auto order
                        for condition in unique_conditions:
                            if condition not in auto_order:
                                self.conditions_listbox.insert(tk.END, condition)
                    else:
                        # Start with default order (Display Order)
                        ordered_conditions = self.conditions_data.sort_values(by='Display Order')
                        
                        # Add conditions in display order
                        for condition in ordered_conditions['Condition']:
                            if condition in unique_conditions:
                                self.conditions_listbox.insert(tk.END, condition)
    
    def move_condition_up(self):
        """Move selected condition up in the list"""
        selection = self.conditions_listbox.curselection()
        if not selection or selection[0] == 0:
            return  # Nothing selected or already at top
        
        index = selection[0]
        condition = self.conditions_listbox.get(index)
        
        # Remove and reinsert at previous position
        self.conditions_listbox.delete(index)
        self.conditions_listbox.insert(index - 1, condition)
        self.conditions_listbox.selection_set(index - 1)
        self.auto_save_custom_order()
    
    def move_condition_down(self):
        """Move selected condition down in the list"""
        selection = self.conditions_listbox.curselection()
        if not selection or selection[0] == self.conditions_listbox.size() - 1:
            return  # Nothing selected or already at bottom
        
        index = selection[0]
        condition = self.conditions_listbox.get(index)
        
        # Remove and reinsert at next position
        self.conditions_listbox.delete(index)
        self.conditions_listbox.insert(index + 1, condition)
        self.conditions_listbox.selection_set(index + 1)
        self.auto_save_custom_order()
    
    def reset_condition_order(self):
        """Reset the condition order to default (Display Order)"""
        self.populate_conditions_listbox()
    
    def on_condition_double_click(self, event):
        """Move double-clicked condition to the top of the list"""
        selection = self.conditions_listbox.curselection()
        if not selection:
            return
        
        index = selection[0]
        if index == 0:
            return  # Already at top
        
        condition = self.conditions_listbox.get(index)
        self.conditions_listbox.delete(index)
        self.conditions_listbox.insert(0, condition)
        self.conditions_listbox.selection_set(0)
        self.auto_save_custom_order()
    
    def auto_save_custom_order(self):
        """Auto-save current custom order to temporary file"""
        try:
            if hasattr(self, 'conditions_listbox'):
                custom_order = [self.conditions_listbox.get(i) for i in range(self.conditions_listbox.size())]
                with open(self.auto_save_file, 'w') as f:
                    json.dump({"custom_order": custom_order}, f)
        except Exception:
            pass  # Ignore errors in auto-save
    
    def auto_load_custom_order(self):
        """Auto-load custom order from temporary file if it exists"""
        try:
            if os.path.exists(self.auto_save_file):
                with open(self.auto_save_file, 'r') as f:
                    data = json.load(f)
                return data.get("custom_order", [])
        except Exception:
            pass  # Ignore errors in auto-load
        return []
    
    def get_condition_order(self, plot_data):
        """Determine the order of conditions for plotting based on user selection"""
        selected_order = self.xorder_combobox.get()
        unique_conditions = plot_data['Condition'].unique()
        
        if selected_order == "Alphabetical":
            return sorted(unique_conditions)
        elif selected_order == "Custom":
            # Get order from listbox
            listbox_order = [self.conditions_listbox.get(i) for i in range(self.conditions_listbox.size())]
            # Filter to only include conditions that exist in the data
            valid_custom_order = [c for c in listbox_order if c in unique_conditions]
            # Add any remaining conditions not specified in custom order
            remaining_conditions = [c for c in unique_conditions if c not in valid_custom_order]
            return valid_custom_order + sorted(remaining_conditions)
        
        # Default: Display Order (from conditions_data)
        if not self.conditions_data.empty:
            # Sort by display order
            ordered_conditions = self.conditions_data.sort_values(by='Display Order')
            # Filter to only include conditions that exist in the plot data
            return [c for c in ordered_conditions['Condition'].tolist() if c in unique_conditions]
        else:
            # Fallback to alphabetical if no conditions data
            return sorted(unique_conditions)
    
    def generate_plot(self):
        if self.measurements_data.empty or self.conditions_data.empty:
            messagebox.showwarning("Warning", "No data available for plotting.")
            return
        
        selected_param = self.plot_param_combobox.get()
        if not selected_param:
            return
        
        # Prepare data for plotting
        plot_data = self.measurements_data.merge(
            self.conditions_data, 
            on='Substrate ID', 
            how='inner'
        )
        
        if plot_data.empty:
            messagebox.showwarning("Warning", "No data available after joining measurements with conditions.")
            return
        
        # Clear previous plot
        for widget in self.condition_plot_frame.winfo_children():
            widget.destroy()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Get the desired condition order
        condition_order = self.get_condition_order(plot_data)
        
        # Get selected color palette
        palette_name = self.color_palette_combobox.get()
        palette = self.color_palettes[palette_name]
        
        # Create box plot with the selected palette and custom order
        sns.boxplot(data=plot_data, x='Condition', y=selected_param, ax=ax, palette=palette, order=condition_order)
        sns.stripplot(data=plot_data, x='Condition', y=selected_param, color='black', size=5, alpha=0.6, ax=ax, order=condition_order)
        
        # Set y-axis limits if provided
        try:
            ymin = self.ymin_entry.get().strip()
            ymax = self.ymax_entry.get().strip()
            
            if ymin and ymax:
                ax.set_ylim(float(ymin), float(ymax))
            elif ymin:
                ax.set_ylim(bottom=float(ymin))
            elif ymax:
                ax.set_ylim(top=float(ymax))
        except ValueError:
            messagebox.showwarning("Warning", "Invalid y-axis limits. Using auto limits.")
        
        ax.set_title(f'{selected_param} by Condition')
        ax.set_xlabel('Condition')
        ax.set_ylabel(selected_param, fontsize=14)
        
        # Add horizontal grid lines only
        ax.grid(True, axis='y', alpha=0.5, linestyle='--')
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=90, fontsize=15)
        
        # Tight layout to ensure everything fits
        plt.tight_layout()
        
        # Save the figure reference and plot data for later use
        self.current_figure = fig
        self.current_param = selected_param
        self.current_plot_data = plot_data.copy()
        
        # Embed plot in tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.condition_plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def save_plot(self):
        if self.current_figure is None:
            messagebox.showwarning("Warning", "No plot to save. Please generate a plot first.")
            return
        
        # Get the save file path
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg"),
                ("PDF files", "*.pdf"),
                ("SVG files", "*.svg"),
                ("All files", "*.*")
            ],
            title="Save Plot As"
        )
        
        if not file_path:
            return  # User cancelled
        
        try:
            # Save the figure
            self.current_figure.savefig(file_path, bbox_inches='tight', dpi=300)
            messagebox.showinfo("Success", f"Plot saved successfully to:\n{file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save plot: {str(e)}")

    def export_condition_data(self):
        """Export the current condition plot data to Excel format"""
        if self.current_plot_data is None or self.current_plot_data.empty:
            messagebox.showwarning("Warning", "No plot data to export. Please generate a plot first.")
            return

        # Get the save file path
        file_path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[
                ("Excel files", "*.xlsx"),
                ("CSV files", "*.csv"),
                ("All files", "*.*")
            ],
            title="Export Condition Data As"
        )

        if not file_path:
            return  # User cancelled

        try:
            # Create a copy of the data for export
            export_data = self.current_plot_data.copy()

            # Get the condition order used in the plot
            condition_order = self.get_condition_order(export_data)

            # Sort data by condition order for better readability
            export_data['Condition_Order'] = export_data['Condition'].map(
                {cond: idx for idx, cond in enumerate(condition_order)}
            )
            export_data = export_data.sort_values(['Condition_Order', 'Substrate ID'])
            export_data = export_data.drop('Condition_Order', axis=1)

            # Select relevant columns for export (excluding internal columns)
            columns_to_export = [
                'Substrate ID', 'Condition', 'Pixel', 'Scan Direction',
                'Voc [V]', 'Jsc [mA/cm2]', 'FF [.]', 'Efficiency [.]',
                'Pmpp [W/m2]', 'Vmpp [V]', 'Jmpp [mA/cm2]',
                'Roc [Ohm.m2]', 'Rsc [Ohm.m2]', 'Scan Speed [V/s]'
            ]

            # Filter to only include columns that exist in the data
            available_columns = [col for col in columns_to_export if col in export_data.columns]
            export_data_filtered = export_data[available_columns]

            # Export based on file extension
            if file_path.lower().endswith('.csv'):
                export_data_filtered.to_csv(file_path, index=False)
            else:
                # Default to Excel format (requires openpyxl: pip install openpyxl)
                export_data_filtered.to_excel(file_path, index=False, sheet_name='Condition Data')

            # Show success message with summary
            total_measurements = len(export_data_filtered)
            unique_conditions = export_data_filtered['Condition'].nunique()
            messagebox.showinfo(
                "Success",
                f"Data exported successfully to:\n{file_path}\n\n"
                f"Exported {total_measurements} measurements across {unique_conditions} conditions."
            )

        except ImportError as e:
            if "openpyxl" in str(e).lower():
                messagebox.showerror(
                    "Missing Dependency",
                    "Excel export requires the 'openpyxl' package.\n\n"
                    "Please install it using:\n"
                    "pip install openpyxl\n\n"
                    "Alternatively, you can export as CSV format."
                )
            else:
                messagebox.showerror("Import Error", f"Failed to export data: {str(e)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export data: {str(e)}")

    def setup_iv_plot(self, parent):
        control_frame = ttk.Frame(parent)
        control_frame.pack(fill="x", padx=10, pady=5)
    
        browse_button = ttk.Button(control_frame, text="Load IV File", command=self.load_iv_data)
        browse_button.pack(side="left", padx=5, pady=5)

        ttk.Label(control_frame, text=" or select from loaded measurements:").pack(side="left", padx=6)
        list_frame = ttk.Frame(control_frame)
        list_frame.pack(side="left", padx=5)
        self.iv_listbox = tk.Listbox(list_frame, selectmode=tk.EXTENDED, height=12, width=90)
        vsb = ttk.Scrollbar(list_frame, orient="vertical", command=self.iv_listbox.yview)
        self.iv_listbox.configure(yscrollcommand=vsb.set)
        self.refresh_iv_selection()
        self.iv_listbox.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)
        select_btn = ttk.Button(control_frame, text="Plot Selected", command=self.load_iv_from_selection)
        select_btn.pack(side="left", padx=5)
        pair_btn = ttk.Button(control_frame, text="Pair FW↔RV (auto)", command=self.auto_pair_and_plot)
        pair_btn.pack(side="left", padx=5)
        pair_sel_btn = ttk.Button(control_frame, text="Pair FW↔RV (selection)", command=self.auto_pair_selection_and_plot)
        pair_sel_btn.pack(side="left", padx=5)
        save_btn = ttk.Button(control_frame, text="Save Plot", command=self.save_current_plot)
        save_btn.pack(side="left", padx=8)

        # --- Axis controls row ---
        axes_frame = ttk.Frame(parent)
        axes_frame.pack(fill="x", padx=10, pady=6)
        ttk.Label(axes_frame, text="Axes:").pack(side="left", padx=(0,6))
        self.iv_xmin = tk.StringVar(); self.iv_xmax = tk.StringVar(); self.iv_ymin = tk.StringVar(); self.iv_ymax = tk.StringVar()
        ttk.Label(axes_frame, text="V min").pack(side="left"); ttk.Entry(axes_frame, textvariable=self.iv_xmin, width=8).pack(side="left", padx=3)
        ttk.Label(axes_frame, text="V max").pack(side="left"); ttk.Entry(axes_frame, textvariable=self.iv_xmax, width=8).pack(side="left", padx=8)
        ttk.Label(axes_frame, text="I min").pack(side="left"); ttk.Entry(axes_frame, textvariable=self.iv_ymin, width=10).pack(side="left", padx=3)
        ttk.Label(axes_frame, text="I max").pack(side="left"); ttk.Entry(axes_frame, textvariable=self.iv_ymax, width=10).pack(side="left", padx=8)

        # Area control for current density conversion
        ttk.Label(axes_frame, text="Area [cm²]").pack(side="left", padx=(15,3))
        self.iv_area = tk.StringVar(value="0.04")
        ttk.Entry(axes_frame, textvariable=self.iv_area, width=8).pack(side="left", padx=3)

        ttk.Button(axes_frame, text="Apply axes", command=self.apply_iv_axes).pack(side="left", padx=6)
        ttk.Button(axes_frame, text="Autoscale", command=self.reset_iv_axes).pack(side="left", padx=2)

        
        # Sort controls
        sort_frame = ttk.Frame(parent)
        sort_frame.pack(fill="x", padx=10, pady=4)
        ttk.Label(sort_frame, text="Sort selection list:").pack(side="left")
        ttk.Button(sort_frame, text="Voc ↑", command=lambda: self.sort_iv_selection_by_voc(True)).pack(side="left", padx=4)
        ttk.Button(sort_frame, text="Voc ↓", command=lambda: self.sort_iv_selection_by_voc(False)).pack(side="left", padx=4)

        self.iv_plot_frame = ttk.Frame(parent)
        self.iv_plot_frame.pack(fill="both", expand=True, padx=10, pady=10)
    
    def load_iv_data(self):
        file_path = filedialog.askopenfilename(filetypes=[("IV Files", "*.iv"), ("All Files", "*.*")])
        if not file_path:
            return
    
        try:
            data = self.parse_iv_data_for_plot(file_path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to parse IV file: {str(e)}")
        try:
            self.plot_iv_curve(data)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to plot IV file: {str(e)}")
            
    
    def parse_iv_data_for_plot(self, filepath):
        with open(filepath, 'r') as f:
            content = f.read()

        # Extract sweep metadata to compute scan speed (V/s)
        vstart = self.extract_value(content, r'Vstart:\s*([-\d\.]+E[+-]\d+|[-\d\.]+)')
        vend = self.extract_value(content, r'Vend:\s*([-\d\.]+E[+-]\d+|[-\d\.]+)')
        npoints = None
        m_np = re.search(r'Number of points:\s*([-\d\.]+E[+-]\d+|[-\d\.]+)', content)
        if m_np:
            try:
                npoints = int(float(m_np.group(1)))
            except:
                npoints = None
        delay_s = self.extract_value(content, r'Delay \[s\]:\s*([-\d\.]+E[+-]\d+|[-\d\.]+)')
        it_s = self.extract_value(content, r'Integration time \[s\]:\s*([-\d\.]+E[+-]\d+|[-\d\.]+)')
        scan_speed = None
        try:
            if vstart is not None and vend is not None and npoints and npoints > 1 and (delay_s is not None) and (it_s is not None):
                step_time = (delay_s + it_s)
                total_time = (npoints - 1) * step_time
                if total_time > 0:
                    scan_speed = abs(vend - vstart) / total_time
        except Exception:
            scan_speed = None

    
        data_section = re.search(    r'% MEASURED IV FRLOOP DATA\s*\nV \(measured\) \[V\]\s+I \(measured\) \[A\].*?\n(.*)', content, re.DOTALL)

        if not data_section:
            raise ValueError("Could not find IV data in file.")
    
        data_lines = data_section.group(1).strip().split('\n')
        voltages, currents = [], []
    
        for line in data_lines:
            values = line.split('\t')
            if len(values) >= 2:
                try:
                    voltages.append(float(values[0]))
                    currents.append(float(values[1]))
                except ValueError:
                    continue
    
        return pd.DataFrame({'Voltage (V)': voltages, 'Current (A)': currents})

    def convert_to_current_density(self, current_data):
        """Convert current (A) to current density (mA/cm²) using the area from UI"""
        try:
            area_cm2 = float(self.iv_area.get())
            if area_cm2 <= 0:
                return current_data  # Return original if invalid area
            # Convert A to mA/cm²: (A * 1000 mA/A) / (area in cm²)
            return current_data * 1000 / area_cm2
        except (ValueError, AttributeError):
            return current_data  # Return original if area parsing fails
    
    def plot_iv_curve(self, data):
        for widget in self.iv_plot_frame.winfo_children():
            widget.destroy()

        fig, ax = plt.subplots(figsize=(7, 6))
        # Convert current to current density
        current_density = self.convert_to_current_density(data['Current (A)'])
        ax.plot(data['Voltage (V)'], current_density, marker='o', linestyle='-')
        ax.set_title("IV Curve")
        ax.set_xlabel("Voltage (V)", fontsize=12)
        ax.set_ylabel("Current Density (mA/cm²)", fontsize=12)
        ax.grid(True)
    
        canvas = FigureCanvasTkAgg(fig, master=self.iv_plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


    def refresh_iv_selection(self):
        """Fill the IV selection list from the first tab table, including conditions, respecting sort order."""
        try:
            if hasattr(self, 'iv_listbox'):
                self.iv_listbox.delete(0, tk.END)
                # Build condition map
                cond_map = {}
                try:
                    if hasattr(self, 'conditions_data') and not self.conditions_data.empty:
                        for _, r in self.conditions_data.iterrows():
                            cond_map[str(r.get('Substrate ID',''))] = r.get('Condition','')
                except Exception:
                    cond_map = {}
                if 'Filename' in self.measurements_data.columns:
                    df = self.measurements_data.copy()
                    df = self._apply_iv_sort(df)
                    for _, row in df.iterrows():
                        fn = row.get('Filename', '')
                        subid = str(row.get('Substrate ID', ''))
                        pix = row.get('Pixel', '')
                        sd = row.get('Scan Direction', '')
                        cond = cond_map.get(subid, '—')
                        disp = f"{fn}  |  {subid}  |  Pixel {pix}  |  {sd}  |  {cond}"
                        self.iv_listbox.insert(tk.END, disp)
        except Exception:
            pass


    def load_iv_from_selection(self):
        """Parse and plot IV curve(s) for the selected entries from the first tab."""
        try:
            indices = self.iv_listbox.curselection()
        except Exception:
            indices = []
        if not indices:
            messagebox.showinfo("Select file(s)", "Please choose one or more measurements from the list.")
            return
        
        filepaths = []
        for idx in indices:
            sel = self.iv_listbox.get(idx)
            filename = sel.split('  |')[0].strip()
            if 'Filepath' in self.measurements_data.columns:
                matches = self.measurements_data[self.measurements_data['Filename'] == filename]
                if not matches.empty:
                    fp = matches.iloc[0].get('Filepath', None)
                    if fp: filepaths.append(fp)
        if not filepaths:
            messagebox.showerror("Not found", "Could not locate file paths for the selections.")
            return
        
        curves = []
        for fp in filepaths:
            try:
                df = self.parse_iv_data_for_plot(fp)
                curves.append((fp, df))
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load {os.path.basename(fp)}: {e}")
        if not curves:
            return
        self.plot_iv_curves_overlaid(curves)

    def _split_fw_rv_if_present(self, df):
        """Split a single IV sweep into FW/RV if a voltage direction reversal is found.
        Returns a list of (label_suffix, sub_df). If no reversal, returns [("", df)]."""
        v = df['Voltage (V)'].values
        if len(v) < 3:
            return [("", df)]
        dir0 = 1 if v[1] >= v[0] else -1
        change_idx = None
        for i in range(2, len(v)):
            d = 1 if v[i] >= v[i-1] else -1
            if d != dir0:
                change_idx = i-1
                break
        if change_idx is None:
            return [("", df)]
        df1 = df.iloc[:change_idx+1].reset_index(drop=True)
        df2 = df.iloc[change_idx+1:].reset_index(drop=True)
        # Label segments
        lab1 = " (FW)" if df1['Voltage (V)'].iloc[-1] > df1['Voltage (V)'].iloc[0] else " (RV)"
        lab2 = " (FW)" if df2['Voltage (V)'].iloc[-1] > df2['Voltage (V)'].iloc[0] else " (RV)"
        return [(lab1, df1), (lab2, df2)]

        def plot_iv_curves_overlaid(self, curves):
            """Overlay arbitrary curves; if a single file contains FW+RV, use split; style as FW/RV with pairwise colors."""
            for w in self.iv_plot_frame.winfo_children():
                w.destroy()
            fig, ax = plt.subplots(figsize=(8, 6))

            # Build condition map
            cond_map = {}
            try:
                if hasattr(self, 'conditions_data') and not self.conditions_data.empty:
                    for _, r in self.conditions_data.iterrows():
                        cond_map[str(r.get('Substrate ID',''))] = r.get('Condition','')
            except Exception:
                cond_map = {}

            import os
            for i, (fp, df) in enumerate(curves):
                parts = self._split_fw_rv_if_present(df)
                subid = None
                try:
                    if 'Filepath' in self.measurements_data.columns:
                        match = self.measurements_data[self.measurements_data['Filepath'] == fp]
                        if not match.empty:
                            subid = str(match.iloc[0].get('Substrate ID',''))
                except Exception:
                    subid = None
                cond = cond_map.get(subid or '', '—')
                c_fw, c_rv = self._pair_colors(i)
                for suf, part in parts:
                    # crude check for RV segment
                    is_rv = ('RV' in suf) or ('rv' in suf)
                    label = f"{subid or ''} | {cond} | {'RV' if is_rv else 'FW'}"
                    if is_rv:
                        ax.plot(part['Voltage (V)'], part['Current (A)'], linestyle='--', color=c_rv, alpha=0.5, label=label)
                    else:
                        ax.plot(part['Voltage (V)'], part['Current (A)'], linestyle='-', color=c_fw, label=label)

            ax.set_title("IV Curves (overlay)")
            ax.set_xlabel("Voltage (V)")
            ax.set_ylabel("Current (A)")
            ax.grid(True)
            ax.legend()
            canvas = FigureCanvasTkAgg(fig, master=self.iv_plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def sort_iv_selection_by_voc(self, ascending=True):
        """Sort the selection listbox by Voc [V]."""
        try:
            self._iv_sort_key = ('voc', ascending)
        except Exception:
            self._iv_sort_key = ('voc', ascending)
        self.refresh_iv_selection()

    def _apply_iv_sort(self, df):
        """Internal: apply current sort to the measurements_data for the selection list."""
        key = getattr(self, '_iv_sort_key', None)
        if not key:
            return df
        kind, asc = key
        try:
            if kind == 'voc' and 'Voc [V]' in df.columns:
                # Convert to numeric to be safe
                dfx = df.copy()
                dfx['__voc__'] = pd.to_numeric(dfx['Voc [V]'], errors='coerce')
                dfx = dfx.sort_values(by='__voc__', ascending=asc, na_position='last')
                return dfx
        except Exception:
            return df
        return df

        def _get_color_cycle(self):
            import matplotlib.pyplot as plt
            return plt.rcParams.get('axes.prop_cycle', None).by_key().get('color', ['C0','C1','C2','C3','C4','C5']) if plt.rcParams.get('axes.prop_cycle', None) else ['C0','C1','C2','C3','C4','C5']

        def _pair_colors(self, idx):
            colors = self._get_color_cycle()
            base = colors[idx % len(colors)]
            return base, base

    def auto_pair_and_plot(self):
        """Pair FW/RV across ALL loaded data and plot (FW solid, RV dashed; RV alpha=0.5)."""
        needed = {'Substrate ID','Pixel','Scan Direction','Filepath','Filename'}
        if not needed.issubset(set(self.measurements_data.columns)):
            messagebox.showwarning("Missing info", "Required columns missing; please load files in the first tab.")
            return

        df = self.measurements_data.copy()
        df['__dir__'] = df['Scan Direction'].astype(str).str.lower().replace({'fw':'fwd','forward':'fwd','rev':'rev','reverse':'rev'})

        cond_map = {}
        try:
            if hasattr(self, 'conditions_data') and not self.conditions_data.empty:
                for _, r in self.conditions_data.iterrows():
                    cond_map[str(r.get('Substrate ID',''))] = r.get('Condition','')
        except Exception:
            cond_map = {}

        pairs = []
        for (subid, pix), g in df.groupby(['Substrate ID','Pixel']):
            fwd_rows = g[g['__dir__']=='fwd']
            rev_rows = g[g['__dir__']=='rev']
            if not fwd_rows.empty and not rev_rows.empty:
                pairs.append((fwd_rows.iloc[0], rev_rows.iloc[0]))
            else:
                def guess_dir(fn):
                    s = str(fn).lower()
                    if any(t in s for t in ['_fw','-fw',' fw','(fw)','_fwd','-fwd',' fwd']): return 'fwd'
                    if any(t in s for t in ['_rv','-rv',' rv','(rv)','_rev','-rev',' rev']): return 'rev'
                    return None
                g2 = g.copy()
                g2['__guess__'] = g2['Filename'].apply(guess_dir)
                fwd_rows = g2[g2['__guess__']=='fwd']
                rev_rows = g2[g2['__guess__']=='rev']
                if not fwd_rows.empty and not rev_rows.empty:
                    pairs.append((fwd_rows.iloc[0], rev_rows.iloc[0]))

        if not pairs:
            messagebox.showinfo("No pairs found", "No FW/RV pairs detected.")
            return

        for w in self.iv_plot_frame.winfo_children():
            w.destroy()
        fig, ax = plt.subplots(figsize=(7, 6))
        self._last_ax = ax
        self._maybe_apply_axes(ax)

        # Get color cycle
        try:
            colors = plt.rcParams['axes.prop_cycle'].by_key().get('color', [])
        except Exception:
            colors = []

        for i, (fw_row, rv_row) in enumerate(pairs):
            subid = str(fw_row['Substrate ID'])
            cond = cond_map.get(subid, '—')
            base_color = colors[i % len(colors)] if colors else None
            # FWD
            try:
                df_fw = self.parse_iv_data_for_plot(fw_row['Filepath'])
                current_density_fw = self.convert_to_current_density(df_fw['Current (A)'])
                ax.plot(df_fw['Voltage (V)'], current_density_fw, linestyle='-', color=base_color, label=f"{subid} | {cond} | FW")
            except Exception as e:
                messagebox.showerror("Error", f"Failed FW: {e}")
            # REV
            try:
                df_rv = self.parse_iv_data_for_plot(rv_row['Filepath'])
                current_density_rv = self.convert_to_current_density(df_rv['Current (A)'])
                ax.plot(df_rv['Voltage (V)'], current_density_rv, linestyle='--', color=base_color, alpha=0.5, label=f"{subid} | {cond} | RV")
            except Exception as e:
                messagebox.showerror("Error", f"Failed RV: {e}")

        ax.set_title("IV Curves – Auto-paired FW/RV")
        ax.set_xlabel("Voltage (V)", fontsize=12)
        ax.set_ylabel("Current Density (mA/cm²)", fontsize=12)
        ax.grid(True)
        ax.legend()
        canvas = FigureCanvasTkAgg(fig, master=self.iv_plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    def auto_pair_selection_and_plot(self):
        """Pair FW/RV ONLY for selected items."""
        needed = {'Substrate ID','Pixel','Scan Direction','Filepath','Filename'}
        if not needed.issubset(set(self.measurements_data.columns)):
            messagebox.showwarning("Missing info", "Required columns missing; please load files in the first tab.")
            return

        try:
            indices = self.iv_listbox.curselection()
        except Exception:
            indices = []
        if not indices:
            messagebox.showinfo("Select file(s)", "Please select at least one measurement in the list.")
            return

        df = self.measurements_data.copy()
        df['__dir__'] = df['Scan Direction'].astype(str).str.lower().replace({'fw':'fwd','forward':'fwd','rev':'rev','reverse':'rev'})

        cond_map = {}
        try:
            if hasattr(self, 'conditions_data') and not self.conditions_data.empty:
                for _, r in self.conditions_data.iterrows():
                    cond_map[str(r.get('Substrate ID',''))] = r.get('Condition','')
        except Exception:
            cond_map = {}

        def guess_dir(fn):
            s = str(fn).lower()
            if any(t in s for t in ['_fw','-fw',' fw','(fw)','_fwd','-fwd',' fwd']): return 'fwd'
            if any(t in s for t in ['_rv','-rv',' rv','(rv)','_rev','-rev',' rev']): return 'rev'
            return None

        def rows_for(subid, pix, direction):
            g = df[(df['Substrate ID']==subid) & (df['Pixel']==pix)]
            g1 = g[g['__dir__']==direction]
            if not g1.empty:
                return g1
            g2 = g.copy()
            g2['__guess__'] = g2['Filename'].apply(guess_dir)
            return g2[g2['__guess__']==direction]

        selected_rows = []
        for idx in indices:
            disp = self.iv_listbox.get(idx)
            fname = disp.split('  |')[0].strip()
            r = df[df['Filename']==fname]
            if not r.empty:
                selected_rows.append(r.iloc[0])

        pairs = []
        for r in selected_rows:
            subid, pix = r['Substrate ID'], r['Pixel']
            d = r['__dir__']
            if d not in ['fwd','rev']:
                d = guess_dir(r['Filename']) or 'fwd'
            other = 'rev' if d=='fwd' else 'fwd'
            cand = rows_for(subid, pix, other)
            if not cand.empty:
                o = cand.iloc[0]
                if d=='fwd':
                    pairs.append((r, o))
                else:
                    pairs.append((o, r))

        if not pairs:
            messagebox.showinfo("No pairs found", "No FW/RV pairs found for the selection.")
            return

        for w in self.iv_plot_frame.winfo_children():
            w.destroy()
        fig, ax = plt.subplots(figsize=(7, 6))
        self._last_ax = ax
        self._maybe_apply_axes(ax)

        try:
            colors = plt.rcParams['axes.prop_cycle'].by_key().get('color', [])
        except Exception:
            colors = []

        for i, (fw_row, rv_row) in enumerate(pairs):
            subid = str(fw_row['Substrate ID'])
            cond = cond_map.get(subid, '—')
            base_color = colors[i % len(colors)] if colors else None
            try:
                df_fw = self.parse_iv_data_for_plot(fw_row['Filepath'])
                current_density_fw = self.convert_to_current_density(df_fw['Current (A)'])
                ax.plot(df_fw['Voltage (V)'], current_density_fw, linestyle='-', color=base_color, label=f"{subid} | {cond} | FW")
            except Exception as e:
                messagebox.showerror("Error", f"Failed FW: {e}")
            try:
                df_rv = self.parse_iv_data_for_plot(rv_row['Filepath'])
                current_density_rv = self.convert_to_current_density(df_rv['Current (A)'])
                ax.plot(df_rv['Voltage (V)'], current_density_rv, linestyle='--', color=base_color, alpha=0.5, label=f"{subid} | {cond} | RV")
            except Exception as e:
                messagebox.showerror("Error", f"Failed RV: {e}")

        ax.set_title("IV Curves – Auto-paired FW/RV (Selection)")
        ax.set_xlabel("Voltage (V)", fontsize=12)
        ax.set_ylabel("Current Density (mA/cm²)", fontsize=12)
        ax.grid(True)
        ax.legend()
        canvas = FigureCanvasTkAgg(fig, master=self.iv_plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    def plot_iv_curves_overlaid(self, curves):
        """Overlay arbitrary curves; if a single file contains FW+RV, split; FW solid, RV dashed, RV alpha=0.5."""
        for w in self.iv_plot_frame.winfo_children():
            w.destroy()
        fig, ax = plt.subplots(figsize=(7, 6))
        self._last_ax = ax
        self._maybe_apply_axes(ax)

        # Build condition map
        cond_map = {}
        try:
            if hasattr(self, 'conditions_data') and not self.conditions_data.empty:
                for _, r in self.conditions_data.iterrows():
                    cond_map[str(r.get('Substrate ID',''))] = r.get('Condition','')
        except Exception:
            cond_map = {}

        import os
        try:
            colors = plt.rcParams['axes.prop_cycle'].by_key().get('color', [])
        except Exception:
            colors = []
        for i, (fp, df) in enumerate(curves):
            parts = self._split_fw_rv_if_present(df)
            subid = None
            try:
                if 'Filepath' in self.measurements_data.columns:
                    match = self.measurements_data[self.measurements_data['Filepath'] == fp]
                    if not match.empty:
                        subid = str(match.iloc[0].get('Substrate ID',''))
            except Exception:
                subid = None
            cond = cond_map.get(subid or '', '—')
            base_color = colors[i % len(colors)] if colors else None
            for suf, part in parts:
                is_rv = ('RV' in suf) or ('rv' in suf)
                label = f"{subid or ''} | {cond} | {'RV' if is_rv else 'FW'}"
                current_density = self.convert_to_current_density(part['Current (A)'])
                if is_rv:
                    ax.plot(part['Voltage (V)'], current_density, linestyle='--', color=base_color, alpha=0.5, label=label)
                else:
                    ax.plot(part['Voltage (V)'], current_density, linestyle='-', color=base_color, label=label)

        ax.set_title("IV Curves (overlay)")
        ax.set_xlabel("Voltage (V)", fontsize=12)
        ax.set_ylabel("Current Density (mA/cm²)", fontsize=12)
        ax.grid(True)
        ax.legend()
        canvas = FigureCanvasTkAgg(fig, master=self.iv_plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _maybe_apply_axes(self, ax):
        try:
            xmin = self.iv_xmin.get().strip() if hasattr(self, 'iv_xmin') else ''
            xmax = self.iv_xmax.get().strip() if hasattr(self, 'iv_xmax') else ''
            ymin = self.iv_ymin.get().strip() if hasattr(self, 'iv_ymin') else ''
            ymax = self.iv_ymax.get().strip() if hasattr(self, 'iv_ymax') else ''
        except Exception:
            xmin = xmax = ymin = ymax = ''
        def to_float(s):
            try: return float(s)
            except: return None
        x0, x1 = to_float(xmin), to_float(xmax)
        y0, y1 = to_float(ymin), to_float(ymax)
        if x0 is not None or x1 is not None:
            cur = list(ax.get_xlim())
            if x0 is not None: cur[0] = x0
            if x1 is not None: cur[1] = x1
            ax.set_xlim(cur)
        if y0 is not None or y1 is not None:
            cur = list(ax.get_ylim())
            if y0 is not None: cur[0] = y0
            if y1 is not None: cur[1] = y1
            ax.set_ylim(cur)

    def apply_iv_axes(self):
        try:
            if hasattr(self, '_last_ax') and self._last_ax:
                self._maybe_apply_axes(self._last_ax)
                self._last_ax.figure.canvas.draw_idle()
        except Exception:
            pass

    def reset_iv_axes(self):
        try:
            if hasattr(self, 'iv_xmin'): self.iv_xmin.set('')
            if hasattr(self, 'iv_xmax'): self.iv_xmax.set('')
            if hasattr(self, 'iv_ymin'): self.iv_ymin.set('')
            if hasattr(self, 'iv_ymax'): self.iv_ymax.set('')
        except Exception:
            pass
        try:
            if hasattr(self, '_last_ax') and self._last_ax:
                self._last_ax.relim(); self._last_ax.autoscale(); self._last_ax.figure.canvas.draw_idle()
        except Exception:
            pass


    def save_current_plot(self):
        """Save the current IV plot to PNG/SVG/PDF."""
        try:
            ax = getattr(self, '_last_ax', None)
            fig = ax.figure if ax is not None else None
        except Exception:
            fig = None
        if fig is None:
            messagebox.showinfo("No plot", "There is no plot to save yet. Create a plot first.")
            return
        try:
            fpath = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG Image","*.png"), ("SVG Vector","*.svg"), ("PDF Document","*.pdf")],
                initialfile="IV_plot.png",
                title="Save IV Plot"
            )
        except Exception as e:
            messagebox.showerror("Error", f"Could not open Save dialog: {e}")
            return
        if not fpath:
            return
        try:
            fig.savefig(fpath, dpi=300, bbox_inches="tight")
            messagebox.showinfo("Saved", f"Plot saved to:\n{fpath}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save plot: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = IVDataAnalyzer(root)
    root.mainloop()