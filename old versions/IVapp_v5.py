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
            'Pmpp [W/m2]', 'Vmpp [V]', 'Jmpp [mA/cm2]', 'Roc [Ohm.m2]', 'Rsc [Ohm.m2]'
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
        notebook.add(plot_frame, text="Plotting")
    
        # New IV Plot tab
        iv_plot_frame = ttk.Frame(notebook)
        notebook.add(iv_plot_frame, text="IV Plot")
    
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
                  'Pmpp [W/m2]', 'Vmpp [V]', 'Jmpp [mA/cm2]', 'Roc [Ohm.m2]', 'Rsc [Ohm.m2]')
        
        self.measurements_tree = ttk.Treeview(measurements_frame, columns=columns, show='headings')
        
        # Set column headings and widths
        for col in columns:
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
        plot_parameters = ['Voc [V]', 'Jsc [mA/cm2]', 'FF [.]', 'Efficiency [.]', 'Pmpp [W/m2]', 'Vmpp [V]', 'Jmpp [mA/cm2]', 'Roc [Ohm.m2]', 'Rsc [Ohm.m2]']
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
        
        # Frame for the plot
        self.plot_frame = ttk.Frame(parent)
        self.plot_frame.pack(fill="both", expand=True, padx=10, pady=10)
    
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
                self.measurements_data = pd.concat([self.measurements_data, pd.DataFrame([data])], ignore_index=True)
                self.measurements_data_original = self.measurements_data.copy()
                
                # Add to treeview
                values = [data[col] for col in self.measurements_data.columns]
                self.measurements_tree.insert('', 'end', values=values)
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to parse file {os.path.basename(filepath)}: {str(e)}")
        
        self.update_measurements_display()
    
    def parse_iv_file(self, filepath):
        filename = os.path.basename(filepath)
        
        # Extract pixel information from filename

        pixel_match = re.search(r'c(\d)', filename)
        pixel = {'1': 'A', '2': 'B', '3': 'C'}.get(pixel_match.group(1), 'Unknown') if pixel_match else 'Unknown'
        
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Extract substrate ID
        substrate_id_match = re.search(r'Deposition ID:\s*([A-Za-z0-9]+)', content)
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
            values = [row[col] for col in self.measurements_data.columns]
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
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
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
        ax.set_ylabel(selected_param)
        
        # Add horizontal grid lines only
        ax.grid(True, axis='y', alpha=0.5, linestyle='--')
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=90, fontsize=15)
        
        # Tight layout to ensure everything fits
        plt.tight_layout()
        
        # Save the figure reference for later use
        self.current_figure = fig
        self.current_param = selected_param
        
        # Embed plot in tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
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
            
    def setup_iv_plot(self, parent):
        control_frame = ttk.Frame(parent)
        control_frame.pack(fill="x", padx=10, pady=5)
    
        browse_button = ttk.Button(control_frame, text="Load IV File", command=self.load_iv_data)
        browse_button.pack(side="left", padx=5, pady=5)
    
        self.plot_frame = ttk.Frame(parent)
        self.plot_frame.pack(fill="both", expand=True, padx=10, pady=10)
    
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
    
    def plot_iv_curve(self, data):
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
    
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(data['Voltage (V)'], data['Current (A)'], marker='o', linestyle='-')
        ax.set_title("IV Curve")
        ax.set_xlabel("Voltage (V)")
        ax.set_ylabel("Current (A)")
        ax.grid(True)
    
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


if __name__ == "__main__":
    root = tk.Tk()
    app = IVDataAnalyzer(root)
    root.mainloop()