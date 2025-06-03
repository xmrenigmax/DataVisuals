import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import os
import numpy as np

# handles the import and loading of data files, selection of tables, and column projections
def import_data(directory_path=None):
    """
    Imports data files from a user-specified directory.
    Returns a list of valid file paths for supported file formats.
    """

    # Supported file formats
    supported_extensions = ['.csv', '.xlsx', '.xls', '.json', '.ods']
    
    # Keep prompting until we get valid files or user quits
    while True:
        # Get directory path if not provided
        if directory_path is None:
            directory_path = input("\n📁 Enter the path to your data files >>> ")
            
            # Allow user to exit
            if directory_path.lower() in ['exit', 'quit', 'q']:
                print("❌ Operation cancelled.")
                return None
        
        # Validate directory
        if not os.path.isdir(directory_path):
            print(f"⚠️ The directory '{directory_path}' does not exist.")
            directory_path = None
            continue
            
        try:
            # Find all supported files
            valid_files = []
            print(f"🔍 Searching for data files in: {directory_path}")
            
            for file in os.listdir(directory_path):
                file_path = os.path.join(directory_path, file)
                if os.path.isfile(file_path) and any(file.endswith(ext) for ext in supported_extensions):
                    valid_files.append(file_path)
            
            # Check if we found any files
            if not valid_files:
                print(f"⚠️ No supported data files found in '{directory_path}'")
                print(f"   Supported formats: {', '.join(supported_extensions)}")
                directory_path = None
                continue
                
            # Success! Return the list of files
            print(f"✅ Found {len(valid_files)} data file(s)")
            return valid_files
            
        except Exception as e:
            print(f"❌ Error accessing directory: {e}")
            directory_path = None



# loads files
def load_file(file_path):
    """
    Loads a data file into a pandas DataFrame based on its file extension.

    """

    print(f"📊 Loading {os.path.basename(file_path)}...")
    
    try:
        # Load file based on extension
        if file_path.endswith('.csv'):
            return pd.read_csv(file_path)
            
        elif file_path.endswith(('.xlsx', '.xls')):
            return pd.read_excel(file_path)
            
        elif file_path.endswith('.json'):
            return pd.read_json(file_path)
            
        elif file_path.endswith('.ods'):
            return pd.read_excel(file_path, engine='odf')
            
        else:
            print(f"❌ Unsupported file format: {os.path.basename(file_path)}")
            return None
            
    except Exception as e:
        print(f"❌ Error loading file: {str(e)}")
        return None



# allows users to select which data files they want to analyze
def get_tables(file_list):
    """
    Allows users to select which data files they want to analyze.
    """

    if not file_list:
        print("❌ No data files available.")
        return None
    
    # Display available files
    print("\n📋 Available Data Files:")
    for i, file_path in enumerate(file_list):
        print(f"  {i+1}. {os.path.basename(file_path)}")
    
    # Get number of files to use
    while True:
        try:
            num_files = input("\n🔢 How many files do you want to work with? >>> ")
            
            # Allow cancellation
            if num_files.lower() in ['exit', 'quit', 'q', 'cancel']:
                print("❌ Operation cancelled.")
                return None
                
            num_files = int(num_files)
            
            if num_files < 1:
                print("⚠️ Please select at least one file.")
            elif num_files > len(file_list):
                print(f"⚠️ You can select at most {len(file_list)} files.")
            else:
                break
                
        except ValueError:
            print("⚠️ Please enter a valid number.")
    

    # Select specific files
    selected_data = []
    print(f"\n🔍 Select {num_files} file(s):")
    
    for i in range(num_files):
        while True:
            try:
                selection = input(f"  File {i+1}: Enter number (1-{len(file_list)}) >>> ")
                
                # Allow cancellation
                if selection.lower() in ['exit', 'quit', 'q', 'cancel']:
                    print("❌ Selection cancelled.")
                    return None
                    
                file_index = int(selection) - 1
                
                # Validate selection
                if file_index < 0 or file_index >= len(file_list):
                    print(f"⚠️ Please enter a number between 1 and {len(file_list)}.")
                    continue
                
                # Check if already selected
                file_path = file_list[file_index]
                filename = os.path.basename(file_path)
                
                if any(filename == fname for fname, _ in selected_data):
                    print(f"⚠️ '{filename}' already selected. Choose a different file.")
                    continue
                
                # Load the data
                data = load_file(file_path)
                
                if data is not None:
                    selected_data.append((filename, data))
                    print(f"✅ Added: {filename} ({len(data)} rows, {len(data.columns)} columns)")
                    break
                else:
                    print(f"⚠️ Failed to load {filename}. Please select another file.")
                    
            except ValueError:
                print("⚠️ Please enter a valid number.")
    
    return selected_data


# allows users to select which columns to include in their analysis
def projection_columns(selected_data):
    """
    Allows users to select which columns to include in their analysis.
    """

    if not selected_data:
        print("❌ No data available for column selection.")
        return None
    
    projection_data = []
    
    for i, (filename, data) in enumerate(selected_data):
        print(f"\n📊 File {i+1}: {filename}")
        
        # Show a preview of the data
        print("\n📋 Data Preview:")
        print(data.head(3).to_string())
        
        # Display column information
        print("\n🔍 Available Columns:")
        for j, column in enumerate(data.columns):
            # Show data type and sample values for better context
            dtype = data[column].dtype
            non_null = data[column].count()
            null_percent = round(100 * (1 - non_null / len(data)), 1) if len(data) > 0 else 0
            
            print(f"  {j+1}. {column} ({dtype}) - {null_percent}% missing")
        
        # Ask if user wants to filter columns
        filter_choice = input("\n🔍 Do you want to select specific columns? (y/n) >>> ").strip().lower()
        
        if filter_choice in ['y', 'yes']:
            while True:
                try:
                    # Get column selection
                    column_input = input("\n📝 Enter column names or numbers separated by commas >>> ")
                    
                    # Allow cancellation
                    if column_input.lower() in ['exit', 'quit', 'q', 'cancel']:
                        print("❌ Column selection cancelled. Using all columns.")
                        projection_data.append((filename, data))
                        break
                    
                    # Parse column selection (support both names and numbers)
                    selected_columns = []
                    for item in [x.strip() for x in column_input.split(',')]:
                        # If item is a number, convert to column name
                        if item.isdigit() and 1 <= int(item) <= len(data.columns):
                            col_index = int(item) - 1
                            selected_columns.append(data.columns[col_index])
                        # Otherwise, treat as column name
                        elif item in data.columns:
                            selected_columns.append(item)
                        else:
                            print(f"⚠️ Column '{item}' not found. Please try again.")
                            selected_columns = []
                            break
                    
                    # If we have valid columns, filter the data
                    if selected_columns:
                        filtered_data = data[selected_columns]
                        projection_data.append((filename, filtered_data))
                        print(f"✅ Selected {len(selected_columns)} columns from {filename}")
                        print(f"   Columns: {', '.join(selected_columns)}")
                        break
                        
                except Exception as e:
                    print(f"❌ Error selecting columns: {e}")
        else:
            # Use all columns
            projection_data.append((filename, data))
            print(f"✅ Using all {len(data.columns)} columns from {filename}")
    
    return projection_data
