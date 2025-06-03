from handling import *
from singletablehandler import visualise_single_table, analyse_single_table
from matplotlib.backends.backend_pdf import PdfPages
from multitablehandler import visualise_multiple_tables, analyse_multiple_tables
import os
import pandas as pd
import matplotlib.pyplot as plt

# Column identification function for single table analysis and visualization
def column_identification(table):
    """
    Identifies column types (timestamp, categorical, numerical) from a dataframe.
    Returns the categorized columns and the dataframe for further processing.
    """
    filename, df = table 

    # User feedback
    print(f"\n Analyzing data from: {filename}")
    print("Identifying column types...")

    # Initialize column categorization
    timestamp_columns = []
    categorical_columns = []
    numerical_columns = []

    # Analyze each column's data type
    for column in df.columns:
        # Check for timestamp columns
        if (pd.api.types.is_datetime64_any_dtype(df[column]) or 
            any(time_word in column.lower() for time_word in ['date', 'time', 'timestamp', 'datetime'])):
            try:
                df[column] = pd.to_datetime(df[column])
                timestamp_columns.append(column)
            except Exception as e:
                print(f" Could not convert {column} to datetime: {e}")
        
        # Check for numerical columns
        elif (pd.api.types.is_numeric_dtype(df[column]) or 
              pd.api.types.is_integer_dtype(df[column]) or 
              pd.api.types.is_float_dtype(df[column])):
            numerical_columns.append(column)
        
        # Everything else is categorical
        else:
            categorical_columns.append(column)
    
    # Display results in a more readable format
    print("\n Column Classification Results:")
    print(f"  ⏱️  Time-based: {', '.join(timestamp_columns) if timestamp_columns else 'None'}")
    print(f"  🔢 Numerical: {', '.join(numerical_columns) if numerical_columns else 'None'}")
    print(f"  📝 Categorical: {', '.join(categorical_columns) if categorical_columns else 'None'}")
    
    # Get user confirmation
    confirm_columns = input("\n✅ Is this classification correct? (y/n) >>> ").strip().lower()
    if confirm_columns != 'y':
        print("🛑 Operation cancelled. Please check your data and try again.")
        return None, None, None, None, None
    
    return timestamp_columns, categorical_columns, numerical_columns, filename, df




# handles the visualization of a single table
def handle_single_table_visualization(table):
    """
    Prepares and calls the visualization function for a single table.
    Handles user feedback and visualization saving.
    """
    # Get column information and table data
    timestamp_columns, categorical_columns, numerical_columns, filename, df = column_identification(table)
    
    # Exit if columns were not confirmed
    if df is None:
        return
    
    print("\n🎨 Creating visualizations...")
    
    # Call the visualization function from singletablehandler
    figures = visualise_single_table(timestamp_columns, categorical_columns, numerical_columns, filename, df)
    
    # Save visualizations if requested
    if figures:
        save_visualisation_to_pdf(figures, filename)
    else:
        print("ℹ️ No visualizations were generated.")



# handles the analysis of a single table
def handle_single_table_analysis(table):
    """
    Prepares and calls the analysis function for a single table.
    Handles user feedback and visualization saving.
    """
    # Get column information and table data
    timestamp_columns, categorical_columns, numerical_columns, filename, df = column_identification(table)
    
    # Exit if columns were not confirmed
    if df is None:
        return
    
    print("\n📊 Performing data analysis...")
    
    # Call the analysis function from singletablehandler
    figures = analyse_single_table(timestamp_columns, categorical_columns, numerical_columns, filename, df)
    
    # Save visualizations if requested
    if figures:
        save_visualisation_to_pdf(figures, filename)
    else:
        print("ℹ️ No analysis visualizations were generated.")

        
# handles the data detection and guides the user through analysis options
def data_detection(selected_data):
    """
    Detects data type and guides the user through appropriate analysis options.
    Handles both single and multiple table scenarios.
    """
    if not selected_data:
        print("❌ No data selected for processing.")
        return None
    
    # Single table processing
    elif len(selected_data) == 1:
        print("\n🔍 Single table detected")
        
        # Provide clear options to the user
        print("\nWhat would you like to do with this data?")
        print("  v - Create visualizations (charts, graphs)")
        print("  a - Perform statistical analysis")
        
        choice = input("\nEnter your choice (v/a) >>> ").strip().lower()
        
        if choice == 'v':
            handle_single_table_visualization(selected_data[0])
        elif choice == 'a':
            handle_single_table_analysis(selected_data[0])
        else:
            print("❌ Invalid option. Please enter 'v' for visualize or 'a' for analyze.")
    
    # Multiple table processing
    else:
        print(f"\n🔍 {len(selected_data)} tables detected")
        
        # Provide clear options to the user
        print("\nWhat would you like to do with these tables?")
        print("  v - Compare tables with visualizations")
        print("  a - Compare tables with statistical analysis")
        
        choice = input("\nEnter your choice (v/a) >>> ").strip().lower()
        
        if choice == 'v':
            handle_multiple_table_visualization(selected_data)
        elif choice == 'a':
            handle_multiple_table_analysis(selected_data)
        else:
            print("❌ Invalid option. Please enter 'v' for visualize or 'a' for analyze.")



# saves visualizations to a PDF file with improved user feedback
def save_visualisation_to_pdf(figures, filename='visualisation'):
    """
    Handles saving visualizations to PDF with improved user feedback.
    Creates a well-organized output directory structure.
    """
    if not figures:
        return
        
    print("\n💾 Save Options")
    save_option = input("Would you like to save these visualizations to PDF? (y/n) >>> ").strip().lower()
    
    if save_option == 'y':
        # Create directory structure
        output_dir = os.path.join(os.getcwd(), 'visualizations')
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate a meaningful filename with timestamp
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        output_file = os.path.join(output_dir, f"{filename}_viz_{timestamp}.pdf")
        
        print(f"\n⏳ Saving visualizations...")
        
        # Save figures to PDF
        with PdfPages(output_file) as pdf:
            figures = [plt.figure(n) for n in plt.get_fignums()]
            for fig in figures:
                pdf.savefig(fig)
        
        print(f"✅ Visualizations saved to: {output_file}")
    else:
        print("ℹ️ Visualizations not saved.")




# handles the visualization of multiple tables
def handle_multiple_table_visualization(tables):
    """
    Prepares and calls the visualization function for multiple tables.
    Handles user feedback and visualization saving.
    """
    print("\n🎨 Creating multi-table visualizations...")
    
    # Call the visualization function from multitablehandler
    figures = visualise_multiple_tables(tables)
    
    # Save visualizations if requested
    if figures:
        # Use the first filename as base for the output file
        base_filename = "_".join([table[0].split('.')[0] for table in tables])
        save_visualisation_to_pdf(figures, f"multi_{base_filename}")
    else:
        print("ℹ️ No multi-table visualizations were generated.")


# handles the analysis of multiple tables
def handle_multiple_table_analysis(tables):
    """
    Prepares and calls the analysis function for multiple tables.
    Handles user feedback and visualization saving.
    """
    print("\n📊 Performing multi-table data analysis...")
    
    # Call the analysis function from multitablehandler
    figures = analyse_multiple_tables(tables)
    
    # Save visualizations if requested
    if figures:
        # Use the first filename as base for the output file
        base_filename = "_".join([table[0].split('.')[0] for table in tables])
        save_visualisation_to_pdf(figures, f"multi_analysis_{base_filename}")
    else:
        print("ℹ️ No multi-table analysis visualizations were generated.")


# main function to run the script
if __name__ == "__main__":
    """Main execution flow with improved user guidance and feedback"""
    # Display welcome header
    print("\n" + "=" * 70)
    print("🚀 Welcome to DataVisuals - Data Visualization and Analysis Tool 📊")
    print("=" * 70)
    
    # Step 1: Import data files with clear guidance
    print("\n📂 STEP 1: Import Data Files")
    print("Select the data files you want to analyze")
    file_list = import_data()
    
    if not file_list:
        print("❌ No valid data files were found. Please check your data and try again.")
    else:
        print(f"\n✅ Found {len(file_list)} valid data file(s)")
        
        # Step 2: Select tables with clear guidance
        print("\n📋 STEP 2: Select Tables")
        print("Choose which tables you want to work with")
        selected_data = get_tables(file_list)
        
        if not selected_data:
            print("❌ No tables were selected. Exiting.")
        else:
            # Step 3: Select columns with clear guidance
            print("\n🔍 STEP 3: Select Columns")
            print("Choose which columns to include in your analysis")
            projected_data = projection_columns(selected_data)
            
            # Step 4: Choose visualization or analysis
            if projected_data:
                print("\n📊 STEP 4: Analysis Options")
                data_detection(projected_data)
    
    # Closing message
    print("\n🙏 Thank you for using DataVisuals!")
