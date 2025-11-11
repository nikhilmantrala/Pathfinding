import pandas as pd
import os
import glob

def standardize_algorithm_name(name):
    # Map algorithm names to standard format
    name_map = {
        'ML': 'ml',
        'ML-Dyn': 'ml_dynamic',
        'A*': 'astar',
        'A*-Dyn': 'astar_dynamic'
    }
    return name_map.get(name, name.lower())

def process_pathfinding_data(csv_file):
    try:
        # Read CSV file
        df = pd.read_csv(csv_file)
        
        print(f"\nProcessing file: {os.path.basename(csv_file)}")
        print("Columns found:", df.columns.tolist())
        print("Number of rows:", len(df))
        
        # Standardize algorithm names
        df['Algorithm'] = df['Algorithm'].apply(standardize_algorithm_name)
        
        # Group by test case
        test_cases = df.groupby('Test Case')
        print(f"Number of test cases found: {len(test_cases)}")
        
        # Initialize list to store valid test cases
        valid_test_cases = []
    except Exception as e:
        print(f"Error reading or processing file {csv_file}: {str(e)}")
        return None
    
    # Process each test case
    for test_case, group in test_cases:
        # Check if all algorithms in this test case were successful
        # Convert 'Success' column to lowercase for comparison
        success_values = group['Success'].str.lower()
        if success_values.all() == 'yes' or success_values.str.startswith('y').all():
            valid_test_cases.append(group)
    
    if not valid_test_cases:
        print(f"No valid test cases found in {csv_file}")
        print("Sample of data:")
        print(df.head())
        print("\nUnique values in Success column:", df['Success'].unique())
        return None
    
    # Combine valid test cases
    valid_df = pd.concat(valid_test_cases)
    
    # Drop wall density column
    valid_df = valid_df.drop('Wall Density', axis=1)
    
    # Split data by algorithm
    algorithms = valid_df['Algorithm'].unique()
    algorithm_data = {}
    
    for algo in algorithms:
        algo_df = valid_df[valid_df['Algorithm'] == algo]
        
        # Further split by layout type
        layout_data = {}
        for layout in algo_df['Layout Type'].unique():
            layout_df = algo_df[algo_df['Layout Type'] == layout]
            layout_data[layout] = layout_df
        
        algorithm_data[algo] = layout_data
    
    return {
        'full_data': valid_df,
        'by_algorithm': algorithm_data
    }

def process_all_files(directory):
    # Get all CSV files in the unprocessed_data directory
    csv_files = glob.glob(os.path.join(directory, '*.csv'))
    
    if not csv_files:
        print(f"No CSV files found in {directory}")
        return None
        
    processed_data = {}
    print(f"Found {len(csv_files)} files to process:")
    for file in csv_files:
        print(f"\nProcessing file: {os.path.basename(file)}")
        base_name = os.path.basename(file)
        try:
            data = process_pathfinding_data(file)
            if data:
                processed_data[base_name] = data
                print(f"Successfully processed {base_name}")
                # Print found algorithms
                algorithms = data['full_data']['Algorithm'].unique()
                print(f"Found algorithms: {', '.join(algorithms)}")
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")
    
    return processed_data

def save_processed_data(processed_data, output_dir):
    if not processed_data:
        print("No processed data to save.")
        return

    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        for file_name, data in processed_data.items():
            base_name = os.path.splitext(file_name)[0]
            
            # Save full dataset
            full_data_path = os.path.join(output_dir, f"{base_name}_processed_full.csv")
            data['full_data'].to_csv(full_data_path, index=False)
            print(f"Saved processed data to: {full_data_path}")
            
            # Save algorithm-specific datasets
            for algo, layout_data in data['by_algorithm'].items():
                # Use standardized algorithm names for directories
                algo_dir = os.path.join(output_dir, base_name, algo)
                os.makedirs(algo_dir, exist_ok=True)
                
                for layout, df in layout_data.items():
                    layout_file = os.path.join(algo_dir, f"{layout.lower()}.csv")
                    df.to_csv(layout_file, index=False)
                    print(f"Saved {algo} - {layout} data to: {layout_file}")
    except Exception as e:
        print(f"Error saving processed data: {str(e)}")

if __name__ == "__main__":
    # Set up directories
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(current_dir, "unprocessed_data")
    output_dir = os.path.join(current_dir, "processed_data")
    
    print(f"Looking for CSV files in: {input_dir}")
    print(f"Output will be saved to: {output_dir}")
    
    # Process all files
    processed_data = process_all_files(input_dir)
    
    if processed_data:
        # Save processed data
        save_processed_data(processed_data, output_dir)
        print("\nData processing complete. Results saved in 'processed_data' directory")
    else:
        print("\nNo data was processed. Please check the input directory and file contents.")
