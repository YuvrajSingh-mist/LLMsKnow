#!/usr/bin/env python3
import os
import sys
import pandas as pd

# Detect if we're in Kaggle environment
def is_kaggle():
    return os.path.exists('/kaggle/working')

# Get the base directory
if is_kaggle():
    BASE_DIR = '/kaggle/working/LLMsKnow'
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Create the probing_groups directory if it doesn't exist
PROBING_GROUPS_DIR = os.path.join(BASE_DIR, 'probing_groups')
os.makedirs(PROBING_GROUPS_DIR, exist_ok=True)

# Function to get the correct path for a dataset
def get_dataset_path(model_name, dataset_name):
    """Get the path to a dataset CSV file, trying multiple locations"""
    possible_paths = [
        # Kaggle paths
        f"/kaggle/working/{model_name}-answers-{dataset_name}.csv",
        f"/kaggle/working/LLMsKnow/{model_name}-answers-{dataset_name}.csv",
        # Local paths
        os.path.join(BASE_DIR, f"{model_name}-answers-{dataset_name}.csv"),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
            
    raise FileNotFoundError(f"Could not find dataset file for {dataset_name}")

# Parameters
def split_dataset(dataset_name, model_friendly_name=None, input_file=None):
    """
    Split a dataset into true and false prediction groups
    
    Args:
        dataset_name: Base name of the dataset (e.g., 'frank_lampard')
        model_friendly_name: Friendly name of the model (e.g., 'llama-3.1-frank-lampard')
        input_file: Optional path to the input CSV file
    """
    if input_file is None:
        if model_friendly_name is None:
            raise ValueError("Either input_file or model_friendly_name must be provided")
        try:
            input_file = get_dataset_path(model_friendly_name, dataset_name)
        except FileNotFoundError as e:
            print(f"Error: {str(e)}")
            print("Available files in current directory:")
            for f in os.listdir('.'):
                if f.endswith('.csv'):
                    print(f"  - {f}")
            return
    
    # Output file paths
    true_file = os.path.join(PROBING_GROUPS_DIR, f"{dataset_name}_true_predictions.csv")
    false_file = os.path.join(PROBING_GROUPS_DIR, f"{dataset_name}_false_predictions.csv")
    
    print(f"Input file: {input_file}")
    print(f"True predictions will be saved to: {true_file}")
    print(f"False predictions will be saved to: {false_file}")
    
    # Load the dataset
    try:
        df = pd.read_csv(input_file)
        print(f"Loaded dataset with {len(df)} rows")
        
        # Display column names to help debugging
        print("Available columns:", df.columns.tolist())
        
        # Check if the required columns exist
        if 'automatic_correctness' not in df.columns:
            if 'correctness' in df.columns:
                print("Using 'correctness' column instead of 'automatic_correctness'")
                df['automatic_correctness'] = df['correctness']
            else:
                raise ValueError("Dataset does not have 'automatic_correctness' or 'correctness' column")
        
        # Add an index column to keep track of the original indices
        df['index'] = df.index
        
        # Split based on correctness
        tp_tn_df = df[df['automatic_correctness'] == 1]
        fp_fn_df = df[df['automatic_correctness'] == 0]
        
        # Save the split dataframes
        tp_tn_df.to_csv(true_file, index=False)
        fp_fn_df.to_csv(false_file, index=False)
        
        # Count examples in each group
        tp_tn_count = len(tp_tn_df)
        fp_fn_count = len(fp_fn_df)
        total_count = len(df)
        
        # Print summary
        print(f"Total examples: {total_count}")
        print(f"True Positive/True Negative examples: {tp_tn_count} ({tp_tn_count/total_count:.1%})")
        print(f"False Positive/False Negative examples: {fp_fn_count} ({fp_fn_count/total_count:.1%})")
        
        # Also count stance types in each group if available
        if 'correct_answer' in df.columns:
            print("\nStance distribution in TP/TN group:")
            print(tp_tn_df['correct_answer'].value_counts())
            
            print("\nStance distribution in FP/FN group:")
            print(fp_fn_df['correct_answer'].value_counts())
        
        print(f"\nFiles created: {true_file} and {false_file}")
        
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python split_probing_groups.py [dataset_name] <model_friendly_name>")
        print("Example: python split_probing_groups.py frank_lampard llama-3.1-frank-lampard")
        print("Or specify a file directly: python split_probing_groups.py frank_lampard --file=path/to/file.csv")
        sys.exit(1)
    
    dataset_name = sys.argv[1]
    
    # Check for file argument
    input_file = None
    model_friendly_name = None
    
    for arg in sys.argv[2:]:
        if arg.startswith('--file='):
            input_file = arg.split('=', 1)[1]
        else:
            model_friendly_name = arg
    
    split_dataset(dataset_name, model_friendly_name, input_file)
