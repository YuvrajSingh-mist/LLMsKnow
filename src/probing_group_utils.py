#!/usr/bin/env python3
import os
import pandas as pd
import torch

# Detect if we're in a Kaggle environment
def is_kaggle():
    """Check if we're running in a Kaggle environment"""
    return os.path.exists('/kaggle/working')

# Determine the appropriate base directories based on environment
if is_kaggle():
    # Kaggle path
    BASE_DIR = '/kaggle/working/LLMsKnow'
    PROBING_GROUPS_DIR = os.path.join(BASE_DIR, 'probing_groups')
else:
    # Local path based on repository structure
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    PROBING_GROUPS_DIR = os.path.join(BASE_DIR, 'probing_groups')

# Path constants for easier access to each file
MARADONA_TRUE_PATH = os.path.join(PROBING_GROUPS_DIR, 'maradona_true_predictions.csv')
MARADONA_FALSE_PATH = os.path.join(PROBING_GROUPS_DIR, 'maradona_false_predictions.csv')
FRANK_LAMPARD_TRUE_PATH = os.path.join(PROBING_GROUPS_DIR, 'frank_lampard_true_predictions.csv')
FRANK_LAMPARD_FALSE_PATH = os.path.join(PROBING_GROUPS_DIR, 'frank_lampard_false_predictions.csv')
LUIS_SUAREZ_TRUE_PATH = os.path.join(PROBING_GROUPS_DIR, 'luis_suarez_true_predictions.csv')
LUIS_SUAREZ_FALSE_PATH = os.path.join(PROBING_GROUPS_DIR, 'luis_suarez_false_predictions.csv')

def get_maradona_true():
    """Load the Maradona true prediction file as a pandas DataFrame"""
    return pd.read_csv(MARADONA_TRUE_PATH)

def get_maradona_false():
    """Load the Maradona false prediction file as a pandas DataFrame"""
    return pd.read_csv(MARADONA_FALSE_PATH)

def get_frank_lampard_true():
    """Load the Frank Lampard true prediction file as a pandas DataFrame"""
    return pd.read_csv(FRANK_LAMPARD_TRUE_PATH)

def get_frank_lampard_false():
    """Load the Frank Lampard false prediction file as a pandas DataFrame"""
    return pd.read_csv(FRANK_LAMPARD_FALSE_PATH)

def get_luis_suarez_true():
    """Load the Luis Suarez true prediction file as a pandas DataFrame"""
    return pd.read_csv(LUIS_SUAREZ_TRUE_PATH)

def get_luis_suarez_false():
    """Load the Luis Suarez false prediction file as a pandas DataFrame"""
    return pd.read_csv(LUIS_SUAREZ_FALSE_PATH)

# Dictionary mapping simple names to loader functions
probing_groups = {
    'maradona_true': get_maradona_true,
    'maradona_false': get_maradona_false,
    'frank_lampard_true': get_frank_lampard_true,
    'frank_lampard_false': get_frank_lampard_false,
    'luis_suarez_true': get_luis_suarez_true,
    'luis_suarez_false': get_luis_suarez_false
}

def load_probing_group(group_name):
    """
    Load a probing group's data from the transformed CSV files.
    
    Args:
        group_name: Name of the probing group to load
        
    Returns:
        pandas DataFrame containing the probing group data
    """
    # Use the transformed directory
    file_path = f"probing_groups_transformed/{group_name}_predictions.csv"
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find probing group file at {file_path}")
    
    df = pd.read_csv(file_path)
    
    return df

# Add these probing groups to the list of available datasets in probing_utils.py
def update_probing_datasets_lists():
    """Update the LIST_OF_DATASETS in probing_utils.py with probing group names"""
    try:
        import sys
        sys.path.append(os.path.dirname(__file__))
        from probing_utils import LIST_OF_DATASETS, LIST_OF_TEST_DATASETS
        
        # Add probing groups to the list if they're not already there
        for group_name in probing_groups.keys():
            if group_name not in LIST_OF_DATASETS:
                LIST_OF_DATASETS.append(group_name)
                # Also add test versions
                LIST_OF_TEST_DATASETS.append(f"{group_name}_test")
        
        return True
    except ImportError:
        return False

# Automatically try to update the dataset lists when this module is imported
update_probing_datasets_lists()

# Function to get input_output_ids for probing datasets, based on base dataset
def get_input_output_ids_for_probing_group(group_name, model_name):
    """
    Get input_output_ids tensor for a probing group based on the original dataset
    
    Args:
        group_name: Name of the probing group (e.g., 'maradona_true')
        model_name: Name of the model used for probing
        
    Returns:
        PyTorch tensor of input_output_ids for the specified group
    """
    from probing_utils import MODEL_FRIENDLY_NAMES
    
    # Extract base dataset name
    base_dataset = None
    if group_name.startswith('maradona'):
        base_dataset = 'maradona'
    elif group_name.startswith('frank_lampard'):
        base_dataset = 'frank_lampard'
    elif group_name.startswith('luis_suarez'):
        base_dataset = 'luis_suarez'
    
    if not base_dataset:
        raise ValueError(f"Could not determine base dataset for probing group {group_name}")
    
    # Get friendly model name
    friendly_name = MODEL_FRIENDLY_NAMES.get(model_name, 'model')
    
    # Define possible file paths in both Kaggle and local environments
    possible_paths = [
        # Kaggle paths for both original and probing group specific files
        f"/kaggle/working/{friendly_name}-input_output_ids-{base_dataset}.pt",
        f"/kaggle/working/{friendly_name}-input_output_ids-{group_name}.pt",
        f"/kaggle/working/llama-3.1-{base_dataset.replace('_', '-')}-input_output_ids-{base_dataset}.pt",
        f"/kaggle/working/LLMsKnow/{friendly_name}-input_output_ids-{base_dataset}.pt",
        # Local paths
        os.path.join(BASE_DIR, f"{friendly_name}-input_output_ids-{base_dataset}.pt"),
        os.path.join(BASE_DIR, f"{friendly_name}-input_output_ids-{group_name}.pt"),
        os.path.join(os.path.dirname(__file__), '..', f"{friendly_name}-input_output_ids-{base_dataset}.pt")
    ]
    
    # Try loading from each possible path
    for pt_path in possible_paths:
        if os.path.exists(pt_path):
            print(f"Found input_output_ids file at: {pt_path}")
            try:
                input_output_ids = torch.load(pt_path)
                
                # Get the relevant subset based on the probing group CSV
                df = load_probing_group(group_name)
                
                # Only keep input_output_ids that correspond to rows in the probing group
                # This assumes the probing group has an 'index' column that matches the original dataset
                if 'index' in df.columns:
                    indices = df['index'].tolist()
                    filtered_input_output_ids = [input_output_ids[i] for i in indices if i < len(input_output_ids)]
                    return filtered_input_output_ids
                else:
                    # If no index column, just return all (not ideal but fallback)
                    print("Warning: No 'index' column found in probing group. Using all input_output_ids.")
                    return input_output_ids
                    
            except Exception as e:
                print(f"Error loading file {pt_path}: {str(e)}")
                continue
    
    # Debug info
    print(f"Could not find input_output_ids file for {base_dataset} dataset.")
    print(f"Looked in the following locations:")
    for path in possible_paths:
        print(f"  - {path} (exists: {os.path.exists(path)})")
        
    # As a last resort, try to generate input_output_ids from scratch
    try:
        from probing_utils import MODEL_FRIENDLY_NAMES, tokenize, load_model_and_validate_gpu
        
        print(f"Attempting to generate input_output_ids for {group_name} from scratch...")
        data = load_probing_group(group_name)
        model, tokenizer = load_model_and_validate_gpu(model_name)
        
        # Create input_output_ids from questions
        questions = data['question'].tolist()
        input_output_ids = []
        
        for question in questions:
            model_input = tokenize(question, tokenizer, model_name)
            input_output_ids.append(model_input.cpu().squeeze())
            
        return input_output_ids
    except Exception as e:
        print(f"Failed to generate input_output_ids: {str(e)}")
            
    raise FileNotFoundError(f"Could not find input_output_ids file for {base_dataset} dataset")

# Example usage:
# from probing_group_utils import load_probing_group, get_input_output_ids_for_probing_group
# 
# # Load the Maradona true predictions data
# maradona_true_df = load_probing_group('maradona_true')
# 
# # Get input_output_ids for probing with a specific model
# input_output_ids = get_input_output_ids_for_probing_group('maradona_true', 'meta-llama/Meta-Llama-3-8B-Instruct')
#
# # Or use the specific function
# from probing_group_utils import get_maradona_true
# maradona_true_df = get_maradona_true()