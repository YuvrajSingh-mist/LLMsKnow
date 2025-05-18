#!/usr/bin/env python3
import os
import pandas as pd

# Base directory for all probing group files
PROBING_GROUPS_DIR = os.path.join('./LLMsKnow', 'probing_groups')

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
    Load a probing group file by its simple name
    
    Args:
        group_name: One of 'maradona_true', 'maradona_false', 'frank_lampard_true',
                   'frank_lampard_false', 'luis_suarez_true', 'luis_suarez_false'
    
    Returns:
        pandas DataFrame containing the requested probing group data
    
    Raises:
        KeyError: If the group_name is not recognized
    """
    if group_name not in probing_groups:
        raise KeyError(f"Unknown probing group: {group_name}. Available groups: {list(probing_groups.keys())}")
    
    return probing_groups[group_name]()

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
    import torch
    from probing_utils import MODEL_FRIENDLY_NAMES
    
    base_dataset = None
    if group_name.startswith('maradona'):
        base_dataset = 'maradona'
    elif group_name.startswith('frank_lampard'):
        base_dataset = 'frank_lampard'
    elif group_name.startswith('luis_suarez'):
        base_dataset = 'luis_suarez'
    
    if not base_dataset:
        raise ValueError(f"Could not determine base dataset for probing group {group_name}")
    
    # Try to find the PT file in the expected locations
    friendly_name = MODEL_FRIENDLY_NAMES.get(model_name, 'model')
    pt_paths = [
        f"/kaggle/working/{friendly_name}-input_output_ids-{base_dataset}.pt",
        os.path.join(os.path.dirname(__file__), '..', f"{friendly_name}-input_output_ids-{base_dataset}.pt")
    ]
    
    for pt_path in pt_paths:
        if os.path.exists(pt_path):
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
                return input_output_ids
    
    raise FileNotFoundError(f"Could not find input_output_ids PT file for {base_dataset} dataset")

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