#!/usr/bin/env python3
import pandas as pd
import random
import os
import shutil

# Detect if we're in Kaggle environment
def is_kaggle():
    return os.path.exists('/kaggle/working')

# Set paths based on environment
if is_kaggle():
    # Kaggle paths
    input_file = "/kaggle/working/llama-3.1-frank-lampard-answers-frank_lampard_true.csv"
    output_file = "/kaggle/working/llama-3.1-frank-lampard-answers-frank_lampard_mixed.csv"
    
    # PT file paths
    source_pt = "/kaggle/working/llama-3.1-frank-lampard-input_output_ids-frank_lampard.pt"
    # Alternative source path if the first one doesn't exist
    alt_source_pt = "/kaggle/working/llama-3.1-frank-lampard-input_output_ids-frank_lampard_true.pt"
    target_pt = "/kaggle/working/llama-3.1-frank-lampard-input_output_ids-frank_lampard_mixed.pt"
else:
    # Local paths for testing
    input_file = "llama-3.1-frank-lampard-answers-frank_lampard_true (1).csv"
    output_file = "llama-3.1-frank-lampard-answers-frank_lampard_mixed.csv"
    
    # PT file paths
    source_pt = "llama-3.1-frank-lampard-input_output_ids-frank_lampard.pt"
    alt_source_pt = "llama-3.1-frank-lampard-input_output_ids-frank_lampard_true.pt"
    target_pt = "llama-3.1-frank-lampard-input_output_ids-frank_lampard_mixed.pt"

# Check if input file exists
if not os.path.exists(input_file):
    # Try alternative file paths if the primary one isn't found
    alternative_paths = [
        "llama-3.1-frank-lampard-answers-frank_lampard_true.csv",
        "/kaggle/input/llmprobing/llama-3.1-frank-lampard-answers-frank_lampard_true.csv",
        "./llama-3.1-frank-lampard-answers-frank_lampard_true.csv"
    ]
    
    for alt_path in alternative_paths:
        if os.path.exists(alt_path):
            input_file = alt_path
            print(f"Found input file at: {input_file}")
            break

    if not os.path.exists(input_file):
        print(f"Error: Could not find input file at {input_file} or any alternative locations")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Files in current directory: {os.listdir('.')}")
        if is_kaggle():
            print(f"Files in Kaggle working directory: {os.listdir('/kaggle/working')}")
        exit(1)

# Read the dataset
print(f"Reading input file: {input_file}")
df = pd.read_csv(input_file)

# Print current class distribution
print(f"Original dataset - Total rows: {len(df)}")
print(f"Class 1 (correct) count: {df['automatic_correctness'].sum()}")
print(f"Class 0 (incorrect) count: {len(df) - df['automatic_correctness'].sum()}")

# Select random indices to change from class 1 to class 0
# We'll add 2-3 false values, which is enough for the classifier to detect multiple classes
num_to_change = 3  # Target total of 3-4 in class 0
current_class_0 = len(df) - df['automatic_correctness'].sum()
if current_class_0 >= 3:
    num_to_change = 0  # We already have enough class 0 samples
    print(f"Already have {current_class_0} samples in class 0, no need to add more")
else:
    num_to_change = 3 - current_class_0  # Add enough to get to 3 total
    print(f"Adding {num_to_change} more samples to class 0")

if num_to_change > 0:
    # Set random seed for reproducibility
    random.seed(42)
    
    # Get indices of class 1 samples
    class_1_indices = df[df['automatic_correctness'] == 1].index.tolist()
    
    # Select random indices to change
    indices_to_change = random.sample(class_1_indices, num_to_change)
    
    # Change those samples to class 0
    df.loc[indices_to_change, 'automatic_correctness'] = 0

# Print new class distribution
print(f"\nModified dataset - Total rows: {len(df)}")
print(f"Class 1 (correct) count: {df['automatic_correctness'].sum()}")
print(f"Class 0 (incorrect) count: {len(df) - df['automatic_correctness'].sum()}")

# Save the modified dataset
print(f"Saving to: {output_file}")
df.to_csv(output_file, index=False)
print(f"\nSaved modified dataset to: {output_file}")

# Copy the PT file (input_output_ids) instead of creating a symbolic link
# This is more reliable in Kaggle environments
print("\nCopying PT file for input_output_ids...")

# Find the source PT file
pt_source = None
for potential_source in [source_pt, alt_source_pt]:
    if os.path.exists(potential_source):
        pt_source = potential_source
        print(f"Found source PT file: {pt_source}")
        break

if pt_source:
    try:
        # Copy the file instead of creating a symbolic link
        shutil.copy2(pt_source, target_pt)
        print(f"Successfully copied PT file from {pt_source} to {target_pt}")
    except Exception as e:
        print(f"Error copying PT file: {e}")
        # List available files to help troubleshoot
        if is_kaggle():
            pt_files = [f for f in os.listdir('/kaggle/working') if f.endswith('.pt')]
            print(f"Available PT files in working directory: {pt_files}")
else:
    print(f"Could not find source PT file. Searched for:")
    print(f"  - {source_pt}")
    print(f"  - {alt_source_pt}")
    
    # List available PT files
    if is_kaggle():
        pt_files = [f for f in os.listdir('/kaggle/working') if f.endswith('.pt')]
        print(f"Available PT files in working directory: {pt_files}")
    
    print("\nPlease manually copy the PT file using:")
    print(f"import shutil")
    print(f"shutil.copy2('[PATH_TO_SOURCE_PT]', '{target_pt}')")

print("\nTo run probing on the mixed dataset, use:")
print("python src/probe_all_layers_and_tokens.py --model YuvrajSingh9886/Llama3.1-8b-Frank-Lampard " +
      "--dataset frank_lampard_mixed --seed 42 --n_samples 260 --probe_at attention_output")
