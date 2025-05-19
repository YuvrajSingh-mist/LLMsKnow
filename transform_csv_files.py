#!/usr/bin/env python
# Script to transform CSV files in the probing_groups directory to have only two columns: Comments and Label

import os
import pandas as pd
import re
import csv

def extract_comment_from_question(question_text):
    """Extract the comment from the question text by finding the input section."""
    input_pattern = r"### Input:\s*(.*?)(?=\s*###|\s*$)"
    match = re.search(input_pattern, question_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None  # Return None if no comment is found

def process_csv_file(input_file, output_file):
    """Process a CSV file to extract Comments and Label columns."""
    try:
        # Read the CSV file
        df = pd.read_csv(input_file)
        
        # Prepare new dataframe with two columns
        new_df = pd.DataFrame(columns=["Comments", "Label"])
        
        # Extract comments from questions and use exact_answer as the label
        for _, row in df.iterrows():
            if "question" in df.columns and "exact_answer" in df.columns:
                comment = extract_comment_from_question(row["question"])
                label = row["exact_answer"]
                if comment and label:
                    # Add to the new dataframe
                    new_df = new_df._append({"Comments": comment, "Label": label}, ignore_index=True)
        
        # Write the new dataframe to CSV
        new_df.to_csv(output_file, index=False)
        print(f"Processed {input_file} -> {output_file}")
    except Exception as e:
        print(f"Error processing {input_file}: {str(e)}")

def main():
    # Directory containing the CSV files
    input_dir = "probing_groups"
    # Directory to save the transformed files
    output_dir = "probing_groups_transformed"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each CSV file in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(".csv"):
            input_file = os.path.join(input_dir, filename)
            output_file = os.path.join(output_dir, filename)
            process_csv_file(input_file, output_file)

if __name__ == "__main__":
    main()
    print("All files have been processed.")