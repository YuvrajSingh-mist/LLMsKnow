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

def clean_comment(comment):
    """
    Clean the comment text by removing unnecessary text or formatting.
    """
    if not isinstance(comment, str):
        return comment

    # Remove any leading/trailing whitespace
    comment = comment.strip()

    # Remove specific unwanted patterns (e.g., "### Input:", "###", etc.)
    unwanted_patterns = ["### Input:", "###"]
    for pattern in unwanted_patterns:
        comment = comment.replace(pattern, "")

    # Further cleaning can be added here if needed

    return comment

def create_compatible_dataset(input_file, output_file, text_column='Comment', label_column='Llama3.1 Not Fine Tuned Output'):
    """
    Create a dataset compatible with generate_model_answers.py by extracting specific columns
    """
    # Check if the input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found!")
        return False

    try:
        # Read the input Excel file
        df = pd.read_excel(input_file)

        # Check if required columns exist
        if text_column not in df.columns:
            print(f"Error: '{text_column}' column not found in {input_file}")
            return False

        if label_column not in df.columns:
            print(f"Error: '{label_column}' column not found in {input_file}")
            return False

        # Extract only the necessary columns
        output_df = df[[text_column, label_column]].copy()

        # Rename columns for compatibility
        output_df.rename(columns={text_column: 'Comment', label_column: 'Llama3.1 Not Fine Tuned Output'}, inplace=True)

        # Save to the output file
        output_df.to_csv(output_file, index=False)
        print(f"Successfully created compatible dataset: {output_file}")
        print(f"Dataset contains {len(output_df)} rows")
        return True

    except Exception as e:
        print(f"Error processing file {input_file}: {str(e)}")
        return False

def create_compatible_dataset_with_ground_truth(original_file, excel_file, output_file, text_column='Comment', label_column='Llama3.1 Not Fine Tuned output'):
    """
    Create a dataset compatible with generate_model_answers.py by ensuring it includes the same 400 comments
    from the original file along with the ground truth labels from the Excel file.
    """
    # Check if the input files exist
    if not os.path.exists(original_file):
        print(f"Error: Original file '{original_file}' not found!")
        return False

    if not os.path.exists(excel_file):
        print(f"Error: Excel file '{excel_file}' not found!")
        return False

    try:
        # Read the original CSV file
        original_df = pd.read_csv(original_file)

        # Read the Excel file containing ground truth labels
        excel_df = pd.read_excel(excel_file)

        # Check if required columns exist in the Excel file
        if text_column not in excel_df.columns:
            print(f"Error: '{text_column}' column not found in {excel_file}")
            return False

        if label_column not in excel_df.columns:
            print(f"Error: '{label_column}' column not found in {excel_file}")
            return False

        # Ensure proper alignment by trimming and normalizing whitespace in both datasets
        original_df['question_normalized'] = original_df['question'].str.strip()
        excel_df['Comment_normalized'] = excel_df[text_column].str.strip()

        # Merge the original comments with the ground truth labels
        merged_df = original_df.merge(
            excel_df[['Comment_normalized', label_column]],
            left_on='question_normalized',
            right_on='Comment_normalized',
            how='left'
        )

        # Clean the comments
        merged_df['question'] = merged_df['question'].apply(clean_comment)

        # Select only the necessary columns
        output_df = merged_df[['question', label_column]].copy()

        # Rename columns for compatibility
        output_df.rename(columns={'question': 'Comment', label_column: 'Llama3.1 Not Fine Tuned output'}, inplace=True)

        # Save to the output file
        output_df.to_csv(output_file, index=False)
        print(f"Successfully created compatible dataset: {output_file}")
        print(f"Dataset contains {len(output_df)} rows")
        return True

    except Exception as e:
        print(f"Error processing files {original_file} and {excel_file}: {str(e)}")
        return False

def extract_comments_and_match_labels(original_file, excel_file, output_file, text_column='Comment', label_column='Llama3.1 Not Fine Tuned output'):
    """
    Extract comments from the original file and match them with the corresponding labels in the Excel file.
    """
    # Check if the input files exist
    if not os.path.exists(original_file):
        print(f"Error: Original file '{original_file}' not found!")
        return False

    if not os.path.exists(excel_file):
        print(f"Error: Excel file '{excel_file}' not found!")
        return False

    try:
        # Read the original CSV file
        original_df = pd.read_csv(original_file)

        # Extract comments from the 'question' column
        original_df['comment_extracted'] = original_df['question'].str.extract(r"### Input:\s*(.*)")

        # Read the Excel file containing ground truth labels
        excel_df = pd.read_excel(excel_file)

        # Check if required columns exist in the Excel file
        if text_column not in excel_df.columns:
            print(f"Error: '{text_column}' column not found in {excel_file}")
            return False

        if label_column not in excel_df.columns:
            print(f"Error: '{label_column}' column not found in {excel_file}")
            return False

        # Ensure proper alignment by trimming and normalizing whitespace in both datasets
        original_df['comment_extracted'] = original_df['comment_extracted'].str.strip()
        excel_df[text_column] = excel_df[text_column].str.strip()

        # Merge the extracted comments with the ground truth labels
        merged_df = original_df.merge(
            excel_df[[text_column, label_column]],
            left_on='comment_extracted',
            right_on=text_column,
            how='left'
        )

        # Select only the necessary columns
        output_df = merged_df[['comment_extracted', label_column]].copy()

        # Rename columns for compatibility
        output_df.rename(columns={'comment_extracted': 'Comment', label_column: 'Llama3.1 Not Fine Tuned output'}, inplace=True)

        # Save to the output file
        output_df.to_csv(output_file, index=False)
        print(f"Successfully created compatible dataset: {output_file}")
        print(f"Dataset contains {len(output_df)} rows")
        return True

    except Exception as e:
        print(f"Error processing files {original_file} and {excel_file}: {str(e)}")
        return False

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

    # Process all three datasets
    datasets = [
        {
            'original': '/mnt/c/Users/yuvra/OneDrive/Desktop/Work/IISERKolkata/LLMsKnow/llama-3.1-frank-lampard-answers-frank_lampard.csv',
            'excel': '/mnt/c/Users/yuvra/OneDrive/Desktop/Work/IISERKolkata/LLMsKnow/Frank Lampard Ghost Goal Labels with Llama3.1_8b_Instruct using Alpaca Prompt Fine Tuned (9).xlsx',
            'output': '/mnt/c/Users/yuvra/OneDrive/Desktop/Work/IISERKolkata/LLMsKnow/frank_lampard_not_fine_tuned.csv'
        },
        {
            'original': '/mnt/c/Users/yuvra/OneDrive/Desktop/Work/IISERKolkata/LLMsKnow/llama-3.1-maradona-answers-maradona.csv',
            'excel': '/mnt/c/Users/yuvra/OneDrive/Desktop/Work/IISERKolkata/LLMsKnow/Maradon Hand of God Labels with Llama3.1_8b_Instruct using Alpaca Prompt Fine Tuned (10).xlsx',
            'output': '/mnt/c/Users/yuvra/OneDrive/Desktop/Work/IISERKolkata/LLMsKnow/maradona_not_fine_tuned.csv'
        },
        {
            'original': '/mnt/c/Users/yuvra/OneDrive/Desktop/Work/IISERKolkata/LLMsKnow/llama-3.1-luis-suarez-answers-luis_suarez.csv',
            'excel': '/mnt/c/Users/yuvra/OneDrive/Desktop/Work/IISERKolkata/LLMsKnow/Luis Suarez Handball Labels with Llama3.1_8b_Instruct using Alpaca Prompt Fine Tuned.xlsx',
            'output': '/mnt/c/Users/yuvra/OneDrive/Desktop/Work/IISERKolkata/LLMsKnow/luis_suarez_not_fine_tuned.csv'
        }
    ]

    # Process each dataset
    for dataset in datasets:
        print(f"\nProcessing dataset: {os.path.basename(dataset['original'])}")
        extract_comments_and_match_labels(
            dataset['original'], 
            dataset['excel'], 
            dataset['output'], 
            text_column='Comment', 
            label_column='Llama3.1 Not Fine Tuned output'
        )

    print("\nTo run generate_model_answers.py with the new datasets:")
    for dataset in datasets:
        dataset_name = os.path.basename(dataset['output']).replace('.csv', '')
        print(f"\npython src/generate_model_answers.py --model meta-llama/Llama-3.1-8B-Instruct --dataset {dataset_name} --n_samples 400 --excel_file \"{os.path.basename(dataset['excel'])}\"")

if __name__ == "__main__":
    main()
    print("All files have been processed.")