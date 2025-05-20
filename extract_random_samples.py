import pandas as pd
import random
import os

def extract_random_samples(excel_file, output_file, n_samples=400):
    """
    Extract n random samples from an Excel file and save to a new CSV file.

    Args:
        excel_file (str): Path to the input Excel file.
        output_file (str): Path to the output CSV file.
        n_samples (int): Number of random samples to extract.
    """
    # Check if the input file exists
    if not os.path.exists(excel_file):
        print(f"Error: Input file '{excel_file}' not found!")
        return False

    try:
        # Read the Excel file
        df = pd.read_excel(excel_file)

        # Check if the file has enough rows
        if len(df) < n_samples:
            print(f"Warning: The file '{excel_file}' has only {len(df)} rows. Extracting all rows.")
            n_samples = len(df)

        # Randomly sample n rows
        sampled_df = df.sample(n=n_samples, random_state=42)

        # Save the sampled rows to a CSV file
        sampled_df.to_csv(output_file, index=False)
        print(f"Successfully extracted {n_samples} samples from '{excel_file}' to '{output_file}'")
        return True

    except Exception as e:
        print(f"Error processing file '{excel_file}': {str(e)}")
        return False

# Define the input Excel files and output CSV files
datasets = [
    {
        'excel': '/mnt/c/Users/yuvra/OneDrive/Desktop/Work/IISERKolkata/LLMsKnow/Frank Lampard Ghost Goal Labels with Llama3.1_8b_Instruct using Alpaca Prompt Fine Tuned (9).xlsx',
        'output': '/mnt/c/Users/yuvra/OneDrive/Desktop/Work/IISERKolkata/LLMsKnow/frank_lampard_random_samples.csv'
    },
    {
        'excel': '/mnt/c/Users/yuvra/OneDrive/Desktop/Work/IISERKolkata/LLMsKnow/Maradon Hand of God Labels with Llama3.1_8b_Instruct using Alpaca Prompt Fine Tuned (10).xlsx',
        'output': '/mnt/c/Users/yuvra/OneDrive/Desktop/Work/IISERKolkata/LLMsKnow/maradona_random_samples.csv'
    },
    {
        'excel': '/mnt/c/Users/yuvra/OneDrive/Desktop/Work/IISERKolkata/LLMsKnow/Luis Suarez Handball Labels with Llama3.1_8b_Instruct using Alpaca Prompt Fine Tuned.xlsx',
        'output': '/mnt/c/Users/yuvra/OneDrive/Desktop/Work/IISERKolkata/LLMsKnow/luis_suarez_random_samples.csv'
    }
]

# Process each dataset
for dataset in datasets:
    print(f"\nProcessing dataset: {os.path.basename(dataset['excel'])}")
    extract_random_samples(
        dataset['excel'], 
        dataset['output'], 
        n_samples=400
    )