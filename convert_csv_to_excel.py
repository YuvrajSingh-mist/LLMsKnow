import pandas as pd
import os

def convert_csv_to_excel(csv_file, excel_file):
    """
    Convert a CSV file to an Excel file.

    Args:
        csv_file (str): Path to the input CSV file.
        excel_file (str): Path to the output Excel file.
    """
    # Check if the input file exists
    if not os.path.exists(csv_file):
        print(f"Error: Input file '{csv_file}' not found!")
        return False

    try:
        # Read the CSV file
        df = pd.read_csv(csv_file)

        # Save as an Excel file
        df.to_excel(excel_file, index=False)
        print(f"Successfully converted '{csv_file}' to '{excel_file}'")
        return True

    except Exception as e:
        print(f"Error converting file '{csv_file}': {str(e)}")
        return False

# Define the input CSV files and output Excel files
datasets = [
    {
        'csv': '/mnt/c/Users/yuvra/OneDrive/Desktop/Work/IISERKolkata/LLMsKnow/frank_lampard_random_samples.csv',
        'excel': '/mnt/c/Users/yuvra/OneDrive/Desktop/Work/IISERKolkata/LLMsKnow/frank_lampard_random_samples.xlsx'
    },
    {
        'csv': '/mnt/c/Users/yuvra/OneDrive/Desktop/Work/IISERKolkata/LLMsKnow/maradona_random_samples.csv',
        'excel': '/mnt/c/Users/yuvra/OneDrive/Desktop/Work/IISERKolkata/LLMsKnow/maradona_random_samples.xlsx'
    },
    {
        'csv': '/mnt/c/Users/yuvra/OneDrive/Desktop/Work/IISERKolkata/LLMsKnow/luis_suarez_random_samples.csv',
        'excel': '/mnt/c/Users/yuvra/OneDrive/Desktop/Work/IISERKolkata/LLMsKnow/luis_suarez_random_samples.xlsx'
    }
]

# Process each dataset
for dataset in datasets:
    print(f"\nProcessing dataset: {os.path.basename(dataset['csv'])}")
    convert_csv_to_excel(
        dataset['csv'], 
        dataset['excel']
    )