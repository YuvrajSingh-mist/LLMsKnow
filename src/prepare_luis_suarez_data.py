#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd
import json
import re
import argparse
from tqdm import tqdm

def extract_finetuned(text):
    """Extract the response label from the model output text"""
    match = re.search(r'### Response:\s*(\w+)', text)
    label = match.group(1) if match else None
    return label

def prepare_alpaca_prompt(instruction, input_text, response=""):
    """Format the input using the Alpaca prompt template"""
    return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input_text}

### Response:
{response}"""

def convert_excel_to_qna(excel_file, output_file, system_instruction):
    """Convert Excel data to QnA format for use with the Luis Suarez model"""
    print(f"Loading Excel file: {excel_file}")
    
    # Read the Excel file
    df = pd.read_excel(excel_file)
    
    # Output dictionary to store QnA pairs
    output_data = []
    
    print("Converting data to QnA format...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        # Extract necessary fields from the Excel
        # Adjust these column names based on your Excel structure
        try:
            text = row.get('Text', '')
            label = row.get('Label', '')
            
            # Skip empty rows
            if not isinstance(text, str) or not text.strip():
                continue
                
            # Create input dict following the movie_qa format
            item = {
                "question": prepare_alpaca_prompt(system_instruction, text),
                "correct_answer": label,
                "automatic_correctness": 1  # Assuming all are correct in the training data
            }
            
            output_data.append(item)
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
    
    # Save as JSON or CSV
    if output_file.endswith('.json'):
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
    else:
        pd.DataFrame(output_data).to_csv(output_file, index=False)
    
    print(f"Converted {len(output_data)} items. Saved to {output_file}")
    return output_data

def main():
    parser = argparse.ArgumentParser(description="Convert Excel data to QnA format for Luis Suarez model")
    parser.add_argument("--excel_file", type=str, required=True, help="Path to the Excel file")
    parser.add_argument("--output_file", type=str, default="luis_suarez_data.json", help="Output file path")
    parser.add_argument("--format", type=str, choices=['json', 'csv'], default='csv', help="Output format")
    args = parser.parse_args()
    
    # Ensure output file has the correct extension
    if not args.output_file.endswith(f'.{args.format}'):
        args.output_file = f"{os.path.splitext(args.output_file)[0]}.{args.format}"
    
    # System instruction for stance detection
    system_instruction = ("You are provided with an input. You are required to perform stance detection "
                         "on the input with output as one of the following labels - Favor, Against, "
                         "Irrelevant, Neutral. The labels are self-explanatory. Remember to only use the labels provided "
                         "and do not mispell its name. Only output the stance detected label and nothing else.")
    
    # Convert Excel to QnA format
    convert_excel_to_qna(args.excel_file, args.output_file, system_instruction)

if __name__ == "__main__":
    main()