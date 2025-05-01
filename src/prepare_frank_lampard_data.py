#!/usr/bin/env python
"""
Script to prepare the Frank Lampard Ghost Goal dataset for probing analysis
"""

import pandas as pd
import torch
import argparse
from tqdm import tqdm
import os
from transformers import AutoTokenizer

from probing_utils import MODEL_FRIENDLY_NAMES, tokenize, encode


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare the Frank Lampard Ghost Goal dataset for analysis")
    parser.add_argument("--model", default="YuvrajSingh9886/Llama3.1-8b-Frank-Lampard", 
                        help="Model to use for tokenization")
    return parser.parse_args()


def load_excel_data(filepath):
    """Load and process the Frank Lampard Ghost Goal dataset from Excel."""
    print(f"Loading data from {filepath}...")
    
    # Load Excel file
    df = pd.read_excel(filepath)
    
    # Rename columns to match expected format
    column_mapping = {
        'Question': 'question',
        'Model Answer': 'answer',
        'Reference Answer': 'correct_answer',
        'Is Correct (1/0)': 'automatic_correctness',
        'Exact Answer': 'exact_answer',
        'Valid Exact Answer': 'valid_exact_answer'
    }
    
    # Apply column mapping where columns exist
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns:
            df[new_col] = df[old_col]
    
    # Ensure required columns exist
    required_cols = ['question', 'answer', 'correct_answer', 'automatic_correctness']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in the dataset")
    
    # Add exact_answer and valid_exact_answer columns if they don't exist
    if 'exact_answer' not in df.columns:
        df['exact_answer'] = df['answer'].apply(extract_exact_answer)
    
    if 'valid_exact_answer' not in df.columns:
        df['valid_exact_answer'] = df['exact_answer'].notna().astype(int)
    
    print(f"Loaded {len(df)} examples")
    print(f"Distribution of correct/incorrect answers: {df['automatic_correctness'].value_counts()}")
    
    return df


def extract_exact_answer(text):
    """Extract the exact answer from the text."""
    # This is a simple implementation
    # You may need more sophisticated extraction based on your data patterns
    if pd.isna(text):
        return "NO ANSWER"
    
    # For example, if answers typically contain "The answer is X" pattern
    if "the answer is" in text.lower():
        parts = text.lower().split("the answer is")
        if len(parts) > 1:
            return parts[1].strip()
    
    # Otherwise return the full answer
    return text


def tokenize_dataset(df, tokenizer, model_name):
    """Tokenize the dataset for the specified model."""
    print("Tokenizing questions and answers...")
    
    input_output_ids = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        question = row['question']
        
        # Tokenize question using the model's format
        token_ids = tokenize(question, tokenizer, model_name)
        
        input_output_ids.append(token_ids)
    
    return input_output_ids


def main():
    args = parse_args()
    model_name = args.model
    
    # File paths
    excel_file = "/mnt/c/Users/yuvra/OneDrive/Desktop/Work/IISERKolkata/LLMsKnow/Frank Lampard Ghost Goal Labels with Llama3.1_8b_Instruct using Alpaca Prompt Fine Tuned (9).xlsx"
    output_csv = f"{MODEL_FRIENDLY_NAMES[model_name]}-answers-frank_lampard.csv"
    output_pt = f"{MODEL_FRIENDLY_NAMES[model_name]}-input_output_ids-frank_lampard.pt"
    
    # Load tokenizer
    print(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load and process data
    df = load_excel_data(excel_file)
    
    # Tokenize dataset
    input_output_ids = tokenize_dataset(df, tokenizer, model_name)
    
    # Save processed data
    print(f"Saving processed data to {output_csv} and {output_pt}...")
    df.to_csv(output_csv, index=False)
    torch.save(input_output_ids, output_pt)
    
    print("Data preparation complete!")


if __name__ == "__main__":
    main()