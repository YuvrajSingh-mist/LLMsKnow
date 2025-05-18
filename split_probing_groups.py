#!/usr/bin/env python3
import pandas as pd

# Load the Frank Lampard dataset
input_file = '/kaggle/working//llama-3.1-frank-lampard-answers-frank_lampard.csv'
df = pd.read_csv(input_file)

# Create output file paths
output_dir = '/kaggle/working/LLMsKnow/probing_groups'
true_file = f"{output_dir}/frank_lampard_true_predictions.csv"
false_file = f"{output_dir}/frank_lampard_false_predictions.csv"

# Filter data into TP/TN and FP/FN groups
# True predictions: automatic_correctness = 1
tp_tn_df = df[df['automatic_correctness'] == 1].copy()

# False predictions: automatic_correctness = 0
fp_fn_df = df[df['automatic_correctness'] == 0].copy()

# Save to CSV files
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

# Also count stance types in each group
print("\nStance distribution in TP/TN group:")
print(tp_tn_df['correct_answer'].value_counts())

print("\nStance distribution in FP/FN group:")
print(fp_fn_df['correct_answer'].value_counts())

print(f"\nFiles created: {true_file} and {false_file}")
