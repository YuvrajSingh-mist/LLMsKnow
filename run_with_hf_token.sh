#!/bin/bash

# Set Hugging Face token for accessing gated models
export HF_TOKEN="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"

# Run the model generation script with proper authentication
python src/generate_model_answers.py \
  --model "unsloth/Llama-3.1-8B-Instruct" \
  --dataset "frank_lampard_not_fine_tuned" \
  --n_samples 400 \
  --excel_file "frank_lampard_random_samples.xlsx"

# Run the probing script with the same authentication
python src/probe_all_layers_and_tokens.py \
  --model "unsloth/Llama-3.1-8B-Instruct" \
  --dataset "frank_lampard_not_fine_tuned" \
  --seed 42 \
  --n_samples 400 \
  --probe_at attention_output

echo "Processing completed."