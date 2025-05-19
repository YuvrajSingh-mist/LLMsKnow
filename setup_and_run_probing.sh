#!/bin/bash
# Script to set up the Kaggle environment and run probing commands

# Set script to exit on any error
set -e

# Default values
MODEL="YuvrajSingh9886/Llama3.1-8b-Frank-Lampard"
DATASET="frank_lampard"
PROBE_TYPE="frank_lampard_true"
PROBE_AT="mlp"
LAYER=20
TOKEN="exact_answer_first_token"
SEEDS="0 1 2"
N_SAMPLES=260

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --model=*)
      MODEL="${1#*=}"
      shift
      ;;
    --dataset=*)
      DATASET="${1#*=}"
      shift
      ;;
    --probe-type=*)
      PROBE_TYPE="${1#*=}"
      shift
      ;;
    --probe-at=*)
      PROBE_AT="${1#*=}"
      shift
      ;;
    --layer=*)
      LAYER="${1#*=}"
      shift
      ;;
    --token=*)
      TOKEN="${1#*=}"
      shift
      ;;
    --seeds=*)
      SEEDS="${1#*=}"
      shift
      ;;
    --n-samples=*)
      N_SAMPLES="${1#*=}"
      shift
      ;;
    --help)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  --model=MODEL           Model name (default: YuvrajSingh9886/Llama3.1-8b-Frank-Lampard)"
      echo "  --dataset=DATASET       Base dataset name (default: frank_lampard)"
      echo "  --probe-type=TYPE       Probe type (default: frank_lampard_true)"
      echo "  --probe-at=LOCATION     Probe location (default: mlp)"
      echo "  --layer=LAYER           Layer to probe (default: 20)"
      echo "  --token=TOKEN           Token to probe (default: exact_answer_first_token)"
      echo "  --seeds=SEEDS           Space-separated list of seeds (default: 0 1 2)"
      echo "  --n-samples=N           Number of samples to use (default: 260)"
      echo "  --help                  Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Determine if we're in Kaggle or not
if [ -d "/kaggle/working" ]; then
  echo "Running in Kaggle environment"
  # In Kaggle, we need to organize our files
  KAGGLE_DIR="/kaggle/working"
  LLMK_DIR="${KAGGLE_DIR}/LLMsKnow"
  
  # Create LLMsKnow directory if it doesn't exist
  mkdir -p "${LLMK_DIR}/src"
  mkdir -p "${LLMK_DIR}/probing_groups"

  # Copy probing groups files
  if [ -d "./probing_groups" ]; then
    cp -r ./probing_groups/* "${LLMK_DIR}/probing_groups/"
    echo "Copied probing group files to ${LLMK_DIR}/probing_groups/"
  else
    echo "Warning: probing_groups directory not found in current directory"
    # Try to create probing groups files from the base dataset
    echo "Attempting to create probing groups from base dataset"
    if [ -f "./split_probing_groups.py" ]; then
      cp ./split_probing_groups.py "${LLMK_DIR}/"
      cd "${LLMK_DIR}"
      python split_probing_groups.py "${DATASET}" "llama-3.1-${DATASET//_/-}" 
      cd -
    fi
  fi
  
  # Copy source files
  if [ -d "./src" ]; then
    cp ./src/*.py "${LLMK_DIR}/src/"
    echo "Copied source files to ${LLMK_DIR}/src/"
  else
    echo "Error: src directory not found"
    exit 1
  fi
  
  # Copy dataset files if they exist
  for file in *-answers-*.csv; do
    if [ -f "$file" ]; then
      cp "$file" "${KAGGLE_DIR}/"
      echo "Copied $file to ${KAGGLE_DIR}/"
    fi
  done
  
  # Copy tensor files if they exist
  for file in *-input_output_ids-*.pt; do
    if [ -f "$file" ]; then
      cp "$file" "${KAGGLE_DIR}/"
      echo "Copied $file to ${KAGGLE_DIR}/"
    fi
  done
  
  # Change to LLMsKnow directory
  cd "${LLMK_DIR}"

  # Run the probing command
  echo "Running probe_all_layers_and_tokens.py with:"
  echo "  Model: $MODEL"
  echo "  Dataset: $PROBE_TYPE"
  echo "  Probe at: $PROBE_AT"
  echo "  Seeds: $SEEDS"
  echo "  N samples: $N_SAMPLES"
  
  for seed in $SEEDS; do
    echo "Running with seed $seed"
    python src/probe_all_layers_and_tokens.py \
      --model "$MODEL" \
      --dataset "$PROBE_TYPE" \
      --seed "$seed" \
      --n_samples "$N_SAMPLES" \
      --probe_at "$PROBE_AT"
  done
  
  echo "All probing runs complete!"
  
else
  # We're not in Kaggle, so we just run directly
  echo "Running in local environment"
  
  # First, ensure probing groups exist
  if [ ! -d "./probing_groups" ] || [ ! -f "./probing_groups/${PROBE_TYPE}_predictions.csv" ]; then
    echo "Creating probing groups from the base dataset"
    python split_probing_groups.py "${DATASET}" "llama-3.1-${DATASET//_/-}"
  fi
  
  # Run the probing command for each seed
  for seed in $SEEDS; do
    echo "Running with seed $seed"
    python src/probe_all_layers_and_tokens.py \
      --model "$MODEL" \
      --dataset "$PROBE_TYPE" \
      --seed "$seed" \
      --n_samples "$N_SAMPLES" \
      --probe_at "$PROBE_AT"
  done
  
  echo "All probing runs complete!"
fi

# Show a summary of the results
echo "Results and output files:"
if [ -d "/kaggle/working/LLMsKnow/wandb" ]; then
  echo "WandB logs: /kaggle/working/LLMsKnow/wandb/"
elif [ -d "./wandb" ]; then
  echo "WandB logs: ./wandb/"
fi

# List generated CSV and heatmap files
find . -name "results_*.csv" -o -name "heatmap_*.png" | sort