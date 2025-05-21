import os
import subprocess
import argparse

def run_with_hf_token(model="unsloth/Llama-3.1-8B-Instruct", 
                      dataset="frank_lampard_not_fine_tuned", 
                      n_samples=400, 
                      excel_file="frank_lampard_random_samples.xlsx",
                      seed=42,
                      probe_at="attention_output",
                      run_generation=True,
                      run_probing=True):
    """
    Run model generation and/or probing tasks with proper Hugging Face authentication.
    
    Args:
        model (str): The model ID to use.
        dataset (str): The dataset to use.
        n_samples (int): Number of samples to process.
        excel_file (str): Excel file with the dataset.
        seed (int): Random seed for reproducibility.
        probe_at (str): What to probe in the model.
        run_generation (bool): Whether to run the generation step.
        run_probing (bool): Whether to run the probing step.
    """
    # Set the Hugging Face token
    os.environ["HF_TOKEN"] = "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"
    
    if run_generation:
        print(f"Running model generation with {model} on {dataset}...")
        cmd = [
            "python", "src/generate_model_answers.py",
            "--model", model,
            "--dataset", dataset,
            "--n_samples", str(n_samples),
            "--excel_file", excel_file
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print("Error in generation step:")
            print(result.stderr)
        else:
            print("Generation completed successfully")
            print(result.stdout)
    
    if run_probing:
        print(f"Running probing with {model} on {dataset}...")
        cmd = [
            "python", "src/probe_all_layers_and_tokens.py",
            "--model", model,
            "--dataset", dataset,
            "--seed", str(seed),
            "--n_samples", str(n_samples),
            "--probe_at", probe_at
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print("Error in probing step:")
            print(result.stderr)
        else:
            print("Probing completed successfully")
            print(result.stdout)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model generation and probing with HF authentication")
    parser.add_argument("--model", default="unsloth/Llama-3.1-8B-Instruct", help="Model ID to use")
    parser.add_argument("--dataset", default="frank_lampard_not_fine_tuned", help="Dataset to use")
    parser.add_argument("--n_samples", type=int, default=400, help="Number of samples to process")
    parser.add_argument("--excel_file", default="frank_lampard_random_samples.xlsx", help="Excel file with the dataset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--probe_at", default="attention_output", help="What to probe in the model")
    parser.add_argument("--skip_generation", action="store_true", help="Skip the generation step")
    parser.add_argument("--skip_probing", action="store_true", help="Skip the probing step")
    
    args = parser.parse_args()
    
    run_with_hf_token(
        model=args.model,
        dataset=args.dataset,
        n_samples=args.n_samples,
        excel_file=args.excel_file,
        seed=args.seed,
        probe_at=args.probe_at,
        run_generation=not args.skip_generation,
        run_probing=not args.skip_probing
    )