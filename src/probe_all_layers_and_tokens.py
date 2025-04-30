import argparse
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import wandb
from transformers import set_seed

from probing_utils import extract_internal_reps_all_layers_and_tokens, load_model_and_validate_gpu, \
    probe_specific_layer_token, N_LAYERS, compile_probing_indices, LIST_OF_DATASETS, LIST_OF_MODELS, \
    MODEL_FRIENDLY_NAMES, prepare_for_probing, LIST_OF_PROBING_LOCATIONS


def parse_args_and_init_wandb():
    parser = argparse.ArgumentParser(
        description='Probe for hallucinations and create plots')
    parser.add_argument("--model", choices=LIST_OF_MODELS)
    parser.add_argument("--probe_at", choices=LIST_OF_PROBING_LOCATIONS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--n_samples", default=1000, help="Usually you would limit to 1000 due to memory constraints")
    parser.add_argument("--dataset", 
                        required=True)  # Removed choices restriction to allow custom datasets
    parser.add_argument("--use_mlp", action='store_true', default=False)

    args = parser.parse_args()

    wandb.init(
        project="probe_hallucinations",
        config=vars(args)
    )

    return args


def probe_all(model, tokenizer, data, input_output_ids, tokens_to_probe, layers_to_probe, training_data_indices,
              validation_data_indices, probe_at, seed, model_name, use_dict_for_tokens=True):
    _, _, input_output_ids_train, input_output_ids_valid, y_train, y_valid, exact_answer_train, exact_answer_valid, \
    validity_of_exact_answer_train, validity_of_exact_answer_valid, questions_train, questions_valid = prepare_for_probing(
        data, input_output_ids, training_data_indices, validation_data_indices)
    
    # Print class distribution to help debug class imbalance issues
    print(f"Class distribution in training data: {sum(y_train)}/{len(y_train)} positive examples")
    print(f"Class distribution in validation data: {sum(y_valid)}/{len(y_valid)} positive examples")

    extracted_embeddings_train = \
        extract_internal_reps_all_layers_and_tokens(model, input_output_ids_train, probe_at, model_name)
    extracted_embeddings_valid = \
        extract_internal_reps_all_layers_and_tokens(model, input_output_ids_valid, probe_at, model_name)

    all_metrics = defaultdict(list)

    for token in tokens_to_probe:
        print(f"############ token {token} ################")
        metrics_per_layer = defaultdict(list)
        for layer in layers_to_probe:

            metrics = probe_specific_layer_token(extracted_embeddings_train,
                                                 extracted_embeddings_valid, layer, token,
                                                 questions_train, questions_valid,
                                                 input_output_ids_train, input_output_ids_valid,
                                                 exact_answer_train, exact_answer_valid,
                                                 validity_of_exact_answer_train,
                                                 validity_of_exact_answer_valid,
                                                 tokenizer,
                                                 y_train, y_valid, seed, model_name,
                                                 use_dict_for_tokens=use_dict_for_tokens)
            for m in metrics:
                metrics_per_layer[m].append(metrics[m])

        for m in metrics_per_layer:
            all_metrics[m].append(metrics_per_layer[m])

    return all_metrics

def log_metrics(all_metrics, tokens_to_probe):
    for metric_name, metric_value in all_metrics.items():
        # if name == 'clf':
        #     continue
        dict_of_results = {}
        lst_of_results = []
        for idx, token in enumerate(tokens_to_probe):
            dict_of_results[str(token).replace('_token', '')] = metric_value[idx]
            for layer, value in enumerate(metric_value[idx]):
                lst_of_results.append(((layer, token), value))
        results_pd = pd.DataFrame.from_dict(dict_of_results)
        lst_of_results.sort(key=lambda x: x[1], reverse=True)

        plt.figure(figsize=(12, 10))
        if metric_name == 'auc':
            center = 0.5
        else:
            center = 0
        ax = sns.heatmap(results_pd, annot=True, cmap='Blues', vmin=center, vmax=1.0)
        plt.xlabel('Token')
        plt.ylabel('Layer')
        
        # Save locally with timestamp to avoid overwriting
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        local_file_path = f"heatmap_{metric_name}_{timestamp}.png"
        ax.figure.savefig(local_file_path, bbox_inches="tight")
        print(f"Saved heatmap for {metric_name} to {local_file_path}")
        
        # Also save results as CSV
        local_csv_path = f"results_{metric_name}_{timestamp}.csv"
        results_pd.to_csv(local_csv_path, index=True)
        print(f"Saved results for {metric_name} to {local_csv_path}")
        
        # Print top 5 results for this metric
        print(f"\nTop 5 results for {metric_name}:")
        for rank in range(0, 5):
            layer = lst_of_results[rank][0][0]
            token = lst_of_results[rank][0][1]
            value = lst_of_results[rank][1]
            print(f"  {rank+1}. Layer: {layer}, Token: {token}, Value: {value:.4f}")
        
        # Continue with wandb logging if available
        try:
            wandb_file = f"{wandb.run.name}_temp_fig.png"
            ax.figure.savefig(wandb_file, bbox_inches="tight")
            wandb.log({f"{metric_name}_heatmap": wandb.Image(wandb_file)})
            for rank in range(0, 5):
                wandb.summary[f"{metric_name}_top_{rank + 1}"] = {"layer": lst_of_results[rank][0][0],
                                                                  "token": lst_of_results[rank][0][1],
                                                                  "value": lst_of_results[rank][1]}

            results_pd.to_csv(f"{wandb.run.name}_temp_df.csv", index=False)
            artifact = wandb.Artifact(name=f"{metric_name}_df", type="dataframe")
            artifact.add_file(local_path=f"{wandb.run.name}_temp_df.csv")
            wandb.log_artifact(artifact)

            os.remove(f"{wandb.run.name}_temp_fig.png")
            os.remove(f"{wandb.run.name}_temp_df.csv")
        except Exception as e:
            print(f"Warning: Weights & Biases logging failed: {str(e)}")


def main():
    args = parse_args_and_init_wandb()
    set_seed(args.seed)
    
    # Use local paths instead of Kaggle paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_output_file = f"{base_dir}/{MODEL_FRIENDLY_NAMES[args.model] if args.model in MODEL_FRIENDLY_NAMES else 'llama-3.1'}-answers-{args.dataset}.csv"
    
    # Check if specific file exists first
    if not os.path.exists(model_output_file):
        # Fall back to files in the current directory that match the dataset
        csv_files = [f for f in os.listdir(base_dir) if f.endswith(f'-{args.dataset}.csv')]
        if csv_files:
            model_output_file = f"{base_dir}/{csv_files[0]}"
            print(f"Using found CSV file: {model_output_file}")
        else:
            print(f"Warning: Could not find a CSV file for dataset {args.dataset}")
    
    print(f"Loading data from: {model_output_file}")
    data = pd.read_csv(model_output_file)
    
    # Check if PT file exists
    pt_file = f"{base_dir}/{MODEL_FRIENDLY_NAMES[args.model] if args.model in MODEL_FRIENDLY_NAMES else 'llama-3.1'}-input_output_ids-{args.dataset}.pt"
    if not os.path.exists(pt_file):
        pt_files = [f for f in os.listdir(base_dir) if f.endswith(f'-input_output_ids-{args.dataset}.pt')]
        if pt_files:
            pt_file = f"{base_dir}/{pt_files[0]}"
            print(f"Using found PT file: {pt_file}")
        else:
            print(f"Warning: Could not find a PT file for dataset {args.dataset}")
    
    print(f"Loading input_output_ids from: {pt_file}")
    input_output_ids = torch.load(pt_file)
    
    model, tokenizer = load_model_and_validate_gpu(args.model)

    if args.dataset == 'imdb':
        tokens_to_probe = ['last_q_token', 'exact_answer_first_token', 'exact_answer_last_token', 'exact_answer_after_last_token',
                           -8, -7, -6, -5, -4, -3, -2, -1]
    elif args.dataset == 'luis_suarez':
        # Custom tokens for stance detection task
        tokens_to_probe = ['last_q_token', 'first_answer_token', 'second_answer_token',
                           'exact_answer_first_token', 'exact_answer_last_token',
                           -8, -7, -6, -5, -4, -3, -2, -1]
        
        # Ensure the data has valid_exact_answer column
        if 'valid_exact_answer' not in data.columns:
            data['valid_exact_answer'] = data['exact_answer'].notna().astype(int)
    else:
        tokens_to_probe = ['last_q_token', 'first_answer_token', 'second_answer_token',
                           'exact_answer_before_first_token',
                           'exact_answer_first_token', 'exact_answer_last_token', 'exact_answer_after_last_token',
                           -8, -7, -6, -5, -4, -3, -2, -1]

    training_data_indices, validation_data_indices = compile_probing_indices(data, args.n_samples,
                                                                             args.seed)

    # Check if the model exists in N_LAYERS, if not use a default
    if args.model not in N_LAYERS:
        print(f"Warning: Model {args.model} not found in N_LAYERS dictionary. Using default layer count of 32.")
        num_layers = 32
    else:
        num_layers = N_LAYERS[args.model]

    all_metrics = probe_all(model, tokenizer, data, input_output_ids, tokens_to_probe,
                            range(0, num_layers),
                            training_data_indices,
                            validation_data_indices, args.probe_at, args.seed,
                            args.model, use_dict_for_tokens=True)

    log_metrics(all_metrics, tokens_to_probe)


if __name__ == "__main__":
    main()
