#!/usr/bin/env python
"""
Specialized script for analyzing the Frank Lampard Ghost Goal dataset,
particularly focused on handling class imbalance issues.
"""

import argparse
import os
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

from probing_utils import extract_internal_reps_specific_layer_and_token, load_model_and_validate_gpu
from probing_utils import extract_internal_reps_all_layers_and_tokens, LIST_OF_MODELS, LIST_OF_DATASETS
from probing_utils import MODEL_FRIENDLY_NAMES, prepare_for_probing, compile_probing_indices


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze Frank Lampard dataset with balanced class handling')
    parser.add_argument("--model", default="YuvrajSingh9886/Llama3.1-8b-Frank-Lampard")
    parser.add_argument("--dataset", default="frank_lampard")
    parser.add_argument("--n_samples", default='all')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--probe_at", default='mlp')
    return parser.parse_args()


def load_data(model_name, dataset_name):
    """Load the dataset with model answers."""
    filename = f"{MODEL_FRIENDLY_NAMES[model_name]}-answers-{dataset_name}.csv"
    
    if not os.path.exists(filename):
        csv_files = [f for f in os.listdir() if f.endswith(f'-answers-{dataset_name}.csv')]
        if csv_files:
            filename = csv_files[0]
        else:
            raise FileNotFoundError(f"Could not find answers file for {dataset_name}")
    
    print(f"Loading data from: {filename}")
    data = pd.read_csv(filename)
    print(f"Loaded {len(data)} examples")
    print(f"Class distribution: {data['automatic_correctness'].value_counts(normalize=True) * 100:.2f}% correct")
    
    # Load input_output_ids
    pt_file = f"{MODEL_FRIENDLY_NAMES[model_name]}-input_output_ids-{dataset_name}.pt"
    if not os.path.exists(pt_file):
        pt_files = [f for f in os.listdir() if f.endswith(f'-input_output_ids-{dataset_name}.pt')]
        if pt_files:
            pt_file = pt_files[0]
        else:
            raise FileNotFoundError(f"Could not find input_output_ids for {dataset_name}")
    
    print(f"Loading embeddings from: {pt_file}")
    input_output_ids = torch.load(pt_file)
    
    return data, input_output_ids


def extract_activations(model, tokenizer, data, input_output_ids, token_positions, layers_range, probe_at, model_name):
    """Extract activations for specific layer-token combinations with balanced class sampling."""
    
    # Get indices of correct and incorrect examples
    correct_indices = data[data['automatic_correctness'] == 1].index.tolist()
    incorrect_indices = data[data['automatic_correctness'] == 0].index.tolist()
    
    print(f"Found {len(correct_indices)} correct examples and {len(incorrect_indices)} incorrect examples")
    
    # Balance classes by undersampling the majority class
    n_samples_per_class = min(len(correct_indices), len(incorrect_indices))
    if len(correct_indices) > n_samples_per_class:
        np.random.seed(42)
        correct_indices = np.random.choice(correct_indices, size=n_samples_per_class, replace=False)
    if len(incorrect_indices) > n_samples_per_class:
        np.random.seed(42)
        incorrect_indices = np.random.choice(incorrect_indices, size=n_samples_per_class, replace=False)
    
    balanced_indices = np.concatenate([correct_indices, incorrect_indices])
    np.random.shuffle(balanced_indices)
    
    print(f"Using {len(balanced_indices)} balanced examples ({n_samples_per_class} per class)")
    
    # Split data
    train_indices, val_indices = train_test_split(balanced_indices, test_size=0.2, random_state=42)
    
    # Extract activations
    activations_by_layer_token = {}
    
    for token in token_positions:
        print(f"\nExtracting activations for token: {token}")
        
        for layer in tqdm(layers_range, desc="Processing layers"):
            # Extract for train set
            train_activations = extract_internal_reps_specific_layer_and_token(
                model, tokenizer, 
                data.iloc[train_indices]['question'].tolist(), 
                [input_output_ids[i] for i in train_indices],
                probe_at, model_name, layer, token,
                data.iloc[train_indices]['exact_answer'].tolist(),
                data.iloc[train_indices]['valid_exact_answer'].astype(int).tolist()
            )
            
            # Extract for val set
            val_activations = extract_internal_reps_specific_layer_and_token(
                model, tokenizer, 
                data.iloc[val_indices]['question'].tolist(), 
                [input_output_ids[i] for i in val_indices],
                probe_at, model_name, layer, token,
                data.iloc[val_indices]['exact_answer'].tolist(),
                data.iloc[val_indices]['valid_exact_answer'].astype(int).tolist()
            )
            
            key = f"layer{layer}_token{token}"
            activations_by_layer_token[key] = {
                'train': {
                    'X': train_activations,
                    'y': data.iloc[train_indices]['automatic_correctness'].values
                },
                'val': {
                    'X': val_activations,
                    'y': data.iloc[val_indices]['automatic_correctness'].values
                }
            }
    
    return activations_by_layer_token


def train_balanced_probes(activations_by_layer_token, seed=42):
    """Train logistic regression probes with handling for class imbalance."""
    results = {}
    
    for key, data in tqdm(activations_by_layer_token.items(), desc="Training probes"):
        X_train = data['train']['X']
        y_train = data['train']['y']
        X_val = data['val']['X'] 
        y_val = data['val']['y']
        
        # Try multiple solvers to find the best one
        solvers = ['liblinear', 'saga', 'lbfgs']
        best_auc = 0
        best_clf = None
        
        for solver in solvers:
            try:
                clf = LogisticRegression(
                    random_state=seed,
                    class_weight='balanced',
                    max_iter=2000,
                    solver=solver,
                    tol=1e-4
                )
                clf.fit(X_train, y_train)
                
                # Check if the classifier is making varied predictions
                val_preds = clf.predict(X_val)
                if len(np.unique(val_preds)) > 1:
                    # Get probabilities for AUC
                    val_probs = clf.predict_proba(X_val)[:, 1]
                    auc = roc_auc_score(y_val, val_probs)
                    
                    if auc > best_auc:
                        best_auc = auc
                        best_clf = clf
            except Exception as e:
                print(f"Error with solver {solver}: {e}")
        
        if best_clf is None:
            print(f"Failed to find good classifier for {key}. Using basic model.")
            clf = LogisticRegression(random_state=seed, class_weight='balanced')
            clf.fit(X_train, y_train)
            val_probs = clf.predict_proba(X_val)[:, 1]
            metrics = {
                'auc': 0.5,  # Default for random performance
                'acc': (y_val == y_val.mean() > 0.5).mean(),  # Accuracy of always predicting majority class
                'f1': 0.0
            }
        else:
            clf = best_clf
            val_preds = clf.predict(X_val)
            val_probs = clf.predict_proba(X_val)[:, 1]
            
            metrics = {
                'auc': roc_auc_score(y_val, val_probs),
                'acc': (y_val == val_preds).mean(),
                'f1': f1_score(y_val, val_preds, average='binary')
            }
        
        results[key] = {
            'clf': clf,
            'metrics': metrics
        }
        
    return results


def create_heatmap(results, layers_range, token_positions, metric_name='auc'):
    """Create a heatmap of results for the specified metric."""
    heatmap_data = np.zeros((len(layers_range), len(token_positions)))
    
    for i, layer in enumerate(layers_range):
        for j, token in enumerate(token_positions):
            key = f"layer{layer}_token{token}"
            if key in results:
                heatmap_data[i, j] = results[key]['metrics'][metric_name]
    
    # Create dataframe for heatmap
    df = pd.DataFrame(
        heatmap_data, 
        index=layers_range, 
        columns=[str(t).replace('_token', '') for t in token_positions]
    )
    
    # Plot heatmap
    plt.figure(figsize=(14, 10))
    if metric_name == 'auc':
        vmin, vmax = 0.5, 1.0
    else:
        vmin, vmax = 0, 1.0
        
    ax = sns.heatmap(df, annot=True, cmap='viridis', fmt='.2f', vmin=vmin, vmax=vmax)
    plt.title(f'{metric_name.upper()} Scores for Frank Lampard Dataset ({args.probe_at})')
    plt.xlabel('Token')
    plt.ylabel('Layer')
    
    # Save plot
    plt.tight_layout()
    plt.savefig(f'frank_lampard_{metric_name}_{args.probe_at}.png', dpi=300)
    print(f"Saved heatmap to frank_lampard_{metric_name}_{args.probe_at}.png")
    
    # Find top 5 layer-token combinations
    flat_indices = np.argsort(heatmap_data.flatten())[-5:][::-1]
    
    print(f"\nTop 5 layer-token combinations for {metric_name}:")
    for i, idx in enumerate(flat_indices):
        layer_idx = idx // len(token_positions)
        token_idx = idx % len(token_positions)
        layer = layers_range[layer_idx]
        token = token_positions[token_idx]
        score = heatmap_data[layer_idx, token_idx]
        print(f"  {i+1}. Layer {layer}, Token {token}: {metric_name}={score:.4f}")
    
    return heatmap_data


def analyze_top_neurons(results, data, input_output_ids, model, tokenizer, top_layer_token_pairs, model_name, probe_at):
    """Analyze which individual neurons are most important for prediction."""
    print("\nAnalyzing top neurons in important layer-token combinations...")
    
    # For each top layer-token pair
    for layer, token in top_layer_token_pairs:
        key = f"layer{layer}_token{token}"
        if key not in results:
            continue
            
        clf = results[key]['clf']
        
        if not hasattr(clf, 'coef_'):
            print(f"Classifier for {key} doesn't have coefficient attributes.")
            continue
            
        # Get feature importance from model coefficients
        feature_importance = np.abs(clf.coef_[0])
        top_neuron_indices = np.argsort(feature_importance)[-10:][::-1]
        
        print(f"\nTop 10 most important neurons for Layer {layer}, Token {token}:")
        for i, neuron_idx in enumerate(top_neuron_indices):
            importance = feature_importance[neuron_idx]
            print(f"  {i+1}. Neuron {neuron_idx}: importance={importance:.4f}")
        
        # Plot neuron importance
        plt.figure(figsize=(10, 6))
        plt.bar(range(10), feature_importance[top_neuron_indices])
        plt.xticks(range(10), [f"N{i}" for i in top_neuron_indices])
        plt.title(f"Top 10 Important Neurons for Layer {layer}, Token {token}")
        plt.xlabel("Neuron Index")
        plt.ylabel("Importance (|Coefficient|)")
        plt.tight_layout()
        plt.savefig(f"frank_lampard_neurons_l{layer}_t{token}_{probe_at}.png", dpi=300)


def main():
    global args
    args = parse_args()
    
    print(f"Analyzing Frank Lampard dataset with {args.probe_at} probing...")
    print(f"Using model: {args.model}")
    
    # Load data
    data, input_output_ids = load_data(args.model, args.dataset)
    
    # Ensure data has required columns
    if 'valid_exact_answer' not in data.columns and 'exact_answer' in data.columns:
        data['valid_exact_answer'] = data['exact_answer'].notna().astype(int)
    
    # Load model
    model, tokenizer = load_model_and_validate_gpu(args.model)
    
    # Define token positions to probe (same as for luis_suarez)
    tokens_to_probe = [
        'last_q_token', 'first_answer_token', 'second_answer_token',
        'exact_answer_first_token', 'exact_answer_last_token',
        -8, -7, -6, -5, -4, -3, -2, -1
    ]
    
    # Define layers to probe
    layers_to_probe = list(range(32))
    
    # Extract balanced activations
    print("\nExtracting activations with balanced classes...")
    activations = extract_activations(
        model, tokenizer, data, input_output_ids, 
        tokens_to_probe, layers_to_probe, args.probe_at, args.model
    )
    
    # Train balanced probes
    print("\nTraining balanced probes...")
    results = train_balanced_probes(activations, seed=args.seed)
    
    # Create heatmaps
    print("\nCreating heatmaps...")
    auc_heatmap = create_heatmap(results, layers_to_probe, tokens_to_probe, 'auc')
    f1_heatmap = create_heatmap(results, layers_to_probe, tokens_to_probe, 'f1')
    
    # Get top 5 layer-token pairs based on AUC
    flat_indices = np.argsort(auc_heatmap.flatten())[-5:][::-1]
    top_pairs = [(layers_to_probe[idx // len(tokens_to_probe)], 
                 tokens_to_probe[idx % len(tokens_to_probe)]) 
                 for idx in flat_indices]
    
    # Analyze top neurons
    analyze_top_neurons(results, data, input_output_ids, model, tokenizer, top_pairs, args.model, args.probe_at)
    
    # Save results
    with open(f'frank_lampard_results_{args.probe_at}.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print("\nAnalysis complete! Results are saved to files.")


if __name__ == "__main__":
    main()