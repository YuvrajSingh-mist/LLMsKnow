import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import wandb
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import set_seed

from compute_correctness import compute_correctness
from probing_utils import load_model_and_validate_gpu, tokenize, generate, LIST_OF_DATASETS, MODEL_FRIENDLY_NAMES, \
    LIST_OF_MODELS

# Import the probing group utilities if available
try:
    from probing_group_utils import load_probing_group, probing_groups
    HAS_PROBING_GROUPS = True
except ImportError:
    HAS_PROBING_GROUPS = False


def parse_args():
    parser = argparse.ArgumentParser(description="A script for generating model answers and outputting to csv")
    parser.add_argument("--model",
                        choices=LIST_OF_MODELS,
                        required=True)
    
    # Allow any dataset name (including probing groups) by removing choices restriction
    parser.add_argument("--dataset", 
                        help='Dataset to use for generating answers', 
                        required=True)
    
    parser.add_argument("--verbose", action='store_true', help='print more information')
    parser.add_argument("--n_samples", type=int, help='number of examples to use', default=None)
    parser.add_argument("--excel_file", type=str, help='path to Excel file or CSV for stance detection', default=None)

    return parser.parse_args()


def load_data_movies(test=False):
    file_name = 'movie_qa'
    if test:
        file_path = f'./data/{file_name}_test.csv'
    else: # train
        file_path = f'./data/{file_name}_train.csv'
    if not os.path.exists(file_path):
        # split into train and test
        data = pd.read_csv(f"./data/{file_name}.csv")
        # spit into test and train - 50% each
        train, test = train_test_split(data, train_size=10000, random_state=42)
        train.to_csv(f"./data/{file_name}_train.csv", index=False)
        test.to_csv(f"./data/{file_name}_test.csv", index=False)

    data = pd.read_csv(file_path)
    questions = data['Question']
    answers = data['Answer']
    return questions, answers


def load_data_nli(split, data_file_names):
    data_folder = './data'
    if split == 'train':
        file_path = f"{data_folder}/{data_file_names['train']}.csv"
    elif split == 'test':
        file_path = f"{data_folder}/{data_file_names['test']}.csv"
    else:
        raise ValueError("split should be either train or test")

    data = pd.read_csv(file_path)
    questions = data['Question']
    answers = data['Answer']
    origin = data['Origin']

    return questions, answers, origin

def load_data_snli(split):
    data_file_names = {
        'train': 'snli_train',
        'test': 'snli_validation'
    }
    return load_data_nli(split, data_file_names)

def load_data_mnli(split):
    data_file_names = {
        'train': 'mnli_train',
        'test': 'mnli_validation'
    }
    return load_data_nli(split, data_file_names)

def load_data_nq(split, with_context=False):
    raw_data_folder = './data'
    data_folder = './data'
    file_name = 'nq_wc' # don't need special file for no context, simply don't use it
    if split == 'train':
        file_path = f'{data_folder}/{file_name}_dataset_train.csv'
    elif split == 'test':
        file_path = f'{data_folder}/{file_name}_dataset_test.csv'
    else:
        raise ValueError("split should be either train or test")
    if not os.path.exists(file_path):
        all_data = pd.read_csv(f"{raw_data_folder}/{file_name}_dataset.csv")
        train, test = train_test_split(all_data, train_size=10000, random_state=42)
        train.to_csv(f"{data_folder}/{file_name}_dataset_train.csv", index=False)
        test.to_csv(f"{data_folder}/{file_name}_dataset_test.csv", index=False)

    data = pd.read_csv(file_path)
    questions = data['Question']
    answers = data['Answer']

    if with_context:
        context = data['Context']
    else:
        context = None

    return questions, answers, context


def load_data_winogrande(split):
    data_folder = './data'
    if split == 'train':
        file_path = f"{data_folder}/winogrande_train.csv"
    elif split == 'test':
        file_path = f"{data_folder}/winogrande_test.csv"
    else:
        raise ValueError("split should be either train or test")

    if not os.path.exists(file_path):
        all_data = pd.read_csv(f"{data_folder}/winogrande.csv")
        train, test = train_test_split(all_data, train_size=10000, random_state=42)
        train.to_csv(f"{data_folder}/winogrande_train.csv", index=False)
        test.to_csv(f"{data_folder}/winogrande_test.csv", index=False)

    data = pd.read_csv(file_path)
    questions = data['Question']
    answers = data['Answer']
    wrong_answers = data['Wrong_Answer']

    return questions, answers, wrong_answers


def load_data_triviaqa(test=False, legacy=False):
    if legacy:
        with open('./data/verified-web-dev.json') as f:
            data_verified = json.load(f)
            data_verified = data_verified['Data']
        with open('./data/web-dev.json') as f:
            data = json.load(f)
            data = data['Data']
        questions_from_verified = [x['Question'] for x in data_verified]
        data_not_verified = []
        for x in data:
            if x['Question'] in questions_from_verified:
                pass
            else:
                data_not_verified.append(x)

        print("Length of not verified data: ", len(data_not_verified))
        print("Length of verified data: ", len(data_verified))

        if test:
            return [ex['Question'] for ex in data_verified], [ex['Answer']['Aliases'] for ex in data_verified]
        else:
            return [ex['Question'] for ex in data_not_verified], [ex['Answer']['Aliases'] for ex in data_not_verified]
    else:
        if test:
            file_path = './data/triviaqa-unfiltered/unfiltered-web-dev.json'
        else:
            file_path = './data/triviaqa-unfiltered/unfiltered-web-train.json'
        with open(file_path) as f:
            data = json.load(f)
            data = data['Data']
        data, _ = train_test_split(data, train_size=10000, random_state=42)
        return [ex['Question'] for ex in data], [ex['Answer']['Aliases'] for ex in data]

def load_data_math(test=False):
    if test:
        data = pd.read_csv("./data/AnswerableMath_test.csv")
    else:
        data = pd.read_csv("./data/AnswerableMath.csv")


    questions = data['question']
    answers = data['answer'].map(lambda x: eval(x)[0])
    return questions, answers

def math_preprocess(model_name, all_questions, labels):
    prompts = []

    if 'instruct' in model_name.lower():
        for q in all_questions:
            prompts.append(q + " Answer shortly.")
    else:
        for q in all_questions:
            prompts.append(f'''Q: {q}
            A:''')
    return prompts

def prepare_winogrande(model_name, all_questions, labels):
    prompts = []
    for q in all_questions:
        prompts.append(f'''Question: {q}\nAnswer:''')
    return prompts

def generate_model_answers(data, model, tokenizer, device, model_name, do_sample=False, output_scores=False,
                           temperature=1.0,
                           top_p=1.0, max_new_tokens=100, stop_token_id=None, verbose=False):

    all_textual_answers = []
    all_scores = []
    all_input_output_ids = []
    all_output_ids = []
    counter = 0
    for prompt in tqdm(data):

        model_input = tokenize(prompt, tokenizer, model_name).to(device)

        with torch.no_grad():

            model_output = generate(model_input, model, model_name, do_sample, output_scores, max_new_tokens=max_new_tokens,
                                    top_p=top_p, temperature=temperature, stop_token_id=stop_token_id, tokenizer=tokenizer)

        answer = tokenizer.decode(model_output['sequences'][0][len(model_input[0]):])
        if output_scores:
            scores = torch.concatenate(model_output['scores']).cpu()  # shape = (new_tokens, len(vocab))
            all_scores.append(scores)
            output_ids = model_output['sequences'][0][len(model_input[0]):].cpu()
            all_output_ids.append(output_ids)

        all_textual_answers.append(answer)
        all_input_output_ids.append(model_output['sequences'][0].cpu())

        if verbose:
            if counter % 100 == 0:
                print(f"Counter: {counter}")
                print(f"Prompt: {prompt}")
                print(f"Answer: {answer}")
            counter += 1

    return all_textual_answers, all_input_output_ids, all_scores, all_output_ids


def init_wandb(args):
    wandb.init(
        project="generating_answers",
        config=vars(args)
    )

def load_data(dataset, excel_file=None):
    # If we have probing groups available and the dataset is a probing group
    if HAS_PROBING_GROUPS and dataset in probing_groups:
        print(f"Loading data from probing group: {dataset}")
        if not excel_file:
            # Use the default probing group file path
            df = load_probing_group(dataset)
        else:
            # Use the specified file path
            if 'probing_groups_transformed' in excel_file:
                # Already using transformed data
                df = pd.read_csv(excel_file)
            else:
                # Switch to transformed directory if using original path
                transformed_file = excel_file.replace('probing_groups/', 'probing_groups_transformed/')
                if os.path.exists(transformed_file):
                    df = pd.read_csv(transformed_file)
                    print(f"Using transformed file: {transformed_file}")
                else:
                    # Fallback to original file
                    df = pd.read_csv(excel_file)
                    print(f"Warning: Transformed file not found, using original: {excel_file}")
        
        # Extract the base dataset name (e.g., "frank_lampard" from "frank_lampard_true")
        base_dataset = None
        if dataset.startswith('frank_lampard'):
            base_dataset = 'frank_lampard'
        elif dataset.startswith('luis_suarez'):
            base_dataset = 'luis_suarez'
        elif dataset.startswith('maradona'):
            base_dataset = 'maradona'
        
        if not base_dataset:
            raise ValueError(f"Could not determine base dataset for probing group {dataset}")
        
        # Handle the new transformed CSV format with Comments and Label columns
        if 'Comments' in df.columns and 'Label' in df.columns:
            print("Using transformed CSV format with Comments and Label columns")
            
            # Prepare the data in the expected format
            from prepare_luis_suarez_data import prepare_alpaca_prompt
            
            # System instruction for stance detection
            system_instruction = ("You are provided with an input. You are required to perform stance detection "
                                "on the input with output as one of the following labels - Favor, Against, "
                                "Irrelevant, Neutral. The labels are self-explanatory. Remember to only use the labels provided "
                                "and do not mispell its name. Only output the stance detected label and nothing else.")
            
            # Prepare Alpaca prompts
            all_questions = [prepare_alpaca_prompt(system_instruction, comment) for comment in df['Comments']]
            labels = df['Label'].tolist()
            
            # Return the data in the expected format
            return all_questions, None, labels, 100, None, None, None, None, None
            
        # Use the original format if Comments/Label columns aren't found
        all_questions = df['question'].tolist()
        labels = df['correct_answer'].tolist()
        wrong_labels = None
        context = None
        preprocess_fn = None
        max_new_tokens = 100
        
        return all_questions, context, labels, max_new_tokens, None, preprocess_fn, None, None, wrong_labels
    
    # Handle custom dataset types for stance detection (frank_lampard, maradona, luis_suarez)
    if dataset.startswith('frank_lampard') and excel_file:
        # Use the dedicated function for Frank Lampard dataset
        all_questions, labels = load_frank_lampard_data(excel_file)
        wrong_labels = None
        context = None
        preprocess_fn = None
        max_new_tokens = 100
        return all_questions, context, labels, max_new_tokens, None, preprocess_fn, None, None, wrong_labels
    
    elif dataset.startswith('maradona') and excel_file:
        # Use the dedicated function for Maradona dataset
        all_questions, labels = load_maradona_data(excel_file)
        wrong_labels = None
        context = None
        preprocess_fn = None
        max_new_tokens = 100
        return all_questions, context, labels, max_new_tokens, None, preprocess_fn, None, None, wrong_labels
    
    elif dataset.startswith('luis_suarez') and excel_file:
        # Use the dedicated function for Luis Suarez dataset
        all_questions, labels = load_luis_suarez_data(excel_file)
        wrong_labels = None
        context = None
        preprocess_fn = None
        max_new_tokens = 100
        return all_questions, context, labels, max_new_tokens, None, preprocess_fn, None, None, wrong_labels
    
    # Otherwise use the standard dataset loading logic
    if dataset == 'triviaqa':
        return load_data_triviaqa()
    elif dataset == 'imdb':
        return load_data_imdb()
    elif dataset == 'winobias':
        return load_winobias('dev')
    elif dataset == 'winobias_test':
        return load_winobias('test')
    elif dataset == 'hotpotqa':
        return load_hotpotqa('train', with_context=False)
    elif dataset == 'hotpotqa_test':
        return load_hotpotqa('validation', with_context=False)
    elif dataset == 'hotpotqa_with_context':
        return load_hotpotqa('train', with_context=True)
    elif dataset == 'hotpotqa_with_context_test':
        return load_hotpotqa('validation', with_context=True)
    elif dataset == 'math':
        return load_data_math(test=False)
    elif dataset == 'math_test':
        return load_data_math(test=True)
    elif dataset == 'movies':
        return load_data_movies(test=False)
    elif dataset == 'movies_test':
        return load_data_movies(test=True)
    elif dataset == 'mnli':
        return load_data_mnli('train')
    elif dataset == 'mnli_test':
        return load_data_mnli('test')
    elif dataset == 'natural_questions':
        return load_data_nq('train')
    elif dataset == 'natural_questions_test':
        return load_data_nq('test')
    elif dataset == 'natural_questions_with_context':
        return load_data_nq('train', with_context=True)
    elif dataset == 'natural_questions_with_context_test':
        return load_data_nq('test', with_context=True)
    elif dataset == 'winogrande':
        return load_data_winogrande('train')
    elif dataset == 'winogrande_test':
        return load_data_winogrande('test')
    else:
        raise TypeError("data type is not supported")

def load_luis_suarez_data(excel_file):
    """Load and prepare data from the Luis Suarez Excel file for stance detection"""
    from prepare_luis_suarez_data import prepare_alpaca_prompt
    
    print(f"Loading Luis Suarez data from: {excel_file}")
    
    # Read the Excel file
    df = pd.read_excel(excel_file)
    
    # Display column names to help debug
    print("Available columns in Excel file:", df.columns.tolist())
    
    # Use the correct column names
    text_column = 'Comment'
    label_column = 'Llama3.1 Fine Tuned output'
    
    # System instruction for stance detection
    system_instruction = ("You are provided with an input. You are required to perform stance detection "
                         "on the input with output as one of the following labels - Favor, Against, "
                         "Irrelevant, Neutral. The labels are self-explanatory. Remember to only use the labels provided "
                         "and do not mispell its name. Only output the stance detected label and nothing else.")
    
    # Extract text and labels
    texts = df[text_column].dropna().tolist()
    labels = df[label_column].dropna().tolist()
    
    print(f"Loaded {len(texts)} texts and {len(labels)} labels")
    
    # Ensure we have the same number of texts and labels
    min_len = min(len(texts), len(labels))
    texts = texts[:min_len]
    labels = labels[:min_len]
    
    # Prepare prompts with Alpaca format
    prompts = [prepare_alpaca_prompt(system_instruction, text) for text in texts]
    
    return prompts, labels

def load_frank_lampard_data(excel_file):
    """Load and prepare data from the Frank Lampard Ghost Goal Excel file for stance detection"""
    from prepare_frank_lampard_data import extract_exact_answer
    
    print(f"Loading Frank Lampard Ghost Goal data from: {excel_file}")
    
    # Read the Excel file
    df = pd.read_excel(excel_file)
    
    # Display column names to help debug
    print("Available columns in Excel file:", df.columns.tolist())
    
    # Use the correct column names - adjust these based on your actual column names
    text_column = 'Comment'
    label_column = 'Llama3.1 Fine Tuned output'
    
    # System instruction for stance detection (same as Luis Suarez)
    system_instruction = ("You are provided with an input. You are required to perform stance detection "
                         "on the input with output as one of the following labels - Favor, Against, "
                         "Irrelevant, Neutral. The labels are self-explanatory. Remember to only use the labels provided "
                         "and do not mispell its name. Only output the stance detected label and nothing else.")
    
    # Extract text and labels
    texts = df[text_column].dropna().tolist()
    labels = df[label_column].dropna().tolist()
    
    print(f"Loaded {len(texts)} texts and {len(labels)} labels")
    
    # Ensure we have the same number of texts and labels
    min_len = min(len(texts), len(labels))
    texts = texts[:min_len]
    labels = labels[:min_len]
    
    # Prepare prompts - for consistency we'll use the same Alpaca format
    try:
        from prepare_luis_suarez_data import prepare_alpaca_prompt
        prompts = [prepare_alpaca_prompt(system_instruction, text) for text in texts]
    except ImportError:
        # Fallback if prepare_alpaca_prompt isn't available
        prompts = [f"System: {system_instruction}\n\nUser: {text}\n\nAssistant:" for text in texts]
    
    return prompts, labels

def load_maradona_data(excel_file):
    """Load and prepare data from the Maradona Hand of God Excel file for stance detection"""
    from prepare_maradona_data import prepare_alpaca_prompt
    
    print(f"Loading Maradona Hand of God data from: {excel_file}")
    
    # Read the Excel file
    df = pd.read_excel(excel_file)
    
    # Display column names to help debug
    print("Available columns in Excel file:", df.columns.tolist())
    
    # Use the correct column names
    text_column = 'Comment'
    label_column = 'Llama3.1 Fine Tuned output'
    
    # System instruction for stance detection
    system_instruction = ("You are provided with an input. You are required to perform stance detection "
                         "on the input with output as one of the following labels - Favor, Against, "
                         "Irrelevant, Neutral. The labels are self-explanatory. Remember to only use the labels provided "
                         "and do not mispell its name. Only output the stance detected label and nothing else.")
    
    # Extract text and labels - only keep rows where both text and label exist
    valid_indices = df[text_column].notna() & df[label_column].notna()
    texts = df.loc[valid_indices, text_column].tolist()
    labels = df.loc[valid_indices, label_column].tolist()
    
    print(f"Loaded {len(texts)} texts and {len(labels)} labels")
    
    # Prepare prompts with Alpaca format
    prompts = [prepare_alpaca_prompt(system_instruction, text) for text in texts]
    
    return prompts, labels


def main():
    args = parse_args()
    init_wandb(args)
    set_seed(0)
    dataset_size = args.n_samples

    model, tokenizer = load_model_and_validate_gpu(args.model)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    stop_token_id = None
    if 'instruct' not in args.model.lower():
        stop_token_id = tokenizer.encode('\n', add_special_tokens=False)[-1]
    all_questions, context, labels, max_new_tokens, origin, preprocess_fn, stereotype, type_, wrong_labels = load_data(args.dataset, args.excel_file)

    if not os.path.exists('/kaggle/working'):
        os.makedirs('/kaggle/working')

    file_path_output_ids = f"/kaggle/working/{MODEL_FRIENDLY_NAMES[args.model]}-input_output_ids-{args.dataset}.pt"
    file_path_scores = f"/kaggle/working/{MODEL_FRIENDLY_NAMES[args.model]}-scores-{args.dataset}.pt"
    file_path_answers = f"/kaggle/working/{MODEL_FRIENDLY_NAMES[args.model]}-answers-{args.dataset}.csv"

    if dataset_size:
        all_questions = all_questions[:dataset_size]
        labels = labels[:dataset_size]
        if 'mnli' in args.dataset:
            origin = origin[:dataset_size]
        if 'winogrande' in args.dataset:
            wrong_labels = wrong_labels[:dataset_size]

    output_csv = {}
    if preprocess_fn:
        if 'winobias' in args.dataset:
            output_csv['raw_question'] = all_questions[0]
        else:
            output_csv['raw_question'] = all_questions
        if 'natural_questions' in args.dataset:
            with_context = True if 'with_context' in args.dataset else False
            print('preprocessing nq')
            all_questions = preprocess_fn(args.model, all_questions, labels, with_context, context)
        else:
            all_questions = preprocess_fn(args.model, all_questions, labels)

    model_answers, input_output_ids, all_scores, all_output_ids = generate_model_answers(all_questions, model,
                                                                                         tokenizer, device, args.model,
                                                                                         output_scores=True, max_new_tokens=max_new_tokens,
                                                                                         stop_token_id=stop_token_id)

    # Determine which correctness function to use based on dataset
    correctness_dataset = args.dataset
    # Strip _true or _false suffix for probing groups to get base dataset
    if correctness_dataset.endswith('_true') or correctness_dataset.endswith('_false'):
        correctness_dataset = '_'.join(correctness_dataset.split('_')[:-1])

    res = compute_correctness(all_questions, correctness_dataset, args.model, labels, model, model_answers, tokenizer, wrong_labels)
    correctness = res['correctness']

    acc = np.mean(correctness)
    wandb.summary[f'acc'] = acc
    print(f"Accuracy:", acc)

    output_csv['question'] = all_questions
    output_csv['model_answer'] = model_answers
    output_csv['correct_answer'] = labels
    output_csv['automatic_correctness'] = correctness

    if 'exact_answer' in res:
        output_csv['exact_answer'] = res['exact_answer']
        output_csv['valid_exact_answer'] = 1
    if 'incorrect_answer' in res:
        output_csv['incorrect_answer'] = res['incorrect_answer']
    if 'winobias' in args.dataset:
        output_csv['stereotype'] = stereotype
        output_csv['type'] = type_

    if 'mnli' in args.dataset:
        output_csv['origin'] = origin

    pd.DataFrame(output_csv).to_csv(file_path_answers)
    torch.save(input_output_ids, file_path_output_ids)
    torch.save({'all_scores': all_scores, 'all_output_ids': all_output_ids}, file_path_scores)


if __name__ == "__main__":
    main()
