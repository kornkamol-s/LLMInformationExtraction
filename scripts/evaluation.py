import logging, json, csv, argparse, evaluate
import pandas as pd
from rouge_score import rouge_scorer
from tools.OpenAIConnection import OpenAIConnection
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Set up logging configuration with timestamps
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main(args):
    """
    Main function to handle data loading, response generation, and metric evaluation.
    """
    question_file = f"data/training/data_partitioning/test/{args.question}.jsonl"
    answer_file = f"data/training/data_partitioning/test/{args.answer}.csv"
    output_dir = f"data/training/result/{args.output}"

    # Initialize OpenAI connection
    openai = OpenAIConnection('gpt-3.5-turbo')

    # Load test data from question and answer files
    test_records = _load_test_data(question_file, answer_file)

    # Generate model's responses based on test records
    logging.info(f"Generating responses for a total of {len(test_records)} records using model identifier: {args.id}.")
    _generate_model_response(openai, test_records, args.id, output_dir)

    # Generate performance metrics based on responses
    logging.info(f"Generating Performance Metrics of responses obtained from model identifier: {args.id}.")
    _generate_metrics(output_dir, args.id)

    
def _load_test_data(question_file, answer_file):
    """
    Load and prepare test data from question and answer files.

    Parameters:
        question_file (str): Path to the JSONL file with question prompts.
        answer_file (str): Path to the CSV file with true answers.

    Returns:
        list: A list of paired question prompts and true answers.
    """
    test = []
    
    # Load questions from JSONL
    with open(question_file, 'r') as f1:
        questions = [json.loads(line.strip())['messages'] for line in f1]

    # Load answers from CSV
    answers_df = pd.read_csv(answer_file, encoding='utf-8')
    answers = answers_df['answers'].tolist()

    # Zip questions and answers together
    for i, (line_prompt, line_answer) in enumerate(zip(questions, answers)):
        test.append([line_prompt, line_answer])

    return test


def _generate_model_response(openai, records, model, output_dir):
    """
    Generate model responses and save to a CSV file.

    Parameters:
        openai (OpenAIConnection): Connection to the OpenAI model.
        records (list): List of question and answer pairs.
        model (str): Model ID to use for generation.
        output_dir (str): Directory to save the responses CSV.
    """
    with open(f'{output_dir}/responses/{model.split(':')[-1]}.csv', mode='a', encoding='utf-8', newline='') as f:

        # Write header if the file is empty
        writer = csv.DictWriter(f, fieldnames=['prompt', 'true_answer', 'model_answer'])
        if f.tell() == 0:
            writer.writeheader()

        # Generate model responses and write them to CSV
        for rec in records:
            answer = openai._evaluate_model(model, rec[0])
            row = {
                'prompt': rec[0],
                'true_answer': rec[1],
                'model_answer': answer,
            }
            writer.writerow(row)


def _generate_metrics(output_dir, model):
    """
    Calculate and save evaluation metrics for model responses.

    Parameters:
        output_dir (str): Directory where the response CSV is located.
        model (str): Model ID used for naming the metrics file.
    """
    inputfile = f'{output_dir}/responses/{model.split(':')[-1]}.csv'
    outputfile = f'{output_dir}/metrics/{model.split(':')[-1]}.csv'

    # Load BLEU and ROUGE scorers
    bleu = evaluate.load("bleu")
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], use_stemmer=True)
    
    # Read model responses from CSV
    df = pd.read_csv(inputfile)
    true_answers = df['true_answer'].tolist()
    model_answers = df['model_answer'].tolist()

    # Compute classification metrics
    accuracy = accuracy_score(true_answers, model_answers)
    precision = precision_score(true_answers, model_answers, average='weighted',zero_division=0)
    recall = recall_score(true_answers, model_answers, average='weighted', zero_division=0)
    f1 = f1_score(true_answers, model_answers, average='weighted', zero_division=0)

    logging.info(f"accuracy: {accuracy}")
    logging.info(f"precision: {precision}")
    logging.info(f"recall: {recall}")
    logging.info(f"f1: {f1}")

    # Compute BLEU scores
    bleu_scores = [bleu.compute(predictions=[pred], references=[[ref]], max_order=2)['bleu']
                   for pred, ref in zip(model_answers, true_answers)]
    
    # Initialize dictionaries to store ROUGE score components
    rouge_scores = {
        'rouge1_precision': [], 'rouge1_recall': [], 'rouge1_f1': [],
        'rouge2_precision': [], 'rouge2_recall': [], 'rouge2_f1': [],
        'rougeL_precision': [], 'rougeL_recall': [], 'rougeL_f1': [],
        'rougeLSum_precision': [], 'rougeLSum_recall': [], 'rougeLSum_f1': []
    }

    # Calculate ROUGE scores and append results to the lists
    for true, pred in zip(true_answers, model_answers):
        scores = scorer.score(true, pred)
        rouge_scores['rouge1_precision'].append(scores['rouge1'].precision)
        rouge_scores['rouge1_recall'].append(scores['rouge1'].recall)
        rouge_scores['rouge1_f1'].append(scores['rouge1'].fmeasure)

        rouge_scores['rouge2_precision'].append(scores['rouge2'].precision)
        rouge_scores['rouge2_recall'].append(scores['rouge2'].recall)
        rouge_scores['rouge2_f1'].append(scores['rouge2'].fmeasure)

        rouge_scores['rougeL_precision'].append(scores['rougeL'].precision)
        rouge_scores['rougeL_recall'].append(scores['rougeL'].recall)
        rouge_scores['rougeL_f1'].append(scores['rougeL'].fmeasure)

        rouge_scores['rougeLSum_precision'].append(scores['rougeLsum'].precision)
        rouge_scores['rougeLSum_recall'].append(scores['rougeLsum'].recall)
        rouge_scores['rougeLSum_f1'].append(scores['rougeLsum'].fmeasure)

    # Append BLEU and ROUGE scores to DataFrame
    df['bleu'] = bleu_scores
    for key, values in rouge_scores.items():
        df[key] = values

    # Save the metrics to output CSV
    df.to_csv(outputfile, index=False)


def _setup_args():
    """
    Set up command-line arguments.

    Returns:
        argparse: The parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('id', type=str, nargs='?', help='Model ID')
    parser.add_argument('--question', type=str, help='File containing prompts')
    parser.add_argument('--answer', type=str, help='File containing answers')
    parser.add_argument('--output', type=str, help='Output Filepath')
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    # Set up command-line arguments
    args = _setup_args()

    # Execute the main function with the parsed arguments
    main(args)