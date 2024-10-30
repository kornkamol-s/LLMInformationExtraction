import argparse
from tools.OpenAIConnection import OpenAIConnection
import logging 
import os 
import json
import csv
import ast
import pandas as pd
from rouge_score import rouge_scorer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import evaluate
# from deepeval.metrics import FaithfulnessMetric
# from deepeval.test_case import LLMTestCase

# Set up logging configuration with timestamps
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main(args):
    test_files = f"data/intermediate/{args.source}/gpt35/{args.source}_test"
    output_dir = f"data/output/{args.pipeline}-gpt35/{args.source}"

    openai = OpenAIConnection(args.model)

    # logging.info(f"Loading test data; {args.n} records ...")
    # tests = _load_test_data(test_files, args.n)

    # # logging.info("Generate response from base model...")
    # # _generate_model_response(openai, tests, args.model, output_dir, f'base_{args.model}')

    # logging.info(f"Generate response from fine-tuned model...")
    # for model_id in args.model_ids:
    #     logging.info(f"Generate response from fine-tuned model: {model_id}...")
    #     _generate_model_response(openai, tests, model_id, output_dir, f'{model_id.split(':')[-1]}')

    _generate_metrics(output_dir)

    
def _load_test_data(test_files, n):
    test = []
    with open(f"{test_files}_prompt.jsonl", 'r') as f1, open(f"{test_files}_answer.jsonl", 'r') as f2:
        for i, (line_prompt, line_answer) in enumerate(zip(f1, f2)):
            test.append([json.loads(line_prompt.strip())['messages'],line_answer.strip()])
            if i+1 == n:
                break
    return test


def _generate_model_response(openai, tests, model, output_dir, type):
    with open(f'{output_dir}/model_response__{type}.csv', mode='a', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['prompt', 'true_answer', 'model_answer'])
        if f.tell() == 0:
            writer.writeheader()
        for test in tests:
            answer = openai._evaluate_model(model, test[0])
            row = {
                'prompt': test[0],
                'true_answer': test[1],
                'model_answer': answer,
            }
            writer.writerow(row)


def extract_context_and_question(prompt):
    data_list = ast.literal_eval(prompt)
    for item in data_list:
        if item['role'] == 'user':
            content = item['content']
            if "Question:" in content:
                question_part, context_part = content.split("Context:", 1)
                question = question_part.strip().replace("Question:", "").strip()
                context = context_part.strip()
                    
    return pd.Series([question, context])


def _generate_metrics(output_dir):
    bleu = evaluate.load("bleu")
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], use_stemmer=True)
    
    for file in os.listdir(output_dir):
        if file.endswith('.csv'):
            print(f"Processing {output_dir}/{file}")
            df = pd.read_csv(f'{output_dir}/{file}')
            bleu_list = []
            rouge1_precision_list, rouge1_recall_list, rouge1_f1_list = [], [], []
            rouge2_precision_list, rouge2_recall_list, rouge2_f1_list = [], [], []
            rougeL_precision_list, rougeL_recall_list, rougeL_f1_list = [], [], []
            rougeLSum_precision_list, rougeLSum_recall_list, rougeLSum_f1_list = [], [], []

            true_answers = df['true_answer'].tolist()
            model_answers = df['model_answer'].tolist()
            accuracy = accuracy_score(true_answers, model_answers)
            precision = precision_score(true_answers, model_answers, average='weighted', zero_division=0)
            recall = recall_score(true_answers, model_answers, average='weighted', zero_division=0)
            f1 = f1_score(true_answers, model_answers, average='weighted', zero_division=0)

            print(f"accuracy: {accuracy}")
            print(f"precision: {precision}")
            print(f"recall: {recall}")
            print(f"f1: {f1}")

            for index, row in df.iterrows():
                true_answer = row['true_answer']
                model_answer = row['model_answer']

                bleu_result = bleu.compute(predictions=[model_answer], references=[[true_answer]],  max_order=2)
                bleu_list.append(bleu_result['bleu'])
                
                scores = scorer.score(true_answer, model_answer)

                rouge1_precision_list.append(scores['rouge1'].precision)
                rouge1_recall_list.append(scores['rouge1'].recall)
                rouge1_f1_list.append(scores['rouge1'].fmeasure)
                
                rouge2_precision_list.append(scores['rouge2'].precision)
                rouge2_recall_list.append(scores['rouge2'].recall)
                rouge2_f1_list.append(scores['rouge2'].fmeasure)

                rougeL_precision_list.append(scores['rougeL'].precision)
                rougeL_recall_list.append(scores['rougeL'].recall)
                rougeL_f1_list.append(scores['rougeL'].fmeasure)

                rougeLSum_precision_list.append(scores['rougeLsum'].precision)
                rougeLSum_recall_list.append(scores['rougeLsum'].recall)
                rougeLSum_f1_list.append(scores['rougeLsum'].fmeasure)
                
            df['bleu'] = bleu_list
            df['rouge1_precision'] = rouge1_precision_list
            df['rouge1_recall'] = rouge1_recall_list
            df['rouge1_f1'] = rouge1_f1_list
            df['rouge2_precision'] = rouge2_precision_list
            df['rouge2_recall'] = rouge2_recall_list
            df['rouge2_f1'] = rouge2_f1_list
            df['rougeL_precision'] = rougeL_precision_list
            df['rougeL_recall'] = rougeL_recall_list
            df['rougeL_f1'] = rougeL_f1_list
            df['rougeLSum_precision'] = rougeLSum_precision_list
            df['rougeLSum_recall'] = rougeLSum_recall_list
            df['rougeLSum_f1'] = rougeLSum_f1_list

            df.to_csv(f'{output_dir}/records_metrics_{file}', index=False)


def _setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, default='gpt-3.5-turbo', nargs='?', help='Model Type')
    parser.add_argument('--model_ids', type=str, nargs='*', help='Model IDs')
    parser.add_argument('--n', type=int, default=1000, help='Model ID')
    parser.add_argument('--source', type=str, default='squad', help='Type of Data')
    parser.add_argument('--pipeline', type=str, default='custom', help='Type of Data')
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = _setup_args()
    main(args)
