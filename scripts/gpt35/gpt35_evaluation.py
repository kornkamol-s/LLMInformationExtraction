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
from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main(args):
    test_files = f"data/intermediate/{args.source}/gpt35/{args.source}_test"
    output_dir = f"data/output/{args.pipeline}-gpt35/{args.source}"

    openai = OpenAIConnection(args.model)

    logging.info(f"Loading test data; {args.n} records ...")
    tests = _load_test_data(test_files, args.n)

    # logging.info("Generate response from base model...")
    # _generate_model_response(openai, tests, args.model, output_dir, f'base_{args.model}')

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
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    for file in os.listdir(output_dir):
        if file.endswith('.csv'):
            df = pd.read_csv(f'{output_dir}/{file}')
            # df[['question', 'context']] = df['prompt'].apply(extract_context_and_question)

            true_answers = df['true_answer'].tolist()
            model_answers = df['model_answer'].tolist()
            # question = df['question'].tolist()
            # context = df['context'].tolist()

            accuracy = accuracy_score(true_answers, model_answers)
            precision = precision_score(true_answers, model_answers, average='weighted', zero_division=0)
            recall = recall_score(true_answers, model_answers, average='weighted', zero_division=0)
            f1 = f1_score(true_answers, model_answers, average='weighted', zero_division=0)

            bleu = evaluate.load("bleu")
            bleu = bleu.compute(predictions=model_answers, references=true_answers)

            rouge = evaluate.load('rouge')
            rouge = rouge.compute(predictions=model_answers, references=true_answers)

            total_precision, total_recall, total_f1 = 0, 0, 0
            for true_answer, model_answer in zip(true_answers, model_answers):
                scores = scorer.score(str(true_answer), str(model_answer))
                total_precision += scores['rougeL'].precision
                total_recall += scores['rougeL'].recall
                total_f1 += scores['rougeL'].fmeasure

            rouge['rougeL-precision'] = total_precision/len(true_answers)
            rouge['rougeL-recall'] = total_recall/len(true_answers)
            rouge['rougeL-f1'] = total_f1/len(true_answers)

            # test_case = LLMTestCase(input=question[0], 
            #                         actual_output=model_answers[0],
            #                         retrieval_context=[context[0]])
            # metric = FaithfulnessMetric(threshold=0.5)
            # metric.measure(test_case)
            # print(metric.score)
            # print(metric.reason)
            # print(metric.is_successful())

            row = {
                'base_model': 'GPT-3.5-turbo-125',
                'model': file.split('__')[-1],
                'source': args.source,
                'pipeline': args.pipeline,
                'common-metric': json.dumps({'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}),
                'bleu': json.dumps(bleu),
                'rouge': json.dumps(rouge)}

            with open(f'data/output/evaluation_metrics.csv', mode='a', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['base_model', 'model', 'source', 'pipeline', 'common-metric', 'bleu', 'rouge'])
                if f.tell() == 0:
                    writer.writeheader()
                writer.writerow(row)


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
