import argparse
from tools.OpenAIConnection import OpenAIConnection
import logging 
import os 
import time

# Set up logging configuration with timestamps
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main(args):
    openai = OpenAIConnection(args.model)
    train_data = f'data/intermediate/{args.source}/gpt35/{args.source}_train.jsonl'
    validate_data = f'data/intermediate/{args.source}/gpt35/{args.source}_validate.jsonl'

    train_id = openai._upload_file(train_data)
    vaildate_id = openai._upload_file(validate_data)

    job_id = openai._create_finetune_job(train_id, vaildate_id, epoch=4, bsize=8, lr=2)
    logging.info(f'Start Fine-tuning, Job: {job_id}')

    while True:
        response = openai._check_job_status(job_id)
        logging.info(f'Model: {response.fine_tuned_model}, Status: {response.status}')
        if response.status in ('succeeded', 'failed'):
            break
        time.sleep(3600)

    output_dir = f'data/output/custom-gpt35/{args.source}/logs/{job_id}/'
    os.makedirs(output_dir, exist_ok=True)    
    openai._download_result(response.result_files, output_dir)
    logging.info(f"Download files successfully.")
        

def _setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, default='gpt-3.5-turbo', nargs='?', help='Model Type')
    parser.add_argument('--source', type=str, default='squad', help='Data file')
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = _setup_args()
    main(args)
