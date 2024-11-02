import argparse, logging, os, time
from tools.OpenAIConnection import OpenAIConnection

# Set up logging configuration with timestamps
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main(args):
    """
    Main function to train GPT model, monitor status, and download training logs to local directory.
    """
    # Initialize connection to OpenAI
    openai = OpenAIConnection('gpt-3.5-turbo')
    
    # File paths for training and validation data
    train_data = f'data/training/data_partitioning/train/{args.train_file}.jsonl'
    validate_data = f'data/training/data_partitioning/validate/{args.validate_file}.jsonl'

    # Upload training and validation files, obtaining file IDs for both
    train_id = openai._upload_file(train_data)
    vaildate_id = openai._upload_file(validate_data)

    # Create fine-tuning job with specific hyperparameters
    job_id = openai._create_finetune_job(train_id, vaildate_id, epoch=args.epoch, bsize=args.bsize, lr=args.lr)
    logging.info(f'Initiate Fine-tuning Task, Job: {job_id}.')
    logging.info(f'Hyperparameter: Epoch-{args.epoch}, Batch Size-{args.bsize}, Learning Rate-{args.lr}.')

    # Continuously check the fine-tuning job's status until it succeeds or fails
    while True:
        # Check current job status
        response = openai._check_job_status(job_id)
        logging.info(f'Model: {response.fine_tuned_model}, Status: {response.status}')
        if response.status in ('succeeded', 'failed'):
            break
        
        # Delay for an hour before checking status again
        time.sleep(3600)

    # Define output directory path for downloaded logs
    output_dir = f'data/training/result/{args.output_dir}/logs/{job_id}/'

    # Create output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)    

    # Download result files from completed job
    openai._download_result(response.result_files, output_dir)
        

def _setup_args():
    """
    Set up command-line arguments.

    Returns:
        argparse: The parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('output_dir', type=str, help='Output Directory for keeping logs')
    parser.add_argument('-train_file', type=str, help='File containing training data')
    parser.add_argument('-validate_file', type=str, help='File containing validation data')
    parser.add_argument('-epoch', type=int, nargs='?', help='Epoch')
    parser.add_argument('-bsize', type=int, nargs='?', help='Batch Size')
    parser.add_argument('-lr', type=float, nargs='?', help='Learning Rate')
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    # Set up command-line arguments
    args = _setup_args()

    # Execute the main function with the parsed arguments
    main(args)
