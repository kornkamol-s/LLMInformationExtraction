from datasets import load_dataset
import pandas as pd


def _transform_record(row, is_test=False):
    """
    Transform a row into the desired JSON format.

    Parameters:
        row (Series): Data row with question, context, and answers.
        is_test (bool): Flag to format records for test data.

    Returns:
        dict: Transformed record with structured messages.
    """
    messages = [
        {"role": "system", "content": "You are tasked with extracting relevant information from the provided context to answer the following question. If no relevant information is found, do not return anything."},
        {"role": "user", "content": f"Question: {row['question']}\n\nContext: {row['context']}"}
    ]
    
    # If it is either training or validation set, append assistant role (answer) to the message
    if not is_test:
        messages.append({"role": "assistant", "content": f"{row['answers']}"})
    
    return {"messages": messages}


def _save_jsonl(data, file_path, transform_func):
    """
    Save transformed data to a JSONL file.
    
    Parameters:
        data (Dataframe): Dataframe to transform.
        file_path (str): Output file path.
        transform_func (function): Function to apply transformation to each row.
    """
    # Transform record into desired structure
    data = data.apply(transform_func, axis=1)

    # Write to local JSONL file
    data.to_json(file_path, orient='records', lines=True)


# Define filepath
filepath = 'data/training/data_partitioning'

# Load SQuAD dataset
dataset = load_dataset('rajpurkar/squad')

# Use the training subset to transform and save only 10,000 for training purpose
train_df = pd.DataFrame(dataset['train']).assign(answers=lambda df: df['answers'].apply(lambda x: x['text']))
_save_jsonl(train_df.head(10000), f'{filepath}/train/squad_train.jsonl', _transform_record)

# Use the validation subset for validation and testing purposes
val_df = pd.DataFrame(dataset['validation']).assign(
    answers=lambda df: df['answers'].apply(lambda x: list(set(x['text']))))

# Split 50% for validation and testing set, transform validation set to required structure, 
# and save only 2,000 records for validation purpose
_save_jsonl(val_df.sample(frac=0.5, random_state=42).head(2000), f'{filepath}/validate/squad_validate.jsonl', _transform_record)
test_df = val_df.drop(val_df.sample(frac=0.5, random_state=42).index)

# Transform testing set and save only 2,000 records for testing purpose with separated prompts and answers
test_df['answers'].head(2000).to_csv(f'{filepath}/test/squad_test_answer.csv', index=False)
_save_jsonl(test_df.head(2000), f'{filepath}/test/squad_test_prompt.jsonl', lambda row: _transform_record(row, is_test=True))