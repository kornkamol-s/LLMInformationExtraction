import ast
import pandas as pd
from config import config
from sklearn.model_selection import train_test_split


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
        {"role": "system", "content": "You are tasked with extracting relevant information from the provided context to answer the following question. If any key information is missing, omit that key from response. If no relevant information is found, do not return anything."},
        {"role": "user", "content": f"Question: {row['question']}\n\nContext: {row['context']}"}
    ]
    
    # If it is either training or validation set, append assistant role (answer) to the message
    if not is_test:
        messages.append({"role": "assistant", "content": f"{row['GHG Emission Reductions']}"})
    
    return {"messages": messages}


def filter_groundtruth(row):
    """
    Remove answer where the answer does not appear in the context.
    
    Parameters:
        row (Series): Data row containing question, context, and GHG Emission Reductions.
        
    Returns:
        Series: Data row where inconsistent answers are set to None.
    """
    try:
        # Convert answer to dictionary for easier retrieval
        value_dict = ast.literal_eval(row['GHG Emission Reductions'])

        if isinstance(value_dict, dict):

            # Retrieve first key of dictionary such as {'2021': 12345}
            first_year = next(iter(value_dict.keys()), '')

            # Check if the first year from GHG Emission Reductions exists in the context
            if first_year and str(first_year) not in row['context']:

                # If the part of answer does not match with context, 
                # remove answer to prevent model hallucination from learning wrong pattern
                row['GHG Emission Reductions'] = None
    except:
        row['GHG Emission Reductions'] = None
        
    return row


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

# Load dataset both contexts and answers
context_df = pd.read_csv('data/training/data_processing/pdd_context_retrieval.csv', encoding='utf-8')
context_df = context_df[context_df['section_category']=='ghg_emission_reductions']
answer_df = pd.read_csv('data/training/data_processing/manual_refined_ground_truth_ghg.csv', encoding='utf-8')

# Merge contexts and answers
df = pd.merge(context_df, answer_df, left_on='id', right_on='Project ID', how='inner')

# Add question for ghg emission reduction
df['question'] = df['section_category'].map(config.QUESTION_MAPPING)

# Remove all whitespace to remain consisten with model's generated answers
df['GHG Emission Reductions'] = df['GHG Emission Reductions'].str.replace(r'\s+', '', regex=True)

# Remove records containing inconsistency between contexts and answers
df = df.apply(filter_groundtruth, axis=1)

# Fill no answer with proper representation
df['GHG Emission Reductions'] = df['GHG Emission Reductions'].fillna('No relevant information found in context')

# Shuffle to reduce bias
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Split 80% of total data for training purpose
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)

# Split 20% of total data for validation and testing purpose equally
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# Transform and save to local file
_save_jsonl(train_df, f'{filepath}/train/ghg_emission_reduction_train.jsonl', _transform_record)
_save_jsonl(val_df, f'{filepath}/validate/ghg_emission_reduction_validate.jsonl', _transform_record)
_save_jsonl(test_df, f'{filepath}/test/ghg_emission_reduction_test_prompt.jsonl', lambda row: _transform_record(row, is_test=True))
test_df = test_df.rename(columns={'GHG Emission Reductions': 'answers'})
test_df['answers'].to_csv(f'{filepath}/test/ghg_emission_reduction_test_answer.csv', index=False)