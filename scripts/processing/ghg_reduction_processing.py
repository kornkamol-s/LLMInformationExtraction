import pandas as pd
from sklearn.model_selection import train_test_split
import ast
from dateutil import parser
import json

with open('config/question_mapping.json', 'r') as f:
        question_mapping = json.load(f)

def _transform_record(row):
    return {
                "messages": [
            {
                "role": "system",
                "content": "You are tasked with extracting relevant information from the provided context to answer the following question. If any key information is missing, omit that key from response. If no relevant information is found, do not return anything."
            },
            {
                "role": "user",
                "content": f"Question: {row['question']}\n\nContext: {row['context']}"
            },
            {
                "role": "assistant",
                "content": f"{row['Value']}"
            }
        ]
    }

# def _transform_record_test(row):
#     return {
#         "input": f"Question: {row['question']}\n\nContext: {row['context']}",
#         "question": f"{row['question']}",
#         "context": f"{row['context']}",
#         "groundtruth": f"{row['GHG Emission Reductions']}"
#     }

def _transform_record_test(row):
    return {
                "messages": [
            {
                "role": "system",
                "content": "You are tasked with extracting relevant information from the provided context to answer the following question. If any key information is missing, omit that key from response. If no relevant information is found, do not return anything."
            },
            {
                "role": "user",
                "content": f"Question: {row['question']}\n\nContext: {row['context']}"
            }
        ]
    }


def filter_groundtruth(row):
    value_dict = ast.literal_eval(row['Value'])
    if isinstance(value_dict, dict):
        first_year = next(iter(value_dict.keys()), '')
        if first_year and str(first_year) not in row['context']:
            row['Value'] = None
    
    return row


context_df = pd.read_parquet('data/intermediate/kita_dataset/refined_pdd_context_retrieval_ghg.parquet', engine='fastparquet')
answer_df = pd.read_csv('data/intermediate/kita_dataset/processed_ground_truth_ghg_refined.csv')

context_df['id'] = context_df['id'].astype('int')
context_df = context_df[context_df['section_category'] == 'ghg_emission_reductions']
answer_df['id'] = answer_df['Project ID'].astype('int')

df = pd.merge(context_df, answer_df, on='id', how='inner')
df['question'] = df['section_category'].map(question_mapping)

df['Value'] = df['Value'].str.replace(r'\s+', '', regex=True)

# df = df.apply(filter_groundtruth, axis=1)
df['Value'] = df['Value'].fillna('No relevant information found in context')

df = df.sample(frac=1, random_state=42).reset_index(drop=True)
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

train_df = train_df.apply(lambda row: _transform_record(row), axis=1)
train_df.to_json('data/intermediate/ghg_reduction/gpt35/ghg_reduction_train.jsonl', orient='records', lines=True)

val_df = val_df.apply(lambda row: _transform_record(row), axis=1)
val_df.to_json('data/intermediate/ghg_reduction/gpt35/ghg_reduction_validate.jsonl', orient='records', lines=True)


jsonl_file_path = 'data/intermediate/ghg_reduction/gpt35/ghg_reduction_test_answer.jsonl'
with open(jsonl_file_path, 'w') as jsonl_file:
    for value in test_df['Value']:
        jsonl_file.write(value + '\n')


test_df = test_df.apply(lambda row: _transform_record_test(row), axis=1)
test_df.to_json('data/intermediate/ghg_reduction/gpt35/ghg_reduction_test_prompt.jsonl', orient='records', lines=True)
