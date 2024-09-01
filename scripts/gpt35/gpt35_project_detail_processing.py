import pandas as pd
from sklearn.model_selection import train_test_split

question_mapping = {'project_proponents': 'What is the project proponents, includind organization name, phone number and email address, city and country of the entity responsible for managing this carbon project?',
                    'methodology': 'What is the methodology of this project?',
                    'project_description': 'What is the project summary, province, country, latitude, and longitude of this project?',
                    'crediting period': 'What are the start and end dates of the crediting period for this project?',
                    'sector': 'What is the project sector, either Renewable Energy or Forestry and Land Use?',
                    }


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
                "content": f"{row['value']}"
            }
        ]
    }


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
        # "input":  f"Question: {row['question']}\n\nContext: {row['context']}",
        # "question": f'{row['question']}',
        # "context": f'{row['context']}',
        # "groundtruth": f'{row['value']}'
    }

context_df = pd.read_parquet('data/intermediate/kita_dataset/refined_pdd_context_retrieval.parquet', engine='fastparquet')
scrape_df = pd.read_csv('data/intermediate/kita_dataset/processed_ground_truth_project_information.csv')

context_df['id'] = context_df['id'].astype('int')
scrape_df['id'] = scrape_df['id'].astype('int')

df = pd.merge(context_df, scrape_df, left_on=['id', 'section_category'], right_on=['id', 'type'], how='inner')
df['question'] = df['section_category'].map(question_mapping)

df = df.sample(frac=1, random_state=42).reset_index(drop=True)
train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)


train_df = train_df.apply(lambda row: _transform_record(row), axis=1)
train_df.head(1000).to_json('data/intermediate/project_description/gpt35/project_description_train.jsonl', orient='records', lines=True)

val_df = val_df.apply(lambda row: _transform_record(row), axis=1)
val_df.head(200).to_json('data/intermediate/project_description/gpt35/project_description_validate.jsonl', orient='records', lines=True)

test_df[['value']].head(200).to_json('data/intermediate/project_description/gpt35/project_description_test_answer.jsonl', orient='records', lines=True)

test_df = test_df.apply(lambda row: _transform_record_test(row), axis=1)
test_df.head(200).to_json('data/intermediate/project_description/gpt35/project_description_test_prompt.jsonl', orient='records', lines=True)
