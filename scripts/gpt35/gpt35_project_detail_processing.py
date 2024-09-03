import pandas as pd
from sklearn.model_selection import train_test_split
from fuzzywuzzy import fuzz
import ast
from dateutil import parser


question_mapping = {'project_proponents': 'What is the project proponents, includind organization name, phone number and email address, state/city and country of the entity responsible for managing this carbon project?',
                    'methodology': 'What is the methodology of this project?',
                    'project_description': 'What is the project province, country, latitude, and longitude of this project?',
                    'crediting period': 'What are the start and end dates of the crediting period for this project?',
                    'sector': 'What is the project sector, either Renewable Energy or Forestry and Land Use?'}


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


def fuzzy_match(query, choices, threshold=70):
    return any(fuzz.partial_ratio(query, choice) >= threshold for choice in choices)


def filter_groundtruth(row):
    value_dict = ast.literal_eval(row['value'])
    if row['question'] == 'What are the start and end dates of the crediting period for this project?':
        start_date_str = value_dict.get('crediting_period_start', '')
        if start_date_str:
            start_date = parser.parse(start_date_str)
            if not str(start_date.year) in row['context']:
                row['value'] = None

    elif row['question'] == 'What is the methodology of this project?':
        methodologies = value_dict.get('project_methodologies', [])
        filtered_methodologies = [methodology for methodology in methodologies if methodology in row['context']]
        if not filtered_methodologies:
            row['value'] = None
        else:
            row['value'] = {'project_methodologies': filtered_methodologies}

    elif row['question'] == 'What is the project province, country, latitude, and longitude of this project?':
        city = value_dict.get('project_state_province', '')
        country = value_dict.get('project_country', '')
        latitude = str(value_dict.get('project_latitude', ''))
        longitude = str(value_dict.get('project_longitude', ''))
        if not city.lower() in row['context'].lower():
            value_dict.pop('project_state_province')
        if not country.lower() in row['context'].lower():
            value_dict.pop('project_country')
        if not latitude in row['context']:
            value_dict.pop('project_latitude')
        if not longitude in row['context']:
            value_dict.pop('project_longitude')
        if len(value_dict)>0:
            row['value'] = value_dict
        else:
            row['value'] = None
    
    elif row['question'] == 'What is the project proponents, includind organization name, phone number and email address, state/city and country of the entity responsible for managing this carbon project?':
        if len(value_dict) == 1:
            value = value_dict[0]
            name = value.get('organization_name', '')
            phone = value.get('telephone', '')
            city = value.get('state/city', '')
            country = value.get('country', '')
            if not fuzzy_match(name.lower(), row['context'].lower()):
                value.pop('organization_name', None)
            if not phone in row['context']:
                value.pop('telephone')
            if not city.lower() in row['context'].lower():
                value.pop('state/city')
            if not country.lower() in row['context'].lower():
                value.pop('country')
            if len(value)>0:
                row['value'] = [value]
            else:
                row['value'] = None
        
    return row 


context_df = pd.read_parquet('data/intermediate/kita_dataset/refined_pdd_context_retrieval.parquet', engine='fastparquet')
scrape_df = pd.read_csv('data/intermediate/kita_dataset/processed_ground_truth_project_information.csv')

context_df['id'] = context_df['id'].astype('int')
scrape_df['id'] = scrape_df['id'].astype('int')

df = pd.merge(context_df, scrape_df, left_on=['id', 'section_category'], right_on=['id', 'type'], how='inner')
df['question'] = df['section_category'].map(question_mapping)
df['value_orig'] = df['value']
df = df.apply(filter_groundtruth, axis=1)
df['value'] = df['value'].fillna('No relevant information found in context')

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

train_df = train_df.apply(lambda row: _transform_record(row), axis=1)
train_df.head(2000).to_json('data/intermediate/project_description/gpt35/project_description_train_subset.jsonl', orient='records', lines=True)

val_df = val_df.apply(lambda row: _transform_record(row), axis=1)
val_df.head(400).to_json('data/intermediate/project_description/gpt35/project_description_validate_subset.jsonl', orient='records', lines=True)

test_df[['value']].head(400).to_json('data/intermediate/project_description/gpt35/project_description_test_answer_subset.jsonl', orient='records', lines=True)

test_df = test_df.apply(lambda row: _transform_record_test(row), axis=1)
test_df.head(400).to_json('data/intermediate/project_description/gpt35/project_description_test_prompt_subset.jsonl', orient='records', lines=True)
