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
                "content": "You are tasked with extracting relevant information from the provided context to answer the following question. If any key information is missing, omit that key from response. If No relevant information found in context is found, do not return anything."
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
                "content": "You are tasked with extracting relevant information from the provided context to answer the following question. If any key information is missing, omit that key from response. If No relevant information found in context is found, do not return anything."
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
        filtered_methodologies = [methodology for methodology in methodologies if methodology[:-1] in row['context']]
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
            if not fuzzy_match(phone.lower(), row['context'].lower()):
                value.pop('telephone', None)
            if not fuzzy_match(city.lower(), row['context'].lower()):
                value.pop('state/city', None)
            if not fuzzy_match(country.lower(), row['context'].lower()):
                value.pop('country', None)
            if len(value)>0:
                row['value'] = [value]
            else:
                row['value'] = None
    return row 


def create_split_crediting_period(df, total_records, no_relevant_count):
    no_info_df = df[df['value'] == 'No relevant information found in context'].head(no_relevant_count)
    remaining_df = df[(df['value'] != 'No relevant information found in context')&(df['value']!='{}')].head(total_records - len(no_info_df))
    final_df = pd.concat([no_info_df, remaining_df])
    return final_df


def create_split_sector(df, total_records, no_relevant_count):
    no_info_df = df[df['value'] == 'No relevant information found in context'].head(no_relevant_count)
    remaining_records = total_records - len(no_info_df)
    half_records = remaining_records // 2
    forestry_df = df[df['value'].str.contains('Forestry', na=False)].head(half_records)
    energy_df = df[df['value'].str.contains('Energy', na=False)].head(half_records)
    final_df = pd.concat([no_info_df, forestry_df, energy_df])
    return final_df


def create_split_methodology(df, no_relevant_count, comma_count, acm0002_count, other_count):
    no_info_df = df[df['value'] == 'No relevant information found in context'].head(no_relevant_count)
    comma_df = df[df['value'].astype('str').str.contains(',', na=False)].head(comma_count)
    acm0002_df = df[df['value'].astype('str').str.contains('ACM0002') &
                  (~df['value'].astype('str').str.contains(',', na=False))].head(acm0002_count)
    other_df = df[(df['value'] != 'No relevant information found in context') & 
                  (~df['value'].astype('str').str.contains('ACM0002', na=False)) &
                  (~df['value'].astype('str').str.contains(',', na=False))].head(other_count)
    final_df = pd.concat([no_info_df, comma_df, acm0002_df, other_df])
    return final_df


def create_split_location(df, no_relevant_count, geo_count, other_count):
    no_info_df = df[df['value'] == 'No relevant information found in context'].head(no_relevant_count)
    geo_df = df[df['value'].astype('str').str.contains('project_longitude', na=False)].head(geo_count)
    
    other_df = df[(df['value'] != 'No relevant information found in context') & 
                  (~df['value'].astype('str').str.contains('project_longitude', na=False))].head(other_count)
    final_df = pd.concat([no_info_df, geo_df, other_df])
    return final_df


def create_split_proponent(df, total_records, no_relevant_count, comma_count, telephone_count, email_count, state_count):
    no_info_df = df[df['value'] == 'No relevant information found in context'].head(no_relevant_count)
    remaining_df = df[df['value'] != 'No relevant information found in context']
    comma_df = remaining_df[remaining_df['value'].astype('str').str.contains(', {', na=False)].head(comma_count)
    remaining_df = remaining_df[~remaining_df['value'].astype('str').str.contains(', {', na=False)]
    telephone_df = remaining_df[remaining_df['value'].astype('str').str.contains('telephone', case=False, na=False)].head(telephone_count)
    remaining_df = remaining_df[~remaining_df.index.isin(telephone_df.index)]
    email_df = remaining_df[remaining_df['value'].astype('str').str.contains('email', case=False, na=False)].head(email_count)
    remaining_df = remaining_df[~remaining_df.index.isin(email_df.index)]
    state_df = remaining_df[remaining_df['value'].astype('str').str.contains('state', case=False, na=False)].head(state_count)
    remaining_df = remaining_df[~remaining_df.index.isin(state_df.index)]
    other_count = total_records - (len(no_info_df) + len(comma_df) + len(telephone_df) + len(email_df) + len(state_df))
    other_df = remaining_df.head(other_count)
    final_df = pd.concat([no_info_df, comma_df, telephone_df, email_df, state_df, other_df])
    return final_df


def data_partitioning(df):
    credit_df = df[df['section_category']=='crediting period']
    sector_df = df[df['section_category']=='sector']
    proponent_df = df[df['section_category']=='project_proponents']
    method_df = df[df['section_category']=='methodology']
    location_df = df[df['section_category']=='project_description']
    
    credit_train_df = create_split_crediting_period(credit_df, total_records=400, no_relevant_count=10)
    remaining_credit_df = credit_df[~credit_df.index.isin(credit_train_df.index)]
    credit_test_df = create_split_crediting_period(remaining_credit_df, total_records=50, no_relevant_count=2)
    remaining_credit_df = remaining_credit_df[~remaining_credit_df.index.isin(credit_test_df.index)]
    credit_val_df = create_split_crediting_period(remaining_credit_df, total_records=50, no_relevant_count=2)

    sector_train_df = create_split_sector(sector_df, total_records=400, no_relevant_count=10)
    remaining_sector_df = sector_df[~sector_df.index.isin(sector_train_df.index)]
    sector_test_df = create_split_sector(remaining_sector_df, total_records=50, no_relevant_count=2)
    remaining_sector_df = remaining_sector_df[~remaining_sector_df.index.isin(sector_test_df.index)]
    sector_val_df = create_split_sector(remaining_sector_df, total_records=50, no_relevant_count=2)

    method_train_df = create_split_methodology(method_df, no_relevant_count=10, comma_count=155, acm0002_count=80, other_count=155)
    remaining_method_df = method_df[~method_df.index.isin(method_train_df.index)]
    method_test_df = create_split_methodology(remaining_method_df, no_relevant_count=2, comma_count=20, acm0002_count=8, other_count=20)
    remaining_method_df = remaining_method_df[~remaining_method_df.index.isin(method_test_df.index)]
    method_val_df = create_split_methodology(remaining_method_df, no_relevant_count=2, comma_count=20, acm0002_count=8, other_count=20)

    location_train_df = create_split_location(location_df, no_relevant_count=10, geo_count=101, other_count=289)
    remaining_location_df = location_df[~location_df.index.isin(location_train_df.index)]
    location_test_df = create_split_location(remaining_location_df, no_relevant_count=2, geo_count=10, other_count=38)
    remaining_location_df = remaining_location_df[~remaining_location_df.index.isin(location_test_df.index)]
    location_val_df = create_split_location(remaining_location_df, no_relevant_count=2, geo_count=10, other_count=38)


    proponent_train_df = create_split_proponent(proponent_df, total_records=400, 
                                                    no_relevant_count=3, comma_count=54, 
                                                    telephone_count=150, email_count=150, 
                                                    state_count=27)
    remaining_proponent_df = proponent_df[~proponent_df.index.isin(proponent_train_df.index)]
    proponent_test_df = create_split_proponent(remaining_proponent_df, total_records=50, 
                                                    no_relevant_count=0, comma_count=9, 
                                                    telephone_count=18, email_count=18, 
                                                    state_count=4)
    remaining_proponent_df = remaining_proponent_df[~remaining_proponent_df.index.isin(proponent_test_df.index)]
    proponent_val_df = create_split_proponent(remaining_proponent_df, total_records=50, 
                                                    no_relevant_count=0, comma_count=9, 
                                                    telephone_count=18, email_count=18, 
                                                    state_count=4)

    train_df = pd.concat([
        credit_train_df,
        sector_train_df,
        method_train_df,
        location_train_df, 
        proponent_train_df
    ])

    test_df = pd.concat([
        credit_test_df,
        sector_test_df,
        method_test_df,
        location_test_df,
        proponent_test_df
    ])

    val_df = pd.concat([
        credit_val_df,
        sector_val_df,
        method_val_df,
        location_val_df,
        proponent_val_df
    ])
    return train_df, test_df, val_df


context_df = pd.read_parquet('data/intermediate/kita_dataset/refined_pdd_context_retrieval.parquet', engine='fastparquet')
scrape_df = pd.read_csv('data/intermediate/kita_dataset/processed_ground_truth_project_information_refined.csv')

context_df['id'] = context_df['id'].astype('int')
scrape_df['id'] = scrape_df['id'].astype('int')

df = pd.merge(context_df, scrape_df, left_on=['id', 'section_category'], right_on=['id', 'type'], how='inner')
df['question'] = df['section_category'].map(question_mapping)
df['value_orig'] = df['value']
df = df.apply(filter_groundtruth, axis=1)
df['value'] = df['value'].fillna('No relevant information found in context')
train_df, test_df, val_df = data_partitioning(df)

train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)
val_df = val_df.sample(frac=1, random_state=42).reset_index(drop=True)

# train_df.to_csv('data/intermediate/project_description/gpt35/project_description_subset_train.csv')
# test_df.to_csv('data/intermediate/project_description/gpt35/project_description_subset_test.csv')
# val_df.to_csv('data/intermediate/project_description/gpt35/project_description_subset_val.csv')

# train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
# val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

train_df = train_df.apply(lambda row: _transform_record(row), axis=1)
train_df.to_json('data/intermediate/project_description/gpt35/project_description_train_subset.jsonl', orient='records', lines=True)

val_df = val_df.apply(lambda row: _transform_record(row), axis=1)
val_df.to_json('data/intermediate/project_description/gpt35/project_description_validate_subset.jsonl', orient='records', lines=True)

test_df[['value']].to_json('data/intermediate/project_description/gpt35/project_description_test_answer_subset.jsonl', orient='records', lines=True)
test_df = test_df.apply(lambda row: _transform_record_test(row), axis=1)
test_df.to_json('data/intermediate/project_description/gpt35/project_description_test_prompt_subset.jsonl', orient='records', lines=True)