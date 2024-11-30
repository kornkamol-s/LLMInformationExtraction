import pandas as pd
import ast
from fuzzywuzzy import fuzz
from dateutil import parser
from config import config


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
        {"role": "system", "content": "You are tasked with extracting relevant information from the provided context to answer the following question. If any key information is missing, omit that key from response. If No relevant information found in context is found, do not return anything."},
        {"role": "user", "content": f"Question: {row['question']}\n\nContext: {row['context']}"}
    ]
    
    # If it is either training or validation set, append assistant role (answer) to the message
    if not is_test:
        messages.append({"role": "assistant", "content": f"{row['value']}"})
    
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


def _fuzzy_match(answer, contexts, threshold=80):
    """
    Perform a fuzzy match to check if the provided answer partially matches any of the contexts above a set threshold.

    Parameters:
        answer (str): The string to search for in the contexts.
        contexts (List): A list of strings to compare with the answer.
        threshold (int): Minimum match score. Default is 80.

    Returns:
        bool: True if any context matches the answer above threshold.
    """
    return any(fuzz.partial_ratio(answer, context) >= threshold for context in contexts)


def _filter_groundtruth(row):
    """
    Remove answer where the answer does not appear in the context.
    
    Parameters:
        row (Series): Data row containing question, context, and value for each information.
        
    Returns:
        Series: Data row where inconsistent answers are set to None.
    """
    # Convert to dictionary
    value_dict = ast.literal_eval(row['value'])

    # Handle Crediting Period by checking the start year of crediting period whether it matches with the context
    if row['section_category'] == 'crediting_period':
        start_date_str = value_dict.get('crediting_period_start', '')
        if start_date_str:
            # Convert to datetime
            start_date = parser.parse(start_date_str)
            # If start year is not in the context, remove the answer
            if not str(start_date.year) in row['context']:
                row['value'] = None

    # Handle Methodology by removing methodology codes that are not included in the context
    elif row['section_category'] == 'methodology':
        methodologies = value_dict.get('project_methodologies', [])
        filtered_methodologies = [methodology for methodology in methodologies if methodology[:-1] in row['context']]
        row['value'] = {'project_methodologies': filtered_methodologies} if filtered_methodologies else None

    # Handle Project location by removing all entities that their information does not appear in context
    elif row['section_category'] == 'project_location':
        if not value_dict.get('project_state_province', '').lower() in str(row['context']).lower():
            value_dict.pop('project_state_province')
        if not value_dict.get('project_country', '').lower() in str(row['context']).lower():
            value_dict.pop('project_country')
        if not str(value_dict.get('project_latitude', '')) in row['context']:
            value_dict.pop('project_latitude')
        if not str(value_dict.get('project_longitude', '')) in row['context']:
            value_dict.pop('project_longitude')

        row['value'] = value_dict if value_dict else None
    
    # Handle Project proponents by removing all entities that their information does not appear in context with partially match
    elif row['section_category'] == 'project_proponents':
        # Process only single proponent, as multiple proponents were manually gathered
        if len(value_dict) == 1:
            value = value_dict[0]
            if not _fuzzy_match(value.get('organization_name', '').lower(), str(row['context']).lower()):
                value.pop('organization_name', None)
            if not _fuzzy_match(value.get('telephone', '').lower(), str(row['context']).lower()):
                value.pop('telephone', None)
            if not _fuzzy_match(value.get('state/city', '').lower(), str(row['context']).lower()):
                value.pop('state/city', None)
            if not _fuzzy_match(value.get('country', '').lower(), str(row['context']).lower()):
                value.pop('country', None)
            
            row['value'] = [value] if value else None

    return row 


def create_split_crediting_period(df, total_records, no_relevant_count):
    """
    Creates a split of crediting period data for training, testing, or validation.
    
    Parameters:
        df (DataFrame): The DataFrame containing crediting period data.
        total_records (int): Total number of records to include in the final split.
        no_relevant_count (int): Number of rows with "No relevant information found in context" to include.

    Returns:
        DataFrame: A subset DataFrame with specified record counts.
    """
    no_info_df = df[df['value'] == 'No relevant information found in context'].head(no_relevant_count)
    remaining_df = df[df['value'] != 'No relevant information found in context'].head(total_records - len(no_info_df))
    final_df = pd.concat([no_info_df, remaining_df])
    return final_df


def create_split_sector(df, total_records, no_relevant_count):
    """
    Creates a split of project sector data for training, testing, or validation.
    
    Parameters:
        df (DataFrame): DataFrame containing sector data.
        total_records (int): Total number of records to include in the final split.
        no_relevant_count (int): Number of rows with "No relevant information found in context" to include.

    Returns:
        DataFrame: A subset DataFrame with specified record counts.
    """
    no_info_df = df[df['value'] == 'No relevant information found in context'].head(no_relevant_count)
    remaining_records = total_records - len(no_info_df)
    # Select both sectors equally
    half_records = remaining_records // 2
    forestry_df = df[df['value'].str.contains('Forestry', na=False)].head(half_records)
    energy_df = df[df['value'].str.contains('Energy', na=False)].head(half_records)
    final_df = pd.concat([no_info_df, forestry_df, energy_df])
    return final_df


def create_split_methodology(df, no_relevant_count, comma_count, acm0002_count, other_count):
    """
    Creates a split of methodology data for training, testing, or validation.
    
    Parameters:
        df (DataFrame): DataFrame containing methodology data.
        no_relevant_count (int): Number of rows with "No relevant information found in context" to include.
        comma_count (int): Number of rows containing multiple methodologies to include.
        acm0002_count (int): Number of rows containing the mojority methodology; 'ACM0002'.
        other_count (int): Number of other type of information to include.

    Returns:
        DataFrame: A subset DataFrame with specified record counts.
    """
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
    """
    Creates a split of location data for training, testing, or validation.
    
    Parameters:
        df (DataFrame): DataFrame containing location data.
        no_relevant_count (int): Number of rows with "No relevant information found in context" to include.
        geo_count (int): Number of rows containing 'project_longitude' to include.
        other_count (int): Number of other relevant rows to include.

    Returns:
        DataFrame: A subset DataFrame with specified record counts.
    """
    no_info_df = df[df['value'] == 'No relevant information found in context'].head(no_relevant_count)
    geo_df = df[df['value'].astype('str').str.contains('project_longitude', na=False)].head(geo_count)
    other_df = df[(df['value'] != 'No relevant information found in context') & 
                  (~df['value'].astype('str').str.contains('project_longitude', na=False))].head(other_count)
    final_df = pd.concat([no_info_df, geo_df, other_df])

    return final_df


def create_split_proponent(df, total_records, no_relevant_count, comma_count, telephone_count, email_count, state_count):
    """
    Creates a split of proponents data for training, testing, or validation.
    
    Parameters:
        df (DataFrame): DataFrame containing proponent data.
        total_records (int): Total number of records to include in the final split.
        no_relevant_count (int): Number of rows with "No relevant information found in context" to include.
        comma_count (int): Number of rows containing multiple proponents to include.
        telephone_count (int): Number of rows containing telephone information to include.
        email_count (int): Number of rows containing email to include.
        state_count (int): Number of rows containing state to include.

    Returns:
        DataFrame: A subset DataFrame with specified record counts.
    """
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


def _data_partitioning(df):
    """
    Partitions a DataFrame into training, testing, and validation sets for various categories.
    
    Parameters:
        df (DataFrame): DataFrame containing all sections and categories to be partitioned.

    Returns:
        Three DataFrames for training, testing, and validation.
    """
    credit_df = df[df['section_category']=='crediting period']
    sector_df = df[df['section_category']=='sector']
    proponent_df = df[df['section_category']=='project_proponents']
    method_df = df[df['section_category']=='methodology']
    location_df = df[df['section_category']=='project_location']
    
    # Split credit data into train-validate-test
    credit_train_df = create_split_crediting_period(credit_df, total_records=400, no_relevant_count=10)
    remaining_credit_df = credit_df[~credit_df.index.isin(credit_train_df.index)]
    credit_test_df = create_split_crediting_period(remaining_credit_df, total_records=50, no_relevant_count=2)
    remaining_credit_df = remaining_credit_df[~remaining_credit_df.index.isin(credit_test_df.index)]
    credit_val_df = create_split_crediting_period(remaining_credit_df, total_records=50, no_relevant_count=2)

    # Split project sector data into train-validate-test
    sector_train_df = create_split_sector(sector_df, total_records=400, no_relevant_count=10)
    remaining_sector_df = sector_df[~sector_df.index.isin(sector_train_df.index)]
    sector_test_df = create_split_sector(remaining_sector_df, total_records=50, no_relevant_count=2)
    remaining_sector_df = remaining_sector_df[~remaining_sector_df.index.isin(sector_test_df.index)]
    sector_val_df = create_split_sector(remaining_sector_df, total_records=50, no_relevant_count=2)

    # Split project methodology data into train-validate-test
    method_train_df = create_split_methodology(method_df, no_relevant_count=10, comma_count=155, acm0002_count=80, other_count=155)
    remaining_method_df = method_df[~method_df.index.isin(method_train_df.index)]
    method_test_df = create_split_methodology(remaining_method_df, no_relevant_count=2, comma_count=20, acm0002_count=8, other_count=20)
    remaining_method_df = remaining_method_df[~remaining_method_df.index.isin(method_test_df.index)]
    method_val_df = create_split_methodology(remaining_method_df, no_relevant_count=2, comma_count=20, acm0002_count=8, other_count=20)

    # Split project location data into train-validate-test
    location_train_df = create_split_location(location_df, no_relevant_count=10, geo_count=101, other_count=289)
    remaining_location_df = location_df[~location_df.index.isin(location_train_df.index)]
    location_test_df = create_split_location(remaining_location_df, no_relevant_count=2, geo_count=10, other_count=38)
    remaining_location_df = remaining_location_df[~remaining_location_df.index.isin(location_test_df.index)]
    location_val_df = create_split_location(remaining_location_df, no_relevant_count=2, geo_count=10, other_count=38)

    # Split project proponent data into train-validate-test
    proponent_train_df = create_split_proponent(proponent_df, total_records=400, no_relevant_count=3, comma_count=54, 
                                    telephone_count=150, email_count=150, state_count=27)
    remaining_proponent_df = proponent_df[~proponent_df.index.isin(proponent_train_df.index)]
    proponent_test_df = create_split_proponent(remaining_proponent_df, total_records=50, no_relevant_count=0, comma_count=9, 
                                    telephone_count=18, email_count=18, state_count=4)
    remaining_proponent_df = remaining_proponent_df[~remaining_proponent_df.index.isin(proponent_test_df.index)]
    proponent_val_df = create_split_proponent(remaining_proponent_df, total_records=50, no_relevant_count=0, comma_count=9, 
                                    telephone_count=18, email_count=18, state_count=4)

    # Merge various questions into train-validate-test
    train_df = pd.concat([credit_train_df, sector_train_df, method_train_df, location_train_df, proponent_train_df])
    test_df = pd.concat([credit_test_df, sector_test_df, method_test_df, location_test_df, proponent_test_df])
    val_df = pd.concat([credit_val_df, sector_val_df, method_val_df, location_val_df, proponent_val_df])

    return train_df, test_df, val_df


filepath = 'data/training/data_partitioning'

# Load contexts and answers
context_df = pd.read_csv('data/training/data_processing/pdd_context_retrieval.csv', encoding='utf-8')
answer_df = pd.read_csv('data/training/data_processing/processed_ground_truth_project_info.csv', encoding='utf-8')

# Convert ID to integer, and merge using id and information type
context_df['id'] = context_df['id'].astype('int')
answer_df['id'] = answer_df['id'].astype('int')
context_df = context_df[~context_df['context'].isna()]
df = pd.merge(context_df, answer_df, left_on=['id', 'section_category'], right_on=['id', 'type'], how='inner')

# Map questions to specific information type
df['question'] = df['section_category'].map(config.QUESTION_MAPPING)

# Remove records containing inconsistency between contexts and answers
df = df.apply(_filter_groundtruth, axis=1)

# Fill no answer with proper representation
df['value'] = df['value'].fillna('No relevant information found in context')

# Separate into train-test-validation
train_df, test_df, val_df = _data_partitioning(df)

# Shuffle to reduce bias
train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)
val_df = val_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Transform and save to local file
_save_jsonl(train_df, f'{filepath}/train/project_info_train.jsonl', _transform_record)
_save_jsonl(val_df, f'{filepath}/validate/project_info_validate.jsonl', _transform_record)
_save_jsonl(test_df, f'{filepath}/test/project_info_test_prompt.jsonl', lambda row: _transform_record(row, is_test=True))
test_df = test_df.rename(columns={'value': 'answers'})
test_df['answers'].to_csv(f'{filepath}/test/project_info_test_answer.csv', index=False)