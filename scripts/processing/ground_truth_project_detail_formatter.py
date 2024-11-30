import us, re, warnings
import pandas as pd
import numpy as np
from flashgeotext.geotext import GeoText
from commonregex import CommonRegex


def _filter_na(data):
    """
    Filters out items in a dictionary with no value.
    
    Parameters:
        data (Dict): The dictionary to filter.
        
    Returns:
        Dict: A new dictionary containing only items with non-nullable values.
    """
    return {k: v for k, v in data.items() if pd.notna(v)}


def _parse_proponent(proponents):
    """
    Parse a list of proponents' details, extracting organization name, telephone, email, city, and country.
    
    Parameters:
        proponents (List): List of strings representing proponent details.
        
    Returns:
        List of dictionaries containing proponent information with keys.
    """

    # Return None when parse information not in a list
    if not isinstance(proponents, list):
        return None
    
    processed = []
    # Process each proponent
    for prop in proponents:
        
        # Use first line as organization name
        contact_name = prop.split('\n')[0].strip()

        # Use CommonRegex to extract phone and email
        text = CommonRegex(prop)

        # Extract cities and countries from text
        data = geotext.extract(prop)
        cities = list(data.get('cities', {}).keys())
        countries = list(data.get('countries', {}).keys())

        # If no city found, replace state abbreviations and re-attempt city extraction
        if not cities:
            pattern = r'\b(?:' + '|'.join(state_mapping.keys()) + r')\b'
            replaced_states = re.sub(pattern, lambda match: state_mapping[match.group(0)], prop)
            data = geotext.extract(input_text=replaced_states)
            cities = list(data.get('cities', {}).keys())
            
        # Append parsed details to list
        processed.append(_filter_na({
            'organization_name': contact_name,
            'telephone': text.phones[0] if text.phones else None,
            'email': text.emails[0] if text.emails else None,
            'state/city': cities[0] if cities else None,
            'country': countries[0] if countries else None
        }))

    return processed


def _combine_columns(df, cols, name):
    """
    Combine specified columns in a DataFrame into a single list column.
    
    Parameters:
        df (DataFrame): Input DataFrame.
        cols (List): List of column names to combine.
        name (str): Name of the new column that will hold combined lists.
        
    Returns:
        DataFrame with a new column containing combined lists.
    """
    df[name] = df[cols].apply(lambda row: [str(i) for i in row if pd.notna(i)], axis=1)

    return df


def _process_columns(row):
    """
    Process a row of project data to extract and organize location, crediting period, sector, methodology, and proponent data.
    
    Parameters:
        row (Series): Series representing a row from the DataFrame.
        
    Returns:
        Dictionary containing processed project information.
    """
    return {
        'id': row['id'],
        'project_location': _filter_na({
            'project_state_province': row['Project State Or Province'],
            'project_country': row['Project Country'],
            'project_latitude': row['Project Latitude'],
            'project_longitude': row['Project Longitude']
        }),
        'crediting_period': _filter_na({
            'crediting_period_start': row['Crediting Period Start Date'],
            'crediting_period_end': row['Crediting Period End Date']
        }),
        'sector': {'project_sector': row['Project Sector']} if pd.notna(row['Project Sector']) else None,
        'methodology': {'project_methodologies': methodologies} if (methodologies := [m for m in str(row.get('VCS Methodology', '')).split(',') if m in methods]) else None,
        'project_proponents': _parse_proponent(row['Proponent'])
    }



# Load project scraping data
scrape_df = pd.read_csv('data/training/data_collection/verra_data.csv', encoding='utf-8')
scrape_df['id'] = scrape_df['id'].astype('int')

# Load project information from AlliedOffsets
project_detail_df = pd.read_csv('data/training/data_collection/AlliedOffsets_project_info.csv', encoding='utf-8', low_memory=False)

# Filter project sector and projects registered in VCS, and extract project ID into numerical format
project_detail_df = project_detail_df[
    (project_detail_df['UID'].str.contains('VCS', na=False)) &
    (project_detail_df['Project Sector'].isin(['Forestry and Land Use', 'Renewable Energy']))
]
project_detail_df['id'] = project_detail_df['UID'].str.extract(r'(\d+)', expand=False)
project_detail_df['id'] = project_detail_df['id'].astype('int')

# Load manual groundtruth containing multiple proponent data
manual_groundtruth = pd.read_csv('data/training/data_collection/multiple_proponents.csv')

# Merge multiple columns holding multiple proponent data into single column
manual_groundtruth = _combine_columns(manual_groundtruth, [f'Proponents{i}' for i in range(1, 11)], 'Combined_Proponents')

# Merge dataframes to the final dataset
df = scrape_df.merge(project_detail_df, on='id', how='outer').merge(manual_groundtruth, left_on='id', right_on='ID', how='left')

# Remove all 'Multiple Proponents' to be replaced by detailed multiple proponents data from 'manual_groundtruth' dataset
df['Proponent'] = df['VCS Proponent'].replace('Multiple Proponents', None)

# Convert single proponent in to list element to be consistent with multiple proponents
df['Proponent'] = df['Proponent'].apply(lambda x: [x] if x is not None and pd.notna(x) else x)

# Fill empty record with multiple proponents if exists
df.fillna({'Proponent': df['Combined_Proponents']}, inplace=True)

# Fill empty value in Verra dataset with AlliedOffsets data
df.fillna({'Project State Or Province': df['State/Province'], 
           'VCS Methodology': df['Project Methodologies'],
           'Project Sector': df['VCS Project Type']}, inplace=True)

# Categorised all variants of project sector to be in 2 categories
df['Project Sector'] = df['Project Sector'].apply(lambda x: 'Forestry and Land Use' if 'forestry' in x.lower() 
                     else ('Renewable Energy' if 'energy' in x.lower() else np.nan))

# Convert crediting period from AlliedOffsets to datetime
df['Crediting Period Start Date'] = pd.to_datetime(df['Crediting Period Start Date'], format='%B %d, %Y').dt.strftime('%Y-%m-%d')
df['Crediting Period End Date'] = pd.to_datetime(df['Crediting Period End Date'], format='%B %d, %Y').dt.strftime('%Y-%m-%d')

# Convert crediting period from Verra to datetime
df[['Start Date', 'End Date']] = df['Crediting Period Term'].str.extractall(r'(\d{2}/\d{2}/\d{4})').unstack().fillna(np.nan)
df['Start Date'] = pd.to_datetime(df['Start Date'], format='%d/%m/%Y', errors='coerce').dt.strftime('%Y-%m-%d')
df['End Date'] = pd.to_datetime(df['End Date'], format='%d/%m/%Y', errors='coerce').dt.strftime('%Y-%m-%d')

# Fill empty value in AlliedOffsets using Verra dataset if exists
df.fillna({'Crediting Period Start Date': df['Start Date'], 'Crediting Period End Date': df['End Date']}, inplace=True)

# Initialize GeoText for geographic extraction
geotext = GeoText()

# Create a mapping of state abbreviations to full state names
state_mapping = us.states.mapping('abbr', 'name')
pattern = '|'.join(re.escape(abbr) for abbr in state_mapping.keys())

# Load methodologies and prepare a list of valid methodology numbers
methods_df = pd.read_excel('data/training/data_collection/CDM methodologies.xlsx', sheet_name='Approved', usecols=['Number'])
methods = methods_df['Number'].tolist()

# Process each row to be in structured format
processed_df = df.apply(_process_columns, axis=1, result_type='expand')

# Stack all columns into single row, separating with type column
final_df = pd.melt(processed_df, id_vars=['id'], 
                  value_vars=['project_location', 'crediting_period', 'sector', 'methodology', 'project_proponents'],
                  var_name='type', 
                  value_name='value')

# Remove all null records
final_df = final_df[~final_df['value'].apply(lambda x: x in ['', {}, [], None])]

# Save the final processed data to CSV
final_df.to_csv('data/training/data_processing/processed_ground_truth_project_info.csv', index=False, encoding='utf-8')