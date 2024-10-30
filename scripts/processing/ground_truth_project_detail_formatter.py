import os
import logging
import argparse
import pandas as pd
import numpy as np
import us
import re
from flashgeotext.geotext import GeoText
from commonregex import CommonRegex

def combine_proponents_and_entities(df):
    df['Combined_Proponents'] = df[['Proponents', 'Proponents2', 'Proponents3', 'Proponents4', 
                                    'Proponents5', 'Proponents6', 'Proponents7', 'Proponents8', 
                                    'Proponents9', 'Proponents10']
                                ].apply(lambda row: [str(proponent) for proponent in row if pd.notna(proponent)], axis=1)
    
    df['Combined_Other_Entities'] = df[['Other Entities', 'Other Entities2', 'Other Entities3', 
                                        'Other Entities4', 'Other Entities5', 'Other Entities6', 
                                        'Other Entities7', 'Other Entities8', 'Other Entities9']
                                    ].apply(lambda row: [str(entity) for entity in row if pd.notna(entity)], axis=1)
    
    return df


# Set up logging configuration with timestamps
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

geotext = GeoText()
state_mapping = us.states.mapping('abbr', 'name')
pattern = '|'.join(re.escape(abbr) for abbr in state_mapping.keys())
url_pattern = r'(https?:\/\/[^\s/$.?#].[^\s]*|www\.[^\s/$.?#].[^\s]*)'

methods_df = pd.read_excel('data/input/CDM methodologies.xlsx', sheet_name='Approved', usecols=['Number'])
methods = methods_df['Number'].tolist()

def main(args):
    manual_groundtruth = pd.read_csv('data/input/manual_groundtruth.csv')
    manual_groundtruth = combine_proponents_and_entities(manual_groundtruth)
    manual_groundtruth = manual_groundtruth[['ID', 'Combined_Proponents', 'Combined_Other_Entities']]
    scrape_df = pd.read_csv('data/input/scraped_data.csv', encoding='utf-8')
    scrape_df = scrape_df[[ 'title'
                            ,'description'
                            ,'State/Province'
                            ,'Proponent'
                            ,'VCS Project Type'
                            ,'VCS Methodology'
                            ,'Crediting Period Term'
                            ,'id']]

    project_detail_df = pd.read_csv('data/input/additional_project_information.csv', encoding='utf-8')
    project_detail_df = project_detail_df[
        (project_detail_df['UID'].str.contains('VCS', na=False)) &
        (project_detail_df['Project Sector'].isin(['Forestry and Land Use', 'Renewable Energy']))
    ]
    project_detail_df['id'] = project_detail_df['UID'].str.extract(r'(\d+)', expand=False)
    project_detail_df = project_detail_df[[ 'id'
                                            ,'Project Name'
                                            ,'Project Description'
                                            ,'Project Sector'
                                            ,'Project Country'
                                            ,'Project State Or Province'
                                            ,'Project Latitude'
                                            ,'Project Longitude'
                                            ,'Project Methodologies'
                                            ,'Crediting Period End Date'
                                            ,'Crediting Period Start Date']]

    scrape_df['id'] = scrape_df['id'].astype('int')
    project_detail_df['id'] = project_detail_df['id'].astype('int')

    df = scrape_df.merge(project_detail_df, how='outer', on='id')
    df = df.merge(manual_groundtruth, how='left', left_on='id', right_on='ID')
    
    df['Proponent'] = df['Proponent'].replace('Multiple Proponents', None)
    df['Proponent'] = df['Proponent'].apply(lambda x: [x] if x is not None and pd.notna(x) else x)
    df['Proponent'] = df['Proponent'].fillna(df['Combined_Proponents'])
    df['Project Name'] = df['Project Name'].fillna(df['title'])
    df['Project Description'] = df['Project Description'].fillna(df['description'])
    df['Project State Or Province'] = df['Project State Or Province'].fillna(df['State/Province'])
    df['VCS Methodology'] = df['VCS Methodology'].fillna(df['Project Methodologies'])

    df['Project Sector'] = df['Project Sector'].fillna(df['VCS Project Type'])
    df['Project Sector'] = df['Project Sector'].apply(lambda x: 'Forestry and Land Use' if 'forestry' in x.lower() 
                                                                    else ('Renewable Energy' if 'energy' in x.lower() else np.nan))
    
    df['Crediting Period Start Date'] = pd.to_datetime(df['Crediting Period Start Date'], format='%B %d, %Y').dt.strftime('%Y-%m-%d')
    df['Crediting Period End Date'] = pd.to_datetime(df['Crediting Period End Date'], format='%B %d, %Y').dt.strftime('%Y-%m-%d')

    df[['Start Date', 'End Date']] = df['Crediting Period Term'].apply(lambda text: pd.Series(re.findall(r'(\d{2}/\d{2}/\d{4})', text) if isinstance(text, str) else [np.nan, np.nan]))
    df['Start Date'] = pd.to_datetime(df['Start Date'], format='%d/%m/%Y').dt.strftime('%Y-%m-%d')
    df['End Date'] = pd.to_datetime(df['End Date'], format='%d/%m/%Y').dt.strftime('%Y-%m-%d')

    df['Crediting Period Start Date'] = df['Crediting Period Start Date'].fillna(df['Start Date'])
    df['Crediting Period End Date'] = df['Crediting Period End Date'].fillna(df['End Date'])


    df.drop(columns=['title', 'Project Methodologies', 'description', 'State/Province', 'VCS Project Type', 'Crediting Period Term', 'Start Date', 'End Date'], inplace=True)

    if args.ids:
        total_ids = args.ids
    else:
        total_ids = df['id'].to_list()

    if os.path.exists(args.output):
        processed_ids = pd.read_csv(args.output)['id'].astype('int').to_list()
        total_ids = list(set(total_ids)-set(processed_ids))

    to_be_processed = df[df['id'].isin(total_ids)]
    processed_df = to_be_processed.copy()

    logging.info(f'=========================================================')
    logging.info(f'Total ID : {len(processed_df)}')
    logging.info(f'=========================================================')

    if len(processed_df):
        processed_df['project_proponents'] = processed_df.apply(_process_proponent, axis=1)
        processed_df['project_description'] = processed_df.apply(_process_project_description, axis=1)
        processed_df['crediting period'] = processed_df.apply(_process_crediting, axis=1)
        processed_df['sector'] = processed_df.apply(_process_sector, axis=1)
        processed_df['methodology'] = processed_df.apply(_process_method, axis=1)

        final_df = processed_df[['id', 'project_proponents', 'project_description', 
                                'crediting period', 'sector', 'methodology']]

        final_df = pd.melt(final_df, 
                            id_vars=['id'], 
                            value_vars=[
                                'project_proponents', 
                                'project_description', 
                                'crediting period', 
                                'sector', 
                                'methodology'
                            ], 
                            var_name='type', 
                            value_name='value')
        
        final_df = final_df.dropna(subset=['value'])
        final_df.to_csv(args.output, index=False, encoding='utf-8')


def _process_project_description(row):
    result = {
            'project_state_province': row['Project State Or Province'],
            'project_country': row['Project Country'],
            'project_latitude': row['Project Latitude'],
            'project_longitude': row['Project Longitude'],
            }
    filtered_data = {k: v for k, v in result.items() if pd.notna(v)}
    return filtered_data


def _process_crediting(row):
    result = {'crediting_period_start': row['Crediting Period Start Date'],
            'crediting_period_end': row['Crediting Period End Date'],
            }
    filtered_data = {k: v for k, v in result.items() if pd.notna(v)}
    return filtered_data


def _process_sector(row):
    if pd.notna(row['Project Sector']) and row['Project Sector']:
        return {'project_sector': row['Project Sector']}


def _process_method(row):
    if pd.notna(row['VCS Methodology']) and row['VCS Methodology']:
        methodologies = row['VCS Methodology'].split(',')
        filtered_methodologies = [method for method in methodologies if method in methods]
        if filtered_methodologies:
            return {'project_methodologies': filtered_methodologies}


def _process_proponent(row):
    proponents = row['Proponent']
    processed_proponents = []
    if isinstance(proponents, list):
        for proponent in proponents:
                presumed_contact_name = proponent.split('\n')[0].strip()
                
                # Extract phone, email, link, city, and country
                text = CommonRegex(proponent)
                phones = text.phones
                emails = text.emails
                links = text.links
                links = re.findall(url_pattern, " ".join(links), re.IGNORECASE)

                # Extract cities and countries
                data = geotext.extract(input_text=proponent)
                cities = list(data.get('cities', {}).keys())
                countries = list(data.get('countries', {}).keys())      

                if not cities:  
                    # Replace state abbreviations with full names to be able to catch
                    pattern = r'\b(?:' + '|'.join(state_mapping.keys()) + r')\b'
                    replaced_states = re.sub(pattern, lambda match: state_mapping[match.group(0)], proponent)
                    data = geotext.extract(input_text=replaced_states)
                    cities = list(data.get('cities', {}).keys())
                
                # Ensure there's at most one of each
                phone = phones[0] if phones else None
                email = emails[0] if emails else None
                link = links[0] if links else None
                city = cities[0] if cities else None
                country = countries[0] if countries else None
                
                processed_proponent = {
                    'organization_name': presumed_contact_name,
                    'telephone': phone,
                    'email': email,
                    'state/city': city,
                    'country': country
                }
                
                # Filter out None values
                filtered_data = {k: v for k, v in processed_proponent.items() if v is not None}
                
                processed_proponents.append(filtered_data)

        return processed_proponents


def _setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, default='analysis/projects_information.csv', nargs='?', help='Input File')
    parser.add_argument('--ids', type=int, nargs='+',help='IDs')
    parser.add_argument('--output', type=str, default='data/intermediate/kita_dataset/processed_ground_truth_project_information.csv', nargs='?',help='Output Folder')
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = _setup_args()
    main(args)