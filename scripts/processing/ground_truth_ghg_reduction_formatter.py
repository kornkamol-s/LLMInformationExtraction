import os
import logging
import argparse
import pandas as pd
import numpy as np
import us
import re
from flashgeotext.geotext import GeoText
from commonregex import CommonRegex

# Set up logging configuration with timestamps
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main(args):
    cb_df = pd.read_excel('data/input/CB_data.xlsx', sheet_name='Emissions Reductions Sum')
    ao_df = pd.read_csv('data/input/AO_data.csv', encoding='utf-8')

    ao_df = ao_df[ao_df['Measure Names']=='Forecast issuance (self-reported)']
    ao_df = ao_df.dropna(subset=['Measure Values'])
    ao_df['id'] = ao_df['uid'].apply(lambda x: re.findall(r'\d+', x)[0]).astype('int')
    ao_df = ao_df[['id', 'Vintage (forecast vs actual issuance)', 'Measure Values']]
    ao_df = ao_df[ao_df['Measure Values']!=0.0]

    cb_df = cb_df.melt(id_vars=['Project ID'], var_name='Year', value_name='Value')
    cb_df = cb_df.dropna(subset=['Year', 'Value'])
    cb_df = cb_df[(cb_df['Year']!=0)&(cb_df['Value']!=0.0)]
    cb_df['Project ID'] = cb_df['Project ID'].astype('int')

    df = pd.merge(ao_df, cb_df, right_on='Project ID', left_on='id', how='outer')

    df['Project ID'] = df['Project ID'].fillna(df['id']).astype('int')
    df['Year'] = df['Year'].fillna(df['Vintage (forecast vs actual issuance)']).astype('int')
    df['Value'] = df['Value'].fillna(df['Measure Values'])
    df = df.drop(columns=['id', 'Vintage (forecast vs actual issuance)', 'Measure Values'])
    df = df.groupby('Project ID').apply(lambda x: dict(zip(x['Year'], x['Value']))).reset_index()
    df.columns = ['Project ID', 'GHG Emission Reductions']
    df.to_csv(args.output, index=False, encoding='utf-8')


def _setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default='data/intermediate/project-details-dataset/processed_ground_truth_ghg.csv', nargs='?',help='Output Folder')
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = _setup_args()
    main(args)