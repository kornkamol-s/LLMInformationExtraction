import re, warnings
import pandas as pd

warnings.filterwarnings("ignore", category=(FutureWarning, DeprecationWarning))

# Load emissions reductions data from Excel and CSV sources
cb_df = pd.read_excel('data/training/data_collection/CarbonMarkets_ghg.xlsx', sheet_name='Emissions Reductions Sum')
ao_df = pd.read_csv('data/training/data_collection/AlliedOffsets_ghg.csv', encoding='utf-8')

# Filter AlliedOffsets data to only include self-reported forecast issuance
ao_df = ao_df[ao_df['Measure Names']=='Forecast issuance (self-reported)']

# Drop rows with missing value
ao_df = ao_df.dropna(subset=['Measure Values'])

# Extract the number from 'uid' column using regex and convert to integer for consistency
ao_df['id'] = ao_df['uid'].apply(lambda x: re.findall(r'\d+', x)[0]).astype('int')

# Keep only necessary columns for merging and further processing
ao_df = ao_df[['id', 'Vintage (forecast vs actual issuance)', 'Measure Values']]

# Exclude any rows where 'Measure Values' is zero, to avoid irrelevant data
ao_df = ao_df[ao_df['Measure Values']!=0.0]

# Reshape data from CarbonMarkets to have emission values across multiple years
cb_df = cb_df.melt(id_vars=['Project ID'], var_name='Year', value_name='Value')

# Remove rows with missing data
cb_df = cb_df.dropna(subset=['Year', 'Value'])

# Filter out rows with zero value
cb_df = cb_df[(cb_df['Year']!=0)&(cb_df['Value']!=0.0)]

# Convert ID to integer
cb_df['Project ID'] = cb_df['Project ID'].astype('int')

# Merge data from both sources
df = pd.merge(ao_df, cb_df, right_on='Project ID', left_on='id', how='outer')

# Fill missing value in carbonmarkets dataset with alliedoffsets
df['Project ID'] = df['Project ID'].fillna(df['id']).astype('int')
df['Year'] = df['Year'].fillna(df['Vintage (forecast vs actual issuance)']).astype('int')
df['Value'] = df['Value'].fillna(df['Measure Values'])

# Remove redundant columns
df = df.drop(columns=['id', 'Vintage (forecast vs actual issuance)', 'Measure Values'])

# Group data by ID and aggregate emission reductions as a dictionary of {Year: Value}
df = df.groupby('Project ID').apply(lambda x: dict(zip(x['Year'], x['Value']))).reset_index()
df.columns = ['Project ID', 'GHG Emission Reductions']

# Save processed data to CSV file in the specified path
df.to_csv('data/training/data_processing/processed_ground_truth_ghg.csv', index=False, encoding='utf-8')