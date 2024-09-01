import os
import json
import pandas as pd
import logging 
import argparse
from tools.PDFExtraction import PDFExtraction
from tools.utils import find_pdf_files
from retry import retry

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

with open('config/heading_mapping.json', 'r') as f:
    headings_mapping = json.load(f)

@retry(tries=50, delay=2)
def main(args):
    pdf_files = find_pdf_files(args.input)

    if args.ids:
        pdf_files = [f for f in pdf_files if int(f.split('_', 1)[0]) in args.ids]
 
    if os.path.exists(args.output):
        processed_files = pd.read_parquet(args.output, engine='fastparquet')['filename'].unique().tolist()
    else:
        processed_files = []

    pdf_files = list(set(pdf_files) - set(processed_files))

    for index, file in enumerate(pdf_files, start=1):
        logging.info(f'=========================================================')
        logging.info(f'Processing [{index}/{len(pdf_files)}] : {file}')
        logging.info(f'=========================================================')

        pdf_extractor = PDFExtraction(f"{args.input}/{file}")
        toc_df = pdf_extractor._get_toc()

        context_df = extract_relevant_section(pdf_extractor, toc_df)
        context_df['id'] = file.split('_', 1)[0]
        context_df['filename'] = file

        if os.path.exists(args.output):
            context_df.to_parquet(args.output, engine='fastparquet', index=False, append=True)
        else:
            context_df.to_parquet(args.output, engine='fastparquet', index=False)


def extract_relevant_section(pdf, toc):
    rows = []
    for section, variants in headings_mapping.items():
        contexts = ''
        pattern = '|'.join(variants)
        matched_df = toc[toc['section'].str.contains(pattern, case=False, na=False)]
        for _, row in matched_df.iterrows():
            context = pdf._extract_page_range(row['start_page'], row['end_page'], row['section'])
            contexts = contexts + '\n' + context
        rows.append({'section_category': section, 'context': contexts})
    df = pd.DataFrame(rows)

    return df 

    
def find_pdf_files(folder_location):    
    if not os.path.isdir(folder_location):
        raise FileNotFoundError("The provided location does not exist")

    pdf_files = [
        file for file in os.listdir(folder_location)
        if file.endswith('.pdf')
    ]
    return pdf_files


def _setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, default='data/input/pdds', nargs='?', help='Input Folder')
    parser.add_argument('--ids', type=int, nargs='+',help='IDs')
    parser.add_argument('--output', type=str, default='data/intermediate/project-details-dataset/pdd_context_retrieval.parquet', nargs='?',help='Input Folder')
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = _setup_args()
    main(args)
