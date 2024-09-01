import os
import logging
import argparse
import pandas as pd
from tools.utils import find_pdf_files, get_filtered_file
from tools.PDFExtraction import PDFExtraction

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

kws = [
        {'Project Proponent': ['proponent']},
        {'GHG Emission Reductions': ['emission reductions', 'emission reduction', 'emissions reductions', 'emissions reduction', 'ghg']},
        {'Methodology': ['methodology', 'methodologies']},
        {'Project Description': ['description', 'location', 'information', 'summary', 'summaries']},
        {'Credit Period': ['crediting period', 'credit period']},
        {'Sector': ['scope', 'sector', 'project type', 'type of project']},
    ]


def main(args):
    pdf_files = get_filtered_file(find_pdf_files(args.input) , args.ids, args.output)
    
    keys = []
    for item in kws:
        for _, keywords in item.items():
            keys.append('|'.join(keywords))

    for index, file in enumerate(pdf_files, start=1):
        logging.info(f'=========================================================')
        logging.info(f'Processing [{index}/{len(pdf_files)}] : {file}')
        logging.info(f'=========================================================')

        pdf_extractor = PDFExtraction(f"{args.input}/{file}")
        results = pdf_extractor._search_keywords(keys)
        
        key_name = [list(d.keys())[0] for d in kws]
        final_results = {key: results[pattern] for key, pattern in zip(key_name, results.keys())}
        flattened_data = [(key, value) for key, values in final_results.items() for value in values]
        df = pd.DataFrame(flattened_data, columns=['Category', 'Page Number'])
        df['id'] = file.split('_')[0]
        df.to_csv(args.output, mode='a', header=not os.path.exists(args.output), index=False, encoding='utf-8')


def _setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, default='data/downloaded_files', nargs='?', help='Input Folder')
    parser.add_argument('--ids', type=int, nargs='+',help='IDs')
    parser.add_argument('--output', type=str, default='data/keywords_found_in_pages_4.csv', nargs='?',help='Output Folder')
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = _setup_args()
    main(args)