import os
import sys
import re
import json
import pdfplumber
import pandas as pd
import logging 
from openai import OpenAI
import argparse

# sk-proj-SSg70hkEZq1A0Q11OxvhT3BlbkFJKDHiJu2KJ7K2m6OdHtuR
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

folder_location = 'downloaded_files/'
output_file = 'heading_mapping_1.csv'

with open('config/information-description.json', 'r') as file:
    description = json.load(file)


def main(args):
    details = {}
    openai = OpenAI(api_key=args.credential)
    pdf_files = get_pdf_files_from(folder_location)
    processed_files = []
    if os.path.exists(output_file):
        processed_files = pd.read_csv(output_file)['filename'].unique()

    pdf_files = list(set(pdf_files) - set(processed_files))
    
    for index, file in enumerate(pdf_files, start=1):
        logging.info(f'=========================================================')
        logging.info(f'Processing [{index}/{len(pdf_files)}] : {file}')
        logging.info(f'=========================================================')

        file_path = os.path.join(folder_location, file)
        pdf = pdfplumber.open(file_path)
        toc_list = find_toc_list_for(pdf)
        mappings = extract_content_by_section(toc_list, pdf, openai)
        details['id'] = file.split('_')[0]
        details['filename'] = file
        details['mappings'] = mappings

        df = flatten_records(details)
        df.to_csv(output_file, mode='a', header=not os.path.exists(output_file), index=False)


def flatten_records(details):
    rows = []
    for heading, category in details['mappings'].items():
        row = {
            'id': details['id'],
            'filename': details['filename'],
            'heading': heading,
            'category': category
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    return df


def mapping_section(headings, openai):
    prompt = f"""Use the below section headings to identify whether which heading relates to which category. Choose only one category that is most relevant to that heading."

        Section headings:
        \"\"\"
        {headings}
        \"\"\"

        Categories:
        \"\"\"
        {description}
        \"\"\"

        return the answer with json format : dict("heading 1":"matched category 1", "heading 2" : "matched category 2", "heading 3": "", ...).  If a heading does not relate to any category, return an empty string for that heading.
    """

    response = openai.chat.completions.create(
        messages=[{"role": "user", "content": prompt},],
        model="gpt-3.5-turbo",
        temperature=0,
    )
    
    response = response.choices[0].message.content

    if response:
        if isinstance(response, dict):
            return response
        
        else:
            catgs = json.loads(response.replace(r'\xad', ''))

            return catgs


def extract_content_by_section(toc_list, pdf, openai):
    headings = [toc[0] for toc in toc_list]
    response = mapping_section(headings, openai)
    print(response)

    return response
        

def find_toc_list_for(pdf):
    toc = []
    floating_number_pattern = r"[1-9]+\.\d+(\.)?(\.\d+)?\s+[A-Z]+"
    has_alphabet_pattern = r"[a-zA-Z]"

    for i, page in enumerate(pdf.pages):
        text = page.extract_text().split("\n")
        filtered_headers = [
            text
            for text in text
            if re.match(floating_number_pattern, text)
            and re.search(has_alphabet_pattern, text)
            and "........." not in text
            and "_________" not in text
        ]

        headers_with_page = map(lambda header: (header, i), filtered_headers)
        toc += headers_with_page

        page.flush_cache()

    return toc


def get_pdf_files_from(folder_location):
    if not os.path.isdir(folder_location):
        sys.exit("The provided location does not exist")

    return list(
        filter(
            lambda file: file.endswith(".pdf") or file.endswith(".PDF"),
            os.listdir(folder_location),
        ),
    )


def _setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('credential', type=str, help='OpenAI Credential')
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = _setup_args()
    main(args)
