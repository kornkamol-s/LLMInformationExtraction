import os
import sys
import re
import json
import pdfplumber
# from sentence_transformers import SentenceTransformer, util
# from gpt4all import GPT4All
import pandas as pd
import logging 
from openai import OpenAI
import argparse
import ast
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# model = GPT4All("mistral-7b-openorca.gguf2.Q4_0.gguf", device="cpu")

folder_location = 'downloaded_files/'
output_file = 'section_mapping_1.csv'

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
    flattened_data = []

    for mapping in details['mappings']:
        for heading, values in mapping.items():
            mapped_categories = values['mapped_categories']
            for category in mapped_categories:
                flattened_entry = {
                    'id': details['id'],
                    'filename': details['filename'],
                    'heading': heading,
                    'mapped_categories': category,
                    'start_page': values['start_page'],
                    'end_page': values['end_page'],
                    'heading_name': values['heading_name'],
                    'heading_number': values['heading_number'],
                    'section': values['section']
                }
                flattened_data.append(flattened_entry)
    df = pd.DataFrame(flattened_data)

    return df


def chunk_text(content, max_tokens=16385):
    words = content.split()
    chunks = []
    current_chunk = []

    for word in words:
        current_chunk.append(word)
        if len(' '.join(current_chunk)) > max_tokens:
            chunks.append(' '.join(current_chunk[:-1]))
            current_chunk = [word]

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks


def mapping_section(heading, content, openai):
    content_chunks = chunk_text(content)
    matched_categories = set()
    for chunk in content_chunks:
    
        prompt = f"""Use the below content for section {heading} to find whether it mentions or relates to any categories."

            Article section:
            \"\"\"
            {chunk}
            \"\"\"

            Categories:
            \"\"\"
            {description}
            \"\"\"

            return the answer with format : ['matched category 1', 'matched category 2', ...],  If the content does not relate, return empty
            """
        
        response = openai.chat.completions.create(
            messages=[{"role": "user", "content": prompt},],
            model="gpt-3.5-turbo",
            temperature=0,
        )
        
        response = response.choices[0].message.content
        print(response)
        if response:
            try:
                catgs = ast.literal_eval(response)
            except:
                catgs = response.strip("[]").split(", ")

            matched_categories = matched_categories | set(catgs)

    return list(matched_categories)


    # with model.chat_session(''):
    #     get_answer(model, prompt)

# def get_answer(model, prompt):
#     output = model.generate(prompt, max_tokens=100)


def extract_content_by_section(toc_list, pdf, openai):
    text = ""
    responses = []
    toc_list = [('First Section', 1)] + toc_list
    for i, (header, start) in enumerate(toc_list):
        logging.info(f'Processing Section: {header}')

        match = re.match(r"(\d+)\.(\d+)\s+(.*)", header)
        if match:
            section = match.group(1)
            heading_number = f"{match.group(1)}.{match.group(2)}"
            heading_name = match.group(3)
        else:
            section = 0
            heading_number = 0
            heading_name = header

        if i == len(toc_list)-1:
            end = len(pdf.pages)-1
            next_header = ' '
        else:
            next_header, end = toc_list[i + 1]

        if start == end:
            page = pdf.pages[start]
            page_text = page.extract_text()

            if header in page_text:
                text = page_text.split(header)[1].split(next_header)[0]
            else:
                text = page_text.split(next_header)[0]
            page.flush_cache()
        else:
            for i in range(start, end + 1):
                page = pdf.pages[i]

                page_text = page.extract_text()
                if i == start:
                    if header in page_text:
                        text += page_text.split(header)[1]
                    else:
                        text += page_text
                elif i == end:
                    text += page_text.split(next_header)[0]
                else:
                    text += page_text

                page.flush_cache()
        time.sleep(3)
        response = mapping_section(header, text, openai)
        responses.append({header:{'mapped_categories':response, 
                                  'start_page':start, 
                                  'end_page':end, 
                                  'heading_name':heading_name,
                                  'heading_number':heading_number,
                                  'section':section}})
    return responses
        

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
