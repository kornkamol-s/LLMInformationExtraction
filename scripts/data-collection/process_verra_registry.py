from selenium import webdriver
import time
import argparse
from bs4 import BeautifulSoup
import pandas as pd
import logging 
import requests
import os
from retry import retry
import re 
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

keys = [
        'State/Province',
        'Proponent',
        'VCS Project Status',
        'Estimated Annual Emission Reductions',
        'VCS Project Type',
        'VCS Methodology',
        'VCS Project Validator',
        'Project Registration Date',
        'Crediting Period Term'
    ]


@retry(tries=50, delay=2)
def _scrape_page_with_project_id(project_id, output_dir):
    url = f'https://registry.verra.org/app/projectDetail/VCS/{project_id}' 
    logging.info(f'URL: {url}')

    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--log-level=3')
    driver = webdriver.Chrome(options=options)
    driver.get(url)
    time.sleep(3)
    page_content = driver.page_source
    driver.quit()

    soup = BeautifulSoup(page_content, 'html.parser')

    result = {}
    title_element = soup.find('div', class_='card-header bg-primary')
    description_element = soup.find('div', class_='card-text p-3')
    result['title'] = title_element.get_text(strip=True)
    result['description'] = description_element.get_text(strip=True)

    for key in keys:
        th_element = soup.find('th', string=key)
        if th_element:
            current_row = th_element.find_next('tr')
            
            elements = []
            
            while current_row and 'attr-row' in current_row.get('class', []):
                span_elements = current_row.find_all('span', class_='text-break')
                elements.extend([span.get_text(strip=True) for span in span_elements])
                current_row = current_row.find_next_sibling('tr')
            
            result[key] = '\n'.join(elements) if elements else None
        else:
            result[key] = None

    result = _filtered_documents(soup, result, output_dir, project_id)

    return result


def _filtered_documents(soup, result, output_dir, project_id):
    pipeline = ['VCS Issuance Documents', 'VCS Registration Documents', 'VCS Pipeline Documents']

    for step in pipeline:    
        document_group_div = soup.find('div', class_='card-header', string=step)
        document_body = document_group_div.find_next('div', class_='card-body')
        document_links = document_body.find_all('a', href=True)
        document_dates = document_body.find_all('td', class_='pr-3 text-right')

        if document_links:
            filenames_raw = [link.text.strip() for link in document_links]
            dates = [date.text.strip() for date in document_dates]
            urls = [link['href'] for link in document_links]
            
            pattern = re.compile(r'(?<![a-zA-Z])(pd|pdd|proj(?:ect)?(?:[^a-zA-Z]*)(desc(?:ription)?))(?![a-zA-Z])', re.IGNORECASE)

            zipped_documents = list(zip(filenames_raw, dates, urls))
            selected_document = sorted(
                [(fn, date, link) for fn, date, link in zipped_documents if pattern.search(fn)],
                key=lambda x: datetime.strptime(x[1], '%d/%m/%Y'),
                reverse=True)
            
            if selected_document:
                url = selected_document[0][2]
                filename_raw = selected_document[0][0]
                filename =  f"{project_id}_{filename_raw.lower().replace(' ', '_').replace('?', '')}"
                updated_date = datetime.strptime(selected_document[0][1], '%d/%m/%Y').strftime('%d/%m/%Y')
                result['link'], result['filename_raw'], result['filename'], result['updated_date'] = url, filename_raw, filename, updated_date
                result = _download_files(result, output_dir)
                
                return result
    
    result['link'], result['filename_raw'], result['filename'], result['updated_date'], result['file_size'] = None, None, None, None, None
    logging.info('No file found.')

    return result


def _download_files(detail, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    response = requests.get(detail['link'], stream=True, timeout=60)
    file_size = 0
    if response.status_code == 200:
        with open(f'{output_dir}/{detail['filename']}', 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
                file_size += len(chunk)

        logging.info(f'Download {detail['filename']} Successfully')
    detail['file_size'] = file_size

    return detail


def _setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Text file containing IDs', nargs='?')
    parser.add_argument('--ids', type=int, nargs='+', help='List of project IDs')
    parser.add_argument('--dirs', type=str, default='downloaded_files', help='Output Directory for downloaded files')
    parser.add_argument('--output', type=str, default='projects_information', help='Output detail file')
    args = parser.parse_args()

    return args


def main(args):
    output_file = f'{args.output}.csv'

    if args.input:
        project_ids = []
        with open(f"{args.input}.txt", 'r') as file:
            for line in file:
                project_ids.append(int(line.strip()))

    else:
        project_ids = args.ids

    if os.path.exists(output_file):
        existing_ids = pd.read_csv(output_file)['id'].unique().tolist()
        project_ids = list(set(project_ids) - set(existing_ids))

    df = pd.DataFrame()
    i = 1
    for project_id in project_ids:
        logging.info(f'=========================================================')
        logging.info(f'Processing [{i}/{len(project_ids)}] ProjectID: {project_id}')
        logging.info(f'=========================================================')
        page_summary = _scrape_page_with_project_id(project_id, args.dirs)
        page_summary['id'] = project_id
        df = pd.DataFrame([page_summary])
        df.to_csv(output_file, mode='a', header=not os.path.exists(output_file), index=False, encoding='utf-8')
        i+=1



if __name__ == "__main__":
    args = _setup_args()
    main(args)
