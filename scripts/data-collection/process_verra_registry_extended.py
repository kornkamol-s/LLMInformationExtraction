from selenium import webdriver
import time
import argparse
from bs4 import BeautifulSoup
import pandas as pd
import logging 
import os
from retry import retry

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

keys = [
        'State/Province',
        'Proponent',
        'VCS Project Status',
        'Estimated Annual Emission Reductions',
        'Total Buffer Pool Credits', 
        'VCS Project Type',
        'AFOLU Activity',
        'VCS Methodology',
        'Acres/Hectares',
        'VCS Project Validator',
        'Project Registration Date',
        'Crediting Period Term',
        'CCB Project Status',
        'CCB Project Type',
        'CCB Project Validator',
        'CCB Standard Edition',
        'Auditor Site Visit To and From Date',
    ]


@retry(tries=50, delay=2)
def _scrape_page_with_project_id(project_id):
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
        th_elements = soup.find_all('th', string=key)
        if th_elements:
            for i, th_element in enumerate(th_elements):  
                current_row = th_element.find_next('tr')
                elements = []
                
                while current_row and 'attr-row' in current_row.get('class', []):
                    span_elements = current_row.find_all('span')
                    elements.extend([span.get_text(strip=True) for span in span_elements])
                    current_row = current_row.find_next_sibling('tr')

                if key not in ('Proponent', 'Estimated Annual Emission Reductions', 'Acres/Hectares'):    
                    result[key] = '\n'.join(elements) if elements else None
                
                else:
                    if i == 0:
                        result[f"VCS {key}"] = '\n'.join(elements) if elements else None
                        if len(th_elements) == 1:
                            result[f"CCB {key}"] = None
                    if i == 1:
                        result[f"CCB {key}"] = '\n'.join(elements) if elements else None
        else:
            if key in ('Proponent', 'Estimated Annual Emission Reductions', 'Acres/Hectares'):    
                result[f"VCS {key}"] = None
                result[f"CCB {key}"] = None
            else:
                result[key] = None

    return result


def _setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Text file containing IDs', nargs='?')
    parser.add_argument('--ids', type=int, nargs='+', help='List of project IDs')
    parser.add_argument('--output', type=str, default='data/input/scraped_data_extended', help='Output detail file')
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
        page_summary = _scrape_page_with_project_id(project_id)
        page_summary['id'] = project_id
        df = pd.DataFrame([page_summary])
        df.to_csv(output_file, mode='a', header=not os.path.exists(output_file), index=False, encoding='utf-8')
        i+=1



if __name__ == "__main__":
    args = _setup_args()
    main(args)
