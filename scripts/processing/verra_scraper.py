import time, argparse, logging, requests, os, re 
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from retry import retry
from datetime import datetime

# Set up logging configuration with timestamps
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Keys to look for in the scraped HTML table
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


def main(args):
    """
    Main function to scrape data for a list of project IDs.
    """
    # Load project IDs from a text file
    if args.input:
        project_ids = []
        with open(f"{args.input}.txt", 'r') as file:
            for line in file:
                project_ids.append(int(line.strip()))

    # Otherwise, process from list of IDs via ids argument
    else:
        project_ids = args.ids

    # Remove already scraped IDs from the list if the output file exists
    if os.path.exists(args.output):
        existing_ids = pd.read_csv(args.output)['id'].unique().tolist()
        project_ids = list(set(project_ids) - set(existing_ids))

    # Iterately process each ID
    for i, project_id in enumerate(project_ids):
        logging.info(f'=========================================================')
        logging.info(f'Processing [{i+1}/{len(project_ids)}] ProjectID: {project_id}')
        logging.info(f'=========================================================')

        # Scrape information on Verra Page on specific project
        page_summary = _scrape_page_with_project_id(project_id, args.dirs)

        # Collect ID into dict result
        page_summary['id'] = project_id

        # Convert to dataframe for easier processing
        df = pd.DataFrame([page_summary])

        # Append data to the output CSV file
        df.to_csv(args.output, mode='a', header=not os.path.exists(args.output), index=False, encoding='utf-8')


@retry(tries=50, delay=2)
def _scrape_page_with_project_id(project_id, output_dir):
    """
    Scrapes the page content for a given project ID.

    Parameters:
        project_id (int): The unique project identifier
        output_dir (str): Directory to save downloaded project documents.

    Returns:
        dict: Scraped information.
    """
    # Verra's URL to a specific project
    url = f'https://registry.verra.org/app/projectDetail/VCS/{project_id}' 
    logging.info(f'URL: {url}')

    # Initialize Chrome WebDriver in headless mode
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')

    # Suppress WebDriver logs
    options.add_argument('--log-level=3')
    driver = webdriver.Chrome(options=options)

    # Open the URL in the WebDriver
    driver.get(url)

    # Pause to ensure page content loads fully
    time.sleep(3)

    # Extract the page's content
    page_content = driver.page_source

    # Close the WebDriver to release resources
    driver.quit()

    # Parse the page content using BeautifulSoup
    soup = BeautifulSoup(page_content, 'html.parser')
    result = {}

    # Extract the project title and description, if available
    title_element = soup.find('div', class_='card-header bg-primary')
    description_element = soup.find('div', class_='card-text p-3')
    result['title'] = title_element.get_text(strip=True)
    result['description'] = description_element.get_text(strip=True)

    # Loop over predefined keys to find and store the corresponding data
    for key in keys:
        th_elements = soup.find_all('th', string=key)

        # Find table headers matching the key
        if th_elements:
            for i, th_element in enumerate(th_elements):  

                # Start at the row containing the key
                current_row = th_element.find_next('tr')
                elements = []
                
                # Collect all rows related to this key
                while current_row and 'attr-row' in current_row.get('class', []):
                    span_elements = current_row.find_all('span')
                    elements.extend([span.get_text(strip=True) for span in span_elements])
                    current_row = current_row.find_next_sibling('tr')

                # Apart from these keys, collect the value without specific to either VCS or CCB
                if key not in ('Proponent', 'Estimated Annual Emission Reductions', 'Acres/Hectares'):    
                    result[key] = '\n'.join(elements) if elements else None
                
                # For these three keys, there will be two set of information under VCS and CCB.
                # This step will collect information two times for each key
                else:
                    if i == 0:
                        result[f"VCS {key}"] = '\n'.join(elements) if elements else None
                        if len(th_elements) == 1:
                            result[f"CCB {key}"] = None
                    if i == 1:
                        result[f"CCB {key}"] = '\n'.join(elements) if elements else None
        
        # if these keys are not exist, fill with None to remain consistent
        else:
            if key in ('Proponent', 'Estimated Annual Emission Reductions', 'Acres/Hectares'):    
                result[f"VCS {key}"] = None
                result[f"CCB {key}"] = None
            else:
                result[key] = None

    # Filter the most relevant document and download to local directory
    result = _filtered_documents(soup, result, output_dir, project_id)

    return result


def _filtered_documents(soup, result, output_dir, project_id):
    """
    Filters and selects the most recent document matching certain criteria for a project.

    Parameters:
        soup (BeautifulSoup): Parsed HTML content of the project page.
        result (dict): Dictionary holding scraped project data.
        output_dir (str): Directory for saving documents.
        project_id (int): The project identifier.

    Returns:
        dict: Updated dictionary with document details.
    """

    # Define the verra's stages where the documents can be uploaded to
    pipeline = ['VCS Issuance Documents', 'VCS Registration Documents', 'VCS Pipeline Documents']

    # Iterate to each project's stage to search for a document, starting with the most recent stage
    for step in pipeline:    
        document_group_div = soup.find('div', class_='card-header', string=step)
        document_body = document_group_div.find_next('div', class_='card-body')
        document_links = document_body.find_all('a', href=True)
        document_dates = document_body.find_all('td', class_='pr-3 text-right')

        # If document exists, it will be downloaded to local dir
        if document_links:

            # Original filename shown on the Verra page
            filenames_raw = [link.text.strip() for link in document_links]

            # The date that document was uploaed
            dates = [date.text.strip() for date in document_dates]

            # The url to download that document
            urls = [link['href'] for link in document_links]
            
            # Pattern to search for the project document (PDD), by searching pd, pdd, pro(ject), or des(cription) in the filename
            pattern = re.compile(r'(?<![a-zA-Z])(pd|pdd|proj(?:ect)?(?:[^a-zA-Z]*)(desc(?:ription)?))(?![a-zA-Z])', re.IGNORECASE)

            # Zip the filename, date, and url altogether
            zipped_documents = list(zip(filenames_raw, dates, urls))

            # Select the documents that match with the pattern, and sort by document's date, to get the most recent document
            selected_document = sorted(
                [(fn, date, link) for fn, date, link in zipped_documents if pattern.search(fn)],
                key=lambda x: datetime.strptime(x[1], '%d/%m/%Y'),
                reverse=True)
            
            if selected_document:
                url = selected_document[0][2]
                filename_raw = selected_document[0][0]
                filename =  f"{project_id}_{filename_raw.lower().replace(' ', '_').replace('?', '')}"
                updated_date = datetime.strptime(selected_document[0][1], '%d/%m/%Y').strftime('%d/%m/%Y')

                # Append information related to the document in dictionary
                result['link'], result['filename_raw'], result['filename'], result['updated_date'] = url, filename_raw, filename, updated_date

                # Download the most recent and relevant project document to local dir
                result = _download_files(result, output_dir)
                
                return result
    
    # If no document found, fill with None to remain consistent
    result['link'], result['filename_raw'], result['filename'], result['updated_date'], result['file_size'] = None, None, None, None, None
    logging.info('No file found.')

    return result


def _download_files(detail, output_dir):
    """
    Downloads a file from a specified URL and saves it to the local directory.

    Parameters:
        detail (dict): Dictionary containing file download details.
        output_dir (str): Directory to save downloaded files.

    Returns:
        dict: Updated dictionary with downloaded file size.
    """
    # Create output directory, if not exists
    os.makedirs(output_dir, exist_ok=True)

    # Send HTTP GET request to download the file from the URL
    # Using stream to download in chunks, which is memory efficient for large files
    response = requests.get(detail['link'], stream=True, timeout=60)
    file_size = 0

    if response.status_code == 200:

        # Open a file in write mode in the specified output directory
        with open(f'{output_dir}/{detail['filename']}', 'wb') as file:

            # Iterate over the content in chunks of 8 KB to prevents loading 
            # the entire file into memory at once
            for chunk in response.iter_content(chunk_size=8192):

                # Write each chunk to the file
                file.write(chunk)
                file_size += len(chunk)

        logging.info(f'Download {detail['filename']} Successfully')

    # Update the dictionary with the total file size
    detail['file_size'] = file_size

    return detail


def _setup_args():
    """
    Set up command-line arguments.

    Returns:
        argparse: The parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Text file containing IDs', nargs='?')
    parser.add_argument('--ids', type=int, nargs='+', help='Project IDs')
    parser.add_argument('--dirs', type=str, default='data/training/data_collection/pdds', help='Local filepath to store PDDs')
    parser.add_argument('--output', type=str, default='data/training/data_collection/verra_data.csv', help='Output filename for keeping scraped information')
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    # Set up command-line arguments
    args = _setup_args()

    # Execute the main function with the parsed arguments
    main(args)
