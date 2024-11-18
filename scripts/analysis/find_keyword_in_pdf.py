import os, logging, argparse
import pandas as pd
from tools.utils import find_pdf_files, get_filtered_file
from tools.PDFExtraction import PDFExtraction

# Set up logging configuration with timestamps
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

kws = [
        {'Project Proponent': ['project proponent']},
        {'GHG Emission Reductions': ['ghg emission reduction']},
        {'Methodology': ['methodology', 'methodologies']},
        {'Project Description': ['location']},
        {'Credit Period': ['crediting period']},
        {'Sector': ['sectoral scope', 'project type']}]


def main(args):
    """
    Main function to extract keyword occurrences from PDF files.
    """

    # If ids are provided, process only those given ids.
    # Otherwise, process entire folder
    # Also, filter out all the processed ids
    pdf_files = get_filtered_file(find_pdf_files(args.input), args.ids, args.output)
    output = f"data/training/data_analysis/{args.output}.csv"

    # Prepare the list of keywords for pattern matching
    keys = []
    for kw in kws:
        for _, keywords in kw.items():
            keys.append('|'.join(keywords))

    # Process each PDF file in the list
    for index, file in enumerate(pdf_files, start=1):
        logging.info(f'=========================================================')
        logging.info(f'Processing [{index}/{len(pdf_files)}] : {file}')
        logging.info(f'=========================================================')

        # Initialize PDF extraction tool
        pdf_extractor = PDFExtraction(f"{args.input}/{file}")
        results = pdf_extractor._search_keywords(keys)
        
        # Map keywords to their categories
        key_name = [list(d.keys())[0] for d in kws]
        final_results = {key: results[pattern] for key, pattern in zip(key_name, results.keys())}

        # Flatten results into a list
        flattened_data = [(key, value) for key, values in final_results.items() for value in values]

        # Convert results into a DataFrame
        df = pd.DataFrame(flattened_data, columns=['Category', 'Page Number'])
        df['id'] = file.split('_')[0]

        # Append data to output CSV
        df.to_csv(output, mode='a', header=not os.path.exists(output), index=False, encoding='utf-8')


def _setup_args():
    """
    Set up command-line arguments.

    Returns:
        argparse: The parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, default='data/training/data_collection/pdds/', nargs='?', help='Input Folder')
    parser.add_argument('--ids', type=int, nargs='+',help='Project IDs')
    parser.add_argument('--output', type=str, default='keywords_found_in_pages', nargs='?',help='Output Folder')
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    # Set up command-line arguments
    args = _setup_args()

    # Execute the main function with the parsed arguments
    main(args)