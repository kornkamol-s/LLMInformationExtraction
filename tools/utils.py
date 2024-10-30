import os
import pandas as pd


def find_pdf_files(folder_location):    
    """
    Get list of PDF files in the provided folder location.

    Parameters:
    - folder_location (str): The local path to the folder to be searched.

    Returns:
    - list: A list of PDF files in the folder.
    """
    if not os.path.isdir(folder_location):
        raise FileNotFoundError("The provided location does not exist")

    # Only search for PDF format
    pdf_files = [
        file for file in os.listdir(folder_location)
        if file.endswith('.pdf')
    ]
    return pdf_files


def get_filtered_file(pdf_files, ids, output):
    """
    Filters the list of PDF files and excludes already processed files.

    Parameters:
    - pdf_files (list): List of PDF files to be filtered.
    - ids (list or None): If provided, filter files that match these IDs.
    - output (str): The path to a CSV file that stores the list of processed files.

    Returns:
    - list: A filtered list of PDF files.
    """
    if ids:
        # If ids are provided, filter files where the ids containing in the filenames
        pdf_files = [f for f in pdf_files if int(f.split('_', 1)[0]) in ids]

    else:    
        # If no ids are provided, check if the output CSV file exists
        if os.path.exists(output):
            # Get all processed ids
            processed_files = pd.read_csv(output)['filename'].unique().tolist()
        else:
            processed_files = []

        # Filter only files have not been processed
        pdf_files = list(set(pdf_files) - set(processed_files))

    return pdf_files