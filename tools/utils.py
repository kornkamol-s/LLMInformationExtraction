import os
import pandas as pd



def find_pdf_files(folder_location):    
    if not os.path.isdir(folder_location):
        raise FileNotFoundError("The provided location does not exist")

    pdf_files = [
        file for file in os.listdir(folder_location)
        if file.endswith('.pdf')
    ]
    return pdf_files


def get_filtered_file(pdf_files, ids, output):
    if ids:
        pdf_files = [f for f in pdf_files if int(f.split('_', 1)[0]) in ids]

    else:    
        if os.path.exists(output):
            processed_files = pd.read_csv(output)['filename'].unique().tolist()
        else:
            processed_files = []

        pdf_files = list(set(pdf_files) - set(processed_files))

    return pdf_files