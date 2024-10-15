import pandas as pd
import os
from multiprocessing import Pool, cpu_count
from tools.PDFExtraction import PDFExtraction
from tools.utils import find_pdf_files

# def process_pdf(file_info):
#     index, file = file_info
#     print(f'=========================================================')
#     print(f'Processing [{index + 1}/{total_files}] : {file}')
#     print(f'=========================================================')

#     pdf_extractor = PDFExtraction(f"data/input/pdds/{file}")
#     toc_df = pdf_extractor._get_toc()
#     toc_df['id'] = file.split('_', 1)[0]
#     toc_df['filename'] = file

#     # Define output filename based on the index
#     output_filename = f'table_of_contents_part_{index // files_per_output}.csv'

#     # Append to the corresponding CSV file for each page
#     toc_df.to_csv(output_filename, mode='a', header=not os.path.exists(output_filename), index=False)

# # Find PDF files in the specified directory
# pdf_files = find_pdf_files('data/input/pdds')

# # Check for already processed files
# if os.path.exists('table_of_contents.csv'):
#     processed_files = pd.read_csv('table_of_contents.csv')['filename'].unique().tolist()
#     pdf_files = list(set(pdf_files) - set(processed_files))

# # Prepare data for multiprocessing
# total_files = len(pdf_files)
# files_per_output = total_files // 10  # Number of files to process per output file
# file_info_list = list(enumerate(pdf_files))  # Create a list of (index, file) tuples

# # Use multiprocessing to process files
# if __name__ == '__main__':
#     with Pool(processes=cpu_count()) as pool:
#         pool.map(process_pdf, file_info_list)





from tools.PDFExtraction import PDFExtraction

pdf_extractor = PDFExtraction(f"data/input/pdds/2623_202305_ccb_vcs_project_description_komaza_clean.pdf")
toc_df = pdf_extractor._get_toc()
print(toc_df)