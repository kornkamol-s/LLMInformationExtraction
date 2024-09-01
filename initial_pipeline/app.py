import os
import sys
import getopt
import re
import hashlib
import json
import logging
import time
from pathlib import Path

from dotenv import load_dotenv
import pdfplumber
from openai import OpenAI
import redis

from section import other_entities, project_proponents, ghg_emission_reductions


search_headers = {
    # "project_proponents": [
    #     "project proponent",
    #     "projectpropoonent",
    #     "promotorul de proiect",
    #     "proponente del proyecto",
    # ],
    # "other_entities": [
    #     "other entities",
    #     "roles and responsibilities, including contact information",
    #     "other project participant",
    #     "alte entități implicate în proiect",
    # ],
    "ghg_emission_reductions": [
        "estimated ghg emission reductions",
        "reduceri și eliminări nete de emisii de ges",
    ],
}


def main(argv):
    now = time.time()
    Path("runs/logs").mkdir(parents=True, exist_ok=True)
    Path(f"runs/outputs/{str(now)}").mkdir(parents=True, exist_ok=True)

    log_file_name = "runs/logs/" + str(now) + "-extract.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.FileHandler(log_file_name), logging.StreamHandler()],
    )
    logging.info("Starting the extraction process")
    load_dotenv()

    folder_location, no_cache = setup_command(argv)
    pdf_files = get_pdf_files_from(folder_location)
    openaiClient = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    redisClient = redis.Redis(
        host=os.environ.get("REDIS_HOST"),
        port=os.environ.get("REDIS_PORT"),
        password=os.environ.get("REDIS_PASSWORD"),
    )
    logging.info(f"Folder location: {folder_location}")
    logging.info(f"Total number of PDF files to be processed: {len(pdf_files)}")

    for index, file in enumerate(pdf_files, start=1):
        logging.info("==========================================================")
        logging.info(f"Processing file {index} of {len(pdf_files)}")
        logging.info(f"Filename: {file}")
        file_path = os.path.join(folder_location, file)
        file_hash = hash_pdf_file(file_path)

        if no_cache is False:
            cache = redisClient.get(file_hash)
            if cache:
                data = json.loads(cache)
                with open(
                    f"runs/outputs/{str(now)}/{file}.json", "w", encoding="utf-8"
                ) as f:
                    json.dump(data, f, ensure_ascii=False, indent=4)
                logging.info("Has already been processed before, used cache")
                continue

        pdf = pdfplumber.open(file_path)
        toc_list = find_toc_list_for(
            pdf,
        )
        spans = get_section_span_for_keys(toc_list, search_headers)
        file_output = {
            "file_name": file,
        }
        for key, span in spans.items():
            logging.info("----------------------------------------------------------")
            logging.info(f"Section: {key}")

            file_output[key] = None
            if span is not None:
                start = span.get("start")
                end = span.get("end")
                header = span.get("found_header")
                next_header = span.get("next_header")
                logging.info(f"Header: {header}")
                logging.info(f"Start: {start + 1} - End: {end + 1}")

                text = extract_text_for_section(
                    pdf, key, start, end, header, next_header
                )
                data = extract_section(openaiClient, key, text, file)
                if data is not None:
                    file_output[key] = data
            else:
                logging.info("Not found")

        if no_cache is False:
            redisClient.set(file_hash, json.dumps(file_output))
        with open(f"runs/outputs/{str(now)}/{file}.json", "w", encoding="utf-8") as f:
            json.dump(file_output, f, ensure_ascii=False, indent=4)
        pdf.close()

    merge_output_files(
        f"runs/outputs/{str(now)}", f"runs/outputs/{str(now)}/Master.json"
    )

    logging.info("Finished the extraction process")


def merge_output_files(directory, output_filename):
    """
    Merge all JSON files in the output directory into a single JSON file.

    :param directory: Directory containing JSON files to merge.
    :param output_filename: Filename for the output merged JSON file.
    """
    merged_data = {}

    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            filepath = os.path.join(directory, filename)

            with open(filepath, "r") as file:
                data = json.load(file)
                key = os.path.splitext(filename)[0]
                merged_data[key] = data

    with open(output_filename, "w") as output_file:
        json.dump(merged_data, output_file, ensure_ascii=False, indent=4)


def extract_section(openaiClient, section_key, text, file):
    """
    Extract the data for a given section key.

    :param openaiClient: The OpenAI key to forward to the handler functions.
    :param section_key: The key of the section.
    :param text: Relevant text to give Open AI more context.
    :param file: Filename to give Open AI more context.
    """
    if section_key == "project_proponents":
        return project_proponents.extract_data(openaiClient, text, file)
    if section_key == "other_entities":
        return other_entities.extract_data(openaiClient, text, file)
    if section_key == "ghg_emission_reductions":
        return ghg_emission_reductions.extract_data(openaiClient, text, file)


def extract_text_for_section(pdf, key, start_page, end_page, header, next_header):
    """
    Extract text for a given section key.
    Extract the text from the start page to the end page but also filter out text that is not between the header and next_header.
    For ghg emission reductions, extract all tables from the pages, keeping the context succinct
    Since there's text in between the tables that usually makes Open AI failed its job.

    :param pdf: The pdf reader instance.
    :param key: The key of the section.
    :param start_page: The start page of the section.
    :param end_page: The end page of the section.
    :param header: Filename to give Open AI more context.
    :param next_header: Filename to give Open AI more context.
    """
    text = ""

    if key == "ghg_emission_reductions":
        for i in range(start_page, end_page + 1):
            page = pdf.pages[i]
            tables = page.extract_tables()
            for table in tables:
                for row in table:
                    for cell in row:
                        if cell:
                            text += cell + "<=>"
                    text += "\n"
                text += "\n"

            page.flush_cache()

    if key != "ghg_emission_reductions":
        if start_page == end_page:
            page = pdf.pages[start_page]
            page_text = page.extract_text()
            text = page_text.split(header)[1].split(next_header)[0]

            page.flush_cache()
            return text

        for i in range(start_page, end_page + 1):
            page = pdf.pages[i]

            page_text = page.extract_text()
            if i == start_page:
                text += page_text.split(header)[1]
            elif i == end_page:
                text += page_text.split(next_header)[0]
            else:
                text += page_text

            page.flush_cache()
    return text


def get_section_span_for_keys(toc_list, search_headers):
    """
    Search for the section in the table of contents and return the start and end page of the section.
    Also returns the search header and the next header.

    :param toc_list: The table of content.
    :param search_headers: The headers we want to search for
    """
    spans = {}

    for search_key, search_values in search_headers.items():
        spans[search_key] = None
        for variant in search_values:
            span = find_section_span(toc_list, variant)
            start, end, found_header, next_header = span
            if start and end:
                spans[search_key] = {
                    "start": start,
                    "end": end,
                    "found_header": found_header,
                    "next_header": next_header,
                }
                break
    return spans


def find_section_span(toc_list, search_text):
    """
    Search for the section in the table of contents and return the start and end page of the section.
    Also returns the search header and the next header.
    There can be up to 2 matches, we should go with the second one
    Since the first one can be the real TOC in the PDF

    :param toc_list: The table of content.
    :param search_text: The header we want to search for
    """
    compressed_search_text = search_text.lower().translate(str.maketrans("", "", "- "))
    start = None
    end = None
    found_header = None
    next_header = None
    for i, (header, page) in enumerate(toc_list):
        compressed_header = header.lower().translate(str.maketrans("", "", "- "))
        # if search_text == "project proponent":
        #     logging.info(compressed_header)
        #     logging.info(compressed_search_text)
        if compressed_search_text in compressed_header:
            next_header_text, next_header_page = toc_list[i + 1]
            start = page
            end = next_header_page
            found_header = header
            next_header = next_header_text
    return (start, end, found_header, next_header)


# The format of the headers is usually a floating number (e.g 1.2) followed by a text with 1 or more spaces in between
# The prefix number can also optionally have 3 numbers (e.g 1.2.3), and it can't start with 0
# It can't contains ......... as it would be the main toc in that case, we don't want that
def find_toc_list_for(pdf):
    """
    Search for the table of content of the PDF and return the headers with the page number.

    The format of the headers is usually a floating number (e.g 1.2) followed by a text with 1 or more spaces in between
    The prefix number can also optionally have 3 numbers (e.g 1.2.3), and it can't start with 0
    It can't contains ......... as it would be the main toc in that case, we don't want that

    :param pdf: The PDF reader instance.
    """
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
        ]

        headers_with_page = map(lambda header: (header, i), filtered_headers)
        toc += headers_with_page

        page.flush_cache()

    return toc


def hash_pdf_file(file_path):
    hashing = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            buf = f.read(8192)
            if not buf:
                break
            hashing.update(buf)
    return hashing.hexdigest()


def get_pdf_files_from(folder_location):
    if not os.path.isdir(folder_location):
        sys.exit("The provided location does not exist")

    return list(
        filter(
            lambda file: file.endswith(".pdf") or file.endswith(".PDF"),
            os.listdir(folder_location),
        ),
    )


def setup_command(argv):
    arg_input = ""
    arg_nocache = False
    arg_help = (
        "{0} -i <location of the folder contains the PDF files> --no-cache".format(
            argv[0]
        )
    )

    try:
        opts, args = getopt.getopt(argv[1:], "hni:", ["help", "no-cache", "input="])
    except:
        logging.info(arg_help)
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            logging.info(arg_help)  # logging.info the help message
            sys.exit(2)
        elif opt in ("-i", "--input"):
            arg_input = arg
        elif opt in ("-nc", "--no-cache"):
            arg_nocache = True

    if arg_input == "":
        sys.exit("Please provide the location of the folder contains the PDF files")

    return (arg_input, arg_nocache)


if __name__ == "__main__":
    main(sys.argv)
