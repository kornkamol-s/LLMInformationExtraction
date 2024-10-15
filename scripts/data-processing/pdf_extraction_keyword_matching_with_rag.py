import os
import re
import json
import pandas as pd
import logging 
import math
import argparse
from tools.PDFExtraction import PDFExtraction
from tools.utils import find_pdf_files
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from retry import retry

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


with open('config/heading_mapping.json', 'r') as f:
    headings_mapping = json.load(f)

with open('config/question_mapping.json', 'r') as f:
    question_mapping = json.load(f)

@retry(tries=50, delay=2)
def main(args):
    pdf_files = find_pdf_files(args.input)

    if args.ids:
        pdf_files = [f for f in pdf_files if int(f.split('_', 1)[0]) in args.ids]
 
    if os.path.exists(args.output):
        processed_files = pd.read_parquet(args.output, engine='fastparquet')['filename'].unique().tolist()
    else:
        processed_files = []

    pdf_files = list(set(pdf_files) - set(processed_files))
    
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") 

    for index, file in enumerate(pdf_files, start=1):
        logging.info(f'=========================================================')
        logging.info(f'Processing [{index}/{len(pdf_files)}] : {file}')
        logging.info(f'=========================================================')
        
        pdf_extractor = PDFExtraction(f"{args.input}/{file}")
        toc_df = pdf_extractor._get_toc()

        context_df = extract_relevant_section(pdf_extractor, toc_df, embedding, file)
        context_df['id'] = file.split('_', 1)[0]
        context_df['filename'] = file

        if os.path.exists(args.output):
            context_df.to_parquet(args.output, engine='fastparquet', index=False, append=True)
        else:
            context_df.to_parquet(args.output, engine='fastparquet', index=False)
            

def extract_relevant_section(pdf, toc, embedding, file):
    toc['next_section'] = toc['section'].shift(-1, fill_value='')
    rows = []
    for i, (section, variants) in enumerate(headings_mapping.items(), start=1):
        logging.info(f'>>> [{i}/{len(headings_mapping)}] Processing Section : {section}')
        logging.info(f'-----------------------------------------------------------------------')

        if not toc.empty:
            logging.info(f'ToC Found.')
            documents = []
            pattern = '|'.join([re.escape(variant) for variant in variants])
            matched_df = toc[toc['section'].str.contains(pattern, case=False, na=False)]
            b_size, b_overlap = 10000, 50

            if not matched_df.empty:
                logging.info(f'Matched Headings Found.')
                for i, row in matched_df.iterrows():
                    context = pdf._extract_page_range(row['start_page'], row['end_page'], row['section'], row['next_section'])
                    doc = Document(page_content=context.encode('utf-8', 'replace').decode('utf-8'))
                    documents.append(doc)

            else:
                logging.info(f'Matched Headings Not Found.')
                for _, row in toc.iterrows():
                    context = pdf._extract_page_range(row['start_page'], row['end_page'], row['section'], row['next_section'])
                    doc = Document(page_content=context.encode('utf-8', 'replace').decode('utf-8'))
                    documents.append(doc)

        else:
            logging.info(f'ToC Not Found.')
            documents = PyPDFLoader(f'{args.input}/{file}').load()
            b_size, b_overlap = 5000, 100

        splitter = RecursiveCharacterTextSplitter(chunk_size=b_size, chunk_overlap=b_overlap)
        documents = splitter.split_documents(documents)          
        if len(documents) > 1:
            logging.info(f'{len(documents)} Docs Found.')
            vectorstore = Chroma.from_documents(documents, embedding,
                                                collection_metadata={"hnsw:space": "cosine"},
                                                persist_directory=f"log/vector-store-1/{file.split('_', 1)[0]}/{section}")
            
            # Reduce by half of total number of documents found
            k = math.ceil(len(documents)/2)
            retriever = vectorstore.as_retriever(search_kwargs={"k":k})
            compressor = DocumentCompressorPipeline(transformers=[EmbeddingsRedundantFilter(embeddings=embedding), 
                                                                EmbeddingsFilter(embeddings=embedding, k=1)])
            compressor_retriever = ContextualCompressionRetriever(base_retriever=retriever, base_compressor=compressor)
            documents = compressor_retriever.invoke(question_mapping[section])

        context = '\n'.join([doc.page_content for doc in documents])    
        rows.append({'section_category': section, 'context': context})

    df = pd.DataFrame(rows)
    return df 

    
def find_pdf_files(folder_location):    
    if not os.path.isdir(folder_location):
        raise FileNotFoundError("The provided location does not exist")

    pdf_files = [
        file for file in os.listdir(folder_location)
        if file.endswith('.pdf')
    ]
    return pdf_files


def _setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, default='data/input/pdds', nargs='?', help='Input Folder')
    parser.add_argument('--ids', type=int, nargs='+',help='IDs')
    parser.add_argument('--output', type=str, default='data/intermediate/kita_dataset/refined_pdd_context_retrieval.parquet', nargs='?',help='Input Folder')

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = _setup_args()
    main(args)
