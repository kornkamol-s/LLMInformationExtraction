from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.document_transformers import EmbeddingsRedundantFilter
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.retrievers.document_compressors import EmbeddingsFilter

import os
import logging
import argparse
from tools.utils import find_pdf_files, get_filtered_file


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def main(args):
    pdf_files = get_filtered_file(find_pdf_files(args.input) , args.ids, args.output)
    
    # Explore chunk size and overlap
    # splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=70)

    # Explore models
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") 
    
    for index, file in enumerate(pdf_files, start=1):
        logging.info(f'=========================================================')
        logging.info(f'Processing [{index}/{len(pdf_files)}] : {file}')
        logging.info(f'=========================================================')

        documents = PyPDFLoader(f'{args.input}/{file}').load()
        print(documents)
        # chunk = splitter.split_documents(documents)

        local_dir = f"/{args.output}/{file.split('_')[0]}"

        if not os.path.exists(local_dir) or not os.listdir(local_dir):

            # Explore on hnsw, and distance matrix
            vectorstore = Chroma.from_documents(documents, embeddings,
                                                collection_metadata={"hnsw:space": "cosine"},
                                                persist_directory=local_dir)
        else:
            vectorstore = Chroma(persist_directory=local_dir, embedding_function=embeddings)

        # Explore on different topn and add threshold on similarity score to not return all n
        retriever = vectorstore.as_retriever(search_kwargs={"k":args.k})
        docs = retriever.get_relevant_documents(args.query)

        for i, doc in enumerate(docs):
            document = doc
            page_number = document.metadata.get('page')
            content = document.page_content

            # logging.info('----------------------------------------')
            # logging.info(f'Base Retrieved Document: {i}, Page: {page_number}')
            # logging.info('----------------------------------------')
            # logging.info(content)

        compressor = DocumentCompressorPipeline(transformers=[EmbeddingsRedundantFilter(embeddings=embeddings), 
                                                              EmbeddingsFilter(embeddings=embeddings, k=5)])
        compressor_retriever = ContextualCompressionRetriever(base_retriever=retriever, base_compressor=compressor)
        filtered_docs = compressor_retriever.get_relevant_documents(query=args.query)
        for i, doc in enumerate(filtered_docs):
            document = doc
            page_number = document.metadata.get('page')
            content = document.page_content

            logging.info('----------------------------------------')
            logging.info(f'Compressed Retrieved Document: {i}, Page: {page_number}')
            logging.info('----------------------------------------')
            logging.info(content)


        # logging.info(f"\n\n{'-'* 100}\nDocument {i+1}\n{'-'* 100}\n\n{doc.page_content}")


def _setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('query', type=str, help='Query')
    parser.add_argument('--input', type=str, default='data/input/pdds', nargs='?', help='Input Folder')
    parser.add_argument('--ids', type=int, nargs='+',help='IDs')
    parser.add_argument('--output', type=str, default='vector-store-1', nargs='?',help='Output Folder')
    parser.add_argument('--k', type=int, default=10, nargs='?',help='Top k relevant documents')
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = _setup_args()
    main(args)
