import os, re, json, logging , math, argparse
import pandas as pd
from tools.PDFExtraction import PDFExtraction
from tools.utils import find_pdf_files, get_filtered_file
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

# Set up logging configuration with timestamps
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@retry(tries=50, delay=2)
def main(args):
    """
    Main function to process PDF files for context extraction based on the table of contents.
    Extracted context is saved to a specified output file.
    """
    # If ids are provided, process only those given ids.
    # Otherwise, process entire folder
    # Also, filter out all the processed ids
    pdf_files = get_filtered_file(find_pdf_files(args.input), args.ids, args.output)

    # Initialize an embedding model using HuggingFace
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") 

    # Iterate over each file and process it
    for index, file in enumerate(pdf_files, start=1):
        logging.info(f'Processing Context Extraction [{index}/{len(pdf_files)}] : {file}')
        
        # Extract table of contents from each PDF file
        pdf_extractor = PDFExtraction(f"{args.input}/{file}")
        toc_df = pdf_extractor._get_toc()
        logging.info('Sucessfully Retrieve ToC')

        # Extract relevant sections
        context_df = _extract_relevant_section(pdf_extractor, toc_df, embedding, file)
        context_df['id'] = file.split('_', 1)[0]
        context_df['filename'] = file

        # Append extracted data to the output file
        context_df.to_csv(args.output,  mode='a', header=not os.path.exists(args.output), index=False, encoding='utf-8')
            

def _extract_relevant_section(pdf, toc, embedding, file):
    """
    Extracts relevant sections from a PDF based on table of contents and returns a DataFrame with context 
    for each type of information.

    Parameters:
        pdf (PDFExtraction): PDF extraction instance for retrieving content.
        toc (DataFrame): Table of contents DataFrame.
        embedding (HuggingFaceEmbeddings): Embedding model to generate vector embeddings.
        file (str): PDF filename to be processed.

    Returns:
        DataFrame: A DataFrame containing category of the section and coresponding extracted context.
    """
    # Load mappings for headings from JSON configuration files
    with open('config/heading_mapping.json', 'r') as f:
        headings_mapping = json.load(f)

    # Load mappings for questions from JSON configuration files
    with open('config/question_mapping.json', 'r') as f:
        question_mapping = json.load(f)
    
    # Set up a next section column to stop extract content when reaching next section
    toc['next_section'] = toc['section'].shift(-1, fill_value='')
    rows = []

    # Process each section specified in the headings mapping
    for i, (section, variants) in enumerate(headings_mapping.items(), start=1):

        if not toc.empty:
            documents = []

            # Pattern to match section headings
            pattern = '|'.join([re.escape(variant) for variant in variants])

            # Filter matched sections
            matched_df = toc[toc['section'].str.contains(pattern, case=False, na=False)]

            # Define chunking parameters
            b_size, b_overlap = 10000, 50

            # Extract text from matched or unmatched sections in the document
            # If there is at least one matched section found in the document,
            # Will extract the contents under all matched sections separately
            # If there is no matched section found in the document,
            # Will extract the contents under all headings found in the document
            if matched_df.empty:
                matched_df = toc

            # Iteratively process each section
            for i, row in matched_df.iterrows():

                # Extract the content of that section using start-end page 
                # and filter out those irrelevant content before and after the focusing section
                context = pdf._extract_page_range(row['start_page'], row['end_page'], row['section'], row['next_section'])

                # Convert to document datatype for chunking to desired size, avoiding token overflow in GPT model
                doc = Document(page_content=context.encode('utf-8', 'replace').decode('utf-8'))
                documents.append(doc)

        # If ToC is empty, load full content of PDF as documents
        else:
            documents = PyPDFLoader(f'{args.input}/{file}').load()
            b_size, b_overlap = 5000, 100

        # Split documents into manageable chunks based on defined size and overlap
        splitter = RecursiveCharacterTextSplitter(chunk_size=b_size, chunk_overlap=b_overlap)
        documents = splitter.split_documents(documents)          

        # If multiple documents are found, Create a vector store to extract the most relevant representation
        if len(documents) > 1:
            logging.info(f'{len(documents)} Docs Found in section {section}')

            # Create a vector store from the document embeddings for retrieval
            vectorstore = Chroma.from_documents(documents, embedding,
                                                # Set retrieval to use cosine similarity
                                                collection_metadata={"hnsw:space": "cosine"},
                                                # Directory to save the vector store per section/file ID
                                                persist_directory=f"log/vector-store/{file.split('_', 1)[0]}/{section}")
            
            # Define the number of top documents to retrieve, setting it to half the total documents to reduce redundancy
            k = math.ceil(len(documents)/2)

            # Create a retriever to find the top-k most relevant documents based on vector similarity
            retriever = vectorstore.as_retriever(search_kwargs={"k":k})
            
            # Set up a document compression pipeline to refine and compress retrieved documents further
            compressor = DocumentCompressorPipeline(transformers=[
                                                                # Filter out redundant or near-duplicate content
                                                                EmbeddingsRedundantFilter(embeddings=embedding), 
                                                                # Further compress by selecting the most relevant subset
                                                                EmbeddingsFilter(embeddings=embedding, k=1)])
            
            # Create a retriever that combines retrieval and compression, producing a refined list of relevant documents
            compressor_retriever = ContextualCompressionRetriever(
                                                                # Base retriever for initial selection
                                                                base_retriever=retriever, 
                                                                # Compressor to further refine retrieved content
                                                                base_compressor=compressor)
            
            # Retrieve and compress the documents by invoking the retriever with specific questions from the mapping
            documents = compressor_retriever.invoke(question_mapping[section])

        # Join the compressed document content to create the context text for the section
        context = '\n'.join([doc.page_content for doc in documents])    

        # Append a dictionary with the section category and extracted context to the rows list
        rows.append({'section_category': section, 'context': context})
    df = pd.DataFrame(rows)

    return df 


def _setup_args():
    """
    Set up command-line arguments.

    Returns:
        argparse: The parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, default='data/input/pdds', nargs='?', help='Input Folder')
    parser.add_argument('--ids', type=int, nargs='+',help='IDs')
    parser.add_argument('--output', type=str, default='data/intermediate/kita_dataset/refined_pdd_context_retrieval.csv', nargs='?',help='Input Folder')

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    # Set up command-line arguments
    args = _setup_args()

    # Execute the main function with the parsed arguments
    main(args)