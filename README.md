PDF Information Extraction using GPT-3.5

## Project Description
This repository is dedicated to the study "Leveraging Large Language Models for Advanced Information Extraction in the Carbon Insurance Domain." The goal of this project is to automate the extraction of critical data from Project Design Documents (PDDs) registered with Verra under the VCS, focusing primarily on carbon activity projects in the Forestry and Land Use or Renewable Energy sectors. 

By automating the extraction process, the project aims to enhance the efficiency and accuracy of risk assessments in the carbon insurance sector while significantly reducing the manual workload.


Key Stages of the Project: 
- Data Collection: Scraping and gathering relevant project data from Verra, including PDDs and associated project details.
- Preprocessing: Converting PDDs from PDFs to machine-readable text, extracting contextual information, and transforming it into structured formats for model training.
- Model Training: Fine-tuning the GPT-3.5 model to extract key details from PDDs, including:
        Project proponents
        Crediting periods
        Project location
        Methodology
        Sector
        Estimated GHG emission reductions
- Model Evaluation: Comprehensive evaluation of the model's performance using unseen datasets with reference answers.
- Information Extraction Pipeline: After fine-tuning, a robust pipeline is developed to use the model for efficiently extracting key information from unseen PDDs. 

---

## Project Structure
```
├── .env                                                # Stores environment variables like API keys.
├── requirements                                        # Project dependencies.
│   ├──requirements.txt                                 # Python dependencies for the project.
│   └── requirements_analysis.yml                       # Conda environment setup file for analysis.
  
├── config                                              # Configuration files.
│   ├── config.py                                       # Central configuration file.
│   ├── heading_mapping.json                            # Maps document headings to categories.
│   └── question_mapping.json                           # Contains questions for each category.
                        
├── data                                                # Data (raw and processed).
│   ├── inference                                       # Data for IE Pipeline.
│   │   ├── input                                       # Input PDDs for IE task.
│   │   ├── intermediate                                # Intermediate results.
│   │   │   └── context.csv                             # Contextual extraction results.
│   │   └── output                                      # Model extraction result files.
│   └── training                                        # Data for training and evaluation.
│       ├── data_analysis                               # Analysis-related files.
│       │   ├── table_of_contents.csv                   # PDD headings.
│       │   ├── keywords_found_in_pages.csv             # Keywords found in PDDs' page.
│       │   ├── result_heading_mapping_using_llm.csv    # Heading mapping results.
│       │   └── result_section_mapping_using_llm.csv    # Section mapping results.
│       ├── data_collection                             # Raw data for training and analysis.
│       │   ├── pdds                                    # Raw PDD files.
│       │   ├── verra_data.csv                          # Verra project information.
│       │   ├── multiple_proponents.csv                 # Information related to multiple project proponents.
│       │   ├── CDM methodologies.csv                   # Methodology metadata.
│       │   ├── CarbonMarkets_ghg.xlsx                  # GHG emission reduction value from Clearblue Markets
│       │   ├── AlliedOffsets_project_info.csv          # Project information from AlliedOffsets
│       │   └── AlliedOffsets_ghg.csv                   # GHG emission reductions from AlliedOffset.
│       ├── data_partitioning                           # Partitioned datasets (train/validate/test).
│       │   ├── test                                    # Test datasets.
│       │   ├── train                                   # Training datasets.
│       │   └── validate                                # Validation datasets.
│       ├── data_processing                             # Data preprocessing and refinement files.
│       │   ├── pdd_context_retrieval.csv               # Extracted PDD context.
│       │   ├── processed_ground_truth_project_info.csv # Processed project data.
│       │   └── manual_refined_ground_truth_ghg.csv     # Processed GHG emission reduction data.
│       └── result                                      # Stores the evaluation and prediction results from the model.
│           ├── ghg_emission_reduction                  # Results related to GHG emission reduction tasks.
│           │   ├── logs                                # Logs generated during the model training process.
│           │   ├── metrics                             # Evaluation metrics for GHG reduction extraction.
│           │   └── responses                           # Responses generated by the model for GHG reduction tasks.
│           ├── project_info                            # Results related to project information tasks.
│           │   ├── logs                                # Logs generated during the model training process.
│           │   ├── metrics                             # Evaluation metrics for project information extraction.
│           │   └── responses                           # Model responses for project information extraction.
│           └── squad                                   # Results from the SQuAD task.
│               ├── logs                                # Logs generated during the model training process.
│               ├── metrics                             # Evaluation metrics for SQuAD extraction.
│               └── responses                           # Model responses for SQuAD extraction.
  
├── log                                                 # Logs (general logs folder).

└── scripts                                             # Scripts for pipeline, training, and analysis.
    ├── evaluation.py                                   # Model evaluation script.
    ├── run_pipeline.py                                 # Main pipeline script for information extraction.
    ├── training.py                                     # Training script for the model.
    ├── analysis                                        # Analysis-specific scripts.
    │   ├── EDA.ipynb                                   # Exploratory Data Analysis notebook.
    │   ├── find_keywords.py                            # Keywords analysis in PDFs.
    │   ├── PDD_categorization.py                       # Analysis of PDDs structure.
    │   └── score_visualization.ipynb                   # Fine-tuning performance visualization.
    └── processing                                      # Data processing scripts.
        ├── verra_scraper.py                            # Verra data scraper and downloader.
        ├── project_ids.txt                             # List of project IDs for Verra scraping.
        ├── context_extractor.py                        # Extracts context from PDDs.
        ├── ground_truth_ghg_formatter.py               # GHG emission reduction ground truth formatter.
        ├── ground_truth_project_detail_formatter.py    # Project detail ground truth formatter.
        ├── squad_dataset_transform.py                  # Transforms SQuAD for question-answer tasks.
        ├── project_detail_dataset_transform.py         # Transforms project detail for question-answer tasks.
        └── ghg_emission_reduction_dataset_transform.py # Transforms GHG emission reduction for question-answer tasks.

├── tools                                               # Helper utilities.
    ├── OpenAIConnection.py                             # Functions for OpenAI API connection.
    ├── PDFExtraction.py                                # PDF text extraction functions.
    └── utils.py                                        # General utilities.
```
---

## Environment Setup Instructions
1. Install Python Version
To install, run the following commands:
`pyenv install 3.12.7`
`pyenv global 3.12.7`

2. Create and Activate Virtual Environment
To create and activate a virtual environment, run the following command:
`python -m venv LLMInformationExtraction`
`\LLMInformationExtraction\Scripts\activate`

3. Install Project Dependencies
To install all required dependencies, run the following command:
`pip install -r requirements/requirements.txt`

4. Configure Environment Variables
Create a .env file and add the following variables:
OPENAI_API_KEY=<your-openai-api-key>
PYTHONPATH=<absolute-path-to-project-folder>

---

## Jupyter Notebook Setup (For Data Analysis)
1. Create a Conda Environment
To create a Conda environment, run the following command:
`conda env create -f requirements/requirements_analysis.yml`

2. Activate the Conda Environment
Activate the environment with the following command:
`conda activate analysis`

3. Launch Jupyter Notebook
Start the Jupyter Notebook with the following command:
`jupyter notebook scripts/analysis/`

---

## How to Execute
### Information Extraction Pipeline
>> Step 1: Prepare Input Files
Place PDF files in the folder:
`data/inference/input`

>> Step 2: Run the Script
To run the the pipeline, use the following command:
`python scripts\run_pipeline.py [1234 1235] [--m ft:gpt-3.5-turbo-0125::APFxmJCP] [--input data/inference/input] [--output data/inference/intermediate/context.csv]`

Arguments:
ids: (Optional) Specific project IDs to process. If not provided, all PDFs in the input folder will be processed.
--m: Model ID (retrieved from fine-tuning process).
--input: (Optional) Input folder containing the PDFs.
--output: (Optional) Absolute path for saving the context extraction results (for debugging purposes).

>> Step 3: View Results
The extracted context will be saved in:
`data/inference/intermediate/context.csv`

The final results from IE task will be saved in:
`data/inference/output/[Model ID].csv`

### Model Training and Evaluation
>> Step 1: Prepare Input Files
Place PDF files in the folder:
`data/training/data_collection/pdds/`

>> Step 2: Extract Context from PDDs
Run the following command to extract context from the PDDs:
`python scripts\processing\context_extractor.py [input data/training/data_collection/pdds] [--ids 1234 1235] [--output data/training/data_processing/pdd_context_retrieval.csv]`

Arguments:
input: (Optional) Folder to search for the PDFs.
--ids: (Optional) Specific project IDs to process. If not provided, all PDFs in the input folder will be processed.
--output: (Optional) Path to save the extracted context.

The extracted context will be saved in:
data/training/data_processing/pdd_context_retrieval.csv

>> Step 3: Prepare Ground Truth
To clean and format the datasets into JSON key-value format, run:

For comprehensive project dataset:
`python scripts\processing\ground_truth_project_detail_formatter.py`

For GHG emission reduction dataset:
`python scripts\processing\ground_truth_ghg_reduction_formatter.py`

The processed ground truth will be saved in:
`data\training\data_processing\processed_ground_truth_project_info.csv`
`data\training\data_processing\processed_ground_truth_ghg.csv`

>> Step 4: Transform Data for Fine-Tuning
To merge context and ground truth, and partition data into train, validation, and test sets, run:

For SQuAD dataset:
`python scripts\processing\squad_dataset_transform.py`

For project detail dataset:
`python scripts\processing\project_detail_dataset_transform.py`

For GHG emission reduction dataset:
`python scripts\processing\ghg_emission_reduction_dataset_transform.py`

The transformed datasets will be saved in:
data/training/data_partitioning/train/
data/training/data_partitioning/validate/
data/training/data_partitioning/test/

>> Step 5: Train the Model
To train the GPT-3.5 model on the training and validation sets, run:
`python scripts\training.py [output_dir squad] [--train_file squad_train] [--validate_file squad_validate] [--epoch 4] [--bsize 8] [--lr 2]`

Arguments:
output_dir: Directory to store logs and learning curves (e.g., squad, project_info, or ghg_emission_reduction).
--train_file: Filename of the training data.
--validate_file: Filename of the validation data.
--epoch: (Optional) Number of epochs.
--bsize: (Optional) Batch size.
--lr: (Optional) Learning rate.

Training logs will be saved in:
data/training/result/{dataset}/logs/

>> Step 6: Evaluate the Model
To evaluate the model’s performance and compare generated responses with reference answers, run:
`python scripts\evaluation.py [id ft:gpt-3.5-turbo-0125::APFxmJCP] [--question squad_test_prompt] [--answer squad_test_answer] [--output squad]`

Arguments:
id: Model ID to generate responses.
--question: Filename containing test prompts.
--answer: Filename containing correct answers.
--output: Folder to store performance metrics.

Evaluation results will be saved in:
data/training/result/{dataset}/responses/{modelID}
data/training/result/{dataset}/metrics/{modelID}

### To Scrape Data and Download PDDs from Verra Website
To scrape project data and download PDDs, run:
`python scripts\processing\verra_scraper.py [input scripts/processing/project_ids.txt] [--ids 1234 1235] [--dirs data/training/data_collection/pdds] [--output data/training/data_collection/verra_data.csv]`

Arguments:
input: (Optional) Text file containing project IDs.
--ids: (Optional) Specific project IDs to process. (Provide either input file or list of IDs)
--dirs: (Optional) Local path to store the downloaded PDDs.
--output: (Optional) Path to store the scraped project data.

Scraped project information will be saved in:
data/training/data_collection/verra_data.csv

Downloaded PDDs will be saved in:
data/training/data_collection/pdds/

### To Run Analysis
For Exploratory Data Analysis:
`jupyter notebook scripts/analysis/EDA.ipynb`

#### To Visualize Fine-Tuning Performance Scores:
`jupyter notebook scripts/analysis/Score Visualisation.ipynb`

#### To Search Keywords in PDDs:
To search for keywords under each category and identify their locations in the PDDs, run:
`python scripts\analysis\find_keyword_in_pdf.py [input data/training/data_collection/pdds/] [-ids 1234 1235] [-output keywords_found_in_pages]`

Arguments:
input: (Optional) Folder to search for the PDFs.
-ids: (Optional) Specific project IDs to process.
-output: (Optional) Output filename to store keyword search results.

Keyword search results will be saved in:
data/training/data_analysis/keywords_found_in_pages.csv

#### To Categorize PDDs by Their Content's Headings Style:
Run:
`python scripts\analysis\PDD_categorization.py`