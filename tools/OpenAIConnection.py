from openai import OpenAI
from config import config
import requests
import base64


class OpenAIConnection:
    def __init__(self, model, purpose='fine-tune'):
        """
        Initializes the OpenAIConnection class.

        Parameters:
        - model (str): The type of model to be used for fine-tuning.
        - purpose (str, optional): The purpose for the file upload (default is 'fine-tune').
        """
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.purpose = purpose
        self.model = model
    

    def _get_model(self):
        """
        Retrieves a list of all available models from OpenAI's account.

        Returns:
        - List of models: A list containing all available models.
        """
        return self.client.models.list()
    

    def _upload_file(self, filepath):
        """
        Uploads a file to OpenAI for a specific purpose.

        Parameters:
        - filepath (str): The local path to the file that needs to be uploaded.

        Returns:
        - file_id (str): The OpenAI's unique ID of the uploaded file.
        """
        response = self.client.files.create(
                        file=open(filepath, 'rb'),
                        purpose=self.purpose)

        return response.id
    

    def _create_finetune_job(self, train_id, validate_id, \
                             epoch=4, bsize=8, lr=2):
        """
        Creates a fine-tuning job for a model.

        Parameters:
        - train_id (str): The file ID of the training dataset.
        - validate_id (str): The file ID of the validation dataset.
        - epoch (int, optional): The number of epochs to run during fine-tuning (default is 4).
        - bsize (int, optional): The batch size for training (default is 8).
        - lr (int, optional): The learning rate multiplier for fine-tuning (default is 2).

        Returns:
        - job_id (str): The OpenAI's unique ID of the created fine-tuning job.
        """
        response = self.client.fine_tuning.jobs.create(
                        training_file=train_id,
                        validation_file=validate_id,
                        model=self.model,
                        hyperparameters={
                                "n_epochs":epoch,
                                "batch_size":bsize,
                                "learning_rate_multiplier":lr,
                            })
        
        return response.id
    

    def _check_job_status(self, job_id):
        """
        Retrieves the status of a fine-tuning job.

        Parameters:
        - job_id (str): The ID of the fine-tuning job to check.

        Returns:
        - Job details (dict): A dictionary containing details and status of the job.
        """
        response = self.client.fine_tuning.jobs.retrieve(job_id)

        return response
    

    def _evaluate_model(self, model_id, prompt, temperature=0, max_token=1000):
        """
        Evaluates the model by sending a prompt and receiving the model's output.

        Parameters:
        - model_id (str): The ID of the model to evaluate.
        - prompt (dict): A dictionary of messages containing the instruction and question to be evaluated.
        - temperature (float, optional): Controls the randomness of the model’s responses (default is 0).
        - max_token (int, optional): The maximum number of tokens in the response (default is 1000).

        Returns:
        - The content of the model's response.
        """
        response = self.client.chat.completions.create(
                        model=model_id, 
                        messages=prompt, 
                        temperature=temperature, 
                        max_tokens=max_token)
        
        return response.choices[0].message.content
    

    def _download_result(self, files, output):
        """
        Downloads the training log files from OpenAI API and saves them locally in the specified directory.

        Parameters:
        - files (list): A list of file IDs to be downloaded.
        - output (str): The local directory path where the downloaded files will be saved.
        """
        headers = {"Authorization": f"Bearer {config.OPENAI_API_KEY}"}  
        for file in files:
            # Request to download the file
            url = f"https://api.openai.com/v1/files/{file}/content"
            response = requests.get(url, headers=headers)
            # Check if the download was successful
            if response.status_code == 200:                
                with open(f"{output}/{file}.csv", 'wb') as file:
                    # Decode and write the file contents
                    file.write(base64.b64decode(response.text))
            else:
                print(f"Failed to download: {response.json()}")