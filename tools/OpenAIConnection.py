from openai import OpenAI
from config import config
import time 
import requests
import base64


class OpenAIConnection:
    def __init__(self, model, purpose='fine-tune'):
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.purpose = purpose
        self.model = model
    
    def _get_model(self):
        return self.client.models.list()
    
    def _upload_file(self, filepath):
        response = self.client.files.create(
                        file=open(filepath, 'rb'),
                        purpose=self.purpose)

        return response.id
    
    def _create_finetune_job(self, training_file_id, validation_file_id, epoch=4, bsize=8, lr=2):
        response = self.client.fine_tuning.jobs.create(
            training_file=training_file_id,
            validation_file=validation_file_id,
            model=self.model,
            hyperparameters={
                    "n_epochs":epoch,
                    "batch_size":bsize,
                    "learning_rate_multiplier":lr,
                }
            )
        
        return response.id
    
    def _check_job_status(self, job_id):
        response = self.client.fine_tuning.jobs.retrieve(job_id)
        if response == 'failed':
            print(f"Job Error; {response}")
            raise

        return response

    def _get_model_events(self, job_id):
        response = self.client.fine_tuning.jobs.list_events(job_id)

        return response


    def _evaluate_model(self, model_id, prompt, temperature=0, max_token=1000):
        response = self.client.chat.completions.create(
            model=model_id, messages=prompt, temperature=temperature, max_tokens=max_token)
        
        return response.choices[0].message.content
    

    def _download_result(self, files, output):
        headers = {"Authorization": f"Bearer {config.OPENAI_API_KEY}"}  
        for file in files:
            url = f"https://api.openai.com/v1/files/{file}/content"

        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:                
                with open(f"{output}/{file}.csv", 'wb') as file:
                    file.write(base64.b64decode(response.text))
            else:
                print(f"Failed to download: {response.status_code}")
                print(f"Response: {response.json()}")
                
        except Exception as e:
            print(f"Error occurred: {e}")


    def _get_usage(self):
        url = "https://api.openai.com/v1/usage"
        headers = {"Authorization": f"Bearer {config.OPENAI_API_KEY}"}
        try:
            response = requests.get(url, headers=headers)
            usage_data = response.json()
            return usage_data['total_usage']['usd']
        
        except Exception as e:
            print(f"Error occurred: {e}")

    def _delete_model(self, model_id):
        self.client.models.delete(model_id)