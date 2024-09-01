# Project description

- This is a standalone python script to extract data from predefined sections for climate report PDFs.
- This can be run with 2 flavors:
  - Barebone: if you already have Python installed locally on your machine.
  - Docker: you'll just need Docker installed locally on your machine.
- The extraction process: use `pdfplumber` to identify the section we're looking to extract, extract the raw text or tables content, depends on the section. Then feeds these raw text & tables to OpenAI completion with defined function to get a structured response based on the section. And use caching so we don't have to process the same file again.

# Dependencies

- pdfplumber: for working with PDF
- OpenAI: for query
- Redis: for caching
- dotenv: for loading environment variables.

# How to run the script

## With Docker (for non tech users)

1. Install Docker to your machine. You can go to https://docs.docker.com/engine/install/ and select the one for your OS. And run the Docker daemon by simply open the application.
2. Clone this repo to your machine with Download ZIP option from Github, and unzip it.
3. Open the terminal and cd into the repo folder. (for example, `cd C:/Downloads/pdf-data-extactrion`)
4. Ask someone for the `.env` file that contains all the credentials needed for the project (OpenAI key, Redis key etc...) and add it to the root folder of the repo.
5. Build the Docker image:

   ```console
   docker build -t extract:v1 .
   ```

6. Put the PDFs that you'd like to process inside any folder that you like inside the repo folder, for example an inputs `mkdir inputs` and copy all the PDFs to this folder
7. Run the container:

   ```console
   docker run -it --rm --name my-extract-script -v "$PWD":/usr/src/extract -w /usr/src/extract extract:v1 python app.py -i {{update this to path to the folder contains the PDF}}
   ```

   For example with the inputs folder above:

   ```console
   docker run -it --rm --name my-extract-script -v "$PWD":/usr/src/extract -w /usr/src/extract extract:v1 python app.py -i ./inputs
   ```

   If you don't want to use the cache for some reasons:

   ```console
   docker run -it --rm --name my-extract-script -v "$PWD":/usr/src/extract -w /usr/src/extract extract:v1 python app.py --no-cache -i ./inputs
   ```

You should see the `outputs` and `logs` inside the `runs` folder in the root of the repo folder.
For consecutive runs, you'll only need to repeat step 6 and 7 accordingly.

## Barebone

If you have Python already installed in your machine.

1. Clone this repo to your machine
2. Install the dependencies

   ```console
   pip install --no-cache-dir -r requirements.txt
   ```

3. Ask someone for the `.env` file that contains all the credentials needed for the project (OpenAI key, Redis key etc...) and add it to the root folder of the repo.
4. Run the script with the input folder as an argument.

   ```console
   python app.py -i {{input folder path}}
   ```

# Common issues

- If you see `Cannot connect to the Docker daemon`, make sure to have Docker already running by open the Docker application.
- Network issues to OpenAI or Redis, make sure to have the `.env` in the root folder before running the script.
