# Hugging Face Documentation Question Answering System

A multi-interface Q&A system that uses Hugging Face's LLM and Retrieval Augmented Generation (RAG) to deliver answers based on Hugging Face documentation. Operable as an API, Discord bot, or Gradio app, it also provides links to the documentation used to formulate each answer.

# Example
![Example](./assets/example.png)

# Table of Contents
- [Setting up](#setting-up)
- [Running](#running)
    - [Gradio](#gradio)
    - [API](#api-serving)
    - [Discord Bot](#discord-bot)
- [Indexes](#indexes-list)
- [Development instructions](#development-instructions)

## Setting up

To execute any of the available interfaces, specify the required parameters in the `.env` file based on the `.env.example` located in the `config/` directory. Alternatively, you can set these as environment variables:

- `QUESTION_ANSWERING_MODEL_ID` - (str) A string that specifies either the model ID from the Hugging Face Hub or the directory containing the model weights
- `EMBEDDING_MODEL_ID` - (str) embedding model ID from the Hugging Face Hub. We recommend using the `hkunlp/instructor-large`
- `INDEX_REPO_ID` - (str) Repository ID from the Hugging Face Hub where the index is stored. List of the most actual indexes can be found in this section: [Indexes](#indexes-list)
- `PROMPT_TEMPLATE_NAME` - (str) Name of the model prompt template used for question answering, templates are stored in the `config/api/prompt_templates` directory
- `USE_DOCS_FOR_CONTEXT` - (bool) Use retrieved documents as a context for a given query
- `NUM_RELEVANT_DOCS` - (int) Number of documents used for the previous feature
- `ADD_SOURCES_TO_RESPONSE` - (bool) Include sources of the retrieved documents used as a context for a given query
- `USE_MESSAGES_IN_CONTEXT` - (bool) Use chat history for conversational experience
- `DEBUG` - (bool) Provides additional logging

Install the necessary dependencies from the requirements file:

```bash
pip install -r requirements.txt
```

## Running

### Gradio

After completing all steps as described in the [Setting up](#setting-up) section, specify the `APP_MODE` environment variable as `gradio` and run the following command:

```bash
python3 app.py
```

### API Serving

By default, the API is served at `http://0.0.0.0:8000`. To launch it, complete all the steps outlined in the [Setting up](#setting-up) section, then execute the following command:

```bash
python3 -m api
```

### Discord Bot

To interact with the system as a Discord bot, add additional required environment variables from the `Discord bot` section of the `.env.example` file in the `config/` directory.

- `DISCORD_TOKEN` - (str) API key for the bot application
- `QA_SERVICE_URL` - (str) URL of the API service. We recommend using: `http://0.0.0.0:8000`
- `NUM_LAST_MESSAGES` - (int) Number of messages used for context in conversations
- `USE_NAMES_IN_CONTEXT` - (bool) Include usernames in the conversation context
- `ENABLE_COMMANDS` - (bool) Allow commands, e.g., channel cleanup
- `DEBUG` - (bool) Provides additional logging

After completing all steps, run:

```bash
python3 -m bot
```

To host bot on Hugging Face Spaces, specify the `APP_MODE` environment variable as `discord`, and the bot will be run automatically from the `app.py` file.

<!-- ### Running in a Docker

Tu run API and bot in a Docker container, run the following command:

```bash
./run_docker.sh
``` -->

## Indexes List

The following list contains the most current indexes that can be used for the system:
- [All Hugging Face repositories over 50 Stars - 512-Character Chunks](https://huggingface.co/datasets/KonradSzafer/index-instructor-large-512-m512-all_repos_above_50_stars)
- [All Hugging Face repositories over 50 Stars - 812-Character Chunks](KonradSzafer/index-instructor-large-812-m512-all_repos_above_50_stars)

## Development Instructions

We use `Python 3.10`

To install all necessary Python packages, run the following command:

```bash
pip install -r requirements.txt
```
We use the pipreqsnb to generate the requirements.txt file. To install pipreqsnb, run the following command:

```bash
pip install pipreqsnb
```
To generate the requirements.txt file, run the following command:

```bash
pipreqsnb --force .
```

To run unit tests, you can use the following command:

```bash
pytest -o "testpaths=tests" --noconftest
```
