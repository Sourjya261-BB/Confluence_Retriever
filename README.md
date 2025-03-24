# Confluence Document Retriever

This repository provides a pipeline to scrape, process, and retrieve Confluence documents efficiently. It includes scripts for web scraping, document processing, and a Streamlit-based application for easy interaction.

## Prerequisites

Before running the project, ensure you have Python installed on your system.

## Setup Instructions

### 1. Create a `.env` File
In your local repository, create a `.env` file to store environment variables.
```sh
# Example of .env file
CONFLUENCE_URL=https://your-confluence-instance.com
USERNAME=your-username
PASSWORD=your-password
```

### 2. Create and Activate Virtual Environment
Use the following commands to create a virtual environment and install dependencies from `requirements.txt`:
```sh
# Create a virtual environment
python -m venv venv

# Activate the virtual environment (Linux/Mac)
source venv/bin/activate

# Activate the virtual environment (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Running the Scraper

Run the `scrape_docs.py` script located in the `scripts/` folder to fetch Confluence pages and save them in the `md_output` folder.
```sh
python scripts/scrape_docs.py
```

### Output Structure
The scraped Confluence pages are stored in `.md` format in the `md_output` folder. Each file is structured as follows:

```
# Title
### Source
### Attachments
### Page_Content
```

- **Attachments Handling:**
  - If a page has attachments, they are saved inside `md_output/<parent_filename>/`.
  - The `###Attachments` section of the markdown file contains local links to these attachment files.

## Processing Scraped Documents

To process the scraped documents and generate attachment summaries, run:
```sh
python scripts/processing_scraped_docs_consolidated_script.py
```

- This script generates attachment summaries and saves them in a file named `<attachment_path>_summary.txt`.
- It then creates parent and child retrievers post-chunking.

## Running the Streamlit Application

Once the documents are processed, you can run the Streamlit application using:
```sh
streamlit run scripts/streamlit_application.py
```

This will start a web interface for interacting with the processed documents and retrieving information efficiently.

## Configuration Settings

The project allows for customizable retrieval mechanisms and chunking strategies:

- **Chunk Sizes:**
  - `PARENT_CHUNK_SIZE`: Number of characters in a parent chunk (default: 1200).
  - `CHILD_CHUNK_SIZE`: Number of characters in a child chunk (default: 400).

- **SOS Mode (`SOS=True/False`)**
  - When `SOS=True`, only a subset of documents containing specific keywords is processed. This is useful for S1 scenarios requiring critical documents.

- **Augmented Chunking (`AUGMENTED_CHUNKING=True/False`)**
  - Uses an LLM to improve chunking quality (recommended only when `SOS=True` due to resource intensity).

- **Retriever Mechanisms:**
  - `complex_parent`: Parent-child retrieval, where parent chunks are retrieved first, followed by sibling enrichment using adjoining paragraphs, code blocks, or relevant keywords.
  - `child`: Single-stage retrieval using child chunks only.

- **Retrieval Modes:**
  - `bm25`: Pure keyword search.
  - `dense`: Semantic search.
  - `hybrid`: Combination of both keyword and semantic search (default).

- **Database & Vector Store:**
  - Stores indexed chunks in `VECTORDB_PATH = "./confluence_db_v2.2"`.
  - Uses `EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"` for embeddings.
  - Uses `RERANKING_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"` for ranking.
  - Supports `KEYWORD_EXTRACTION_MODE = "llm"` for extracting key phrases.

- **SOS Patterns:**
  - Keywords used to filter SOS documents include `"sop", "resilience", "rps", "jdvar", "solr", "grafana", "traffic drop", "rate limit"` and more.

## Folder Structure
```
project_root/
│── md_output/
│   │── <confluence_page_1>.md
│   │── <confluence_page_2>.md
│   │── <parent_filename>/
│   │   │── <attachment_1>
│   │   │── <attachment_2>
│── scripts/
│   │── commons.py
│   │── processing_scraped_docs_consolidated_script.py
│   │── retrieval_pipeline_final_consolidated.py
│   │── scrape_docs.py
│   │── streamlit_application.py
│   │── test_script.ipynb
│   │── retrieval_helpers/
│   │   │── common_utils.py
│   │   │── keyword_extraction_utils.py
│   │   │── parent_doc_retrieval_utils.py
│   │   │── __init__.py
│── requirements.txt
│── .env
```

## Summary of Commands

| Action                          | Command                                      |
|---------------------------------|----------------------------------------------|
| Create virtual environment      | `python -m venv venv`                        |
| Activate virtual environment (Linux/Mac) | `source venv/bin/activate`       |
| Activate virtual environment (Windows)   | `venv\Scripts\activate`         |
| Install dependencies            | `pip install -r requirements.txt`            |
| Run scraper                     | `python scripts/scrape_docs.py`              |
| Process scraped documents       | `python scripts/processing_scraped_docs_consolidated_script.py` |
| Run Streamlit application       | `streamlit run scripts/streamlit_application.py` |

## License
This project is licensed under the MIT License.

## Author
Developed by Sourjya Mukherjee. Feel free to reach out for any queries or contributions!

