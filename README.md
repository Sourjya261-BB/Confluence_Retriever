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

Run the `scrape_docs.py` script to fetch Confluence pages and save them in the `md_output` folder.
```sh
python scrape_docs.py
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
python processing_scraped_docs_consolidated_script.py
```

- This script generates attachment summaries and saves them in a file named `<attachment_path>_summary.txt`.
- It then creates parent and child retrievers post-chunking.

## Running the Streamlit Application

Once the documents are processed, you can run the Streamlit application using:
```sh
streamlit run streamlit_application.py
```

This will start a web interface for interacting with the processed documents and retrieving information efficiently.

## Folder Structure
```
project_root/
│── md_output/
│   │── <confluence_page_1>.md
│   │── <confluence_page_2>.md
│   │── <parent_filename>/
│   │   │── <attachment_1>
│   │   │── <attachment_2>
│── scrape_docs.py
│── processing_scraped_docs_consolidated_script.py
│── streamlit_application.py
│── retrieval_pipeline_final_consolidated.py
│── requirements.txt
│── .env
```

## Summary of Commands

| Action                          | Command |
|---------------------------------|------------------------------------------------|
| Create virtual environment      | `python -m venv venv`                           |
| Activate virtual environment (Linux/Mac) | `source venv/bin/activate`               |
| Activate virtual environment (Windows)   | `venv\Scripts\activate`                 |
| Install dependencies            | `pip install -r requirements.txt`               |
| Run scraper                     | `python scrape_docs.py`                         |
| Process scraped documents       | `python processing_scraped_docs_consolidated_script.py`                |
| Run Streamlit application       | `streamlit run streamlit_application.py`        |

## License
This project is licensed under the MIT License.

## Author
Developed by Sourjya Mukherjee. Feel free to reach out for any queries or contributions!

