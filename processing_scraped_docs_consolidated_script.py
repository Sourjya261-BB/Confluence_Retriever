import os
import hashlib
import tiktoken
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import base64
from io import BytesIO
from PIL import Image
import pandas as pd
import chardet
import fitz  
from langchain_core.messages import HumanMessage
from langchain_community.chat_models import AzureChatOpenAI
from docx import Document 
from dotenv import load_dotenv
import faulthandler
import asyncio
from langchain_chroma import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
import concurrent.futures
import csv
import sqlite3
import ast 
import chromadb
from chromadb.utils.embedding_functions import EmbeddingFunction
from typing import List



embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5", 
                                                     model_kwargs={"device": "cpu"})


faulthandler.enable() #Im using this for tracking where the code breaks....have found many malformed pdf attachments.

load_dotenv() 

#openAI env variabls
AZURE_OPENAI_VERSION=os.environ.get("AZURE_OPENAI_VERSION")
AZURE_OPENAI_DEPLOYMENT=os.environ.get("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_ENDPOINT=os.environ.get("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY=os.environ.get("AZURE_OPENAI_KEY")
AZURE_OPENAI_DEPLOYMENT_GPT4=os.environ.get("AZURE_OPENAI_DEPLOYMENT_GPT4")

multimodal_llm = AzureChatOpenAI(
    openai_api_version=AZURE_OPENAI_VERSION,
    azure_deployment="gpt-4o",
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_KEY,
    temperature=0.8,
    )

# Initialize the tokenizer
tokenizer = tiktoken.get_encoding('cl100k_base')

# Function to calculate token length
def tiktoken_len(text):
    tokens = tokenizer.encode(text, disallowed_special=())
    return len(tokens)

# Hash function to generate an 8-digit ID from the source URL
def generate_id(source_url):
    return int(hashlib.sha256(source_url.encode('utf-8')).hexdigest(), 16) % 10**8

# Function to chunk the page content into pieces not exceeding max_tokens
# def chunk_content(text, max_tokens=10000):
#     tokens = tokenizer.encode(text, disallowed_special=())
#     chunks = []
#     start = 0
#     while start < len(tokens):
#         end = min(start + max_tokens, len(tokens))
#         chunk_tokens = tokens[start:end]
#         chunk_text = tokenizer.decode(chunk_tokens)
#         chunks.append(chunk_text)
#         start = end
#     return chunks
def chunk_content(text, max_tokens=10000, overlap_percent=10):
    """
    Chunk text into pieces not exceeding max_tokens with a specified overlap percentage.
    
    Args:
        text (str): The text to chunk
        max_tokens (int): Maximum number of tokens per chunk
        overlap_percent (int): Percentage of tokens to overlap between chunks
        
    Returns:
        list: List of text chunks with overlap
    """
    tokens = tokenizer.encode(text, disallowed_special=())
    chunks = []
    
    # Calculate overlap size in tokens
    overlap_size = int(max_tokens * (overlap_percent / 100))
    
    # Adjust step size based on overlap
    step_size = max_tokens - overlap_size
    
    # Edge case: if step_size becomes too small or negative
    if step_size <= 0:
        step_size = max(1, max_tokens // 2)
        overlap_size = max_tokens - step_size
    
    start = 0
    while start < len(tokens):
        # Calculate end position for current chunk
        end = min(start + max_tokens, len(tokens))
        
        # Get tokens for this chunk
        chunk_tokens = tokens[start:end]
        
        # Decode tokens back to text
        chunk_text = tokenizer.decode(chunk_tokens)
        
        # Add chunk to results
        chunks.append(chunk_text)
        
        # Move start position for next chunk, accounting for overlap
        start += step_size
    
    return chunks

def extract_structured_data_from_md_files(folder_path):
    data_list = []
    # Read all .md files in the folder
    for filename in tqdm(os.listdir(folder_path), desc="Processing .md files"):
        if filename.endswith('.md'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                
                # Extract title
                title_start = content.find("#") + 1
                title_end = content.find("\n", title_start)
                title = content[title_start:title_end].strip()
                
                # Extract attachments
                attachments_start = content.find("### Attachments:") + len("### Attachments:")
                attachments_end = content.find("### Source:", attachments_start)
                attachments_text = content[attachments_start:attachments_end].strip()
                attachments = [line.strip() for line in attachments_text.split('\n') if line.strip()]
                
                # Extract source
                source_start = content.find("### Source:") + len("### Source:")
                source_end = content.find("### Page_content:", source_start)
                source = content[source_start:source_end].strip()
                
                # Generate ID from the source URL
                id = generate_id(source)
                
                # Extract Page Content
                page_content_start = content.find("### Page_content:") + len("### Page_content:")
                page_content = content[page_content_start:].strip()
                
                # Count tokens in Page Content
                token_count = tiktoken_len(page_content)
                
                # Store data in a dictionary
                data = {
                    'title': title,
                    'attachments': attachments,
                    'source': source,
                    'id': id,
                    'page_content': page_content,
                    'token_count': token_count
                }
                
                data_list.append(data)

    return data_list
    

def plot_token_histogram(new_data_list):
    token_counts = [
        tiktoken_len(content) 
        for data in new_data_list 
        for content in (data['page_content'] if isinstance(data['page_content'], list) else [data['page_content']])
    ]
    
    sns.set_style("whitegrid")
    sns.set_palette("muted")
    plt.figure(figsize=(12, 6))
    sns.histplot(token_counts, kde=False, bins=50)
    plt.title("Token Counts Histogram")
    plt.xlabel("Token Count")
    plt.ylabel("Frequency")
    plt.show()

#-------------------------------Summarization Functions--------------------------------
def sanitize_attachment_path(attachment_path):
    """Sanitize attachment paths by removing leading characters."""
    return attachment_path.lstrip("- ")

# def generate_attachment_summaries(data_list):
#     """Process list of data dictionaries and generate summaries."""
#     error_messages = [
#         "PDF processing error:", "DOCX processing error:",
#         "Spreadsheet processing error:", "Text processing error:",
#         "Image processing error:"
#     ]

#     for data in tqdm(data_list, desc="Processing data list"):
#         title = data.get('title', 'Untitled')
#         attachment_paths = data.get('attachments', [])
        
#         sanitized_paths = [p.lstrip("- ") for p in attachment_paths]
#         for path in sanitized_paths:
#             if os.path.exists(path):
#                 base_path = os.path.splitext(path)[0]
#                 summary_path = f"{base_path}_summary.txt"

#                 # Check if summary file exists
#                 if os.path.exists(summary_path):
#                     with open(summary_path, "r", encoding="utf-8") as f:
#                         content = f.read()

#                     # Extract summary text
#                     summary_start = content.find("Summary:\n")
#                     if summary_start != -1:
#                         existing_summary = content[summary_start + len("Summary:\n"):].strip()

#                         # If existing summary contains an error message, redo summarization
#                         if any(existing_summary.startswith(err) for err in error_messages):
#                             print(f"Retrying summarization for {path} due to error...")
#                             summaries = summarize_attachment(path, title)
#                         else:
#                             summaries = existing_summary
#                     else:
#                         summaries = summarize_attachment(path, title)
#                 else:
#                     summaries = summarize_attachment(path, title)
#             else:
#                 summaries = "File not found"
#             #when I wrote it my intention was to pass the dict of all paths and summaries but storing them inmemory lead to kernel crashes...hence the workaround
#             save_summaries({path:summaries}, title)


async def generate_attachment_summaries(data_list, batch_size=5):
    """Process list of data dictionaries and generate summaries."""
    error_messages = [
        "PDF processing error:", "DOCX processing error:",
        "Spreadsheet processing error:", "Text processing error:",
        "Image processing error:"
    ]

    async def generate_summary_for_attachment_path(path,title):

        if os.path.exists(path):
            base_path = os.path.splitext(path)[0]
            summary_path = f"{base_path}_summary.txt"

            # Check if summary file exists
            if os.path.exists(summary_path):
                with open(summary_path, "r", encoding="utf-8") as f:
                    content = f.read()

                # Extract summary text
                summary_start = content.find("Summary:\n")
                if summary_start != -1:
                    existing_summary = content[summary_start + len("Summary:\n"):].strip()

                    # If existing summary contains an error message, redo summarization
                    if any(existing_summary.startswith(err) for err in error_messages):
                        print(f"Retrying summarization for {path} due to error...")
                        summary = summarize_attachment(path, title)
                    else:
                        summary = existing_summary
                else:
                    summary = summarize_attachment(path, title)
            else:
                summary = summarize_attachment(path, title)
        else:
            summary = "File not found"
        #when I wrote it my intention was to pass the dict of all paths and summaries but storing them inmemory lead to kernel crashes...hence the workaround
        save_summaries({path:summary}, title)

    attachment_paths_dict = {}
    for data in data_list:
        if len(data["attachments"])!=0:
            for attachment_path in data["attachments"]:
                attachment_paths_dict[sanitize_attachment_path(attachment_path)] = data.get('title', 'Untitled')

    attachment_paths_items = list(attachment_paths_dict.items())
    for i in tqdm(range(0, len(attachment_paths_items), batch_size),desc="Processing attachments"):
        batch = attachment_paths_items[i:i + batch_size]
        tasks = []
        for path, title in batch:
            tasks.append(generate_summary_for_attachment_path(path,title))
        await asyncio.gather(*tasks)

def summarize_attachment(file_path, title):
    """Generate summary for different file types using appropriate methods."""
    try:
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext in [".png", ".jpg", ".jpeg"]:
            return generate_image_summary(file_path, title)
        elif file_ext == ".txt":
            return generate_text_summary(file_path, title)
        elif file_ext in [".csv", ".xlsx", ".xlsm", ".xlsb", ".ods", ".xls"]:
            return generate_spreadsheet_summary(file_path, title)
        elif file_ext in [".docx"]:
            return generate_docx_summary(file_path, title)
        elif file_ext in [".pdf"]:
            return generate_pdf_summary(file_path, title)
        else:
            return f"Unsupported file type: {file_ext}"
    except Exception as e:
        return f"Error processing file: {str(e)}"

def generate_image_summary(image_path, title, max_size=1024):
    """Process and summarize images with compression."""
    try:
        # Compress and encode image
        with Image.open(image_path) as img:
            img.thumbnail((max_size, max_size))
            buffered = BytesIO()
            img.save(buffered, format=img.format)
            base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        # Create multimodal message
        message = HumanMessage(content=[
            {"type": "text", "text": f"Summarize this image related to {title}"},
            {"type": "image_url", "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }}
        ])
        
        return multimodal_llm.invoke([message]).content
    
    except Exception as e:
        return f"Image processing error: {str(e)}"

def generate_text_summary(file_path, title, max_length=4000):
    """Process text files with encoding detection and truncation."""
    try:
        # Detect file encoding
        with open(file_path, "rb") as f:
            raw_data = f.read()  # Read raw bytes for detection
            detected_encoding = chardet.detect(raw_data)["encoding"]

        # Use detected encoding to read file
        with open(file_path, "r", encoding=detected_encoding or "utf-8", errors="ignore") as f:
            content = f.read(max_length)

        return multimodal_llm.invoke([
            HumanMessage(content=f"Summarize this text related to {title}:\n\n{content}")
        ]).content

    except Exception as e:
        return f"Text processing error: {str(e)}"

def generate_spreadsheet_summary(file_path, title, sample_rows=10):
    """Process tabular data with sampling for large files."""
    try:
        file_ext = os.path.splitext(file_path)[1].lower()

        if file_ext == ".csv":
            df = pd.read_csv(file_path)
        elif file_ext in [".xlsx", ".xlsm", ".xlsb", ".ods"]:
            df = pd.read_excel(file_path, engine="openpyxl")
        elif file_ext == ".xls":
            df = pd.read_excel(file_path, engine="xlrd")  # Use xlrd for .xls files

        sample = df.head(sample_rows).to_markdown()

        return multimodal_llm.invoke([
            HumanMessage(content=f"Summarize this data related to {title}:\n\n{sample}")
        ]).content

    except Exception as e:
        return f"Spreadsheet processing error: {str(e)}"

def generate_docx_summary(file_path, title):
    """Extract and summarize text from DOCX files."""
    try:
        doc = Document(file_path)
        content = "\n".join([para.text for para in doc.paragraphs])

        return multimodal_llm.invoke([
            HumanMessage(content=f"Summarize this document related to {title}:\n\n{content}")
        ]).content

    except Exception as e:
        return f"DOCX processing error: {str(e)}"

# Need to understand how this works....pdf processing is a nightmare
def generate_pdf_summary(file_path, title, max_pages=5):
    """
    Extract and summarize text from PDF files with process isolation
    to prevent crashes from affecting the main application.
    
    Args:
        file_path (str): Path to the PDF file
        title (str): Title or topic of the PDF for context in summarization
        max_pages (int, optional): Maximum number of pages to process. Defaults to 5.
    
    Returns:
        str: Summary of the PDF content or error message
    """
    import os
    import subprocess
    import tempfile
    import json
    from langchain.schema import HumanMessage
    
    if not os.path.exists(file_path):
        return f"Error: File not found at {file_path}"
    
    # Create an isolated Python script to handle the PDF extraction
    # This script will be executed as a separate process
    pdf_extractor_script = """
import sys
import json
import traceback

def extract_with_pymupdf(file_path, max_pages):
    try:
        import fitz
        text = ""
        try:
            doc = fitz.open(file_path)
            page_count = min(max_pages, doc.page_count)
            
            for page_num in range(page_count):
                try:
                    page = doc[page_num]
                    page_text = page.get_text()
                    text += f"--- Page {page_num + 1} ---\\n{page_text}\\n\\n"
                except Exception as e:
                    text += f"[Error extracting page {page_num + 1}: {str(e)}]\\n"
            
            doc.close()
            return {"success": True, "text": text, "method": "pymupdf"}
        except Exception as e:
            return {"success": False, "error": str(e), "method": "pymupdf"}
    except ImportError:
        return {"success": False, "error": "PyMuPDF not installed", "method": "none"}

def extract_with_pypdf2(file_path, max_pages):
    try:
        import PyPDF2
        text = ""
        
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                # Check if PDF is encrypted
                if pdf_reader.is_encrypted:
                    try:
                        pdf_reader.decrypt('')  # Try empty password
                    except:
                        pass
                
                page_count = min(max_pages, len(pdf_reader.pages))
                
                for page_num in range(page_count):
                    try:
                        page = pdf_reader.pages[page_num]
                        page_text = page.extract_text()
                        text += f"--- Page {page_num + 1} ---\\n{page_text}\\n\\n"
                    except Exception as e:
                        text += f"[Error extracting page {page_num + 1}: {str(e)}]\\n"
                
                return {"success": True, "text": text, "method": "pypdf2"}
        except Exception as e:
            return {"success": False, "error": str(e), "method": "pypdf2"}
    except ImportError:
        return {"success": False, "error": "PyPDF2 not installed", "method": "none"}

if __name__ == "__main__":
    file_path = sys.argv[1]
    max_pages = int(sys.argv[2])
    
    # Try each method in sequence
    result = extract_with_pymupdf(file_path, max_pages)
    
    if not result["success"]:
        result = extract_with_pypdf2(file_path, max_pages)
    
    # Output the result as JSON
    print(json.dumps(result))
    sys.exit(0)  # Ensure we exit cleanly
"""

    # Create a temporary Python file
    with tempfile.NamedTemporaryFile(suffix='.py', mode='w', delete=False) as temp_file:
        temp_file_path = temp_file.name
        temp_file.write(pdf_extractor_script)
    
    try:
        # Execute the script in a separate process with a timeout
        # This ensures that if the PDF processing crashes, it won't affect the main application
        process = subprocess.run(
            ["python3", temp_file_path, file_path, str(max_pages)],
            capture_output=True,
            text=True,
            timeout=60  # Set a reasonable timeout (60 seconds per PDF)
        )
        
        # Try to parse the output as JSON
        if process.returncode == 0 and process.stdout.strip():
            try:
                result = json.loads(process.stdout)
                
                if result["success"]:
                    text = result["text"]
                    method = result["method"]
                    
                    # Generate summary with available text
                    summary = multimodal_llm.invoke([
                        HumanMessage(content=f"Summarize this PDF document related to {title}. "
                                            f"Note that content was extracted using {method}:\n\n{text}")
                    ]).content
                    
                    return summary
                else:
                    return f"Error extracting PDF content: {result.get('error', 'Unknown error')}"
            except json.JSONDecodeError:
                return f"Error parsing PDF extractor output: {process.stdout[:100]}..."
        else:
            # Process failed or no output
            return f"PDF extraction process failed: {process.stderr}"
    
    except subprocess.TimeoutExpired:
        return "PDF processing timed out after 60 seconds"
    except Exception as e:
        return f"Error in PDF extraction process: {str(e)}"
    finally:
        # Clean up the temporary file
        try:
            os.unlink(temp_file_path)
        except:
            pass

def save_summaries(summaries, title):
    """Save generated summaries to files."""
    error_messages = [
        "PDF processing error:", "DOCX processing error:",
        "Spreadsheet processing error:", "Text processing error:",
        "Image processing error:"
    ]
    for path, summary in summaries.items():
            try:
                base_name = os.path.splitext(path)[0]
                summary_path = f"{base_name}_summary.txt"
                # if not os.path.exists(summary_path):
                if any(summary.startswith(err) for err in error_messages):
                    print(f"Failed to generate summary for {path}: {summary}")
                    continue
                else:
                    with open(summary_path, "w") as f:
                        f.write(f"Title: {title}\n\nSummary:\n{summary}")
            except Exception as e:
                print(f"Failed to save summary for {path}: {str(e)}")
#------------------------Chunking and Vectorization------------------
def enrich_structured_data_with_summaries(data_list):
    for data in tqdm(data_list,total=len(data_list),desc="enriching data list with attachment summaries"):
        attachment_paths = data.get('attachments', [])
        sanitized_paths = [p.lstrip("- ") for p in attachment_paths]
        summaries = {}
        for path in sanitized_paths:
            base_name = os.path.splitext(path)[0]
            summary_path = f"{base_name}_summary.txt"
            if os.path.exists(summary_path):
                try:
                    with open(summary_path, "r", encoding="utf-8") as f:
                        summaries[path] = f.read()
                except Exception as e:
                    print(f"Error reading {summary_path}: {e}")
                    summaries[path] = ""
            else:
                summaries[path] = ""
        data['attachment_summaries'] = summaries

    return data_list

def get_parent_chunks(data_list, parent_chunk_size):
    data_list = enrich_structured_data_with_summaries(data_list)
    chunked_data_list = []
    
    for data in data_list:
        page_content = data.get("page_content", "")
        token_count = data.get("token_count", 0)

        if token_count > parent_chunk_size:
            chunks = chunk_content(page_content, max_tokens=parent_chunk_size, overlap_percent=10)
            
            for i, chunk in enumerate(chunks):
                # Remember never to do this : child_chunked_data["title"] = data.get("title","")
                # child_chunked_data["source"] = data.get("source","")..dictionaries are mutable!!!
                chunked_data = {
                    "title": data.get("title", ""),
                    "source": data.get("source", ""),
                    "id": f"{data.get('id', '')}-{i+1}",
                    "page_content": chunk,
                    "token_count": tiktoken_len(chunk),
                    "attachments": data.get("attachments", []),
                    "attachment_summaries": data.get("attachment_summaries", {})
                }
                chunked_data_list.append(chunked_data)

        else:
            chunked_data_list.append(data)  
            
    return chunked_data_list

def get_child_chunks(chunked_data_list, child_chunk_size):
    chunked_child_data_list = []
    
    for data in chunked_data_list:
        page_content = data.get("page_content", "")
        token_count = data.get("token_count", 0)
        
        if token_count > child_chunk_size:
            chunks = chunk_content(page_content, max_tokens=child_chunk_size, overlap_percent=10)
            
            for i, chunk in enumerate(chunks):
                # Create a new dictionary for each child chunk
                child_chunked_data = {
                    "title": data.get("title", ""),
                    "source": data.get("source", ""),
                    "id": f"{data.get('id', '')}-c{i+1}",
                    "parent_id": str(data.get("id", "")),
                    "page_content": f"### Title: {data.get('title', '')}\n{chunk}",
                    "token_count": tiktoken_len(chunk),
                    "attachments": data.get("attachments", [])
                }
                chunked_child_data_list.append(child_chunked_data)

        else:
            chunked_child_data_list.append(data)  # Append original data if no chunking is needed

    return chunked_child_data_list

def index_parent_chunks(parent_chunks, db_path="parent_chunks.db"):
    """Inserts parent chunks into an SQLite database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create table if it doesn't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS parent_chunks (
            id TEXT PRIMARY KEY,
            title TEXT,
            source TEXT,
            page_content TEXT,
            token_count INTEGER,
            attachments TEXT,
            attachment_summaries TEXT
        )
    ''')

    for chunk in parent_chunks:
        cursor.execute('''
            INSERT OR REPLACE INTO parent_chunks 
            (id, title, source, page_content, token_count, attachments, attachment_summaries)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            chunk.get("id", ""),
            chunk.get("title", ""),
            chunk.get("source", ""),
            chunk.get("page_content", ""),
            chunk.get("token_count", 0),
            str(chunk.get("attachments", [])),  # Convert list to string for storage
            str(chunk.get("attachment_summaries", {}))  # Convert dict to string for storage
        ))

    conn.commit()
    conn.close()

def retrieve_parent_chunk(chunk_id, db_path="parent_chunks.db"):
    """Retrieves a parent chunk based on its ID."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute('SELECT * FROM parent_chunks WHERE id = ?', (chunk_id,))
    row = cursor.fetchone()
    
    conn.close()

    if row:
        return {
            "id": row[0],
            "title": row[1],
            "source": row[2],
            "page_content": row[3],
            "token_count": row[4],
            "attachments": eval(row[5]),  # Convert string back to list
            "attachment_summaries": eval(row[6])  # Convert string back to dict
        }
    return None

def generate_embeddings_and_save_in_df(chunked_data_list, embedding_model, output_file_path="./documents_with_embeddings.csv"):
    
    def compute_embedding(text):
        return embedding_model.embed_query(text)

    if os.path.exists(output_file_path):
        df = pd.read_csv(output_file_path)
        existing_ids = set(df["id"].tolist())
    else:
        df = pd.DataFrame(columns=["id", "parent_id", "title", "source", "attachments", "page_content", "embeddings"])
        existing_ids = set()

    new_data = [data for data in chunked_data_list if data.get("id", "") not in existing_ids]

    if not new_data:
        print("All documents already processed. No new embeddings to compute.")
        return
    
    file_exists = os.path.exists(output_file_path)
    
    with open(output_file_path, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["id", "parent_id", "title", "source", "attachments", "page_content", "embeddings"])

        if not file_exists:
            writer.writeheader()

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_to_data = {executor.submit(compute_embedding, data["page_content"]): data for data in new_data}

            for future in tqdm(concurrent.futures.as_completed(future_to_data), total=len(future_to_data), desc="Computing embeddings"):
                data = future_to_data[future]
                try:
                    embedding = future.result()
                    row = {
                        "id": data.get("id", ""),
                        "parent_id": data.get("parent_id", ""),
                        "title": data.get("title", ""),
                        "source": data.get("source", ""),
                        "attachments": data.get("attachments", []),
                        "page_content": data.get("page_content", ""),
                        "embeddings": embedding
                    }
                    writer.writerow(row)
                except Exception as exc:
                    print(f"Error processing document {data.get('id', '')}: {exc}")
    print(f"Embeddings have been computed and saved to '{output_file_path}'.")

def index_child_embeddings(csv_path = "./child_chunks_400_with_embeddings.csv",batch_size=100,collection_name = "confluence_child_retriever_400"):

    class MyEmbeddingFunction(EmbeddingFunction):
        def __init__(self, model_name: str, device: str = "cpu"):
            self.model = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={"device": device})

        def __call__(self, input: List[str]) -> List[List[float]]:
            embeddings = self.model.embed_documents(input)
            return embeddings.tolist()

    def create_chroma_client():
        return chromadb.PersistentClient(
            "./confluence_db_v2"
        )

    def process_batch(collection, batch_df):
        embeddings = batch_df["embeddings"].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        ).tolist()
        
        documents = batch_df["page_content"].tolist()
        metadatas = batch_df[["title", "parent_id", "source", "attachments"]].to_dict('records')
        ids = batch_df["id"].astype(str).tolist()

        collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings
        )

    client = create_chroma_client()
    
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=MyEmbeddingFunction(model_name="BAAI/bge-base-en-v1.5", device="cpu"),
    )

    try:
        with pd.read_csv(csv_path, chunksize=batch_size) as reader:
            for batch in tqdm(reader, desc="Processing batches"):
                process_batch(collection, batch)

                
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
        return

    print(f"Successfully upserted all documents into Chroma DB")
    vectordb = Chroma(persist_directory="./confluence_db_v2", collection_name=collection_name, embedding_function=embedding_model)
    print(f"Vector Count: {vectordb._collection.count()}")

def clean_and_save_df(df, column="id", output_file="cleaned_data.csv"):
    duplicates = df[df.duplicated(subset=[column], keep=False)]  # keep=False to show all duplicates
    
    if not duplicates.empty:
        print("Duplicate Rows:")
        print(duplicates)

    df_cleaned = df.drop_duplicates(subset=[column], keep="first").reset_index(drop=True)
    df_cleaned.to_csv(output_file, index=False)
    print(f"\nCleaned DataFrame saved to '{output_file}'.")
    return df_cleaned

async def main():
    folder_path = './md_output'  # Change this path as needed
    data_list = extract_structured_data_from_md_files(folder_path)
    # plot_token_histogram(data_list)
    await generate_attachment_summaries(data_list)
    parent_chunks = get_parent_chunks(data_list,parent_chunk_size=1200)
    child_chunks = get_child_chunks(parent_chunks,child_chunk_size=400)
    index_parent_chunks(parent_chunks,db_path="parent_chunks.db")
    generate_embeddings_and_save_in_df(child_chunks,embedding_model=embedding_model,output_file_path="./child_chunks_400_with_embeddings.csv")
    clean_and_save_df(pd.read_csv("./child_chunks_400_with_embeddings.csv"),output_file="cleaned_child_chunks_400_with_embeddings.csv")
    index_child_embeddings(csv_path="./cleaned_child_chunks_400_with_embeddings.csv",batch_size=100,collection_name="confluence_child_retriever_400")
    

if __name__ == "__main__":
    asyncio.run(main())
