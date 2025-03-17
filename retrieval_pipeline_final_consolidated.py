import os
import sqlite3
import numpy as np
from langsmith import Client 
from langsmith import traceable
import base64
from langchain_core.messages import HumanMessage
from io import BytesIO
from PIL import Image
import pandas as pd
from langchain_core.output_parsers import JsonOutputParser
from typing import List, Dict
import json
import re
import asyncio
from tqdm.asyncio import tqdm
import torch
from dotenv import load_dotenv
from config import PARENT_CHUNKS_DB_PATH, TOP_K
from commons import gpt_35,gpt_4o,get_retriever
from sentence_transformers import CrossEncoder
import tiktoken
import spacy
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

# Load spaCy model - you'll need to install it with: python -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm")


torch.classes.__path__ = []


load_dotenv() 


os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ["LANGCHAIN_PROJECT"] = "Confluence Retriever"
os.environ["LANGSMITH_API_KEY"] = str(os.environ.get("LANGCHAIN_API_KEY"))
os.environ["LANGSMITH_ENDPOINT"]="https://api.smith.langchain.com"

client = Client()

retriever = get_retriever()

def extract_json(text):
    match = re.search(r'```json\s*({.*})\s*```', text, re.DOTALL)
    if match:
        try:
            json_data = json.loads(match.group(1))
            return json_data
        except json.JSONDecodeError:
            print("Error: Invalid JSON format.")
            return None
    return None

def tiktoken_len(text):
    """Returns the number of tokens in a text using tiktoken."""
    encoding = tiktoken.get_encoding('cl100k_base')
    return len(encoding.encode(text))

@traceable( run_type="chain",
    name="Image_processing_agent",
    tags=[f"{gpt_4o.model_name}"],
    project_name = "Confluence Retriever",
    metadata={"llm": f"{gpt_4o.model_name}"}
)
async def process_image(image_path,max_size=1024,user_query=None,multimodal_llm=gpt_4o):
    """Process and summarize images with compression."""
    prompt = f"""
        Given the image you need to answer the user query. Do not be too elaborate with yor answer just try to answer the user query truthfully without making up stuff. It is also possible the the image may not have relevant information. The image url is provided. The name of the image_path might give you some insight in answering the question. If the query is not not too specific try to atleast tell what you gleaned from the image.

        user_query : {user_query}
        image_path : {image_path}

        **Output Format**
        ```json
        {{
            "answer": "write your answer to the user query here as a string",
            "source": "source url"
        
        }}
        ```
        in case if the image is not relevant return empty jsonObject
        ```json
        {{ }}
        ```
    """
    try:
        # Compress and encode image
        with Image.open(image_path) as img:
            img.thumbnail((max_size, max_size))
            buffered = BytesIO()
            img.save(buffered, format=img.format)
            base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        # Create multimodal message
        message = HumanMessage(content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }}
        ])
        chain = multimodal_llm | JsonOutputParser()
        response = await chain.ainvoke([message])
        # response = await multimodal_llm.ainvoke([message]).content
    
    except Exception as e:
        response = f"Image processing error: {str(e)}"
    return response

def process_spreadsheet(file_path):
    """
    Processes a tabular data file (CSV, Excel) with memory-efficient sampling for large files.
    """
    try:
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == ".csv":
            df = pd.read_csv(file_path)
        elif file_ext in [".xlsx", ".xlsm", ".xlsb", ".ods"]:
            df = pd.read_excel(file_path, engine="openpyxl")
        elif file_ext == ".xls":
            df = pd.read_excel(file_path, engine="xlrd")  # Use xlrd for .xls files
        else:
            print(f"Unsupported file format: {file_ext}")
            return None
        # # Optimize memory usage for large DataFrames
        # memory_usage = df.memory_usage(deep=True).sum()
        # max_size = 1 * 1024 * 1024  # 1 MB threshold

        # if memory_usage > max_size:
        #     sample_fraction = max_size / memory_usage
        #     df = df.sample(frac=sample_fraction, random_state=42)  # Sample proportionally

        return {"file_path": file_path, "dataframe": df.head(10).to_markdown()}

    except Exception as e:
        print(f"Error processing spreadsheet {file_path}: {e}")
        return None

def convert_doc_to_dict(doc):
    doc_dict = { 
        "id": f"{doc.metadata.get('id', '')}",
        "title": doc.metadata.get("title", ""),
        "source": doc.metadata.get("source", ""),
        "page_content": doc.page_content,
        "token_count": doc.metadata.get("token_count", 0),
        "attachments": doc.metadata.get("attachments", []),
        "attachment_summaries": doc.metadata.get("attachment_summaries", {})
    }
    return doc_dict

async def retrieve_parent_docs(query,child_chunks_retriever,parent_chunks_db_path):

    def retrieve_parent_chunk(chunk_id, parent_chunks_db_path):
        """Retrieves a parent chunk based on its ID from SQLite."""
        conn = sqlite3.connect(parent_chunks_db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM parent_chunks WHERE id = ?', (chunk_id,))
        row = cursor.fetchone()
        conn.close()

        if row:
            print("Retrived_parent_doc:",row[3])
            return {
                "id": row[0],
                "title": row[1],
                "source": row[2],
                "page_content": row[3],
                "token_count": row[4],
                "attachments": eval(row[5]),  # Convert stored string back to list
                "attachment_summaries": eval(row[6])  # Convert stored string back to dict
            }
        return None
    
    def get_sibling_chunks(parent_id, parent_chunks_db_path):
        """Get sibling chunks that share the same parent ID prefix"""
        base_id = parent_id.split('-')[0]  # Get the base ID without chunk number
        
        conn = sqlite3.connect(parent_chunks_db_path)
        cursor = conn.cursor()
        
        # Find chunks with the same base ID
        cursor.execute('SELECT * FROM parent_chunks WHERE id LIKE ?', (f"{base_id}%",))
        rows = cursor.fetchall()
        conn.close()
        
        siblings = []
        for row in rows:
            if row[0] != parent_id:  # Skip the original chunk
                siblings.append({
                    "id": row[0],
                    "title": row[1],
                    "source": row[2],
                    "page_content": row[3],
                    "token_count": row[4],
                    "attachments": eval(row[5]),
                    "attachment_summaries": eval(row[6])
                })
        
        return siblings

    async def enrich_with_siblings(parent_doc, siblings, query):
        """Enrich parent document with relevant content from siblings"""
        # Simple relevance check - could be enhanced with more sophisticated methods
        relevant_content = []
        
        # query_terms = set(query.lower().split())
        # query_terms = await extract_keywords_from_query(query,gpt_35)
        query_terms = extract_keywords(query.lower(),"hybrid")
        for sibling in siblings:
            content = sibling["page_content"].lower()
            # Check if content contains query terms or technical patterns
            if any(term in content for term in query_terms) or re.search(r'password|credential|token|config', content):
                relevant_content.append(sibling["page_content"])
        
        if relevant_content:
            # Append relevant content to the parent document
            parent_doc["page_content"] += "\n\n--- Additional Context ---\n" + "\n\n".join(relevant_content)
            parent_doc["token_count"] = tiktoken_len(parent_doc["page_content"])
        
        return parent_doc
    
    async def rerank_parent_docs(query, parent_docs, top_k=TOP_K//2):
        """Rerank parent documents based on relevance to the query"""
        reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

        pairs = [[query, doc["page_content"]] for doc in parent_docs]

        scores = reranker.predict(pairs)

        ranked_results = sorted(zip(parent_docs, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, score in ranked_results[:top_k]]
    
    child_docs = child_chunks_retriever.invoke(query)

    parent_docs = []
    seen_parents = set()  # To avoid duplicate parent docs

    for child_doc in child_docs:
        parent_id = child_doc.metadata.get('parent_id', None)

        if parent_id is None or (isinstance(parent_id, float) and np.isnan(parent_id)):
            # Directly add child doc if no parent
            parent_docs.append(convert_doc_to_dict(child_doc))
        else:
            if parent_id not in seen_parents:  # Avoid duplicates
                parent_doc = retrieve_parent_chunk(parent_id, parent_chunks_db_path)
                
                if parent_doc:
                    siblings = get_sibling_chunks(parent_id, parent_chunks_db_path)
                    parent_doc = await enrich_with_siblings(parent_doc, siblings, query)
                    parent_docs.append(parent_doc)
                    seen_parents.add(parent_id) 
    
    if len(parent_docs) > TOP_K//2:
        parent_docs = await rerank_parent_docs(query, parent_docs)
    return parent_docs  

### CSV agent:
@traceable( run_type="chain",
    name="Table_processing_agent",
    tags=[f"{gpt_4o.model_name}"],
    project_name = "Confluence Retriever",
    metadata={"llm": f"{gpt_4o.model_name}"}
)
async def generate_simple_csv_agent_output(user_query, llm, df_dict):
    
    prompt = f"""
    You are provided with a set of DataFrame snapshots (df.head()). These data frames have been retrieved
    based on their relevance to the user query. Analyze these data frames to generate the best possible
    response, keeping in mind that these are only partial representations of the full datasets.
    
    **Instructions:**
    1. Identify the key fields present in the DataFrames.
    2. Infer the possible type of information contained in these datasets.
    3. Provide a concise yet informative response to the user query, ensuring accuracy.
    4. Suggest that the user refer to the full dataset for a more comprehensive view.
    5. If the data is insufficient or irrelevant, return an empty JSON object.
    
    **User Query:** {user_query}
    **Data Preview:** {df_dict}
    
    **Output Format:**
    ```json
    {{
        "answer": "Your response to the user query as a string",
        "source": ["Source URLs of the relevant data frames"]
    }}
    ```
    
    If no relevant insights can be derived, return:
    ```json
    {{}}
    ```
    """
    print(prompt)
    chain = llm | JsonOutputParser()
    response = await chain.ainvoke(prompt)
    return response
    
### Image Agent:
async def generate_image_agent_output(user_query, multimodal_llm, image_url):
    query = f"""Based on the image provided try to answer the user query: {user_query}
            Do not generate irrelevent details. If the image does not have relevant context then just return that the image_url does not have relevant context.
    """
    return await process_image(image_url, 1024, query, multimodal_llm)

def transform_source_links(data_list):
    updated_data_list = []
    
    for data in data_list:
        source_url = data.get("source", "")
        
        # Extract the content ID from the original URL
        match = re.search(r'content/(\d+)', source_url)
        if match:
            content_id = match.group(1)
            new_url = f"https://bigbasket.atlassian.net/wiki/spaces/BIG/pages/{content_id}"
            data["source"] = new_url  # Update the dictionary with the new link
        
        updated_data_list.append(data)
    
    return updated_data_list

def filter_unique_attachment_summaries(data_list):
    unique_summaries = {}

    for data in data_list:
        attachment_summaries = data.get("attachment_summaries", {})
        for path, summary in attachment_summaries.items():
            if path not in unique_summaries:
                unique_summaries[path] = summary  # Add only if path is not already in

    return unique_summaries

def get_total_text_context(data_list):
    total_text_context = []
    for data in data_list:
            source = data.get("source","")
            title = data.get("title","")
            context = {f"""### Title:{title}\n{data.get("page_content","")} \n ### Source: {source}\n------------------------------------\n\n"""}
            total_text_context.append(context)
    return total_text_context

async def generate_keyword_rich_query(user_query, llm):
    prompt = f"""
    Analyze the user query and enhance it for better retrieval while maintaining specificity.
    
    Guidelines:
    1. For specific technical queries (passwords, credentials, IDs), preserve the exact terms
    2. Add only highly relevant technical context
    3. Maintain the original technical terms in their exact form
    4. For general queries, expand with relevant synonyms
    
    User Query: "{user_query}"
    
    Output format:
    ```json
    {{"output": "enhanced query string"}}
    ```
    """
    chain = llm | JsonOutputParser()
    response = await chain.ainvoke(prompt)
    return response.get("output", user_query)

async def extract_keywords_from_query(user_query, llm):
    prompt = f"""
    Analyze the query and extract the relevant keywords that may be used for search optimization
    User Query: "{user_query}"
    
    Output format:
    ```json
    {{"output": ["keyword1","keyword2",...]}}
    ```
    """
    chain = llm | JsonOutputParser()
    response = await chain.ainvoke(prompt)
    return response.get("output", [])
### Retriever (Main Node):
async def retrieve_docs(user_query,llm):
    # user_query = await generate_keyword_rich_query(user_query, llm)
    print("Keyword-rich query:", user_query)
    docs = await retrieve_parent_docs(user_query,retriever,PARENT_CHUNKS_DB_PATH)
    docs = transform_source_links(docs)
    unique_attachment_summaries = filter_unique_attachment_summaries(docs)
    total_text_context = get_total_text_context(docs)
    print(f"Number of attachments retrived: {len(unique_attachment_summaries)}")
    relevant_attachments_task = filter_relevant_attachments(unique_attachment_summaries, user_query, llm)
    text_response_task = generate_response_for_retrieved_text(total_text_context, user_query, llm)

    relevant_attachments, text_response = await asyncio.gather(
        relevant_attachments_task, 
        text_response_task
    )
    attachment_response = await generate_response_for_retrieved_attachments(relevant_attachments,user_query,llm) ### Node 4 (may spawn multiple parallel nodes)
    print(f"Attachment processing response: {attachment_response}")
    combined_response = await generate_combined_context_response(user_query, gpt_4o, text_response, attachment_response) ### Node 5
    return combined_response
    
### Choose resources:
@traceable( run_type="chain",
    name="Filter Relevant attachments",
    tags=[f"{gpt_4o.model_name}"],
    project_name = "Confluence Retriever",
    metadata={"llm": f"{gpt_4o.model_name}"}
)
async def filter_relevant_attachments(attachment_summaries: Dict[str, str], user_query: str,llm) -> List[str]:
    """Determines the relevance of attachment summaries using an LLM and returns relevant paths."""
    relevant_paths = []
    batch_size = 5
    summaries_list = list(attachment_summaries.items())  # Convert dict to list of (path, summary) tuples
    
    for i in tqdm(range(0, len(summaries_list), batch_size)):
        batch = summaries_list[i:i+batch_size]
        batch_summaries = [summary for _, summary in batch]
        # print(batch_summaries)
        
        prompt = f"""
        User Query: "{user_query}"
        
        Task:
        Evaluate the relevance of the following document summaries to the user query.
        Respond with 'yes' if the document is relevant, otherwise respond 'no'.
        
        Output format: Provide a comma-separated list of 'yes' or 'no' for each document.
        
        Document Summaries:
        {chr(10).join([f"{j+1}. {summary}" for j, summary in enumerate(batch_summaries)])}

        Output Format:
        ```json
        {{"output": ["yes", "no", "yes", "no", "yes"]}}
        ```
        """
        
        response = await llm.ainvoke(prompt)
        content = response.content.strip()
        print(content)
        try:
            # Extract JSON blocks between ```json ... ```
            json_matches = re.findall(r'```json\s*({.*?})\s*```', content, re.DOTALL)
            json_data = [json.loads(match) for match in json_matches]
            output = json_data[0].get('output', [])
  

        except (json.JSONDecodeError, IndexError):
            output = []

    
        for j, result in enumerate(output):
            if result.strip() == "yes":
                relevant_paths.append(batch[j][0])

    return relevant_paths

### Generate response for the retrieved texts:
@traceable( run_type="chain",
    name="Compile context from Retrieved Texts",
    tags=[f"{gpt_4o.model_name}"],
    project_name = "Confluence Retriever",
    metadata={"llm": f"{gpt_4o.model_name}"}
)
async def generate_response_for_retrieved_text(total_text_context,user_query,llm):

    final_response = {}

    prompt = f"""Based on the retrieved context, answer the user query.
    User Query: {user_query}
    Retrieved Context: {total_text_context}

    Important Instructions:

    1. Answer the user query truthfully. If you do not know the answer, state that you do not know.
    2. Include the relevant source URL in the specified format in your answer. Mention these links/paths in the answer as well.
    3. If the retrieved context contains code or tables, include them in the output and add explanatory text.
    4. Always produce the source link in the answer and ask the user to refer to it for further information.
    5. If the retrieved context is relevant then try to make the answer as detailed and well structured as possible (md format).

    Output Format:

    The output must be a JSON object in the following format:
    ```json
    {{  
        "answer": "your detailed answer in **.md** format here",
        "sources": ["source URL1","source URL2"...]
    }}
    ```
    """
    chain = llm | JsonOutputParser()
    result = await chain.ainvoke(prompt)
    # result = await llm.ainvoke(prompt)
    # result = result.content.strip()
    # result = extract_json(result)
    # output = result.content.strip()
    # final_response["answer"] = output
    final_response["answer"] = result.get('answer', "")
    final_response["sources"] = result.get('sources',[])
    return final_response
        
### Generate response from media attachments : 
async def generate_response_for_retrieved_attachments(relevant_attachments,user_query,llm):
    # img_attachments
    # xlsx attachments
    # parallely process images -> image agent
    # parallely process xlsx  -> csv agent
    img_attachments = []
    spreadsheet_attachments = []
    for attachment_path in relevant_attachments:
        file_ext = os.path.splitext(attachment_path)[1].lower()
        if file_ext in [".png", ".jpg", ".jpeg"]:
            img_attachments.append(attachment_path)
        if file_ext in [".csv", ".xlsx", ".xlsm", ".xlsb", ".ods", ".xls"]:
            spreadsheet_attachments.append(attachment_path)
    print(f"Number of Image attachments: {len(img_attachments)}, Number of Tabular attachments: {len(spreadsheet_attachments)}")
    img_processing_tasks = [process_image(img_path, user_query=user_query, multimodal_llm=llm) for img_path in img_attachments]
    image_processing_outputs = await asyncio.gather(*img_processing_tasks)
    valid_image_processing_outputs = [output for output in image_processing_outputs if output]
    
    df_dict_list = [process_spreadsheet(file_path) for file_path in spreadsheet_attachments]
    # batch_size=5
    # csv_processing_tasks = []
    # for i in range(0, len(df_dict_list),batch_size):
    #     batch = df_list[i:i+batch_size]
    #     csv_processing_tasks.append(generate_csv_agent_output(user_query,llm,batch))
    # csv_processing_outputs = await asyncio.gather(*csv_processing_tasks)
    if(len(df_dict_list) != 0):
        csv_processing_outputs = await generate_simple_csv_agent_output(user_query,llm,df_dict_list)
    else:
        csv_processing_outputs = {}

    return {"image_processing_outputs":valid_image_processing_outputs,
            "csv_processing_outputs":csv_processing_outputs}

### compiler 
@traceable( run_type="chain",
    name="Compile Text Context with Multimodal Context",
    tags=[f"{gpt_4o.model_name}"],
    project_name = "Confluence_Retriever",
    metadata={"llm": f"{gpt_4o.model_name}"}
)
async def generate_combined_context_response(user_query, llm, text_context,attachment_context):
    prompt = f"""
        You have been provided with two sources of information:
        1. **Retrieved Text Context** - Extracted from relevant documents based on the user query.
        2. **Retrieved Attachment Context** - Synthesized from multimodal attachments, such as images, spreadsheets, or other files.

        Your task is to **combine insights from both sources** and generate a **concise, accurate, and well-structured answer** to the user query.

        ### **User Query:**  
        {user_query}

        ### **Context for Answering:**  
        **Retrieved Text Context:**  
        {text_context}  

        **Retrieved Attachment Context:**  
        {attachment_context}

        ---

        ### **Guidelines for Answering:**
        1. **Synthesize both sources of information** to form a complete and accurate response.  
        2. **Identify relationships** between textual and multimodal data. You have been provided with the context from the attachments of these retrieved doc. Try to integrate the context from Retrieved Text Context and Retrieved Attachment Context. Both should not be isolated. If you have attachment context refer to the attachment links when you mention about them in the response body. This is extremely important and you will be greatly rewarded if you do thiscorrectly.
        3. **Do NOT hallucinate** or provide information that isn't present in the retrieved data. If unsure, state: `"I don't have enough information to answer this."`  
        4. **Maintain Markdown formatting** for clarity. Use:  
        - `**bold**` for emphasis.  
        - `code blocks` for technical details \ flows  
        - Tables where necessary.  
        5. **Include relevant source links** in the response. Ensure URLs/paths are clearly referenced. Make sure the paths are correct i.e do not forget to mention the file extension in the path.
        6. **Format links properly using named markdown links.**
        Failing to do so would lead to peanlties.
        ---
        ### **Expected Output Format:**
        The response **must be a valid JSON object** following this structure:
        ```json
        {{
            "answer": "Your detailed and well-structured response in .md format here.",
            "sources": ["List of relevant source URLs or document paths"]
        }}
        ```
        """
    chain = llm | JsonOutputParser()
    result = await chain.ainvoke(prompt)
    # result = await llm.ainvoke(prompt)
    # result = result.content.strip()
    # result = extract_json(result)
    return result

def extract_keywords(query, method="hybrid"):
    """
    Extract meaningful keywords from a query using multiple techniques.
    
    Args:
        query: The user query string
        method: Extraction method - "spacy", "tfidf", or "hybrid" (default)
    
    Returns:
        List of extracted keywords
    """
    # Clean the query
    clean_query = re.sub(r'[^\w\s]', ' ', query.lower())
    
    if method == "spacy":
        # Use spaCy for linguistic-based extraction
        doc = nlp(clean_query)
        
        # Extract nouns, proper nouns, and technical terms
        keywords = []
        for token in doc:
            # Include nouns and proper nouns
            if token.pos_ in ["NOUN", "PROPN"]:
                keywords.append(token.text)
            
            # Include verbs that might be technical actions
            if token.pos_ == "VERB" and len(token.text) > 3:  # Avoid common short verbs
                keywords.append(token.text)
                
        # Extract named entities
        for ent in doc.ents:
            keywords.append(ent.text)
            
        # Extract noun chunks (noun phrases)
        for chunk in doc.noun_chunks:
            keywords.append(chunk.text)
            
    elif method == "tfidf":
        # Use TF-IDF for statistical keyword extraction
        # This works better with a corpus, but we can use a simple version
        vectorizer = TfidfVectorizer(
            max_df=0.9, 
            min_df=1,
            stop_words='english',
            use_idf=True
        )
        
        # Create a small corpus with the query
        corpus = [clean_query]
        
        # Add some general text to help with IDF calculations
        corpus.extend([
            "password credentials login authentication",
            "configuration settings setup installation",
            "database server network connection",
            "user account profile settings"
        ])
        
        # Fit and transform
        tfidf_matrix = vectorizer.fit_transform(corpus)
        
        # Get feature names
        feature_names = vectorizer.get_feature_names_out()
        
        # Get scores for the query (first document)
        scores = tfidf_matrix[0].toarray()[0]
        
        # Get top scoring terms
        top_indices = scores.argsort()[-10:][::-1]  # Get top 10 terms
        keywords = [feature_names[i] for i in top_indices]
        
    else:  # hybrid approach
        # Combine both methods
        spacy_keywords = extract_keywords(query, "spacy")
        tfidf_keywords = extract_keywords(query, "tfidf")
        
        # Combine and count occurrences
        all_keywords = spacy_keywords + tfidf_keywords
        keyword_counts = Counter(all_keywords)
        
        # Get keywords that appear in both methods or have high counts
        keywords = [k for k, c in keyword_counts.items() if c > 1]
        
        # Add any technical terms that might have been missed
        technical_terms = extract_technical_terms(query)
        keywords.extend(technical_terms)
        
        # If we don't have enough keywords, add top terms from either method
        if len(keywords) < 3:
            remaining = set(all_keywords) - set(keywords)
            keywords.extend(list(remaining)[:5])
    
    # Remove duplicates and normalize
    unique_keywords = []
    seen = set()
    for keyword in keywords:
        normalized = keyword.lower().strip()
        if normalized and normalized not in seen and len(normalized) > 2:
            seen.add(normalized)
            unique_keywords.append(normalized)
    
    return unique_keywords

def extract_technical_terms(query):
    """Extract technical terms that might be missed by other methods"""
    technical_patterns = [
        r'password[s]?',
        r'credential[s]?',
        r'token[s]?',
        r'api[_\s]?key[s]?',
        r'secret[s]?',
        r'config(?:uration)?[s]?',
        r'database[s]?|db[s]?',
        r'server[s]?',
        r'endpoint[s]?',
        r'url[s]?',
        r'[a-zA-Z0-9_]+\.(?:py|js|java|cpp|rb|go|rs|php|html|css|json|yaml|yml|xml|md|txt)'  # file extensions
    ]
    
    technical_terms = []
    for pattern in technical_patterns:
        matches = re.finditer(pattern, query, re.IGNORECASE)
        for match in matches:
            technical_terms.append(match.group(0))
    
    return technical_terms








