import tiktoken

def tiktoken_len(text):
    """Calculate the token length of a text using tiktoken."""
    try:
        tokenizer = tiktoken.get_encoding("cl100k_base")  # GPT-4 tokenizer
        tokens = tokenizer.encode(text)
        return len(tokens)
    except:
        # Fallback if tiktoken is not available
        return len(text) // 4  # Rough approximation

def convert_doc_to_dict(doc):
    """Convert a Document object to a dictionary."""
    return {
        "id": doc.metadata.get("id", "unknown"),
        "title": doc.metadata.get("title", ""),
        "source": doc.metadata.get("source", ""),
        "page_content": doc.page_content,
        "token_count": tiktoken_len(doc.page_content),
        "attachments": doc.metadata.get("attachments", []),
        "attachment_summaries": doc.metadata.get("attachment_summaries", []),
        "metadata": doc.metadata
    }

def transform_source_links(docs):
    """Transform source links in documents for better display."""
    for doc in docs:
        if "source" in doc and doc["source"]:
            # Clean up source URLs or paths
            source = doc["source"]
            # Add any source transformation logic here
            doc["source"] = source
    return docs