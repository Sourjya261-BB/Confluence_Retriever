# Define chunk sizes
# Reminder these are not token lengths instead they are the number of characters
PARENT_CHUNK_SIZE = 1200
CHILD_CHUNK_SIZE = 400
#options include "complex_parent" and "simple_parent" for parent-child , "child" for child only retriver

SOS = False
AUGMENTED_CHUNKING=True

if AUGMENTED_CHUNKING:
    RETRIEVER_MECHANISM="child"
else:
    RETRIEVER_MECHANISM = "complex_parent"

# "bm25": for pure keyword search or "hybrid": for keyword+semantic search or "dense" : semantic search 
# RETRIEVAL_MODE = "hybrid" 
RETRIEVAL_MODE = "hybrid"
# Paths
MD_FOLDER_PATH = "./md_output"

if not SOS:
    PARENT_CHUNKS_DB_PATH = f"parent_chunks_{PARENT_CHUNK_SIZE}.db"

    if not AUGMENTED_CHUNKING:
        CHILD_CHUNKS_CSV_FILE_PATH = f"./child_chunks_{CHILD_CHUNK_SIZE}_with_embeddings.csv"
        CLEANED_CHILD_CHUNKS_CSV_FILE_PATH = f"./cleaned_child_chunks_{CHILD_CHUNK_SIZE}_with_embeddings.csv"
        COLLECTION_NAME = f"confluence_child_retriever_{CHILD_CHUNK_SIZE}"
    
    else:
        CHILD_CHUNKS_CSV_FILE_PATH = f"./child_chunks_{CHILD_CHUNK_SIZE}_with_embeddings_augmented.csv"
        CLEANED_CHILD_CHUNKS_CSV_FILE_PATH = f"./cleaned_child_chunks_{CHILD_CHUNK_SIZE}_with_embeddings_augmented.csv"
        COLLECTION_NAME = f"confluence_child_retriever_{CHILD_CHUNK_SIZE}_augmented"

else:

    PARENT_CHUNKS_DB_PATH = f"parent_chunks_{PARENT_CHUNK_SIZE}_SOS.db"

    if not AUGMENTED_CHUNKING:
        CHILD_CHUNKS_CSV_FILE_PATH = f"./child_chunks_{CHILD_CHUNK_SIZE}_with_embeddings_SOS.csv"
        CLEANED_CHILD_CHUNKS_CSV_FILE_PATH = f"./cleaned_child_chunks_{CHILD_CHUNK_SIZE}_with_embeddings_SOS.csv"
        COLLECTION_NAME = f"confluence_child_retriever_{CHILD_CHUNK_SIZE}_SOS"
    else:
        CHILD_CHUNKS_CSV_FILE_PATH = f"./child_chunks_{CHILD_CHUNK_SIZE}_with_embeddings_SOS_augmented.csv"
        CLEANED_CHILD_CHUNKS_CSV_FILE_PATH = f"./cleaned_child_chunks_{CHILD_CHUNK_SIZE}_with_embeddings_SOS_augmented.csv"
        COLLECTION_NAME = f"confluence_child_retriever_{CHILD_CHUNK_SIZE}_SOS_augmented"


VECTORDB_PATH = "./confluence_db_v2.2" #2.2 is indexed fir the entire space.
TOP_K = 30
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
RERANKING_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
#"spacy","hybrid", "tfidf" and "llm"
KEYWORD_EXTRACTION_MODE = "llm" 

SEMANTIC_WEIGHT = 0.6
SPARSE_WEIGHT = 0.4
SOS_PATTERNS = ["sop", "sop resilience" "resilience", "resiliency", "rps", "jdvar", "dvar", "solr", "grafana", "traffic drop", "traffic-drop", "traffic_drop", "rate-limit", "rate limit", "rate_limit"]