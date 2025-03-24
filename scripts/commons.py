import os
import sys
import torch
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import VECTORDB_PATH, COLLECTION_NAME, EMBEDDING_MODEL_NAME, TOP_K, RETRIEVAL_MODE, SEMANTIC_WEIGHT, SPARSE_WEIGHT


torch.classes.__path__ = []


load_dotenv() 


AZURE_OPENAI_VERSION=os.environ.get("AZURE_OPENAI_VERSION")
AZURE_OPENAI_DEPLOYMENT=os.environ.get("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_ENDPOINT=os.environ.get("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY=os.environ.get("AZURE_OPENAI_KEY")
AZURE_OPENAI_DEPLOYMENT_GPT4=os.environ.get("AZURE_OPENAI_DEPLOYMENT_GPT4")

gpt_35 = AzureChatOpenAI(
    openai_api_version=AZURE_OPENAI_VERSION,
    azure_deployment=AZURE_OPENAI_DEPLOYMENT,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_KEY,
    temperature=0.75,
    )

gpt_4o = AzureChatOpenAI(
    openai_api_version=AZURE_OPENAI_VERSION,
    azure_deployment="gpt-4o",
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_KEY,
    temperature=0.5,
    )

from langchain_anthropic import ChatAnthropic

haiku = ChatAnthropic(
    model="claude-3-haiku-20240307",
    temperature=0.7
)


def convert_chroma_response_to_docs(chroma_response):
    """
    Converts Chroma's get() method response into a list of LangChain Document objects.
    
    Args:
        chroma_response (dict): The dictionary response from ChromaDB's get() method.
    
    Returns:
        List[Document]: A list of Document objects with `page_content` and `metadata`.
    """
    documents = chroma_response.get("documents", [])
    metadatas = chroma_response.get("metadatas", [])

    print(f"Number of documents for BM25_index: {len(documents)}")

    if len(documents) != len(metadatas):
        raise ValueError("Mismatch between number of documents and metadata entries.")

    return [Document(page_content=doc, metadata=meta) for doc, meta in zip(documents, metadatas)]

embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={"device": "cpu"})

def get_BM25_retriever(vectordb):
    chroma_response = vectordb.get()
    all_child_docs = convert_chroma_response_to_docs(chroma_response)
    bm25_retriever = BM25Retriever.from_documents(all_child_docs)
    bm25_retriever.k = TOP_K
    return bm25_retriever

def get_retriever():
    """Returns the retriever based on the configured retrieval type."""

    metadata_filter = {"search_priority": {"$eq": "SOS"}}

    vectordb = Chroma(persist_directory=VECTORDB_PATH, collection_name=COLLECTION_NAME, embedding_function=embedding_model)
    dense_retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={'k': TOP_K})
    try:
        bm25_retriever = get_BM25_retriever(vectordb)
    except ValueError as e:
        print(f"BM25 Retriever Error: {e}")
        bm25_retriever = None 

    if bm25_retriever:
        print("creating ensemble retriever")
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, dense_retriever], 
            weights=[SPARSE_WEIGHT, SEMANTIC_WEIGHT]
        )
    else:
        ensemble_retriever = dense_retriever  # Fallback to dense if BM25 fails

    if RETRIEVAL_MODE == "bm25":
        return bm25_retriever
    elif RETRIEVAL_MODE == "dense":
        return dense_retriever
    elif RETRIEVAL_MODE == "hybrid":
        print("returning ensemble retriever")
        return ensemble_retriever
    else:
        raise ValueError(f"Invalid retrieval_mode: {RETRIEVAL_MODE}")