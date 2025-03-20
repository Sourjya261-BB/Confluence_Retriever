import re
import sqlite3
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from retrieval_helpers.keyword_extraction_utils import extract_keywords, extract_relevant_section
from retrieval_helpers.common_utils import tiktoken_len, convert_doc_to_dict

class ComplexParentDocRetriever:
    """Class for retrieving and enriching parent documents."""
    
    def __init__(self, parent_chunks_db_path: str, top_k: int, keyword_extraction_mode: str):
        """
        Initialize the parent document retriever.
        
        Args:
            parent_chunks_db_path: Path to the SQLite database containing parent chunks
        """
        self.parent_chunks_db_path = parent_chunks_db_path
        self.top_k = top_k
        self.keyword_extraction_mode = keyword_extraction_mode
    
    def retrieve_parent_docs(self, query: str, child_chunks_retriever, rerank: bool = True) -> List[Dict[str, Any]]:
        """
        Retrieve parent documents based on child chunks and enrich with relevant siblings.
        
        Args:
            query: The user query
            child_chunks_retriever: Retriever for child chunks
            rerank: Whether to rerank the results
            
        Returns:
            List of parent documents
        """
        # Extract meaningful keywords from the query for better filtering
        keywords = extract_keywords(query,self.keyword_extraction_mode)
        print(f"Extracted keywords for retrieval: {keywords}")
        
        # Get initial child documents
        child_docs = child_chunks_retriever.invoke(query)
        
        # Process parent documents
        parent_docs = []
        seen_parents = set()
        
        for child_doc in child_docs:
            parent_id = child_doc.metadata.get('parent_id', None)
            
            if parent_id is None or (isinstance(parent_id, float) and np.isnan(parent_id)):
                # Handle documents without parents
                parent_docs.append(convert_doc_to_dict(child_doc))
            else:
                if parent_id not in seen_parents:
                    parent_doc = self.retrieve_parent_chunk(parent_id)
                    if parent_doc:
                        # Get sibling chunks for context expansion using extracted keywords
                        sibling_chunks = self.get_sibling_chunks(parent_id)
                        
                        # Enrich parent document with relevant sibling content
                        if sibling_chunks:
                            parent_doc = self.enrich_with_siblings(parent_doc, sibling_chunks, query, keywords)
                        
                        parent_docs.append(parent_doc)
                        seen_parents.add(parent_id)
        
        # If we have enough parent docs and reranking is enabled, apply reranking
        if rerank and len(parent_docs) > 3:
            reranked_parent_docs = self.rerank_parent_docs(query, parent_docs, keywords)
            return reranked_parent_docs
        
        return parent_docs
    
    def retrieve_parent_chunk(self, parent_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a parent chunk by ID from the SQLite database.
        
        Args:
            parent_id: ID of the parent chunk
            
        Returns:
            Parent document dictionary or None if not found
        """
        conn = sqlite3.connect(self.parent_chunks_db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM parent_chunks WHERE id = ?', (parent_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                "id": row[0],
                "title": row[1],
                "source": row[2],
                "page_content": row[3],
                "token_count": row[4],
                "attachments": eval(row[5]),
                "attachment_summaries": eval(row[6])
            }
        return None
    
    def get_sibling_chunks(self, parent_id: str) -> List[Dict[str, Any]]:
        """
        Get sibling chunks that share the same parent document.
        
        Args:
            parent_id: ID of the parent chunk
            
        Returns:
            List of sibling documents
        """
        # Extract document ID from the chunk ID (assuming format like "doc123-chunk4")
        doc_id_match = re.match(r'(.*?)(?:-chunk\d+)?$', parent_id)
        doc_id = doc_id_match.group(1) if doc_id_match else parent_id.split('-')[0]
        
        conn = sqlite3.connect(self.parent_chunks_db_path)
        cursor = conn.cursor()
        
        # Find chunks with the same document ID but different from the parent
        cursor.execute('SELECT * FROM parent_chunks WHERE id LIKE ? AND id != ?', 
                      (f"{doc_id}%", parent_id))
        rows = cursor.fetchall()
        conn.close()
        
        siblings = []
        for row in rows:
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
    
    def enrich_with_siblings(self, parent_doc: Dict[str, Any], siblings: List[Dict[str, Any]], 
                            query: str, keywords: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Enrich parent document with relevant content from siblings using extracted keywords.
        
        Args:
            parent_doc: The parent document to enrich
            siblings: List of sibling documents
            query: The original query
            keywords: Pre-extracted keywords (optional)
            
        Returns:
            Enriched parent document
        """
        if not siblings:
            return parent_doc
        
        # Use provided keywords or extract them if not provided
        if keywords is None:
            keywords = extract_keywords(query,self.keyword_extraction_mode)
        
        # Technical patterns to look for
        technical_patterns = [
            r'password[s]?[\s]*[=:]+[\s]*[^\s]+',  # Password assignments
            r'credential[s]?[\s]*[=:]+',           # Credential assignments
            r'token[s]?[\s]*[=:]+',                # Token assignments
            r'api[_\s]?key[s]?[\s]*[=:]+',         # API key assignments
            r'secret[s]?[\s]*[=:]+',               # Secret assignments
            r'config(?:uration)?[\s]*[=:{\[]',     # Configuration blocks
            r'connection[\s]*string[\s]*[=:]+',    # Connection strings
            r'```[a-z]*\n',                        # Code blocks
            r'<[^>]+>',                            # HTML/XML tags
        ]
        
        # Score siblings based on keyword matches and technical patterns
        scored_siblings = []
        for sibling in siblings:
            content = sibling["page_content"].lower()
            
            # Base score
            score = 0
            
            # Score based on keyword matches
            for keyword in keywords:
                if keyword.lower() in content:
                    # Increase score based on how many times the keyword appears
                    occurrences = content.count(keyword.lower())
                    score += min(occurrences, 3)  # Cap at 3 to avoid over-weighting
            
            # Bonus for technical pattern matches
            for pattern in technical_patterns:
                if re.search(pattern, sibling["page_content"], re.IGNORECASE):
                    score += 5  # Higher weight for technical patterns
            
            # Bonus for siblings that are close to the parent in document order
            if "id" in sibling and "id" in parent_doc:
                try:
                    # Extract chunk numbers if they exist
                    parent_chunk_match = re.search(r'-chunk(\d+)$', parent_doc["id"])
                    sibling_chunk_match = re.search(r'-chunk(\d+)$', sibling["id"])
                    
                    if parent_chunk_match and sibling_chunk_match:
                        parent_chunk_num = int(parent_chunk_match.group(1))
                        sibling_chunk_num = int(sibling_chunk_match.group(1))
                        proximity = abs(parent_chunk_num - sibling_chunk_num)
                        
                        if proximity <= 1:
                            score += 2  # Adjacent chunks
                        elif proximity <= 3:
                            score += 1  # Nearby chunks
                except (ValueError, IndexError, AttributeError):
                    pass  # Skip if chunk IDs don't follow expected format
            
            # Only include siblings with a minimum relevance score
            if score > 0:
                scored_siblings.append((sibling, score))
        
        # Sort by score in descending order
        scored_siblings.sort(key=lambda x: x[1], reverse=True)
        
        # Take top N siblings
        top_siblings = [s[0] for s in scored_siblings[:3]]  # Adjust number as needed
        
        # Add relevant content from top siblings
        if top_siblings:
            relevant_content = []
            for sibling in top_siblings:
                # Extract the most relevant section from the sibling
                relevant_section = extract_relevant_section(sibling["page_content"], keywords)
                if relevant_section:
                    # Add source information
                    source_info = f"From {sibling['title']}" if 'title' in sibling and sibling['title'] else f"From document section"
                    relevant_content.append(f"--- {source_info} ---\n{relevant_section}")
            
            if relevant_content:
                # Append relevant content to the parent document
                parent_doc["page_content"] += "\n\n--- Additional Context ---\n" + "\n\n".join(relevant_content)
                parent_doc["token_count"] = tiktoken_len(parent_doc["page_content"])
                
                # Add metadata to indicate enrichment
                if "metadata" not in parent_doc:
                    parent_doc["metadata"] = {}
                parent_doc["metadata"]["enriched_with_siblings"] = True
                parent_doc["metadata"]["sibling_count"] = len(top_siblings)
        
        return parent_doc
    
    def rerank_parent_docs(self, query: str, parent_docs: List[Dict[str, Any]], 
                          keywords: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Rerank parent documents based on relevance to the query and extracted keywords.
        
        Args:
            query: The user query
            parent_docs: List of parent documents to rerank
            keywords: Pre-extracted keywords (optional)
            top_k: Number of top documents to return
            
        Returns:
            Reranked list of parent documents
        """
        # If we have a reranker model available
        try:
            from sentence_transformers import CrossEncoder
            reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            
            # Create pairs of query and document content
            pairs = [[query, doc["page_content"]] for doc in parent_docs]
            
            # Get relevance scores
            scores = reranker.predict(pairs)
            
            # Sort by score and return top_k
            ranked_results = sorted(zip(parent_docs, scores), key=lambda x: x[1], reverse=True)
            return [doc for doc, score in ranked_results[:self.top_k]]
            
        except (ImportError, Exception) as e:
            print(f"Reranker error: {e}. Falling back to keyword-based ranking.")
            # Fall back to keyword-based ranking if reranker is not available
            return self.keyword_based_reranking(query, parent_docs, keywords)
    
    def keyword_based_reranking(self, query: str, parent_docs: List[Dict[str, Any]], 
                               keywords: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Rerank parent documents based on keyword matching when a neural reranker is not available.
        
        Args:
            query: The user query
            parent_docs: List of parent documents to rerank
            keywords: Pre-extracted keywords (optional)
            top_k: Number of top documents to return
            
        Returns:
            Reranked list of parent documents
        """
        # Use provided keywords or extract them
        if keywords is None:
            keywords = extract_keywords(query,self.keyword_extraction_mode)
        
        # Score documents based on keyword matches
        scored_docs = []
        for doc in parent_docs:
            content = doc["page_content"].lower()
            
            # Base score
            score = 0
            
            # Score based on keyword matches
            for keyword in keywords:
                if keyword.lower() in content:
                    # Increase score based on how many times the keyword appears
                    occurrences = content.count(keyword.lower())
                    score += min(occurrences, 3)  # Cap at 3 to avoid over-weighting
            
            # Bonus for technical content
            if re.search(r'password|credential|token|config|secret|api[_\s]?key', content, re.IGNORECASE):
                score += 3
                
            # Bonus for code blocks
            if re.search(r'```|`.*`|\{.*\}|\[.*\]', content):
                score += 2
                
            scored_docs.append((doc, score))
        
        # Sort by score
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Return top_k documents
        return [doc for doc, _ in scored_docs[:self.top_k]]
    

class SimpleParentDocRetriever:
    def __init__(self,parent_chunks_db_path, top_k, keyword_extraction_mode) :
        self.parent_chunks_db_path = parent_chunks_db_path
        self.top_k = top_k
        self.keyword_extraction_mode = keyword_extraction_mode

    def retrieve_parent_chunk(self, chunk_id):
        """Retrieves a parent chunk based on its ID from SQLite."""
        conn = sqlite3.connect(self.parent_chunks_db_path)
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
    
    def get_sibling_chunks(self,parent_id):
        """Get sibling chunks that share the same parent ID prefix"""
        base_id = parent_id.split('-')[0]  # Get the base ID without chunk number
        
        conn = sqlite3.connect(self.parent_chunks_db_path)
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

    def enrich_with_siblings(self,parent_doc, siblings, query):
        """Enrich parent document with relevant content from siblings"""
        # Simple relevance check - could be enhanced with more sophisticated methods
        relevant_content = []
        
        # query_terms = set(query.lower().split())
        # query_terms = await extract_keywords_from_query(query,gpt_35)
        query_terms = extract_keywords(query.lower(),self.keyword_extraction_mode)
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
    
    def rerank_parent_docs(self,query, parent_docs):
        """Rerank parent documents based on relevance to the query"""
        try:
            from sentence_transformers import CrossEncoder
            reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        except Exception as e:
            print(f"Reranker error: {e}. Returning original parent docs")
            return parent_docs

        pairs = [[query, doc["page_content"]] for doc in parent_docs]

        scores = reranker.predict(pairs)

        ranked_results = sorted(zip(parent_docs, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, score in ranked_results[:self.top_k]]
    
    def retrieve_parent_docs(self,query,child_chunks_retriever):
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
                    parent_doc = self.retrieve_parent_chunk(parent_id)
                    
                    if parent_doc:
                        siblings = self.get_sibling_chunks(parent_id)
                        parent_doc = self.enrich_with_siblings(parent_doc, siblings, query)
                        parent_docs.append(parent_doc)
                        seen_parents.add(parent_id) 
        
        if len(parent_docs) > 3:
            parent_docs = self.rerank_parent_docs(query, parent_docs)
        return parent_docs  
        
          