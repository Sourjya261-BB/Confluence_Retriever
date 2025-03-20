import re
import os
import sys
import spacy
from collections import Counter
from langchain_core.output_parsers import JsonOutputParser
from sklearn.feature_extraction.text import TfidfVectorizer
from scripts.commons import gpt_35

try:
    nlp = spacy.load("en_core_web_sm")
except:
    # Fallback if spaCy model isn't installed
    nlp = None
    print("Warning: spaCy model not loaded. Install with: python -m spacy download en_core_web_sm")

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
    
    if method == "spacy" and nlp is not None:
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

    elif method == "llm":
        keywords = extract_keywords_from_query_using_llm(query,gpt_35)

        
    else:  # hybrid approach or fallback if spaCy isn't available
        # If spaCy is available, get spaCy keywords
        spacy_keywords = extract_keywords(query, "spacy") if nlp is not None else []
        
        # Get TF-IDF keywords
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

def extract_relevant_section(content, keywords):
    """Extract the most relevant section from content based on keywords"""
    # If content is short, return it all
    if len(content) < 500:
        return content
    
    # Split into paragraphs
    paragraphs = re.split(r'\n\s*\n', content)
    
    # Score each paragraph
    scored_paragraphs = []
    for para in paragraphs:
        if not para.strip():
            continue
            
        score = 0
        for keyword in keywords:
            if keyword.lower() in para.lower():
                score += 1
        
        # Bonus for paragraphs with technical content
        if re.search(r'password|credential|token|config|secret|api[_\s]?key', para, re.IGNORECASE):
            score += 2
            
        # Bonus for code blocks
        if re.search(r'```|`.*`|\{.*\}|\[.*\]', para):
            score += 2
            
        if score > 0:
            scored_paragraphs.append((para, score))
    
    # Sort by score
    scored_paragraphs.sort(key=lambda x: x[1], reverse=True)
    
    # Return top paragraphs (up to a reasonable length)
    result = []
    total_length = 0
    for para, _ in scored_paragraphs:
        if total_length + len(para) <= 1000:  # Limit to ~1000 chars
            result.append(para)
            total_length += len(para)
        else:
            break
            
    return "\n\n".join(result) if result else ""

def extract_keywords_from_query_using_llm(user_query, llm):
    """
    LLM chain to extract keywords from the user query
    Args: user_query (str)
    Output: List(str)
    """
    prompt = f"""
    Analyze the query and extract the relevant keywords that may be used for search optimization
    User Query: "{user_query}"
    Output format:
    ```json
    {{"output": ["keyword1","keyword2",...]}}
    ```
    """
    chain = llm | JsonOutputParser()
    response = chain.invoke(prompt)
    return response.get("output", [])