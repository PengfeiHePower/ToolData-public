"""
Simple Tool Retriever Functions

Two functions: load_retriever_model() and retrieve_tools()
"""

from typing import List, Dict, Any, Tuple
import pickle
import os
import json

try:
    from sentence_transformers import SentenceTransformer, util
    import torch
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

def load_retriever_model(emb_model_dir: str,model_name: str = "all-MiniLM-L6-v2"):
    """
    Load a sentence transformer model for tool retrieval
    
    Args:
        model_name: Name of the sentence transformer model
        
    Returns:
        Loaded SentenceTransformer model
    """
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        raise ImportError("sentence-transformers package is required. Install with: pip install sentence-transformers")
    
    if model_name == "ToolBench_IR":
        return SentenceTransformer(os.path.join(emb_model_dir, "ToolBench_IR_bert_based_uncased"))
    elif model_name == "bge-large":
        return SentenceTransformer(os.path.join(emb_model_dir, "bge-large-en-v1.5"))
    elif model_name == "all-MiniLM":
        return SentenceTransformer(os.path.join(emb_model_dir, "all-MiniLM-L6-v2"))
    else:
        supported_models = ["ToolBench_IR", "bge-large", "all-MiniLM"]
        raise ValueError(f"Embedding model '{model_name}' is not supported. Supported models: {supported_models}")



def load_encoded_tools(tools: List[Dict[str, Any]], domain_name: str, emb_model_name: str, base_data_dir: str, tool_args: List = ['tool description']) -> Tuple[List[Dict[str, Any]], any]:
    """
    Load embeddings for tools. If cached embeddings exist, load them. Otherwise encode and save.
    
    Args:
        tools: List of tool dictionaries (loaded from JSON externally)
        domain_name: Name of the domain (e.g., "Finance", "Travel") 
        emb_model_name: Name of the embedding model (e.g., "all-MiniLM")
        base_data_dir: Base directory for data
        
    Returns:
        Tuple of (tools, embeddings)
    """
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        raise ImportError("sentence-transformers package is required")
    
    # Create cache directory and file paths  
    cache_dir = os.path.join(base_data_dir, domain_name, emb_model_name.replace("/", "_"))
    embeddings_file = os.path.join(cache_dir, "embeddings.pt")
    
    # Check if cached embeddings exist
    if os.path.exists(embeddings_file):
        print(f"Loading cached embeddings for {domain_name} with {emb_model_name}")
        embeddings = torch.load(embeddings_file)
        print(f"Loaded {len(tools)} tools and cached embeddings")
        return tools, embeddings
    
    # Cache doesn't exist, need to encode
    print(f"No cache found. Encoding {len(tools)} tools for {domain_name} with {emb_model_name}")
    
    # Load model and encode
    model = load_retriever_model(emb_model_name)
    
    def _get_tool_text(tool: Dict[str, Any], tool_args: List) -> str:
        """Extract searchable text from tool"""
        parts = []
        
        for element in tool_args:
            parts.append(tool[element])
        
        return " ".join(parts)
    
    print(f"Encoding {len(tools)} tools...")
    tool_texts = [_get_tool_text(tool, tool_args) for tool in tools]
    embeddings = model.encode(tool_texts, convert_to_tensor=True)
    
    # Save embeddings to cache
    os.makedirs(cache_dir, exist_ok=True)
    torch.save(embeddings, embeddings_file)
    print(f"Saved embeddings to cache: {cache_dir}")
    
    return tools, embeddings


def retrieve_tools(model, query: str, tools: List[Dict[str, Any]], embeddings, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Retrieve relevant tools using pre-computed embeddings
    
    Args:
        model: SentenceTransformer model (from load_retriever_model)
        query: Query string
        tools: List of tool dictionaries (from load_encoded_tools)
        embeddings: Pre-computed tool embeddings (from load_encoded_tools)
        top_k: Number of tools to return
        
    Returns:
        List of selected tools with similarity scores
    """
    # Create query embedding and search
    query_embedding = model.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, embeddings, top_k=top_k)
    
    # Return selected tools with scores
    results = []
    for hit in hits[0]:
        tool = tools[hit['corpus_id']].copy()
        tool['score'] = float(hit['score'])
        results.append(tool)
    
    return results