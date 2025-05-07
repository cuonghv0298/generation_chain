from langchain_community.embeddings import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
import weaviate
from typing import Dict, List, Any, Optional
import os
from dotenv import load_dotenv
load_dotenv()

WEAVIATE_HOST = "WEAVIATE_HOST"
def get_client():
    try:
        client = weaviate.connect_to_custom(
            http_host= os.environ.get("WEAVIATE_HOST"), #"10.100.224.34",  # URL only, no http prefix
            http_port= os.environ.get("WEAVIATE_HOST_PORT"), #"8080",
            http_secure=False ,   # Set to True if https
            grpc_host= os.environ.get("WEAVIATE_GPC_URL"),#  "10.100.224.34",
            grpc_port= os.environ.get("WEAVIATE_GPC_URL_PORT"),#"50051",      # Default is 50051, WCD uses 443
            grpc_secure=False,   # Edit as needed
            skip_init_checks=True,
        )
    except:
        client = weaviate.connect_to_local()
    return client
    # return weaviate.connect_to_local()


def get_embed_model_vicuna(
        embedding_model: Any = OllamaEmbeddings,
        embedding_model_params: Optional[Dict[str, Any]] = {
            'model': 'gemma:2b', 'show_progress': 'True'},
) -> Any:
    return embedding_model(**embedding_model_params)


def get_embed_model_openAI(
    embedding_model: Any = OpenAIEmbeddings,
        embedding_model_params: Optional[Dict[str, Any]] = {'openai_api_key': os.environ.get(
            'OPENAI_API_KEY'), "model": "text-embedding-3-small", 'show_progress_bar': True},
) -> Any:
    return embedding_model(**embedding_model_params)
