import re
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from llama_index.core.node_parser import SentenceSplitter

import requests
import yaml

def chunking(method: str):
    """return TextSplitter with one chunking method"""
    if method == 'RecursiveCharacterTextSplitter':
        return RecursiveCharacterTextSplitter(
            # Set a really small chunk size, just to show.
            chunk_size=512,
            chunk_overlap=10,
            length_function=len,
        )
    elif method == 'CharacterTextSplitter':
        return CharacterTextSplitter(
            separator=" ",
            chunk_size=512,
            chunk_overlap=10,
            length_function=len,
            is_separator_regex=False,
        )
    elif method == 'SentenceSplitter':
        return SentenceSplitter(
            chunk_size=512,
            chunk_overlap=10,
        )

def reformat_text(text):
    """Drop multiple spaces, tabs, endlines."""
    return " ".join(text.split())

def check_link_type(url):
    try:
        response = requests.head(url, allow_redirects=True)
        content_type = response.headers.get('Content-Type', '').lower()

        if 'text/html' in content_type:
            return True #"This link is a webpage."
        else:
            return False #f"This link is not a webpage. Content-Type: {content_type}"

    except requests.RequestException as e:
        print(f'This link: {url} lead to result: {e}')
        return False
    
def find_links(text):
    # Regular expression pattern to match URLs
    url_pattern = re.compile(r'https?://\S+')
    # Find all matches in the text
    links = re.findall(url_pattern, text)
    set_links = list(set(links))
    verified_links = [link for link in set_links if check_link_type(link)]
    return verified_links

def count_words(content):
    # Split the content by whitespace to count words
    words = content.split()
    # Return the length of the list of words
    return len(words)

def load_config(file_path):
    with open(file_path, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            
def check_is_none(text: str) -> bool:
    return "none" in text.lower()

def preprocess_text_for_markdown(raw_text):
    # Replace newline characters with spaces
    cleaned_text = raw_text.replace("\n", " ")

    # Remove null character markers (\x00)
    cleaned_text = cleaned_text.replace("\x00", "")

    # Remove extra spaces
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

    # Use regex to identify bullet points (●) and ensure proper formatting
    cleaned_text = re.sub(r"●", "\n- ", cleaned_text)
    
    return cleaned_text