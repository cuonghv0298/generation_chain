from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

def get_llm_model(chatmodel: str, model_name: str, param: dict, stream_handler=None):
    param['model'] = model_name    
    if stream_handler is not None:
        param['streaming'] = True
        param['callbacks'] = [stream_handler]
    if chatmodel == 'OpenAI':
        llm = ChatOpenAI(**param)
    elif chatmodel == 'Ollama':
        llm = ChatOllama(**param)            
    else:
        raise ValueError(f"We currently just support OpenAI and Ollama, your client is {chatmodel}")
    
    return llm