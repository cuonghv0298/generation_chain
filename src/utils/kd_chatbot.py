from service.langchainbot import LangChainBot
import os 

INDEX = os.environ.get("COLLECTION_ID")
TEXT_KEY = "text"
def start_chatbot(config_llm, tenant_name, index=INDEX, text_key=TEXT_KEY, debug=False, multi_tenancy=True):
    try:
        config = config_llm.chose_llm_model(use_redis=True)
        # config = config_llm.chose_llm_embedding()
        config = config_llm.chose_llm_embedding(
            llm_model = 'OpenAI',
            model = 'text-embedding-3-small'
        )
        config = config_llm.config_db(index, text_key, tenant_name,multi_tenancy)
        chatbot = LangChainBot.bare_init(**config)
        
    except Exception as e:
        if debug:
            raise e  # Re-raise exception in debug mode
        else:
            chatbot = None
    return chatbot
