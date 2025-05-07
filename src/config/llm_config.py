from langchain_openai import ChatOpenAI, OpenAI
from langchain_community.llms import Ollama
# from langchain_community.chat_models import ChatOllama #angChainDeprecationWarning: The class `ChatOllama` was deprecated in LangChain 0.3.1 and will be removed in 1.0.0. An updated version of the class exists in the :class:`~langchain-ollama package and should be used instead. To use it run `pip install -U :class:`~langchain-ollama` and import as `from :class:`~langchain_ollama import ChatOllama``.
from langchain_ollama import ChatOllama
from langchain.chains import LLMChain, ConversationalRetrievalChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from driver import redisdb, weaviatedb
import utils.utils as utils


class ChatConfig:
    def __init__(self, prompt_config, model_config) -> None:
        '''
        We currently support openAI with local use ollama
        So llm_model have 2 options OpenAI and Ollama
        '''
        
        self.config = {}
        if prompt_config:
            self.prompt_config = prompt_config
            print('Loading config from file_config')
        else:
            self.prompt_config = utils.load_config('config/files/prompt_config.yml')
            print('Loading config from default: config/files/prompt_config.yml')
        self.condense_question_prompt = self.prompt_config['prompt']['condense_question_prompt']
        self.combine_docs_prompt = self.prompt_config['prompt']['combine_docs_prompt']
        # self.mode = model_config['ask']['client']
        self.model_config = model_config
        
    def build_model(self, name):
        client  = self.model_config[name]['client']
        model = self.model_config[name]['model_name']
        if client == 'OpenAI':
            CHAT_MODELS = ChatOpenAI
            LLL_PARAMS = {
                "temperature": 0,
                "model": model,
                # "model" : "gpt-3.5-turbo-0613",
            }
        elif client == 'Ollama':
            CHAT_MODELS = ChatOllama
            LLL_PARAMS = {
                "temperature": 0,
                "model": model,
                "max_retries": 2,
            }
        else:
            raise ('chose llm OpenAI or Ollama')
        return CHAT_MODELS, LLL_PARAMS
    def chose_llm_model(self, use_redis=True):
        CHAT_MODELS_CONDENSE, LLL_PARAMS_CONDENSE = self.build_model("condense_question")
        CHAT_MODELS_COMBINE, LLL_PARAMS_COMBINE = self.build_model("combine_docs")
        self.config["knowledge_configure"] = {
            "knowledge_driver": weaviatedb.WeaviateDB}
        self.config["history_store"] = {"history_driver": redisdb.RedisDB}
        self.config["condense_question_configure"] = {"llm_core": CHAT_MODELS_CONDENSE,
                                                      "llm_core_params": LLL_PARAMS_CONDENSE, "prompt_core_template": self.condense_question_prompt}
        # self.config["memory_configure"] = {
        #     "memory_core": ConversationBufferMemory}
        self.config["combine_docs_configure"] = {"llm_core": CHAT_MODELS_COMBINE, 
                                                 "llm_core_params": LLL_PARAMS_COMBINE, "prompt_core_template": self.combine_docs_prompt}
        if use_redis:
            self.config["stack_chain"] = {"runnable_chain": RunnableWithMessageHistory}
        else:
            self.config["stack_chain"] = {"runnable_chain": False}
        return self.config
    # The chose_llm_embedding function is rewirte in src/tasks_embedding.py
    def chose_llm_embedding(self, llm_model=None, model=None):
        if llm_model == None and model == None:
            llm_model = self.model_config['llm_embeding']['client']
            model = self.model_config['llm_embeding']['model_name']
        
        if llm_model == 'OpenAI':
            embedding = OpenAIEmbeddings(model=model)
        elif llm_model == 'Ollama':
            embedding = OllamaEmbeddings(model=model)
        else:
            raise 'WE ONLY SUPPORT OpenAIEmbeddings AND OllamaEmbeddings'
        self.config['embedding_configure'] = {'embedding': embedding}

        return self.config

    def config_db(self, index, text_key, tenant_name, multi_tenancy=True):            
        self.config["condense_question_configure"]['index_db'] = index
        self.config["condense_question_configure"]['text_key'] = text_key
        self.config["condense_question_configure"]['tenant_name'] = tenant_name
        self.config["condense_question_configure"]['multi_tenancy'] = multi_tenancy

        return self.config

    # def langsmith(self, is_langsmith=False):
    #     self.config["trace_by_langsmith"] = {"is_langsmith": is_langsmith}

    #     return self.config
