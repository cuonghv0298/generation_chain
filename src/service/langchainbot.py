import traceback
from typing import Any, Dict, List, Optional
from itertools import groupby
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)

from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama

from langchain.schema import HumanMessage, SystemMessage
from langchain.chains import LLMChain, ConversationalRetrievalChain, ConversationChain, create_qa_with_sources_chain
from langchain.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory, CombinedMemory
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.output_parsers.retry import RetryOutputParser
from langchain.schema import OutputParserException, PromptValue
from langchain_openai import OpenAI


# Additional imports for database drivers and system instructions
from driver import redisdb, weaviatedb
import utils.utils as utils
from uuid import uuid4
from langchain_core.tracers.context import collect_runs

# from llm.conversationchain import ConversationChain, ConversationalRetrievalChain
# from embedding.systeminstruct import SystemInstruct
# mapping title with gglink
    
class LangChainBot:
    __history_configure = False
    __knowledge_configure = False
    #

    def __init__(
            self,
            debug=True,
            init_params: Dict[str, Any] = {},
    ):
        # self.retry_parser, self.prompt_value = self.create_retry_parser_for_final_answer()
        self.debug = debug
        print(f"LangChainBot:\tInitialization")
    #

    @classmethod
    def bare_init(
            cls,
            # system_instruction : Dict[str, Optional[Dict[str, Any]]] = {
            # 	"instruct_embedding" : SystemInstruct,
            # },
            history_store: Dict[str, Optional[Dict[str, Any]]] = {
                "history_driver": redisdb.RedisDB,
            },
            knowledge_configure: Dict[str, Optional[Dict[str, Any]]] = {
                "knowledge_driver": weaviatedb.WeaviateDB,
            },
            condense_question_configure: Dict[str, Optional[Dict[str, Any]]] = {
                "chain_core": LLMChain,
                # "llm_core" : ChatOpenAI,
                "llm_core": ChatOllama,
                "llm_core_params": {
                    "temperature": 0,
                    "model": "phi",
                }
            },
            combine_docs_configure: Dict[str, Optional[Dict[str, Any]]] = {
                "stuff_chain_core": StuffDocumentsChain,
                "chain_core": LLMChain,
                # "llm_core" : OpenAI,
                "llm_core": Ollama,
                "llm_core_params": {
                    "temperature": 0,
                    "model": "phi",
                }
            },
            memory_configure: Dict[str, Optional[Dict[str, Any]]] = {
                "memory_core": ConversationBufferMemory,
            },
            stack_chain: Dict[str, Optional[Dict[str, Any]]] = {
                "chain_core": ConversationalRetrievalChain,
                "runnable_chain": RunnableWithMessageHistory,
            },
            embedding_configure: Dict[str, Optional[Dict[str, Any]]] = {
                'embedding': OpenAIEmbeddings()
            }
    ):
        try:
            bot = cls()
            #
            # if "instruct_embedding" in system_instruction:
            # 	bot.system_instruction(**system_instruction)
            #
            if "knowledge_driver" in knowledge_configure:
                bot.knowledge_configure(**knowledge_configure)
            # Use default or specific hsitory store
            if "history_driver" in history_store:
                bot.history_store(**history_store)
            # Or connect to the history store of api app (available)
            elif "history_store" in history_store:
                bot.set_history_store(**history_store)
            # Chose llm embedding
            bot.embedding_configure(**embedding_configure)
            # Chain to RAG: combine user question and session history to query revelant documents from vectorstore.
            bot.condense_question_configure(**condense_question_configure)
            # Memory: Only using ConversationBufferMemory as default to handle recent conversation, no chat history setup here
            # bot.memory_configure(**memory_configure)
            # Chain to interpreter: combine user question and returned revelant documents to produce answer.
            bot.combine_docs_configure(**combine_docs_configure)
            # Chatbot RAG chain: stacking with chat history, RAG chain and interpreter chain
            bot.stack_chain(**stack_chain)
            return bot
        except Exception as e:
            print(str(e))
            traceback.print_exc()
            return None
    #

    def embedding_configure(
            self,
            embedding
    ):
        self.__llm_embedding = embedding
        print(f"LangChainBot:\t embedding_configure: {embedding}")
        return True


    #
    def knowledge_configure(
            self,
            knowledge_driver: Any = weaviatedb.WeaviateDB,
            knowledge_driver_params: Optional[Dict[str, Any]] = {},
    ) -> bool:
        try:
            driver = knowledge_driver(**knowledge_driver_params)
            self.__retriever = driver
            self.__knowledge_configure = True
            print(
                f"LangChainBot:\tknowledge base retriever: {knowledge_driver}")
            return True
        except Exception as e:
            print(f'Cannot connect self.__retriever with weaviate {str(e)}')
            self.__retriever = None
            self.__knowledge_configure = False
            return False

    def history_store(
            self,
            history_driver: Any = redisdb.RedisDB,
            kwargs: Optional[Dict[str, Any]] = None,
    ) -> bool:
        try:
            driver = history_driver()
            if not driver.is_connected():
                raise Exception(
                    f"LangChainBot:\tHistory database is not connected.")
            self.__history = driver
            self.__history_configure = True
            print(f"LangChainBot:\tchat history retriever: {history_driver}")
            return True
            print()
        except Exception as e:
            print(str(e))
            self.__history = None
            self.__history_configure = False
            return False
    #

    def has_history_store(self,):
        return self.__history_configure == True
    #

    def get_history_store(self,):
        if not self.__history_configure:
            return None
        return self.__history
    #

    def set_history_store(
            self,
            history_store,
            overwrite: bool = True,
            kwargs: Optional[Dict[str, Any]] = None,
    ) -> bool:
        try:
            if not self.__history_configure or overwrite:
                if not history_store.is_connected():
                    raise Exception(
                        f"LangChainBot:\tHistory database is not connected.")
                self.__history = history_store
                self.__history_configure = True
            print(f"LangChainBot:\tchat history store: using FastAPIApp history store.")
            return True
        except Exception as e:
            print(str(e))
            self.__history = None
            self.__history_configure = False
            return False
    #

    def condense_question_configure(
            self,
            llm_core,
            prompt_core_template,
            llm_core_params: Dict[str, Any] = {},
            tenant_name: str = "Admin",
            index_db: str = "None",
            text_key: str = 'text',
            multi_tenancy: bool = True, 
    ) -> bool:
        '''
        The default prompt is 
        combine_docs_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, just "
            "reformulate it if needed and otherwise return it as is."
        )
        '''
        try:            
            print('multi_tenancy_langchainbot:',multi_tenancy)
            llm = llm_core(**llm_core_params)
            retriever = self.__retriever.get_langchain_vectorstore(
                as_retriever=True,
                tentant_name=tenant_name,
                index_name=index_db,
                text_key=text_key,
                embedding=self.__llm_embedding,
                k = 5,
                multi_tenancy = multi_tenancy
            )
            
            llm = llm_core(**llm_core_params)
            # Contextualize question
            contextualize_q_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", prompt_core_template),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            )
            history_aware_retriever = create_history_aware_retriever(
                llm, retriever, contextualize_q_prompt
            )
            self.__condense_question_chain = history_aware_retriever
            return True
        except Exception as e:
            print(str(e))
            self.__condense_question_chain = None
            return False
    #

    def combine_docs_configure(
            self,
            prompt_core_template,
            llm_core,
            llm_core_params: Dict[str, Any] = {},
    ) -> bool:
        '''
        The default prompt is 
        qa_system_prompt = (
                "You are an assistant for question-answering tasks. Use "
                "the following pieces of retrieved context to answer the "
                "question. If you don't know the answer, just say that you "
                "don't know. Use three sentences maximum and keep the answer "
                "concise."
                "\n\n"
                "{context}"
            )
        '''
        try: 
            llm = llm_core(**llm_core_params)
            # Answer question
            qa_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", prompt_core_template),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            )
            customDocumentPrompt_template= """
-   metadata:
        object_id: {uuid}
    quote: |- 
    {page_content}
    
"""
            customDocumentPrompt = PromptTemplate(
                    input_variables=['page_content', 'source'],
                    template=customDocumentPrompt_template,
                )
            question_answer_chain = create_stuff_documents_chain(
                llm=llm, 
                prompt = qa_prompt,
                document_prompt = customDocumentPrompt,
                document_variable_name = "context",
                )
            self.__combine_docs_configure = question_answer_chain
            return True 
        
        except Exception as e:
            print(str(e))
            self.__combine_docs_configure = None
            return False

    def memory_configure(
            self,
            memory_core: ConversationBufferMemory,
            memory_core_params: Dict[str, Any] = {},
    ) -> bool:
        try:
            # memory = memory_core(**memory_core_params)
            # Try ConversationBufferMemory() with default initial
            memory = memory_core(
                memory_key="chat_history",
                input_key="question",
                output_key="answer",
                return_messages=True,
                # **memory_core_params,
            )
            print(f"LangChainBot:\tMemory configure: {memory_core}")
            self.__memory = memory
            self.__memory_configure = True
            return True
        except Exception as e:
            print(str(e))
            self.__memory = None
            self.__memory_configure = False
            return False
    #

    def create_retry_parser_for_final_answer(self):
        final_answer = ResponseSchema(
            name="final_answer",
            description="return 'no value' if the anwser is 'None' else return the answer",
        )
        output_parser = StructuredOutputParser.from_response_schemas(
            [final_answer]
        )
        retry_parser = RetryOutputParser.from_llm(
            llm=OpenAI(temperature=0),
            parser=output_parser,
            max_retries=3
        )
        prompt = PromptTemplate(
            template="Reply the query\n{format_instructions}\n{query}\n",
            input_variables=["query"],
            partial_variables={
                "format_instructions": output_parser.get_format_instructions()},
        )
        prompt_value = prompt.format_prompt(
            query='Find the last answer, provide the evidence and reasoning of this answer.')
        return retry_parser, prompt_value
    def chain_constructor(self,tenant_name,index_db,text_key ):
            new_memory = ConversationBufferMemory(
                memory_key="chat_history",
                input_key="question",
                output_key="answer")            
            return ConversationalRetrievalChain(
                retriever=self.__retriever.get_langchain_vectorstore(as_retriever=True,
                                                                     tentant_name=tenant_name,
                                                                     index_name=index_db,
                                                                     text_key=text_key,
                                                                     embedding=self.__llm_embedding,
                                                                     k = 4
                                                                     ),
                memory = new_memory, 
                question_generator=self.__condense_question_chain,
                # Chat
                combine_docs_chain=self.__combine_docs_chain,)
    def stack_chain(
            self,
            runnable_chain: Optional[RunnableWithMessageHistory] = None,

    ) -> bool:
        
        # self.evaluation_chain = self.chain_constructor(tenant_name,index_db,text_key)
        history_aware_retriever = self.__condense_question_chain
        question_answer_chain = self.__combine_docs_configure
        qa = create_retrieval_chain(
            history_aware_retriever, question_answer_chain
        )

        # self.evaluation_chain = chain_constructor()
        # Deal with chat history using session_id
        print('runnable_chain:',runnable_chain)
        print('self.has_history_store():',self.has_history_store())
        if runnable_chain and self.has_history_store():
            print('--------WE USE RunnableWithMessageHistory')
            chain = RunnableWithMessageHistory(
                qa,
                lambda session_id: self.__history.get_langchain_chat_message_history(
                    session_id=session_id),
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer",
            )
            self.__runnable_chain = True
        # Without chat history, buffer memory only
        else:
            print('--------WE buffer memory only')
            chain = question_answer_chain
            self.__runnable_chain = False

        # chain stacking
        self.__chain = chain
        # For evaluation

        # self.evaluation_chain = chain
        self.__stack_chain = True
        return True
    
    

    async def ask(
            self,
            session_id: str,
            question: str,

    ):
        try:
            msg = ""
            with collect_runs() as runs_cb:
                if self.__runnable_chain:
                    print('LANGCHAINBOT|	WE,RE USING SESSION ID')
                    output = await self.__chain.ainvoke(
                        {
                            "input": question,  
                        },
                        config={"configurable": {"session_id": session_id}},
                        # include_run_info= True, 
                    )
                else:
                    output = self.__chain(
                        {
                            "question": question,
                        },
                    )
                run_id = runs_cb.traced_runs[0].id
            # print(f'------The answer response: {output}')
            # print(f'------The run_id: {run_id}')
            answer = f'{output["answer"]}'
            # result = self.retry_parser.parse_with_prompt(str(output), self.prompt_value)
            is_none = utils.check_is_none(answer)
            answer = {
                'chatbot_answer': f'{output["answer"]}',
                'is_none': is_none,
                "__run":{
                    'run_id': run_id
                }
                
            }
            return answer, msg
        except Exception as e:
            print(str(e))
            traceback.print_exc()
            # return {"bot" : "this is a sample response for debugger"}, ""
            return "_", e
    