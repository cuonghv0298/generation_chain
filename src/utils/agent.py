from llama_index.core.tools import QueryEngineTool
from langchain.agents import initialize_agent, tool
from langchain_openai import ChatOpenAI
from typing import Literal
from langchain.chains.conversation.memory import ConversationBufferMemory
import sys
from langgraph.prebuilt import InjectedStore
from typing import List
from langgraph.store.base import BaseStore
from langgraph.graph.state import CompiledStateGraph
from langchain_core.prompts import PromptTemplate
from langgraph.prebuilt import create_react_agent
from typing import Optional
import os
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
import streamlit as st
from langchain_core.runnables.config import RunnableConfig
import gc
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from loguru import logger
from datetime import datetime
from time import sleep
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from langchain.embeddings import init_embeddings
import uuid
from typing import Annotated

from typing_extensions import TypedDict
from langchain.schema import HumanMessage

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import tools_condition
from langgraph.prebuilt import ToolNode
from langfuse.callback import CallbackHandler
from langfuse import Langfuse
from datetime import datetime
from langfuse.decorators import langfuse_context, observe
from langgraph.types import Command
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from langfuse.decorators import langfuse_context
from typing_extensions import Any, Annotated
from langchain_core.tools.base import BaseTool
from langchain_core.messages import trim_messages, AIMessage, SystemMessage
import weaviate
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from llama_index.core import VectorStoreIndex

# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import streamlit as st
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.postprocessor.colpali_rerank import ColPaliRerank

# colpali_reranker = ColPaliRerank(
#     top_n=3,
#     model="vidore/colpali-v1.2",
#     keep_retrieval_score=True,
#     device="cuda",  # or "cpu" or "cuda:0" or "mps" for Apple
# )
colpali_reranker = None

cohere_rerank = CohereRerank(
    api_key="yWEEBc7lndS6XabMlFW75bFk9giKWAs5xTojtiwI", top_n=3
)
# Weaviate client setup
client = weaviate.connect_to_custom(
    http_host=os.environ.get(
        "WEAVIATE_HOST"
    ),  # "10.100.224.34",  # URL only, no http prefix
    http_port=os.environ.get("WEAVIATE_HOST_PORT"),  # "8080",
    http_secure=False,  # Set to True if https
    grpc_host=os.environ.get("WEAVIATE_GPC_URL"),  #  "10.100.224.34",
    grpc_port=os.environ.get(
        "WEAVIATE_GPC_URL_PORT"
    ),  # "50051",      # Default is 50051, WCD uses 443
    grpc_secure=False,  # Edit as needed
    skip_init_checks=True,
)
EMBED_MODEL = OpenAIEmbedding(model="text-embedding-3-small")


def load_index(collection_name="LlamaIndex_da9b7bb158e64c93bea491df09894psd", **kwargs):
    vector_store = WeaviateVectorStore(
        weaviate_client=client, index_name=collection_name
    )

    # Create index
    return MultiModalVectorStoreIndex.from_vector_store(
        vector_store=vector_store, embed_model=EMBED_MODEL
    )


# Bước 1: Khởi tạo state cho agent
class AgentState(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]


# The in_memory_store works hand-in-hand with the checkpointer: the checkpointer saves state to threads, as discussed above, and the in_memory_store allows us to store arbitrary information for access across threads
in_memory_store = InMemoryStore(
    index={
        "embed": init_embeddings(
            model="openai:text-embedding-3-small"
        ),  # Embedding provider
        "dims": 1536,  # Embedding dimensions
        "fields": ["text"],
    }
)


# We need this because we want to enable threads (conversations)s
checkpointer = MemorySaver()

template = """You are a helpful assistant in Kyanon Digital.

You should get the following rules:

- Totally trust the information from the tools not your prior knowledge.
- Always response the answer with user's one.
- If the question is complext, you should break it into sub-queries then retrieve documents for each one.
"""


def get_messages_info(messages):
    return [SystemMessage(content=template)] + messages


def create_agent(tools: List[BaseTool]) -> CompiledStateGraph:
    model = ChatOpenAI(model="gpt-4o", temperature=0, streaming=True)
    model_with_tools = model.bind_tools(tools)

    # Bước 3: Định nghĩa logic của Nodes

    def should_continue(state: AgentState):
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools"
        return END

    def call_model(state: AgentState, config: RunnableConfig, *, store):
        response = AIMessage("Xin lỗi vì đang gặp sự cố, vui lòng hỏi câu khác.")
        messages = trim_messages(
            state["messages"],
            token_counter=len,  # <-- len will simply count the number of messages rather than tokens
            max_tokens=5,  # <-- allow up to 5 messages.
            strategy="last",
            # Most chat models expect that chat history starts with either:
            # (1) a HumanMessage or
            # (2) a SystemMessage followed by a HumanMessage
            # start_on="human" makes sure we produce a valid chat history
            start_on="human",
            # Usually, we want to keep the SystemMessage
            # if it's present in the original history.
            # The SystemMessage has special instructions for the model.
            include_system=True,
            allow_partial=False,
        )
        # Get the user id from the config
        user_id = config["configurable"]["user_id"]
        # Namespace the memory
        namespace = (user_id, "memories")
        question = messages[-1].content

        messages = get_messages_info(messages)
        response = model_with_tools.invoke(messages)
        if (
            response.content != ""
            and response.response_metadata["finish_reason"] == "stop"
        ):
            mem_id = uuid.uuid4()
            store.put(
                namespace,
                key=str(mem_id),
                value={"text": f"{question}"},
            )
        return {"messages": [response]}

    # Bước 4: Định nghĩa Agent nodes
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_model)
    tool_node = ToolNode(tools)
    workflow.add_node("tools", tool_node)

    # Bước 5: Định nghĩa flow trong agent (EDGES)
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue, ["tools", END])
    workflow.add_edge("tools", "agent")

    # Bước 6: Build Agent
    agent = workflow.compile(checkpointer=checkpointer, store=in_memory_store)
    return agent


def _get_session():
    from streamlit.runtime import get_instance
    from streamlit.runtime.scriptrunner import get_script_run_ctx

    runtime = get_instance()
    ctx = get_script_run_ctx()
    if ctx is None:
        raise RuntimeError("No Streamlit script run context found")
    session_id = ctx.session_id
    session_info = runtime._session_mgr.get_session_info(session_id)
    if session_info is None:
        raise RuntimeError("Couldn't get your Streamlit Session object.")
    return session_info.session


from llama_index.core.query_engine import (
    CustomQueryEngine,
    SimpleMultiModalQueryEngine,
)
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import (
    ImageNode,
    NodeWithScore,
    MetadataMode,
    TextNode,
)
from llama_index.core.prompts import PromptTemplate
from llama_index.core.base.response.schema import Response
from llama_index.core.indices.query.schema import QueryBundle
from typing import Optional

QA_PROMPT_TMPL = """\
Below we give parsed text and images as context.

Use both the parsed text and images to answer the question. 

---------------------
{context_str}
---------------------
Given the context information and not prior knowledge, answer the query. Explain whether you got the answer
from the text or image, and if there's discrepancies, and your reasoning for the final answer.

Query: {query_str}
Answer: """

QA_PROMPT = PromptTemplate(QA_PROMPT_TMPL)


class MultimodalQueryEngine(CustomQueryEngine):
    """Custom multimodal Query Engine.

    Takes in a retriever to retrieve a set of document nodes.
    Also takes in a prompt template and multimodal model.

    """

    qa_prompt: PromptTemplate
    retriever: BaseRetriever
    multi_modal_llm: OpenAIMultiModal

    def __init__(self, qa_prompt: Optional[PromptTemplate] = None, **kwargs) -> None:
        """Initialize."""
        super().__init__(qa_prompt=qa_prompt or QA_PROMPT, **kwargs)

    def custom_query(self, query_str: str):
        # retrieve text nodes
        nodes = self.retriever.retrieve(query_str)
        image_nodes = [n for n in nodes if isinstance(n.node, ImageNode)]
        text_nodes = [n for n in nodes if isinstance(n.node, TextNode)]

        # Make QueryBundle
        query_bundle = QueryBundle(query_str)
        # reranking text nodes
        if cohere_rerank:
            text_nodes = cohere_rerank.postprocess_nodes(text_nodes, query_bundle)

        # reranking image nodes
        if colpali_reranker:
            image_nodes = colpali_reranker.postprocess_nodes(image_nodes, query_bundle)

        # create context string from text nodes, dump into the prompt
        context_str = "\n\n".join(
            [
                r.get_content(metadata_mode=MetadataMode.LLM) for r in text_nodes
            ]  # later can use reranked_text_nodes
        )
        fmt_prompt = self.qa_prompt.format(context_str=context_str, query_str=query_str)

        # synthesize an answer from formatted text and images
        llm_response = self.multi_modal_llm.complete(
            prompt=fmt_prompt,
            image_documents=[n.node for n in image_nodes],
        )
        return Response(
            response=str(llm_response),
            source_nodes=nodes,
            metadata={
                "text_nodes": text_nodes,
                "image_nodes": image_nodes,
            },
        )
