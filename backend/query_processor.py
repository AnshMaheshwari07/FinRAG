import streamlit as st
from langchain_groq import ChatGroq
from langchain.agents import initialize_agent,AgentType
from langchain.callbacks import StreamlitCallbackHandler ##for communicating with agents
import os
from dotenv import load_dotenv
from pymongo import MongoClient
from websearch import Search
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain.tools import Tool
from yfinane import FinanceTool
from langchain.callbacks.manager import CallbackManager
from pymongo.server_api import ServerApi
from llama_index.core.evaluation import(
    RetrieverEvaluator,
    FaithfulnessEvaluator,
    RelevancyEvaluator
)
from llama_index.core.llms import LLM
import numpy as np

load_dotenv()

from langchain_huggingface import HuggingFaceEmbeddings

llm = ChatGroq(
    model="Gemma2-9b-It"
    )
os.environ['HUGGING_FACE_API_KEY']=os.getenv("HUGGING_FACE_API_KEY")
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
os.environ["TAVILY_API_KEY"]=os.getenv("TAVILY_API_KEY")
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]="FinRAG"
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")

client = MongoClient(os.getenv("uri"), server_api=ServerApi('1'))
db = client["Financial_RAG"]
collection=db["vector_store_finance"]

search=MongoDBAtlasVectorSearch(collection=collection,embedding=embeddings)

finance_tool=FinanceTool()

def rag_tool(query:str)->str:
    ##your vector search logic here
    retriever=search.as_retriever(search_kwargs={'k':2})
    docs=retriever.get_relevant_documents(query)
    return "\n\n".join(doc["content"] for doc in docs)



tools=[
    Tool(
        name="Web search",
        func=Search.tavily,
        description="Perform a real-time web search using Tavily and give latest updates."
    ),
    Tool(
        name="Yahoo Finance",
        func=finance_tool.get_stock_data,
        description="Fetch historical stock data for a given ticker and date range"
    ),
    Tool(
        name="RAG_Search",
        func=rag_tool,
        description="Retrieve related documents from MongoDB vector store"
    )
]

# Create your handler
streamlit_handler = StreamlitCallbackHandler(st)

# Wrap it in a manager
callback_manager = CallbackManager([streamlit_handler])

agent=initialize_agent(
    llm=llm,
    tools=tools,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    callback_manager=callback_manager,
    return_intermediate_steps=True,
    max_iterations=3,
)

#I want that throught ui when user ask any query and click enter 


def query_agent(user_input:str):
    response=agent({"input":user_input})
    eval=get_rag_evaluation(query=user_input,response=response,retrieved=rag_tool(user_input),judge_LLM=LLM)
    
    steps=[]

    for (action,obs) in response["intermediate_steps"]:
        cleaned_obs=str(obs).replace("Thought","").replace("Observation","").strip()
        steps.append({
            "thought":action.log.strip(),
            "action":action.tool,
            "observation":cleaned_obs
        })
    return{
        
        "chain_of_thought":steps,
        "final_answer":response["output"],
        "Is_relevant":eval["Is_relevant"],
        "relevancy_score":eval["relevancy_score"],
        "Is_faithful":eval["Is_faithful"],
        "faithfulness_score":eval["faithfulness_score"],
        "error":eval["error"]
        
    }


import streamlit as st
from langchain_groq import ChatGroq
from langchain.agents import initialize_agent,AgentType
from langchain.callbacks import StreamlitCallbackHandler ##for communicating with agents
import os
from dotenv import load_dotenv
from pymongo import MongoClient
from websearch import Search
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain.tools import Tool
from yfinane import FinanceTool
from langchain.callbacks.manager import CallbackManager
from pymongo.server_api import ServerApi
from llama_index.core.evaluation import(
    RetrieverEvaluator,
    FaithfulnessEvaluator,
    RelevancyEvaluator
)
from llama_index.core.llms import LLM
import numpy as np

load_dotenv()

from langchain_huggingface import HuggingFaceEmbeddings

llm = ChatGroq(
    model="Gemma2-9b-It"
    )
os.environ['HUGGING_FACE_API_KEY']=os.getenv("HUGGING_FACE_API_KEY")
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
os.environ["TAVILY_API_KEY"]=os.getenv("TAVILY_API_KEY")
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]="FinRAG"
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")

client = MongoClient(os.getenv("uri"), server_api=ServerApi('1'))
db = client["Financial_RAG"]
collection=db["vector_store_finance"]

search=MongoDBAtlasVectorSearch(collection=collection,embedding=embeddings)

finance_tool=FinanceTool()

def rag_tool(query:str)->list[str]:
    ##your vector search logic here
    retriever=search.as_retriever(search_kwargs={'k':2})
    docs=retriever.get_relevant_documents(query)
    contents = [doc["content"] for doc in docs if doc.get("content")]
    return contents



tools=[
    Tool(
        name="Web search",
        func=Search.tavily,
        description="Perform a real-time web search using Tavily and give latest updates."
    ),
    Tool(
        name="Yahoo Finance",
        func=finance_tool.get_stock_data,
        description="Fetch historical stock data for a given ticker and date range"
    ),
    Tool(
        name="RAG_Search",
        func=rag_tool,
        description="Retrieve related documents from MongoDB vector store"
    )
]

# Create your handler
streamlit_handler = StreamlitCallbackHandler(st)

# Wrap it in a manager
callback_manager = CallbackManager([streamlit_handler])

agent=initialize_agent(
    llm=llm,
    tools=tools,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    callback_manager=callback_manager,
    return_intermediate_steps=True,
    max_iterations=3,
)

#I want that throught ui when user ask any query and click enter 


def query_agent(user_input:str):
    response=agent({"input":user_input})

    eval=get_rag_evaluation(query=user_input,response_str=response["output"],retrieved_contexts=rag_tool(user_input),judge_LLM=LLM)
    
    steps=[]

    for (action,obs) in response["intermediate_steps"]:
        cleaned_obs=str(obs).replace("Thought","").strip()
        steps.append({
            "thought":action.log.strip(),
            "action":action.tool,
            "observation":cleaned_obs
        })
    return{
        
        "chain_of_thought":steps,
        "final_answer":response["output"],
        "Is_relevant":eval["Is_relevant"],
        "relevancy_score":eval["relevancy_score"],
        "Is_faithful":eval["Is_faithful"],
        "faithfulness_score":eval["faithfulness_score"],
        "error":eval["error"]
        
    }


def get_rag_evaluation(query: str, response_str: str, retrieved_contexts: list[str], judge_LLM: LLM):
    # Validate inputs
    if not query or not response_str or not judge_LLM:
        return {
            "Is_relevant": False,
            "relevancy_score": 0.0,
            "Is_faithful": False,
            "faithfulness_score": 0.0,
            "error": "Missing required input"
        }
    
    if not retrieved_contexts:
        return {
            "Is_relevant": False,
            "relevancy_score": 0.0,
            "Is_faithful": False,
            "faithfulness_score": 0.0,
            "error": "No RAG search performed"
        }
    faith_eval = FaithfulnessEvaluator(llm=judge_LLM)
    relev_eval = RelevancyEvaluator(llm=judge_LLM)

    try:
        # Faithfulness check
        faith_result = faith_eval.evaluate(
            query=query,
            retrieved_contexts=retrieved_contexts,
            response=response_str
        )

        # Relevancy check
        relev_result = relev_eval.evaluate(
            query=query,
            retrieved_contexts=retrieved_contexts,
            response=response_str
        )

        return {
            "Is_relevant": relev_result.passing,
            "relevancy_score": relev_result.score,
            "Is_faithful": faith_result.passing,
            "faithfulness_score": faith_result.score,
            "error": None
        }
    except Exception as e:
        return {
            "Is_relevant": False,
            "relevancy_score": 0.0,
            "Is_faithful": False,
            "faithfulness_score": 0.0,
            "error": str(e)
        }