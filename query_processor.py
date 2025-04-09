import streamlit as st
from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor,initialize_agent,AgentType
from langchain.callbacks import StreamlitCallbackHandler ##for communicating with agents
import os
from dotenv import load_dotenv
from pymongo import MongoClient
from websearch import Search
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain.tools import Tool
from yfinane import FinanceTool
from langchain.callbacks.manager import CallbackManager
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import json
import pandas as pd
load_dotenv()

from langchain_huggingface import HuggingFaceEmbeddings

llm = ChatGroq(
    model="Gemma2-9b-It"
    )
os.environ['HUGGING_FACE_API_KEY']=os.getenv("HUGGING_FACE_API_KEY")
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
os.environ["TAVILY_API_KEY"]=os.getenv("TAVILY_API_KEY")

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
    callbacks=[streamlit_handler],
    return_intermediate_steps=True,
    max_iterations=3,
)


st.title("FinRAG")
query = st.text_input("Ask me about stocks, markets, or trends")

if st.button("Run"):
    with st.spinner("Thinking..."):
        response = agent({"input":query})
    

    st.markdown("Chain of thought:")
    for idx,(action,obs) in enumerate(response["intermediate_steps"]):
       ## st.markdown(f"step {idx+1}")
        st.write(f"Thought: {action.log}")
        st.write(f"Action: {action.tool}")
        st.write(f"Extracted params: {action.tool_input}")
        st.write(f"Observation: {obs}")
        st.write(" --- ")
    st.markdown("## Final Answer")
    st.write(response["output"])
