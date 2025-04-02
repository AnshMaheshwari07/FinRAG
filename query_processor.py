import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader,WebBaseLoader,CSVLoader
from langchain.agents import AgentExecutor,initialize_agent,AgentType
from langchain.callbacks import StreamlitCallbackHandler ##for communicating with agents
import os
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain.tools.retriever import create_retriever_tool

os.environ['HUGGING_FACE_API_KEY']=os.getenv("HUGGING_FACE_API_KEY")
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

MONGO_DB_STRING=os.getenv("MONGO_DB")
client =MongoClient(MONGO_DB_STRING,tls=True, tlsAllowInvalidCertificates=True)
db = client["Financial_RAG"]
collection=db["finlatics"]

def make_using_pdf(uploaded_files):
    if uploaded_files:

        temppdf=f"./temp.pdf"
        with open(temppdf,"wb") as file:
            file.write(uploaded_files.read())
                

        loader=PyPDFLoader(temppdf)
        docs=loader.load()
      
# Split and create embeddings for the documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        splits = text_splitter.split_documents(docs)
        
        for doc in splits:
            chunk_text=doc.page_content
            embedding_vector=embeddings.embed_query(chunk_text)

            collection.insert_one({"text":chunk_text,"embedding":embedding_vector.tolist()})
        os.remove(temppdf)





st.title("Chat with search on web")

st.sidebar.title("Settings")
api_key=st.sidebar.text_input("Enter your GROQ API Key",type="password")

link=st.sidebar.text_input("Enter the link you want to crawl")
if link:
    retriever_Tool_link=make_embeddings_using_link(link)
    tools.append(retriever_Tool_link)

pdfload=st.sidebar.file_uploader("Enter the pdf you want to chat with (optional)",type="pdf", accept_multiple_files=False)


if pdfload:
    retriever_pdf=make_using_pdf(pdfload)
    tools.append(retriever_pdf)
    print("tools used are",tools)


if "messages" not in st.session_state:
  st.session_state["messages"]=[
      {"role":"assistant","content":"Hii I am a chatbot who can search the web. How may I help you?"}
  ]  

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if not api_key:
    st.error("Please enter api key to continue")



if prompt:=st.chat_input(placeholder="Write your question"):
    st.session_state.messages.append({"role":"user","content":prompt})
    st.chat_message("user").write(prompt)

    llm=ChatGroq(groq_api_key=api_key,model="Llama3-8b-8192",streaming=True)
    search_agent=initialize_agent(tools,llm,agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION
                            ,handle_parsing_errors=True,verbose=True)


    with st.chat_message("assistant"):
        st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
        response=search_agent.run(st.session_state.messages,callbacks=[st_cb])
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)



