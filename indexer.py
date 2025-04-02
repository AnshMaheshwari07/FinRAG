import os
from dotenv import load_dotenv
load_dotenv()

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from llama_parse import LlamaParse
from pymongo import MongoClient
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_huggingface import HuggingFaceEmbeddings
# only these filetypes will be allowed to parsed
allowed_filetypes = ['pdf', 'doc', 'docx', 'txt', 'xlsx', 'csv', 'ppt', 'pptx', 'md']

parser=LlamaParse(
    api_key=os.getenv("LLAMA_PARSE"),
    result_type="markdown",
    num_workers=4,
    verbose=True,
    language="en",
)

def check_filetype(filename:str):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in allowed_filetypes

def parse_segmentation(file_path:str,chunk_size:int,chunk_overlap:int)->list[Document]:
    """
    Parse the file and segment it into chunks.
    """
    
    file=parser.load_data(file_path)
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    text="".join([doc.text for doc in file])
    chunks=text_splitter.split_text(text)
    source=os.path.basename(file_path)
    documents=[
        Document(
            page_content=chunk,
            metadata={
                'source':source,
                'chunk_index':source+str(idx)
            }
        ) for idx,chunk in enumerate(chunks)
    ]
    return documents

def find_file(source : str,
              vectorStore : MongoDBAtlasVectorSearch) -> bool:
    """
    Check if the document with the given filename already exists in the vector store.

    :param filename: The file name of the file to check.
    :param vectorStore: The MongoDBAtlasVectorSearch instance to query
    :return: True if the file exits, False otherwise
    """
    query_result = vectorStore.similarity_search(f"metadata.source:{source}")
    return len(query_result) > 0

if not os.getenv("uri"):
    raise RuntimeError("MongoDB API Key not found. Check .env file.")


from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi


# Create a new client and connect to the server
client = MongoClient(os.getenv("uri"), server_api=ServerApi('1'))

# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)
db_name="Financial_RAG"
collection_name="vector_store_finance"
collection=client[db_name][collection_name]


embeddings=HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
    )


data_directory='data'
chunk_size=1000
chunk_overlap=100
vectorstore=MongoDBAtlasVectorSearch(collection,embeddings)

for filename in os.listdir(data_directory):
    if not check_filetype(filename):
        raise RuntimeError(f"File not supported{filename}")
    

    if find_file(source=filename, vectorStore=vectorstore):
        print(f"Document with file name `{filename}` already exists in the vectorstore. Skipping upsertion")
        continue
    file_path=os.path.join(data_directory,filename)

    #parse and chunk the data
    documents=parse_segmentation(file_path,chunk_size,chunk_overlap)
    #create embeddings for the documents

    vectorstore.add_documents(documents)
    print(f"Parsed {filename} and added to MongoDB Atlas Vector Search.")



    