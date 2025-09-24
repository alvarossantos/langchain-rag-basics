import os
from decouple import config

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import OpenAIEmbeddings


os.environ['GOOGLE_API_KEY'] = config('GEMINI_API_KEY')

model = ChatGoogleGenerativeAI(
    model='gemini-2.5-pro',
    temperature=1,
)

pdf_path = 'laptop_manual.pdf'
loader = PyPDFLoader(pdf_path)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=200,
)
chunks = text_splitter.split_documents(
    documents=docs
)

persist_directory = 'db'

embedding = OpenAIEmbeddings(
    model='nomic-ai/nomic-embed-text-v1.5',
    openai_api_key=config('OPENAI_API_KEY'),
    openai_api_base=config('OPENAI_API_BASE'),
    tiktoken_enabled=False,
    max_retries=3,
    request_timeout=120
)

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embedding,
    persist_directory=persist_directory,
    collection_name='laptop_manual'
)
