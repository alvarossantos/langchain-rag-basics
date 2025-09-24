import os
from decouple import config

from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


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
    chunk_overlap=200
)
chunks = text_splitter.split_documents(docs)

embedding = OpenAIEmbeddings(
    model='nomic-ai/nomic-embed-text-v1.5',
    openai_api_key=config('OPENAI_API_KEY'),
    openai_api_base=config('OPENAI_API_BASE'),
    tiktoken_enabled=False,
    max_retries=3,
    request_timeout=120
)

try:
    print('Criando vector store...')

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        collection_name='laptop_manual',
        persist_directory='./chroma_db'
    )

    retriever = vector_store.as_retriever()

    prompt = hub.pull('rlm/rag-prompt')

    rag_chain = (
        {
            'context': retriever,
            'question': RunnablePassthrough(),
        } | prompt | model | StrOutputParser()
    )

    try:
        while True:
            question = input('Qual a sua dúvida? ')
            response = rag_chain.invoke(question)
            print(response)

    except KeyboardInterrupt:
        exit()

except Exception as e:
    print(f'❌ Erro: {e}')

    print(f'Status Code: {response.status_code}')
    if response.status_code == 200:
        result = response.json()
        print(f'✅ API funcionando! Embedding length: {len(result['data'][0]['embedding'])}')
