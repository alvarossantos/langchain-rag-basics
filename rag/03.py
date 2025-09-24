import os
from decouple import config

from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import OpenAIEmbeddings


os.environ['GOOGLE_API_KEY'] = config('GEMINI_API_KEY')

model = ChatGoogleGenerativeAI(
    model='gemini-2.5-pro',
    temperature=1,
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

vector_store = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding,
    collection_name='laptop_manual',
)
retriever = vector_store.as_retriever()

system_prompt = '''
Use o contexto para responder as perguntas.
Contexto: {context}
'''

prompt = ChatPromptTemplate.from_messages(
    [
        ('system', system_prompt),
        ('human', '{input}')
    ]
)
question_answer_chain = create_stuff_documents_chain(
    llm=model,
    prompt=prompt,
)
chain = create_retrieval_chain(
    retriever=retriever,
    combine_docs_chain=question_answer_chain,
)

query = 'Qual a marca e modelo do notebook?'

response = chain.invoke(
    {'input': query}
)
print(response)
