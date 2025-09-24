import os
from decouple import config
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import CSVLoader  # TextLoader, PyPDFLoader


os.environ['GOOGLE_API_KEY'] = config('GEMINI_API_KEY')

llm = ChatGoogleGenerativeAI(model='gemini-2.5-pro', temperature=1)

# loader = TextLoader('base_conhecimento.txt')
# documents = loader.load()

# loader = PyPDFLoader('base_conhecimento.pdf')
# documents = loader.load()

loader = CSVLoader('base_conhecimento.csv')
documents = loader.load()

prompt_base_conhecimento = PromptTemplate(
    input_variables=['contexto', 'pergunta'],
    template='''Use o seguinte contexto para responder à pergunta.
    Responda apenas com base nas informações fornecidas.
    Não utilize informações externas ao contexto:
    Contexto: {contexto}
    Pergunta: {pergunta}
    '''
)

chain = (
    prompt_base_conhecimento | llm | StrOutputParser()
)

response = chain.invoke(
    {
        'contexto': '\n'.join(doc.page_content for doc in documents),
        'pergunta': 'Qual carro é mais novo?',
    }
)

print(response)
