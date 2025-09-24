import os
from decouple import config
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


os.environ['GOOGLE_API_KEY'] = config('GEMINI_API_KEY')

llm = ChatGoogleGenerativeAI(model='gemini-2.5-pro', temperature=1)

prompt_template = PromptTemplate.from_template(
    'Me fale sobre o carro {carro}.'
)

# # Verifique se o encadeamento está correto
# runnable_sequence = prompt_template | llm | StrOutputParser()

# try:
#     result = runnable_sequence.invoke({'carro': 'Fiat Uno Way'})
#     print(result)  # Verifique o resultado
# except Exception as e:
#     print(f'Ocorreu um erro: {e}')

chain = (
    PromptTemplate.from_template(
        'Me fale sobre o carro {carro}.'
    ) | llm | StrOutputParser()
)

response = chain.invoke({'carro': 'Fiat Uno Way'})

print(response)
