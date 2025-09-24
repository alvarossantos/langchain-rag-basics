import os
from decouple import config

from langchain.prompts import PromptTemplate
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents import initialize_agent, AgentType
from langchain_google_genai import ChatGoogleGenerativeAI


os.environ['GOOGLE_API_KEY'] = config('GEMINI_API_KEY')

model = ChatGoogleGenerativeAI(model='gemini-2.5-pro', temperature=1)

wikipedia_tool = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper(
        lang='pt'
    )
)

agent_executor = initialize_agent(
    llm=model,
    tools=[wikipedia_tool],
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
)

prompt_template = PromptTemplate(
    input_variables=['query'],
    template='''
    Pesquise na web sobre {query} e fornece um resumo sobre o assunto.
    '''
)

query = 'Alan Turing'
prompt = prompt_template.format(query=query)

response = agent_executor.invoke({'input': prompt})
print(response.get('output'))
