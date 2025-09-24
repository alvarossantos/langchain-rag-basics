import os
from decouple import config
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_google_genai import ChatGoogleGenerativeAI


os.environ['GOOGLE_API_KEY'] = config('GEMINI_API_KEY')

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash', temperature=1)

db = SQLDatabase.from_uri('sqlite:///ipca.db')

toolkit = SQLDatabaseToolkit(
    db=db,
    llm=model,
)
system_message = hub.pull('hwchase17/react')

agent = create_react_agent(
    llm=model,
    tools=toolkit.get_tools(),
    prompt=system_message,
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=toolkit.get_tools(),
    verbose=True,
)

prompt = '''
Use as ferramentas necessárias para responder perguntas relacionadas ao hstórico do IPCA ao logo dos anos.
Responda tudo em português brasileiro.
Perguntas: {q}
'''

prompt_templat = PromptTemplate.from_template(prompt)

question = 'Qual mês e ano tiveram o maior IPCA?'

output = agent_executor.invoke({
    'input': prompt_templat.format(q=question)
})

print(output.get('output'))
