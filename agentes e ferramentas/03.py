import os
from decouple import config

from langchain.agents import Tool
from langchain.prompts import PromptTemplate
from langchain_experimental.utilities import PythonREPL
from langchain.agents import initialize_agent, AgentType
from langchain_google_genai import ChatGoogleGenerativeAI


os.environ['GOOGLE_API_KEY'] = config('GEMINI_API_KEY')

model = ChatGoogleGenerativeAI(model='gemini-2.5-pro', temperature=1)

python_repl = PythonREPL()
python_repl_tool = Tool(
    name='Python REPL',
    description='Um sheel Pytohon. Use essa ferramenta para executar código Python. '
                'Se você precisar de retorne do código, use a função "print(...)".',
    func=python_repl.run
)

agent_executor = initialize_agent(
    llm=model,
    tools=[python_repl_tool],
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
)

prompt_template = PromptTemplate(
    input_variables=['query'],
    template='''
    Resolva o problema: {query}.
    '''
)

query = 'Quanto é 20% de 3000'
prompt = prompt_template.format(query=query)

response = agent_executor.invoke({'input': prompt})
print(response.get('output'))
