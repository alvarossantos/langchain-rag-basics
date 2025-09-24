import os
from decouple import config
from langchain.agents import Tool, AgentExecutor, initialize_agent, AgentType
from langchain.prompts import PromptTemplate
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_experimental.utilities import PythonREPL
from langchain_google_genai import ChatGoogleGenerativeAI


os.environ["GOOGLE_API_KEY"] = config("GEMINI_API_KEY")

model = ChatGoogleGenerativeAI(model='gemini-2.5-pro', temperature=1)

prompt = '''
Como assistente financeiro pessoal, você responderá perguntas dando dicas financeiras e de investimentos.
Responda tudo e somente em português brasileiro.
Pergunta: {question}
'''
prompt_template = PromptTemplate.from_template(prompt)

python_repl = PythonREPL()
python_repl_tool = Tool(
    name='Python REPL',
    description=(
        'Um shell Python. Use isso para executar código Python. Execute apenas códigos Python válidos. '
        'Use "print(...)" para obter os resultados. '
        'Use para cálculos financeiros necessários para responder as perguntas.'
    ),
    func=python_repl.run
)

search = DuckDuckGoSearchRun()
duckduckgo_tool = Tool(
    name='Busca DuckDuckGo',
    description=(
        'Útil para encontrar informações e dicas de economia e opções de investimento. '
        'Você sempre deve pesquisar na internet as melhores dicas usando esta ferramenta. '
        'A resposta deve informar que houve pesquisa online.'
    ),
    func=search.run,
)

# Instruções para o agente
react_instructions = PromptTemplate.from_template("""
Você é um assistente financeiro pessoal.
Responda perguntas financeiras de forma clara, usando dados calculados quando necessário.
Pergunta: {input}
""")

tools = [python_repl_tool, duckduckgo_tool]

agent = initialize_agent(
    llm=model,
    tools=tools,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    prompt=react_instructions,
    verbose=True,
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
)

question = '''
Minha renda é de R$10000 por mês, tenho muitos cartões de crédito no total de R$12000 por mês.
Tenho mais despesa de aluguel e combustível de R$1500.
Quais dicas você me dá?
'''

try:
    output = agent.invoke(
        {'input': prompt_template.format(question=question)}
    )
    print(output.get('output'))
except Exception as e:
    print(f"Erro ao executar o agente: {e}")
