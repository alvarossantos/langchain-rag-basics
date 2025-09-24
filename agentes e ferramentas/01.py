from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_experimental.utilities import PythonREPL


# DuckDuckGo Search
ddg_search = DuckDuckGoSearchRun()
search_result = ddg_search.run('Quem foi Alan Turing?')
print(search_result)

# Python REPL
python_repl = PythonREPL()
result = python_repl.run('print(5+5)')
print(result)

# Wikipedia Query
wikipedia = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper(
        lang='pt'
    )
)
wikipedia_results = wikipedia.run('Quem foi Alan Turing?')
print(wikipedia_results)
