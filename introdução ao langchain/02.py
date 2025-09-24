import os
from decouple import config
from langchain_google_genai import GoogleGenerativeAI
from langchain_community.cache import SQLiteCache
from langchain.globals import set_llm_cache


os.environ['GOOGLE_API_KEY'] = config('GEMINI_API_KEY')

llm = GoogleGenerativeAI(model='gemini-2.5-pro', temperature=1)

set_llm_cache(
    SQLiteCache(database_path='cache.db')
)

prompt = 'Quem foi Alan Turing?'

llm_response1 = llm.invoke(
    prompt,
    config={'max_output_tokens': 500}
)

print(llm_response1)

llm_response2 = llm.invoke(
    prompt,
    config={'max_output_tokens': 500}
)

print(llm_response2)
