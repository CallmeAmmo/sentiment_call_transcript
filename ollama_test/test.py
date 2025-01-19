# import asyncio
# from ollama import AsyncClient

# async def chat():
#   message = {'role': 'user', 'content': 'Why is the sky blue?'}
#   async for part in await AsyncClient().chat(model='llama3.2', messages=[message], stream=True):
#     print(part['message']['content'], end='', flush=True)

# asyncio.run(chat())



from ollama import Client
client = Client(
  host='http://localhost:11434',
  headers={'x-some-header': 'some-value'}
)
response = client.chat(model='llama3.2', messages=[
  {
    'role': 'user',
    'content': 'What is you name in 2 words?',
  },
])