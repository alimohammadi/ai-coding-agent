from client.llm_client import LLMClient
from dotenv import load_dotenv
import asyncio

load_dotenv()


async def main():
    client = LLMClient()
    messages = [{"role": "user", "content": "what's up?"}]

    async for event in client.chat_completion(messages, False):
        print(event)

    print("done")


asyncio.run(main())
