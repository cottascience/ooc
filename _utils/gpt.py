import os, asyncio, nest_asyncio, random
from openai import AsyncOpenAI

nest_asyncio.apply()
client = AsyncOpenAI()

async def predict(system, user, user_input, max_tokens, temperature, extra_system=""):
    system = system.format(**user_input)
    user = user.format(**user_input)
    response = await client.chat.completions.create(
    model = os.environ["CHAT_MODEL"],
    messages=[
        { "role": "system",
        "content": system },
        { "role": "user",
        "content": user },
        { "role": "system",
        "content": extra_system } ],
    temperature=temperature,
    max_tokens=max_tokens,
    top_p=1
    )
    return response.choices[0].message.content
