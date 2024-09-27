import asyncio

from openai import AsyncOpenAI


class Assistant:
    def __init__(self, client: AsyncOpenAI, config: dict = {}):
        self.client = client
        self.config = config
        self.assistant = asyncio.run(self.create_assistant())

    async def create_assistant(self):
        # Create the assistant
        assistant = await self.client.beta.assistants.create(
            name=self.config["name"],
            instructions=self.config["instructions"],
            model=self.config.get("model", "gpt-3.5-turbo"),
            tools=self.config.get("tools", []),
            tool_resources=self.config.get("tool_resources", {}),
        )
        return assistant
