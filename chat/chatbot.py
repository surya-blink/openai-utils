import asyncio
from abc import abstractmethod
from openai import AsyncOpenAI

from openai_tools.utils.threads import OpenAIThread


class ChatBotConfig:
    def __init__(
        self,
        model: str,
        temperature: float,
        max_tokens: int,
        top_p: int,
        frequency_penalty: int,
        presence_penalty: int,
        initial_messages: list = [],
        assistant_id: str = None,
        tool_resources: str = None,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.assistant_id = assistant_id
        self.initial_messages = initial_messages
        self.tool_resources = tool_resources

    @staticmethod
    def load_config(config: dict):
        model = config.get("model", "gpt-3.5-turbo-16k")
        temperature = config.get("temperature", 0.7)
        max_tokens = config.get("max_tokens", 256)
        top_p = config.get("top_p", 1)
        frequency_penalty = config.get("frequency_penalty", 0)
        presence_penalty = config.get("presence_penalty", 0)
        initial_messages = config.get("initial_messages", [])
        assistant_id = config.get("assistant_id")
        tool_resources = config.get("tool_resources")
        return ChatBotConfig(
            model,
            temperature,
            max_tokens,
            top_p,
            frequency_penalty,
            presence_penalty,
            initial_messages,
            assistant_id,
            tool_resources,
        )

    def get_thread_config(self):
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "initial_messages": self.initial_messages,
            "assistant_id": self.assistant_id,
            "tool_resources": self.tool_resources,
        }

    def get_chat_config(self):
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
        }


class ChatBotInterface:
    def __init__(self, client: AsyncOpenAI, config: dict = {}):
        self.client: AsyncOpenAI = client
        self.config: ChatBotConfig = ChatBotConfig.load_config(config)

    @abstractmethod
    def run(self, messages: list):
        raise NotImplementedError

    @abstractmethod
    async def chat_async(self, messages: list):
        return await self.run(messages)

    @abstractmethod
    def chat(self, messages: list):
        raise NotImplementedError


class ChatBotOpenAIChat(ChatBotInterface):
    messages = []

    def __init__(self, client: AsyncOpenAI, config: dict = {}):
        super().__init__(client, config)
        self.messages = self.config.initial_messages

    def set_messages_payload(self, messages: list):
        self.messages = self.messages + messages

    def update_assistant_message(self, message: str):
        self.messages.append({"role": "assistant", "content": message})

    async def run(self, messages: list):
        self.set_messages_payload(messages)
        try:
            response = await self.client.chat.completions.create(
                messages=self.messages,
                **self.config.get_chat_config(),
            )
            print(response)
            message = (
                response.choices[0].message.content
                if response.choices
                else "No response"
            )
            self.update_assistant_message(message)
            return message
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    async def chat_async(self, messages: list):
        return await self.run(messages)

    def chat(self, messages: list):
        response = asyncio.run(self.chat_async(messages))
        return response


class ChatBotAssistantThread(ChatBotInterface):

    def __init__(self, client: AsyncOpenAI, config: dict = {}):
        super().__init__(client, config)
        self.thread = OpenAIThread(client, config)

    async def run(self, messages: list) -> str | None:
        try:
            message = await self.thread.run_thread(messages)
            return message
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    async def chat_async(self, messages: list):
        return await self.run(messages)

    def chat(self, messages: list):
        response = asyncio.run(self.chat_async(messages))
        return response
