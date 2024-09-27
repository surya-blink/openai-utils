import asyncio
from openai import AsyncOpenAI
from openai.types.beta import Thread


class OpenAIThread:
    def __init__(self, client: AsyncOpenAI, config: dict):
        """
        Initialize the OpenAI thread interface with necessary configuration.
        :param client: AsyncOpenAI, the OpenAI client to use for thread handling.
        :param config: dict, configuration settings for the thread handling.
        """
        self.client = client
        self.config = config
        self.assistant_id = config.get("assistant_id")
        self.thread = asyncio.run(self.create_thread())

    async def create_thread(self) -> Thread:
        """
        Create a new thread to start a conversation.
        :return: thread_id of the newly created thread.
        """
        return await self.client.beta.threads.create(
            messages=self.config.get("messages"),
            tool_resources=self.config.get("tool_resources"),
        )

    async def add_message_to_thread(self, message):
        """
        Add a message to an existing thread and get a response from the model.
        :param message: str, the user's input message.
        :return: str, the AI's response.
        """
        response = await self.client.beta.threads.messages.create(
            self.thread.id, role="user", content=message
        )
        return response

    @staticmethod
    def get_first_message_text(messages):
        try:
            # Attempt to extract the desired text based on the assumed structure
            text = messages[0][1][0].content[0].text.value
            return text
        except (IndexError, KeyError, AttributeError) as error:
            # Log the error or handle it as needed
            print(f"An error occurred while accessing message text: {error}")
            # Return a default value or None if the structure is not as expected
            return None

    async def run_thread(self, messages: list) -> str:
        """
        Asynchronously run the thread to get the AI's response.
        :param messages: list, the user's input message.
        :return: str, the AI's response.
        """
        run = await self.client.beta.threads.runs.create_and_poll(
            thread_id=self.thread.id,
            assistant_id=self.assistant_id,
            additional_messages=messages,
        )
        messages = list(
            await self.client.beta.threads.messages.list(
                thread_id=self.thread.id, run_id=run.id
            )
        )
        return self.get_first_message_text(messages)
