import time

from openai_tools.chat.chatbot import ChatBotOpenAIChat, ChatBotAssistantThread
from openai_tools.utils.assistant import Assistant

if __name__ == "__main__":
    from openai_tools.openai_client import client

    config = {
        "initial_messages": [
            {"role": "system", "content": "You are a helpful finance assistant."}
        ],
        "instructions": "You are a helpful finance assistant.",
        "name": "Finance Assistant",
        "model": "gpt-3.5-turbo",
    }
    chatbot = ChatBotOpenAIChat(client, config=config)

    while True:
        user_input = input("You: ")
        t = time.time()
        rsp = chatbot.chat([{"role": "user", "content": user_input}])
        print(rsp)
        print(f"Time taken: {time.time() - t}")
