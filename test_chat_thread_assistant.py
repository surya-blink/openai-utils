import time

from openai_tools.chat.chatbot import ChatBotOpenAIChat, ChatBotAssistantThread
from openai_tools.utils.assistant import Assistant

if __name__ == "__main__":
    from openai_tools.openai_client import client

    config = {
        "initial_messages": [
            {"role": "system", "content": "You are a helpful assistant for agents to answer user queries."},
        ],
        "instructions": "<>",
        "name": "My Assistant",
        "model": "gpt-4o",
    }

    # config.update({"assistant_id": Assistant(client, config=config).assistant.id})
    config.update({"assistant_id":"<>"})
    chatbot = ChatBotAssistantThread(client, config=config)

    while True:
        user_input = input("You: ")
        t = time.time()
        rsp = chatbot.chat([{"role": "user", "content": user_input}])
        print(rsp)
        print(f"Time taken: {time.time() - t}")
