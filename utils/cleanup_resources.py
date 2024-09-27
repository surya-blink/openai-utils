from openai import OpenAI


def cleanup_vector_stores(client: OpenAI):
    try:
        vector_stores = client.beta.vector_stores.list()
        for vector_store_id in vector_stores:
            print(vector_store_id.id)
            client.beta.vector_stores.delete(vector_store_id.id)
    except Exception as e:
        print(e)


def cleanup_files(client: OpenAI):
    try:
        files = client.files.list()
        for file in files:
            print(file.id)
            client.files.delete(file.id)
    except Exception as e:
        print(e)


def cleanup_assistants(client: OpenAI):
    try:
        assistants = client.beta.assistants.list()
        for assistant in assistants:
            print(assistant.id)
            client.beta.assistants.delete(assistant.id)
    except Exception as e:
        print(e)
