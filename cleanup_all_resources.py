from openai_tools.utils.cleanup_resources import (
    cleanup_assistants,
    cleanup_files,
    cleanup_vector_stores,
)

if __name__ == "__main__":
    from openai_tools.openai_client import client_sync as client

    cleanup_vector_stores(client)
    cleanup_files(client)
    cleanup_assistants(client)
