from openai import OpenAI
import os
client = OpenAI()


# Create a vector store caled "Financial Statements"
vector_store = client.beta.vector_stores.create(name="Knowledge-Base-Public")

# Ready the files for upload to OpenAI
file_paths  = [f"../data/kb/{file}" for file in os.listdir("../data/kb/")]
file_streams = [open(path, "rb") for path in file_paths]

# Use the upload and poll SDK helper to upload the files, add them to the vector store,
# and poll the status of the file batch for completion.
file_batch = client.beta.vector_stores.file_batches.upload_and_poll(
    vector_store_id=vector_store.id, files=file_streams
)

# You can print the status and the file counts of the batch to see the result of this operation.
# print(vector_store)
print(file_batch.status)
print(file_batch.file_counts)
