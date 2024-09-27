from langchain_community.document_loaders.text import TextLoader
from langchain_openai import OpenAIEmbeddings

from langchain_community.vectorstores.pgvector import PGVector

import os
import time

from langchain_text_splitters import CharacterTextSplitter

from rag.services.pgvector_service import PgvectorService
from utils.document_loader import DocumentLoader
from langchain_community.embeddings import HuggingFaceEmbeddings

model_name = "BAAI/bge-small-en"
model_name = "BAAI/bge-m3"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}
hf = HuggingFaceEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)

embedding = hf.embed_query("hi this is harrison")
print(len(embedding))

# --------------------------------------------------------------
# Load the documents
# --------------------------------------------------------------

doc_loader = DocumentLoader("../data/zendesk_articles")
docs = doc_loader.load()

print(docs)
embeddings = hf

query = "where is my order?"

# --------------------------------------------------------------
# Query the index with LanChain
# --------------------------------------------------------------


def calculate_average_execution_time(func, *args, **kwargs):
    total_execution_time = 0
    num_runs = 1
    for _ in range(num_runs):
        start_time = time.time()
        result = func(*args, **kwargs)  # Execute the function with its arguments
        end_time = time.time()
        execution_time = end_time - start_time
        total_execution_time += execution_time
    average_execution_time = round(total_execution_time / num_runs, 2)
    print(result)
    print(
        f"\nThe function took an average of {average_execution_time} seconds to execute."
    )
    return


# --------------------------------------------------------------
# Create a PGVector Store
# --------------------------------------------------------------

"""
"""

COLLECTION_NAME = "The Project Gutenberg eBook of A Christmas Carol in Prose"

CONNECTION_STRING = PGVector.connection_string_from_db_params(
    driver=os.environ.get("PGVECTOR_DRIVER", "psycopg2"),
    host=os.environ.get("PGVECTOR_HOST", "localhost"),
    port=int(os.environ.get("PGVECTOR_PORT", "5444")),
    database=os.environ.get("PGVECTOR_DATABASE", "pgvector"),
    user=os.environ.get("PGVECTOR_USER", "admin"),
    password=os.environ.get("PGVECTOR_PASSWORD", "password"),
)
print(CONNECTION_STRING)
# create the store
db = PGVector.from_documents(
    embedding=embeddings,
    documents=docs,
    collection_name=COLLECTION_NAME,
    connection_string=CONNECTION_STRING,
    pre_delete_collection=False,
)


# load the store
pgvector_docsearch = PGVector(
    collection_name=COLLECTION_NAME,
    connection_string=CONNECTION_STRING,
    embedding_function=embeddings,
)

# --------------------------------------------------------------
# Query the index with PGVector
# --------------------------------------------------------------


def run_query_pgvector(docsearch, query):
    docs = docsearch.similarity_search(query, k=4)
    result = docs[0].page_content
    return result


calculate_average_execution_time(
    run_query_pgvector, docsearch=pgvector_docsearch, query=query
)


# --------------------------------------------------------------
# Add more collections to the database
# --------------------------------------------------------------
#
loader = TextLoader("../data/The Project Gutenberg eBook of Romeo and Juliet.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=100000, chunk_overlap=0)
new_docs = text_splitter.split_documents(documents)


COLLECTION_NAME_2 = "The Project Gutenberg eBook of Romeo and Juliet"

db = PGVector.from_documents(
    embedding=embeddings,
    documents=new_docs,
    collection_name=COLLECTION_NAME_2,
    connection_string=CONNECTION_STRING,
    pre_delete_collection=False,
)


# --------------------------------------------------------------
# Query the index with multiple collections
# --------------------------------------------------------------

pg = PgvectorService(CONNECTION_STRING)


def run_query_multi_pgvector(docsearch, query):
    docs = docsearch.custom_similarity_search_with_scores(query, k=4)
    result = docs[0][0].page_content
    print(result)


run_query_multi_pgvector(pg, query)

# # --------------------------------------------------------------
# # Delete the collection
# # --------------------------------------------------------------
# pg.delete_collection(COLLECTION_NAME)
# pg.delete_collection(COLLECTION_NAME_2)
#
# # --------------------------------------------------------------
# # Update the collection
# # --------------------------------------------------------------
# pg.update_collection(docs=docs, collection_name=COLLECTION_NAME)
