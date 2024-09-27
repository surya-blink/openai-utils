from sentence_transformers import SentenceTransformer  # else give segmentation fault
from langchain_community.embeddings import (
    HuggingFaceEmbeddings,
)
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
import os
import faiss
import cloudpickle


class FAISSRetriever:
    def __init__(self):
        self.retriever = None

    async def async_get_relevant_documents(self, query_string, k=3):
        docs = await self.retriever.ainvoke(query_string)
        return docs[:k]

    def get_relevant_documents(self, query_string, k=4):
        docs = self.retriever.invoke(query_string)
        return docs[:k]

    def load_vector_db(self, path):
        # Initialize variables for the components of the database.
        memory_doc_store_dict = {}
        index_to_doc_store_id_dict = {}

        # Check if the database already exists. If it does, load its components.
        if os.path.exists(path):
            memory_doc_store_dict = cloudpickle.load(
                open(path + "memoryDocStoreDict.pkl", "rb")
            )
            index_to_doc_store_id_dict = cloudpickle.load(
                open(path + "indexToDocStoreIdDict.pkl", "rb")
            )
            index = faiss.read_index(path + "faiss.index")
        else:
            index = faiss.IndexFlatL2(1024)

        # Create the FAISS vector database with the loaded or new components.
        vector_db = FAISS(
            index=index,
            docstore=InMemoryDocstore(memory_doc_store_dict),
            index_to_docstore_id=index_to_doc_store_id_dict,
            embedding_function=HuggingFaceEmbeddings(model_name="BAAI/bge-m3"),
        )
        print("Loading local vector DB")
        self.retriever = vector_db.as_retriever()
        return self


# db_retriever = FAISSRetriever().load_vector_db(path="../data/db/")
#
# while True:
#     query_string = input("Enter query: ")
#     results = db_retriever.get_relevant_documents(query_string, k=2)
#
#     for result in results:
#         print(result)
