from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma  
from langchain_huggingface import HuggingFaceEmbeddings

loader = PyPDFLoader("dl-curriculum.pdf")
docs = loader.load()

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_store = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="chroma_db",
    collection_name="sample",
)


vector_store.add_documents(docs)
# retriever = vector_store.as_retriever(
#     search_type="similarity", search_kwargs={"k": 3}
# )