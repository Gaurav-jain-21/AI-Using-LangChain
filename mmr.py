
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
# from langchain_community.retrievers import 

load_dotenv()

loader= PyPDFLoader('ml.pdf')
docs= loader.load()

embedding_model= HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore= FAISS.from_documents(
    documents=docs,
    embedding=embedding_model
)

retriever= vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k":3,"lambda_mult":1}
)


query="What is machine learning"
result= retriever.invoke(query)

for i, doc in enumerate(result):
    print(doc.page_content)
    