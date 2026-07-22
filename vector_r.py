from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader

loader= PyPDFLoader('dl-curriculum.pdf')

docs= loader.load()

embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore= Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    collection_name="chroma_db"
)

retriever= vectorstore.as_retriever(search_kwargs={"k":2})

query= "what is machine learning"

result  = retriever.invoke(query)

for i , doc in enumerate(result):
    print(doc.page_content)