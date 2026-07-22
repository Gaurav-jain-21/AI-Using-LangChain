from langchain_community.document_loaders import PyPDFLoader
from langchain

loader= PyPDFLoader('dl-curriculum.pdf')
docs= loader.load()

print(docs)