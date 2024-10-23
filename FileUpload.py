import sys
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.document_loaders import OnlinePDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama

def upload_new_doc(local_path):
    if local_path:
      loader = UnstructuredPDFLoader(file_path=local_path)
      data = loader.load()
    else:
      return "Upload a PDF file"

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
    chunks = text_splitter.split_documents(data)

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=OllamaEmbeddings(model="nomic-embed-text", show_progress=True),
        collection_name="akf-docs",
        persist_directory="localDB",
    )

    vector_db.persist()

    '''query = "Quando foi criada a fundação?"
    docs = vector_db.similarity_search(query)
    print(docs[0].page_content)'''

if __name__ == "__main__":
  path = sys.argv[1]
  print(upload_new_doc(path))