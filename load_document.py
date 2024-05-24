import os
from langchain_community.document_loaders import GitLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.schema import Document

os.environ["OPENAI_API_KEY"] = "xxx"

def file_filter(file_path):
  return file_path.endswith(".py")

def load_documents():
  loader = GitLoader(
    clone_url="https://github.com/langchain-ai/langchain",
    repo_path="./langchain",
    branch="master",
    file_filter=file_filter,
  )
  return loader.load()

def split_text(documents: list[Document]):
  # text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
  text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300, # Size of each chunk in characters
    chunk_overlap=100, # Overlap between consecutive chunks
    length_function=len, # Function to compute the length of the text
    add_start_index=True, # Flag to add start index to each chunk
  )

  # Split documents into smaller chunks using text splitter
  chunks = text_splitter.split_documents(documents)
  print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

  # Print example of page content and metadata for a chunk
  document = chunks[0]
  print(document.page_content)
  print(document.metadata)
  return chunks # Return the list of split text chunks

CHROMA_PATH = "chroma"
def save_to_chroma(chunks: list[Document]):
  # Clear out the existing database directory if it exists
  if os.path.exists(CHROMA_PATH):
    shutil.rmtree(CHROMA_PATH)

  # Create a new Chroma database from the documents using OpenAI embeddings
  db = Chroma.from_documents(
    chunks,
    OpenAIEmbeddings(),
    persist_directory=CHROMA_PATH,
  )
  # Persist the database to disk
  db.persist()
  print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

def generate_data_store():
  documents = load_documents()
  chunks = split_text(documents)
  save_to_chroma(chunks)
generate_data_store()
