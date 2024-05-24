import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

os.environ["OPENAI_API_KEY"] = "xxx"
CHROMA_PATH = "chroma"
PROMPT_TEMPLATE = """
## 制約事項:
DirectoryLoaderを使用してください

## 文脈:
{context}

## 上記文脈に基づいた回答:
{question}
"""

query_text = "文脈の元になるナレッジがフォルダ内に複数のテキストファイルとして存在しており、全てのテキストファイルをLangChainで読み込ませる方法を教えて"
def query_rag(query_text):
  embedding_function = OpenAIEmbeddings()
  db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
  results = db.similarity_search_with_relevance_scores(query_text, k=3)
  if len(results) == 0 or results[0][1] < 0.7:
    print(f"Unable to find matching results.")

  context_text = "\n\n - -\n\n".join([doc.page_content for doc, _score in results])
  prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
  print(context_text)
  prompt = prompt_template.format(context=context_text, question=query_text)
  model = ChatOpenAI()
  response_text = model.predict(prompt)
  sources = [doc.metadata.get("source", None) for doc, _score in results]
  formatted_response = f"Response: {response_text}\nSources: {sources}"
  return formatted_response, response_text

formatted_response, response_text = query_rag(query_text)
print(response_text)
