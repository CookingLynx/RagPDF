import sys

from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

vector_db = Chroma(collection_name="akf-docs", persist_directory="localDB", embedding_function=OllamaEmbeddings(model="nomic-embed-text", show_progress=True),)

def query(text):
    # LLM from Ollama
    local_model = "llama3.1"
    llm = ChatOllama(model=local_model)
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""És um modelo de AI cuja A sua tarefa é gerar 5 versões diferentes da pergunta fornecida pelo utilizador para recuperar documentos relevantes de uma base de dados vetorial. 
        Ao gerar múltiplas perspetivas sobre a pergunta do utilizador, o seu objetivo é ajudar o utilizador a ultrapassar algumas das limitações da pesquisa baseada em similaridade por distância. 
        Forneça estas perguntas alternativas separadas por novas linhas.
        Pergunta original: {question}""",
    )

    retriever = vector_db.as_retriever()

    # RAG prompt
    template = """Responde a esta pergunta de forma muito completa e detalhada, baseado APENAS no seguinte contexto:
    {context}
    
    Pergunta: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )

    print(chain.invoke(text))


if __name__ == "__main__":
    text = sys.argv[1]
    query(text)