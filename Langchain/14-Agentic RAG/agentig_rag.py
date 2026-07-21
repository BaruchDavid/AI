import os
from dotenv import load_dotenv
import discord
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools.retriever import create_retriever_tool
from typing import Annotated, Sequence

load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

### Datenquelle 1

urls = [
    "https://www.nature.com/articles/d41586-026-01344-8",
    "https://www.nature.com/articles/d41586-026-01341-x",
    "https://www.nature.com/articles/d41586-026-01257-6",
]

# erzeuge eine Liste von Dokumenten, indem du die URLs lädst
docs = [WebBaseLoader(urls).load() for url in urls]

# text_splitter erzeugen, um die Dokumente in kleinere Teile zu zerlegen, damit sie besser verarbeitet werden können
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# die Dokumente in kleinere Teile zerlegen
docs_split = text_splitter.split_documents(docs)

# einen Vektorstore erstellen, um die Dokumente zu speichern und später abzufragen
vectorstore = FAISS.from_documents(documents=docs_split, embedding=OpenAIEmbeddings())

# einen Retriever erstellen, um die Dokumente abzufragen
retriever = vectorstore.as_retriever()

### Retriever to Retriever_Tool, because we need  to intergrate it with other LLMs
# erstelle ein Retriever-Tool, das den Retriever verwendet, um relevante Informationen aus den Dokumenten abzurufen
# das ist eine Quelle mit Informationen, die von anderen LLMs abgefragt werden kann, um Antworten auf Benutzeranfragen zu generieren
### Jetzt können wir das Retriever-Tool in einem Agenten verwenden, um Antworten auf Benutzeranfragen zu generieren, indem wir relevante Informationen aus den Dokumenten abrufen.
retriever_tool = create_retriever_tool(
    retriever=retriever,
    name="naturedotcomRetriever",
    description="Use this tool to retrieve relevant information from the Groq API based on the user's query.",
)

### Datenquelle 2

urls2 = [
    "https://www.wissenschaft.de/naturplus/aus-dem-schatten-des-urwalds/",
    "https://www.wissenschaft.de/erde-umwelt/die-wiege-der-menschheit-bricht-auseinander/",
    "https://www.wissenschaft.de/astronomie-physik/gravitationskonstante-bleibt-eine-harte-nuss/",
]

# erzeuge eine Liste von Dokumenten, indem du die URLs lädst
docs2 = [WebBaseLoader(urls2).load() for url in urls2]

# die Dokumente in kleinere Teile zerlegen
docs_split2 = text_splitter.split_documents(docs)

# einen Vektorstore erstellen, um die Dokumente zu speichern und später abzufragen
vectorstore2 = FAISS.from_documents(documents=docs_split2, embedding=OpenAIEmbeddings())

# einen Retriever erstellen, um die Dokumente abzufragen
retriever2 = vectorstore2.as_retriever()

retriever_tool_2 = create_retriever_tool(
    retriever=retriever,
    name="wissenschaftdotdeRetriever",
    description="Use this tool to retrieve relevant information from the Groq API based on the user's query.",
)

# der Agentive RAG Agent kann jetzt beide Retriever-Tools verwenden, um relevante Informationen aus beiden Datenquellen abzurufen
# um abschließend von der zweiten LLM entscheiden zu können, ob die Antwort  aus der Quelle1 oder Quelle2 oke ist. 
# falls die Antwort oke ist, dann wird eine Antwort generiert
# falls die Antwort nicht oke ist, dann wird es dem Agenten mitgeteilt und es geht von vorne los.
tools = [retriever_tool, retriever_tool_2]