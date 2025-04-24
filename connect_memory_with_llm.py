import os
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from huggingface_hub import InferenceClient
from langchain_core.language_models import LLM
from typing import List

# Charger les variables d'environnement
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

# Custom Prompt
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Don't provide anything out of the given context.

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

def set_custom_prompt(template: str):
    return PromptTemplate(template=template, input_variables=["context", "question"])

# 🔥 Client HuggingFace
client = InferenceClient(model=HUGGINGFACE_REPO_ID, token=HF_TOKEN)

# ✅ LangChain-compatible wrapper autour d'InferenceClient
class HFLLM(LLM):
    def _call(self, prompt: str, stop: List[str] = None) -> str:
        response = client.text_generation(
            prompt,
            max_new_tokens=512,
            temperature=0.5,
            stop_sequences=stop or []
        )
        return response

    @property
    def _llm_type(self) -> str:
        return "huggingface_inference_client"

# 🔁 Load vectorstore
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# 🔗 Construction de la chaîne
qa_chain = RetrievalQA.from_chain_type(
    llm=HFLLM(),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

# 🚀 Utilisation
user_query = input("Posez votre question ici : ")
try:
    response = qa_chain.invoke({"query": user_query})
    print("\n🟢 RÉPONSE :")
    print(response["result"])
    print("\n📚 DOCUMENTS SOURCES :")
    for doc in response["source_documents"]:
        print(f"-> Page {doc.metadata.get('page_label', 'N/A')} - {doc.page_content[:200]}...\n")
except Exception as e:
    print("❌ Une erreur s'est produite :", str(e))
