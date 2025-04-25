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
load_dotenv()#cette fct charge a partie nde env 
HF_TOKEN = os.getenv("HF_TOKEN")#os permet interagie avec le sys dexp pour lire des ficher ...
#ici os get va chercher la variable dans .env
#definir id du modele 
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

# creation du prompt pour guider et expliquer comment se comporter  llm dune maniere de formuler la reponse 
# dans context on va sauvgarder le contenu extrait a partir du pdf 
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Don't provide anything out of the given context.

Context: {context} 
Question: {question}

Start the answer directly. No small talk please.
"""
# creataion dune template personnalise de prompt 
def set_custom_prompt(template: str):
    return PromptTemplate(template=template, input_variables=["context", "question"])

# creer un interface pouur que le client interagir avec api et don doit definir le model 
client = InferenceClient(model=HUGGINGFACE_REPO_ID, token=HF_TOKEN)
#un wrapper 
#une class qui errite depuis LLM pour connectyer avec le model llm sur api 
#une methode pour generer une reponse avec un ppt 
class HFLLM(LLM):
    def _call(self, prompt: str, stop: List[str] = None) -> str:
        response = client.text_generation( #une methode de api puur generer une reponse et la stoker dans reponse  
            prompt,
            max_new_tokens=512,#limiter la reponse a 512 mots 
            temperature=0.5, #controle de la creativite 
            stop_sequences=stop or []
        )
        return response
    
    # @ pour rendre la methode accessible sans parentheses 
    # definir quil sagit de quel api 
    @property
    def _llm_type(self) -> str:
        return "huggingface_inference_client"

# chargement des  vectorstore
DB_FAISS_PATH = "vectorstore/db_faiss"
#creation dune  instance du modele deja entraine 
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#charger une base de donne a partie dun disque local 
# deserialization pour ignorer des verifications de securite 
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Construction de la chaîne question reponse 
#retrievalqa pour rcuperer les doc et usiliser llm 
qa_chain = RetrievalQA.from_chain_type(
    llm=HFLLM(),#definir le model 
    chain_type="stuff", #stuff pour recuper des doc assez long et les concatene
    #utiliser base de donne faiis pour recuper 3 doc les plus pertinents
    retriever=db.as_retriever(search_kwargs={"k": 3}),
    #retoune de la source 
    return_source_documents=True,
    #dictionnaire qui permet de specifier le model ppt perso 
    chain_type_kwargs={"prompt": set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

# 🚀 Utilisation
user_query = input("Posez votre question ici : ")
try:
    #qa_chain deja definit pour questsio  reponse invke permet denvoyer la q de user 
    response = qa_chain.invoke({"query": user_query})
    print("\n🟢 RÉPONSE :")
    print(response["result"])
    print("\n📚 DOCUMENTS SOURCES :")
    for doc in response["source_documents"]:
        #doc.metadata ppur recuperer le label de la page
        #doc.page pour afficher le contenu de 200 premier cara
        print(f"-> Page {doc.metadata.get('page_label', 'N/A')} - {doc.page_content[:200]}...\n")
except Exception as e:
    print("❌ Une erreur s'est produite :", str(e))
