from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


#chargement du fichier 
DATA_PATH="data/"#lemplacement 
def load_pdf_files(data):
    #ouvrir le dossier par directoryloader avec lextention pdf
    #avec pypdf ouvrir les ppdf separemrnt 
    loader = DirectoryLoader(data,
                             glob='*.pdf',
                             loader_cls=PyPDFLoader)
    
    documents=loader.load()
    return documents
#utiliser la fct deja declarer 
documents=load_pdf_files(data=DATA_PATH)#donner le path a ouvrir 
print("Length of PDF pages: ", len(documents))
# creation des morceaux les chunks
#utiliser des fct pour lle decoupage 
#chunk overlap le chevauchement pour ne pas perdre le sens 
def create_chunks(extracted_data):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,
                                                 chunk_overlap=50)
    text_chunks=text_splitter.split_documents(extracted_data)
    return text_chunks
#appliquer le sur notre doc 
text_chunks=create_chunks(extracted_data=documents)

print("Length of Text Chunks: ", len(text_chunks))
#creation des embedding les vecteurs 
#utiliser un model deja entraine de hugging face 
def get_embedding_model():
    embedding_model=HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")
    return embedding_model

embedding_model=get_embedding_model()

# FAISS biblio creer par fcb AI pour la recherche rapide des donnees vectorielle 
#definir le path de ficher qui va etre creer
DB_FAISS_PATH="vectorstore/db_faiss"
#construire une base vectorille via le model de embedding 
db=FAISS.from_documents(text_chunks, embedding_model)
#enregister la base localement 
db.save_local(DB_FAISS_PATH)

#NLP naturel language processing 
# cest un model NLP , il sagit de la transformation de question de user en vecteur alors ce model
#sert a comprnedre le sens de la question de user 
#LLM large language model comme chatgpt llama
#prendre un text en entre avec un prompt (cest la question ) et produire une reponse


#la relation entre tous 
#le NLP : Analyse de text 
#embedding : convertire en vecteur 
#FAISS: chercher les doc pertinents 
#LLM: generer une reponse clair 