import os
import streamlit as st

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

DB_FAISS_PATH = "vectorstore/db_faiss"
#cacher la ressource  retourner par fct et le stocker en memoire 
@st.cache_resource
#chargement du model 
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def set_custom_prompt(custom_prompt_template):
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
#charger le model de generation du text 
def load_llm(huggingface_repo_id, HF_TOKEN):
    return HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        task="text-generation",
        model_kwargs={"token": HF_TOKEN, "max_length": "512"}
    )

def main():
    st.title("🧠 Medical Chatbot")

    # stocker dans st les meg entre uder et assistant pour garder un historique

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        #que ca soit le msg de user ou lassistant  il va etre afficher dans markdown  
        st.chat_message(message['role']).markdown(message['content'])

    prompt = st.chat_input("❓ Posez votre question ici")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        CUSTOM_PROMPT_TEMPLATE = """
        Use the pieces of information provided in the context to answer user's question.
        If you don't know the answer, just say that you don't know, don't try to make up an answer. 
        Don't provide anything out of the given context.

        Context: {context}
        Question: {question}

        Start the answer directly. No small talk please.
        """

        HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
        HF_TOKEN = os.environ.get("HF_TOKEN")

        try:
            #charger la base de donnee 
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("❌ Failed to load the vector store.")
                return
            
            #LANGCHAIN 
            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(HUGGINGFACE_REPO_ID, HF_TOKEN),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )

            response = qa_chain.invoke({'query': prompt})
            result = response["result"]
            sources = response["source_documents"]

            result_to_show = f"### ✅ Réponse :\n{result}\n\n"
            result_to_show += f"---\n### 📚 Extraits des sources :\n"

            for doc in sources:
                snippet = doc.page_content.strip().replace('\n', ' ')
                source_info = doc.metadata.get('source', 'Document inconnu')
                result_to_show += f"**Source**: `{source_info}`\n> {snippet[:400]}...\n\n"

            st.chat_message('assistant').markdown(result_to_show)
            st.session_state.messages.append({'role': 'assistant', 'content': result_to_show})

        except Exception as e:
            st.error(f"❌ Une erreur est survenue : {str(e)}")

if __name__ == "__main__":
    main()
