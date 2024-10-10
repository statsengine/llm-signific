import os
import streamlit as st
import regex as re
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    StorageContext,
    load_index_from_storage
)
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter

invalid_question_response = 'Jag kunde inte tolka din fråga, var god försök igen.'

with open('./data/personalhandbok/instruktioner.txt', 'r') as file:
    instructions = file.read()

tab1, tab2 = st.tabs(["Personalhandbokshjälpare", "TBA"])

@st.cache_resource(show_spinner=False)
def load_data():
    PERSIST_DIR = "./storage/personalhandbok_cleaned"

    if not os.path.exists(PERSIST_DIR):
        os.makedirs(PERSIST_DIR, exist_ok=True)
        reader = SimpleDirectoryReader(input_dir="./data/personalhandbok/clean", recursive=True)
        documents = reader.load_data()

        Settings.llm = OpenAI(model="gpt-4")  
        Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002")  
        Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)

        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir=PERSIST_DIR)
    else:
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context)

    return index

def get_qa_prompt(instructions, user_input):
    return (
        "Instruktioner:\n"
        "---------------------\n"
        f"{instructions}\n"
        "---------------------\n"
        "Svara på följande användarfråga.\n"
        f"Fråga: {user_input}\n"
    )

def handle_enter():
    user_input = st.session_state.text_input_value
    if bool(re.match(r'^[a-zA-Z0-9\s\?\!\å\ä\öÅÄÖ\.\-\/]*$', user_input)):
        try:
            query_engine = index.as_query_engine()  
            response = query_engine.query(get_qa_prompt(instructions, user_input)).response
            st.session_state.response = response
        except Exception as e:
            st.session_state.response = f"{invalid_question_response}: {e}"
    else:
        st.session_state.response = f"{invalid_question_response}."

with tab1:
    st.header("")
    st.title("Har du personalhandboksfrågor? :sunglasses:")

    index = load_data()
    text_input_value = st.text_input("Ställ en fråga och tryck enter:", placeholder="Lösenord till WiFi?", key='text_input_value', on_change=handle_enter)

    if 'response' in st.session_state:
        st.write(st.session_state.response)

with tab2:
    st.header("")
    st.title("TBA :mage:")
