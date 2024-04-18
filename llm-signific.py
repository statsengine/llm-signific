import os
import streamlit as st
import regex as re
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    ServiceContext,
    StorageContext,
    load_index_from_storage
)
from llama_index.llms.openai import OpenAI

invalid_question_response = 'Jag kunde inte tolka din fråga, var god försök igen'

with open('./data/instructions.txt', 'r') as file:
    instructions = file.read().replace('\n', '')

tab1, tab2 = st.tabs(["Personalhandbokshjälpare", "TBA"])

@st.cache_resource(show_spinner=False)
def load_data():
    PERSIST_DIR = "./storage"

    if not os.path.exists(PERSIST_DIR):
        os.makedirs(PERSIST_DIR, exist_ok=True)
        reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
        documents = reader.load_data()
        llm = OpenAI(temperature=0.1, max_tokens=1024, model="gpt-4-turbo")
        service_context = ServiceContext.from_defaults(llm=llm)
        index = VectorStoreIndex.from_documents(documents, service_context=service_context)
        index.storage_context.persist(persist_dir=PERSIST_DIR)
    else:
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context)

    return index

def get_qa_prompt(instructions, user_input):
    return (
        "Instruktioner:.\n"
        "---------------------\n"
        f"{instructions}\n"
        "---------------------\n"
        "Svara på följande användarfråga.\n"
        f"Fråga: {user_input}\n"
    )

def handle_enter():
    user_input = st.session_state.text_input_value
    if bool(re.match(r'^[a-zA-Z0-9\s\?\å\ä\öÅÄÖ]*$', user_input)):
        try:
            query_engine = index.as_query_engine()
            print(get_qa_prompt(instructions, user_input))
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