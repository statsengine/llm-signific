import streamlit as st
import regex as re
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    ServiceContext,
)
from llama_index.llms.openai import OpenAI

invalid_question_response = 'Jag kunde inte tolka din fråga, var god försök igen'

with open('./data/instructions.txt', 'r') as file:
    instructions = file.read().replace('\n', '')

@st.cache_resource(show_spinner=False)
def load_data():
    reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
    docs = reader.load_data()
    llm = OpenAI(temperature=0.1, max_tokens=1024, model="gpt-4-turbo")
    service_context = ServiceContext.from_defaults(llm=llm)
    index = VectorStoreIndex.from_documents(docs, service_context=service_context)
    return index

index = load_data()

st.title("Har du personalhandboksfrågor? :sunglasses:")

def handle_enter():
    user_input = st.session_state.text_input_value
    if bool(re.match(r'^[a-zA-Z0-9\s\?\å\ä\öÅÄÖ]*$', user_input)):
        try:
            query_engine = index.as_query_engine()
            enhanced_input = f"{instructions} {user_input}"
            response = query_engine.query(enhanced_input).response
            st.session_state.response = response
        except Exception as e:
            st.session_state.response = f"{invalid_question_response}: {e}"
    else:
        st.session_state.response = f"{invalid_question_response}."

text_input_value = st.text_input("Ställ en fråga och tryck enter:", placeholder="Lösenord till WiFi?", key='text_input_value', on_change=handle_enter)
spinner_placeholder = st.empty()

if 'response' in st.session_state:
    st.write(st.session_state.response)