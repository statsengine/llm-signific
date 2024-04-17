from sys import argv
import os.path
import streamlit as st
import openai

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    ServiceContext,
    load_index_from_storage
)
from llama_index.llms.openai import OpenAI


@st.cache_resource(show_spinner=False)
def load_data():
    reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
    docs = reader.load_data()
    llm = OpenAI(temperature=0, max_tokens=1024, model="gpt-4-turbo")
    service_context = ServiceContext.from_defaults(llm=llm)
    index = VectorStoreIndex.from_documents(docs, service_context=service_context)
    return index

index = load_data()

st.title("Har du någon fråga om personalhandboken? :sunglasses:")

def handle_enter():
    user_input = st.session_state.text_input_value
    with spinner_placeholder.container():
        st.spinner("Processing...")
        try:
            query_engine = index.as_query_engine()
            response = query_engine.query(user_input).response
            st.session_state.response = response
        except Exception as e:
            st.session_state.response = f"Failed to process the query: {e}"

text_input_value = st.text_input("Ställ en fråga och tryck enter:", key='text_input_value', on_change=handle_enter)

spinner_placeholder = st.empty()

if 'response' in st.session_state:
    st.write(st.session_state.response)