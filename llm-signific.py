from sys import argv
import streamlit as st
import os.path
from llama_index.core import (
    KeywordTableIndex,
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings

PERSIST_DIR = "./storage"
if not os.path.exists(PERSIST_DIR):
    documents = SimpleDirectoryReader("data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)

llm = OpenAI(temperature=0, max_tokens=1024, model="gpt-4-turbo")
documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents, llm=llm)

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