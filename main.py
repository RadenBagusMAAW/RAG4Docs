import os
import streamlit as st
import time
import langchain
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

st.title('RAG4Docs - Retrieval-Augmented Generation for PDFs 📚')
st.sidebar.title('Add Your Documents (.pdf only)')

pdfs = []
for i in range(3):
    pdf = st.sidebar.file_uploader(f"Upload PDF {i+1}",
                                  type=["pdf"],
                                  key=f"pdf_uploader_{i}")
    pdfs.append(pdf)

process_button_clicked = st.sidebar.button('Process')
file_path = 'vector_index_openAI'

main_placeholder = st.empty()
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.5, max_tokens=1000)
embeddings = OpenAIEmbeddings()

if process_button_clicked:
    if not any(pdfs):
        st.error("Please upload at least one PDF before processing.")
    else:
        saved_files = []
        for i, pdf in enumerate(pdfs):
            if pdf is not None:
                temp_filename = f"temp_pdf_{i}.pdf"
                with open(temp_filename, "wb") as f:
                    f.write(pdf.getbuffer())
                saved_files.append(temp_filename)

        # Load data
        pdf_loaders = [PyPDFLoader(file) for file in saved_files]
        main_placeholder.text("📥 Loading PDF data... Please wait.")
        pdf_data = [loader.load() for loader in pdf_loaders]
        all_pdf_data = [doc for sublist in pdf_data for doc in sublist]

        # Split data
        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '.', ','],
            chunk_size=1000
        )
        main_placeholder.text("🔍 Splitting text into chunks for processing...")
        docs = text_splitter.split_documents(all_pdf_data)

        # Create embeddings and save to FAISS index
        vectorindex_openai = FAISS.from_documents(docs, embeddings)
        main_placeholder.text("🧠 Generating embeddings and building the vector index...")
        time.sleep(2)

        vectorindex_openai.save_local(file_path)

        # Optional: Clean up temporary files
        for file in saved_files:
            os.remove(file)

query = main_placeholder.text_input("Question: ")

if query:
     if os.path.exists(file_path):
        vectorIndex = FAISS.load_local(file_path, embeddings, allow_dangerous_deserialization=True)
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorIndex.as_retriever())
        result = chain({"question": query}, return_only_outputs=True)
        st.header("Answer")
        st.write(result["answer"])


st.subheader('By: Adryan Putra')
