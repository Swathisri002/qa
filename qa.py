
import os
import streamlit as st
import nltk
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from transformers import pipeline

# Ensure NLTK package is downloaded
nltk.download('punkt')

# Directories
UPLOAD_FOLDER = "pdf"
DB_FOLDER = "db"

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DB_FOLDER, exist_ok=True)
# Load environment variables

# Load models and setup
folder_path = "db"
pdf_folder_path = "pdf"
os.makedirs(folder_path, exist_ok=True)
os.makedirs(pdf_folder_path, exist_ok=True)

groq_api_key = st.secrets["secrets"]["GROQ_API_KEY"]
google_api_key = st.secrets["secrets"]["GOOGLE_API_KEY"]

cached_llm = ChatGroq(groq_api_key=groq_api_key, model="Gemma-7b-It")
embedding = GoogleGenerativeAIEmbeddings(api_key=google_api_key, model="models/embedding-001")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50, length_function=len)
raw_prompt = PromptTemplate.from_template(""" 
    <s>You are a technical assistant skilled at searching documents. Provide accurate answers using information present from the uploaded PDFs only. Search in the uploaded pdfs and give the best answer from it. If you do not have an answer from the provided information, say so. If the question is related to the words present in pdf you can also answer that.</s>
    [INST] {input} 
            Context: {context}
            Answer: 
    [/INST]
""")

def process_ask_pdf(query):
    index_path = os.path.join(folder_path, "index.faiss")
    if not os.path.exists(index_path):
        return "Error: FAISS index file does not exist. Please upload PDF files first to create the index."
    
    vector_store = FAISS.load_local(folder_path, embedding, allow_dangerous_deserialization=True)
    retriever = vector_store.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k": 5, "score_threshold": 0.2})
    document_chain = create_stuff_documents_chain(cached_llm, raw_prompt)
    chain = create_retrieval_chain(retriever, document_chain)
    result = chain.invoke({"input": query})
    return result["answer"]

def process_pdf(files):
    all_chunks = []
    file_names = []
    for file in files:
        file_name = file.name
        file_names.append(file_name)
        save_file = os.path.join(pdf_folder_path, file_name)
        with open(save_file, "wb") as f:
            f.write(file.getbuffer())
        loader = PDFPlumberLoader(save_file)
        docs = loader.load_and_split()
        chunks = text_splitter.split_documents(docs)
        all_chunks.extend(chunks)
    vector_store = FAISS.from_documents(documents=all_chunks, embedding=embedding)
    vector_store.save_local(folder_path)
    return {"status": "Successfully Uploaded", "filenames": file_names, "total_docs": len(all_chunks)}

def process_ask(question):
    index_path = os.path.join(folder_path, "index.faiss")
    if not os.path.exists(index_path):
        return "Error: FAISS index file does not exist. Please upload PDF files first to create the index."
    
    vector_store = FAISS.load_local(folder_path, embedding, allow_dangerous_deserialization=True)
    retriever = vector_store.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k": 5, "score_threshold": 0.2})
    document_chain = create_stuff_documents_chain(cached_llm, raw_prompt)
    chain = create_retrieval_chain(retriever, document_chain)
    result = chain.invoke({"input": question})
    return result["answer"]

def main():
    st.title('Chatbot')
    option = st.sidebar.selectbox('Choose an option:', ('Home', 'Ask PDF', 'Upload PDF'))

    if option == 'Home':
        st.write('Welcome to the Chatbot')

    elif option == 'Ask PDF':
        query = st.text_input('Enter your question for PDF:')
        if st.button('Ask'):
            response = process_ask_pdf(query)
            st.write('PDF Response:', response)

    elif option == 'Upload PDF':
        st.write('Upload your PDF file(s) here:')
        uploaded_files = st.file_uploader('Choose PDF files', type=['pdf'], accept_multiple_files=True)
        if st.button('Upload'):
            response = process_pdf(uploaded_files)
            st.write(response)

if __name__ == '__main__':
    main()