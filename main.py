import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import LlamaCpp
from langchain.chains import RetrievalQA
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.text_splitter import RecursiveCharacterTextSplitter
import warnings
import os

warnings.filterwarnings("ignore")

DATA_PATH = 'input_pdfs/'
DB_FAISS_PATH = 'vectorstore/db_faiss'

## Default LLaMA-2 prompt style
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
"""

def get_prompt(instruction, new_system_prompt=DEFAULT_SYSTEM_PROMPT):
    SYSTEM_PROMPT = B_SYS + new_system_prompt + E_SYS
    prompt_template =  B_INST + SYSTEM_PROMPT + instruction + E_INST
    return prompt_template

sys_prompt = """select response only from stored database."""

instruction = """CONTEXT:/n/n {context}/n

Question: {question}"""
get_prompt(instruction, sys_prompt)

prompt_template = get_prompt(instruction, sys_prompt)

llama_prompt = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

chain_type_kwargs = {"prompt": llama_prompt}

def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                    chain_type='stuff',
                                    retriever=db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold":0.3,"k": 4}),
                                    return_source_documents=True,
                                    chain_type_kwargs=chain_type_kwargs
                                    )
    return qa_chain

def load_llm():
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    llm = LlamaCpp(
        model_path="llama-2-7b-chat.Q4_0.gguf",
        n_gpu_layers=256,
        n_batch=512,
        max_tokens=2000,
        temperature=0.4,
        callback_manager=callback_manager,
        verbose=True,
    )
    return llm

def qa_bot(upload_option, uploaded_file):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    
    if upload_option:
        st.sidebar.success("PDF Uploaded successfully!")
        
        DATA_PATH = 'input_pdfs/'
        file_path = os.path.join(DATA_PATH, uploaded_file.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        texts = text_splitter.split_documents(documents)
        db = FAISS.from_documents(texts, embeddings)
        db.save_local(DB_FAISS_PATH)
        st.sidebar.success("Vector DB created and saved locally.")
        
        llm = load_llm()
        qa_prompt = llama_prompt
        qa = retrieval_qa_chain(llm, qa_prompt, db)
        st.sidebar.success("Retrieval QA chain created.")
        
        return qa

    else:
        db = FAISS.load_local(DB_FAISS_PATH, embeddings)
        st.sidebar.info("Using existing Vector DB.")

        llm = load_llm()
        qa_prompt = llama_prompt
        qa = retrieval_qa_chain(llm, qa_prompt, db)
        st.sidebar.success("Retrieval QA chain created.")
        
        return qa

def main():
    st.title("Test_Cases Automation Tool📝")
    st.markdown("#### Tool that generate test cases by uploading SRS documents")

    # Sidebar option for PDF document upload
    st.sidebar.title("Upload PDF Document")
    upload_option = st.sidebar.checkbox("Upload a PDF?")
    
    if upload_option:
        uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type=["pdf"])
        if uploaded_file is not None:
            if st.sidebar.button("Start Ingestion"):
                progress_bar = st.sidebar.progress(0)
                st.sidebar.info("Ingestion in progress...")
                qa_result = qa_bot(upload_option, uploaded_file)
                progress_bar.success("Ingestion completed!")
                
                # User input section
                query = st.text_input("Enter your query:")
                if st.button("Submit"):
                    progress_bar_main = st.progress(0)
                    with st.spinner("Searching for answers..."):
                        response = qa_result({'query': query})
                        st.success("Found an answer!")

                        st.markdown("#### Answer:")
                        st.write(response["result"])
                        st.markdown("#### Source Documents:")
                        st.write(response["source_documents"])

                        progress_bar_main.empty()

    else:
        query = st.text_input("Enter your query:")
        
        if st.button("Submit"):
            progress_bar_main = st.progress(0)

            with st.spinner("Searching for answers..."):
                qa_result = qa_bot(upload_option, None)
                response = qa_result({'query': query})
                st.success("Found an answer!")

            st.markdown("#### Answer:")
            st.write(response["result"])
            st.markdown("#### Source Documents:")
            st.write(response["source_documents"])

            progress_bar_main.empty()

if __name__ == "__main__":
    main()
