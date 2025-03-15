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
import time
from comparison_metrics import calculate_rouge_score,calculate_semantic_similarity,calculate_bleu_score

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

sys_prompt = """Generate comprehensive test cases based on the provided document. Ensure the test cases cover functional, non-functional, edge cases, and boundary conditions. The output should be structured and clear, including test case ID, description, steps, expected result, and priority level."""

instruction = """CONTEXT:\n\n {context}\n

Based on the provided document, generate detailed test cases covering all relevant aspects. The test cases should include:
- **Test Case ID**
- **Description**
- **Preconditions**
- **Test Steps**
- **Expected Result**
- **Priority (High/Medium/Low)**

Ensure completeness, accuracy, and relevance to the document content.

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
        max_tokens=4096 ,
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
        
        start_time = time.time()  # Start time for ingestion
        db = FAISS.from_documents(texts, embeddings)
        db.save_local(DB_FAISS_PATH)
        ingestion_time = time.time() - start_time  # Calculate ingestion time
        st.sidebar.success(f"Vector DB created and saved locally. Ingestion time: {ingestion_time:.2f} seconds.")
        
        llm = load_llm()
        qa_prompt = llama_prompt
        qa = retrieval_qa_chain(llm, qa_prompt, db)
        st.sidebar.success("Retrieval QA chain created.")
        
        return qa, ingestion_time

    else:
        # Allow dangerous deserialization
        db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
        st.sidebar.info("Using existing Vector DB.")

        llm = load_llm()
        qa_prompt = llama_prompt
        qa = retrieval_qa_chain(llm, qa_prompt, db)
        st.sidebar.success("Retrieval QA chain created.")
        
        return qa, 0  # No ingestion time if using existing DB

def main():
    st.title("Test_Cases Automation Tool📝")
    st.markdown("#### Tool that generate test cases by uploading SRS documents")

    # Create tabs for different functionalities
    tab1, tab2 = st.tabs(["Generate Test Cases", "Compare Answers"])

    with tab1:
        # Sidebar option for PDF document upload
        st.sidebar.title("Upload PDF Document")
        upload_option = st.sidebar.checkbox("Upload a PDF?")
        
        if upload_option:
            uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type=["pdf"])
            if uploaded_file is not None:
                if st.sidebar.button("Start Ingestion"):
                    progress_bar = st.sidebar.progress(0)
                    st.sidebar.info("Ingestion in progress...")
                    qa_result, ingestion_time = qa_bot(upload_option, uploaded_file)
                    progress_bar.success("Ingestion completed!")
                    
                    # User input section
                    query = st.text_input("Enter your query:")
                    if st.button("Submit"):
                        progress_bar_main = st.progress(0)
                        with st.spinner("Searching for answers..."):
                            start_time = time.time()  # Start time for query response
                            response = qa_result({'query': query})
                            response_time = time.time() - start_time  # Calculate response time
                            st.success(f"Found an answer! Response time: {response_time:.2f} seconds.")

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
                    qa_result, _ = qa_bot(upload_option, None)
                    start_time = time.time()  # Start time for query response
                    response = qa_result({'query': query})
                    response_time = time.time() - start_time  # Calculate response time
                    st.success(f"Found an answer! Response time: {response_time:.2f} seconds.")

                st.markdown("#### Answer:")
                st.write(response["result"])
                st.markdown("#### Source Documents:")
                st.write(response["source_documents"])

                progress_bar_main.empty()

    with tab2:
        st.markdown("### Compare Answers and Calculate Metrics")
        correct_answer = st.text_area("Paste the correct answer here:")
        generated_answer = st.text_area("Paste the generated answer here:")

        if st.button("Calculate Metrics"):
            if correct_answer and generated_answer:
                # Calculate semantic similarity
                similarity = calculate_semantic_similarity(correct_answer, generated_answer)
                st.markdown("#### Semantic Similarity:")
                st.write(f"Similarity Score: {similarity:.2f}")

                # Interpret the similarity score
                if similarity >= 0.7:
                    st.success("Strong match: The generated answer is very similar to the correct answer.")
                elif similarity >= 0.3:
                    st.warning("Partial match: The generated answer has some similarity to the correct answer.")
                else:
                    st.error("Poor match: The generated answer is not similar to the correct answer.")

                # Calculate BLEU score
                bleu_score = calculate_bleu_score(correct_answer, generated_answer)
                st.markdown("#### BLEU Score:")
                st.write(f"BLEU Score: {bleu_score:.2f}")

                # Calculate ROUGE scores
                rouge_scores = calculate_rouge_score(correct_answer, generated_answer)
                st.markdown("#### ROUGE Scores:")
                st.write(f"ROUGE-1: {rouge_scores['rouge1'].fmeasure:.2f}")
                st.write(f"ROUGE-2: {rouge_scores['rouge2'].fmeasure:.2f}")
                st.write(f"ROUGE-L: {rouge_scores['rougeL'].fmeasure:.2f}")
            else:
                st.warning("Please provide both the correct answer and the generated answer.")
if __name__ == "__main__":
    main()