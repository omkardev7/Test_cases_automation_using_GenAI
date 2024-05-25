import streamlit as st
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
import warnings
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import LlamaCpp
warnings.filterwarnings("ignore")

DB_FAISS_PATH = 'vectorstore/db_faiss'

## Default LLaMA-2 prompt style
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

def get_prompt(instruction, new_system_prompt=DEFAULT_SYSTEM_PROMPT ):
    SYSTEM_PROMPT = B_SYS + new_system_prompt + E_SYS
    prompt_template =  B_INST + SYSTEM_PROMPT + instruction + E_INST
    return prompt_template

sys_prompt = """give response only from context provided"""

instruction = """CONTEXT:/n{context}/n

Question: {question}"""
get_prompt(instruction, sys_prompt)

prompt_template = get_prompt(instruction, sys_prompt)

llama_prompt = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

chain_type_kwargs = {"prompt": llama_prompt}

# custom_prompt_template = """ Retrieve information only from the following documents: DFPDS2021, DPM2009, and GFR2017, which are stored in 'vectorstore/db_faiss'.

# Context: {context}
# Question: {question}

# Please ensure that your answer is based solely on the information available in these documents. If you don't know the answer, please respond with 'I don't know.'
# """


# def set_custom_prompt():
#     """
#     Prompt template for QA retrieval for each vectorstore
#     """
#     prompt = PromptTemplate(template=custom_prompt_template,
#                             input_variables=['context', 'question'])
#     return prompt

# Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                    chain_type='stuff',
                                    retriever=db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold":0.3,"k": 4}),
                                    return_source_documents=True,
                                    chain_type_kwargs=chain_type_kwargs
                                    )
    return qa_chain

# Loading the model
def load_llm():
    # Load the locally downloaded model here
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

# QA Model Function
def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    
    db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    llm = load_llm()
    qa_prompt = llama_prompt
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    return qa

# Streamlit UI
def main():
    st.title("Test_Cases_Automation🤖")

    # Customizing the UI
    st.markdown("#### Tool to generate automate test cases by uploading SRS document")
    query = st.text_input("Enter your query:")
    
    if st.button("Submit"):
        progress_bar = st.progress(0)  # Create a progress bar

        # Show spinner
        with st.spinner("Searching for answers..."):
            # Perform the search and get the result
            qa_result = qa_bot()
            response = qa_result({'query': query})
            st.success("Found an answer!")

        # Display the answer and source documents
        st.markdown("#### Answer:")
        st.write(response["result"])
        st.markdown("#### Source Documents:")
        st.write(response["source_documents"])

        # Close the progress bar
        progress_bar.empty()

if __name__ == "__main__":
    main()

