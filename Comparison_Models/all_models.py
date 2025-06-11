import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings
from langchain.llms import CTransformers
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from qdrant_client import QdrantClient
from langchain_community.llms import LlamaCpp
from langchain_community.vectorstores import Qdrant
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Initialize logging
import logging
logging.basicConfig(level=logging.INFO)

# App Config
st.set_page_config(page_title="Maternity Chatbot", layout="wide")

st.title("Maternity Chatbot with Multiple Models")
st.markdown('<style>h1{text-align: center;}</style>', unsafe_allow_html=True)

# Load documents and embeddings
@st.cache_resource
def load_data_and_embeddings():
    loader = DirectoryLoader('data/', glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_chunks = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': "cpu"})
    vector_store = FAISS.from_documents(text_chunks, embeddings)

    pubmed_embeddings = SentenceTransformerEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")
    qdrant_client = QdrantClient(url="http://localhost:6333", prefer_grpc=False)
    qdrant_db = Qdrant(client=qdrant_client, embeddings=pubmed_embeddings, collection_name="vector_db")

    return vector_store, qdrant_db

vector_store, qdrant_db = load_data_and_embeddings()

# Initialize LLMs
@st.cache_resource
def initialize_models():
    llm_mistral = CTransformers(model="mistral-7b-instruct-v0.1.Q4_K_M.gguf", config={'max_new_tokens': 128, 'temperature': 0.01})
    llm_llama2 = CTransformers(model="llama-2-7b-chat.ggmlv3.q4_0.bin", model_type="llama", config={'max_new_tokens': 128, 'temperature': 0.01})
    llm_biomistral = LlamaCpp(model_path="BioMistral-7B.Q4_K_M.gguf", temperature=0.5, max_tokens=4096, top_p=1)
    return llm_mistral, llm_llama2, llm_biomistral

llm_mistral, llm_llama2, llm_biomistral = initialize_models()

# RAG Chain
@st.cache_resource
def initialize_rag_chain():
    prompt_template = PromptTemplate(
        template="""You are a specialized AI assistant for maternity-related queries.
        Context: {context}
        Question: {question}
        Detailed Answer:""",
        input_variables=['context', 'question']
    )
    retriever = qdrant_db.as_retriever(search_kwargs={"k": 1})
    return RetrievalQA.from_chain_type(
        llm=llm_biomistral,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt_template},
        verbose=True
    )

rag_chain = initialize_rag_chain()

# Session state initialization
if "history" not in st.session_state:
    st.session_state.history = []
if "model_responses" not in st.session_state:
    st.session_state.model_responses = {
        "Mistral": [],
        "Llama2": [],
        "BioMistral": [],
        "RAG": []
    }
if "context_data" not in st.session_state:
    st.session_state.context_data = []

# Chat Function
def get_model_responses(query):
    try:
        # Ensure the query is passed as a list of prompts
        mistral_response = llm_mistral.generate(prompts=[query])
        llama2_response = llm_llama2.generate(prompts=[query])
        biomistral_response = llm_biomistral.generate(prompts=[query])

        # Access the text from the LLMResult object (handle different structures)
        mistral_response_text = mistral_response.generations[0][0].text if hasattr(mistral_response, 'generations') else "No response"
        llama2_response_text = llama2_response.generations[0][0].text if hasattr(llama2_response, 'generations') else "No response"
        biomistral_response_text = biomistral_response.generations[0][0].text if hasattr(biomistral_response, 'generations') else "No response"

        # Generate RAG response by calling the RetrievalQA instance
        rag_result = rag_chain({"query": query})

        # Extract RAG context and response
        source_documents = rag_result.get("source_documents", [])
        if source_documents:
            rag_context = source_documents[0].page_content  # Get the first document's content
        else:
            rag_context = "No context available"
        rag_response = rag_result.get("result", "No response available")

        # Update session state
        st.session_state.history.append(query)
        st.session_state.model_responses["Mistral"].append(mistral_response_text)
        st.session_state.model_responses["Llama2"].append(llama2_response_text)
        st.session_state.model_responses["BioMistral"].append(biomistral_response_text)
        st.session_state.model_responses["RAG"].append(rag_response)
        st.session_state.context_data.append(rag_context)

        logging.info("Responses generated successfully.")
    except Exception as e:
        logging.error(f"Error generating responses: {e}")
        st.error("An error occurred while generating responses.")

# Layout
query_input = st.text_input("Ask your maternity-related question:", key="query_input")
if st.button("Submit") and query_input:
    get_model_responses(query_input)

# Display responses
col1, col2 = st.columns(2)
col3, col4 = st.columns(2)

with col1:
    st.subheader("Mistral")
    for response in st.session_state.model_responses["Mistral"]:
        st.text_area("Response", response, height=200)

with col2:
    st.subheader("Llama2")
    for response in st.session_state.model_responses["Llama2"]:
        st.text_area("Response", response, height=200)

with col3:
    st.subheader("BioMistral")
    for response in st.session_state.model_responses["BioMistral"]:
        st.text_area("Response", response, height=200)

with col4:
    st.subheader("RAG")
    for response in st.session_state.model_responses["RAG"]:
        st.text_area("Response", response, height=200)
