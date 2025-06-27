import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain_community.llms import LlamaCpp
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# Load and process PDF
loader = PyPDFLoader("healthyheart.pdf")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = text_splitter.split_documents(docs)

embeddings = SentenceTransformerEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")
vectorstore = Chroma.from_documents(chunks, embeddings)

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Use small LLM model or replace with HuggingFaceHub
llm = LlamaCpp(
    model_path="small_model.gguf",  # Update accordingly
    temperature=0.2,
    max_tokens=2048,
    top_p=1
)

template = """
<|context|>
You are a Medical Assistant that follows the instructions and generate the accurate response based on the query and the context provided.
Please be truthful and give direct answers.
</s>
<|user|>
{query}
</s>
<|assistant|>"""

prompt = ChatPromptTemplate.from_template(template)

rag_chain = (
    {"context": retriever, "query": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Streamlit UI
st.title("Heart Disease Q&A Assistant")
user_query = st.text_input("Ask something about the heart research report")

if user_query:
    response = rag_chain.invoke(user_query)
    st.write("**Answer:**", response)
