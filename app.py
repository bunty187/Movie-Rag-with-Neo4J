from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.document_loaders import PyPDFDirectoryLoader
from langchain_groq import ChatGroq
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_milvus.retrievers import MilvusCollectionHybridSearchRetriever
from langchain_milvus.utils.sparse import BM25SparseEmbedding


from langchain_core.runnables import RunnablePassthrough,RunnableParallel

from langchain.chains import create_retrieval_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

# from langchain.memory import ChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHis

# from openai import OpenAI
import streamlit as st




import os

# os.environ['GOOGLE_API_KEY'] = 'AIzaSyDvzWbyMBX-WIL-1pungOZ0XCOqd5tZaqw'
os.environ['GROQ_API_KEY'] = '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'

model = ChatGroq(model_name="Llama3-8b-8192")

# from milvus import default_server

# default_server.start()


from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    WeightedRanker,
    connections,
)



CONNECTION_URI = "database/milvus_demo.db"
connections.connect(uri=CONNECTION_URI)


pk_field = "doc_id"
dense_field = "dense_vector"
sparse_field = "sparse_vector"
text_field = "text"
fields = [
    FieldSchema(
        name=pk_field,
        dtype=DataType.VARCHAR,
        is_primary=True,
        auto_id=True,
        max_length=100,
    ),
    FieldSchema(name=dense_field, dtype=DataType.FLOAT_VECTOR, dim=384),
    FieldSchema(name=sparse_field, dtype=DataType.SPARSE_FLOAT_VECTOR),
    FieldSchema(name=text_field, dtype=DataType.VARCHAR, max_length=65_535),
]


schema = CollectionSchema(fields=fields, enable_dynamic_field=False)
collection = Collection(
    name="IntroductionToTheRaptors", schema=schema, consistency_level="Strong"
)



import ast

# Read list from the file
with open('list.txt', 'r') as file:
    contents = file.read()
    all_texts = ast.literal_eval(contents)


sparse_embedding_func = BM25SparseEmbedding(corpus=all_texts)   

embd=HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLm-L3-v2")


# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
sparse_search_params = {"metric_type": "IP"}
dense_search_params = {"metric_type": "IP", "params": {}}
retriever = MilvusCollectionHybridSearchRetriever(
    collection=collection,
    rerank=WeightedRanker(0.5, 0.5),
    anns_fields=[dense_field, sparse_field],
    field_embeddings=[embd, sparse_embedding_func],
    field_search_params=[dense_search_params, sparse_search_params],
    top_k=3,
    text_field=text_field
)

## Define the Chat template
chat_template = ChatPromptTemplate.from_messages([
    # System Message Prompt Template
    SystemMessage(content="""You are a Helpful AI Bot.
    You take the context and question from user. Your answer should be based on the specific context."""),

    # Human Message Prompt Template
    HumanMessagePromptTemplate.from_template("""Answer the question based on the given context.
    Context:
    {context}

    Question:
    {question}

    Answer: """)
])

### Define the structure of the output
output_parser = StrOutputParser()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

## Create the chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | chat_template
    | model
    | output_parser
)

st.title("💬Leave No Context Behind Paper Q/A RAG System")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hi, How may I help you today?"),
    ]

# conversation
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

## Take the user input
user_prompt = st.chat_input()

if user_prompt is not None and user_prompt != "":
    st.session_state.chat_history.append(HumanMessage(content=user_prompt))

    with st.chat_message("Human"):
        st.markdown(user_prompt)

    with st.chat_message("AI"):
        processing_message = st.markdown("Processing...")
        response = rag_chain.run({"context": retriever, "question": user_prompt})
        processing_message.empty()
        st.markdown(response)

    st.session_state.chat_history.append(AIMessage(content=response))
