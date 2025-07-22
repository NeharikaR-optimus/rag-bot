import os
import asyncio
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter

# --- 1. Load environment variables ---
load_dotenv()

# --- Azure OpenAI Configuration ---
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_MODEL_NAME = os.getenv("AZURE_OPENAI_MODEL_NAME")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")

AZURE_OPENAI_EMBEDDING_MODEL_NAME = os.getenv("AZURE_OPENAI_EMBEDDING_MODEL_NAME", "text-embedding-ada-002")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME", "text-embedding-ada-002-deployment")

required_vars = [
    "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_DEPLOYMENT_NAME", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_API_VERSION",
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"
]
for var in required_vars:
    if not os.getenv(var):
        raise ValueError(f"Environment variable {var} is not set. Please check your .env file or system environment.")

print(f"Connecting to Azure OpenAI Chat Deployment: {AZURE_OPENAI_DEPLOYMENT_NAME} at {AZURE_OPENAI_ENDPOINT}")
print(f"Connecting to Azure OpenAI Embedding Deployment: {AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME}")

# --- 2. Initialize the Language Model (LLM) and Embedding Model ---
llm = AzureChatOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    azure_deployment=AZURE_OPENAI_DEPLOYMENT_NAME,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    temperature=0.7,
    streaming=True
)
print("Azure OpenAI Chat Model initialized.")

embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    azure_deployment=AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION
)
print("Azure OpenAI Embedding Model initialized.")

# --- 3. RAG Setup: Load, Split, Embed, and Store Documents ---
print("\nSetting up Retrieval-Augmented Generation (RAG) pipeline...")

loader = TextLoader("travel_info.txt")
docs = loader.load()
print(f"Loaded {len(docs)} document(s) from travel_info.txt.")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
splits = text_splitter.split_documents(docs)
print(f"Split document into {len(splits)} chunks.")

vectorstore = FAISS.from_documents(splits, embeddings)
print("Vector store created and populated with embeddings.")

retriever = vectorstore.as_retriever()
print("Retriever initialized.")

# --- 4. Define Prompts for Multi-Turn Conversation and RAG ---
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", "Given a chat history and the latest user question "
               "which might reference context in the chat history, "
               "formulate a standalone question which can be understood "
               "without the chat history. Do NOT answer the question, "
               "just rephrase it if necessary and otherwise return it as is."),
    MessagesPlaceholder("chat_history"),
    ("user", "{input}"),
])
print("Contextualize question prompt defined.")

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant. Answer the user's questions truthfully and concisely "
               "based on the following retrieved context. If the answer is not in the context, "
               "state that you don't know and avoid making up information.\n\n"
               "Retrieved Context: {context}"),
    MessagesPlaceholder("chat_history"),
    ("user", "{input}")
])
print("QA prompt defined.")

# --- 5. Create the LangChain Chains using LCEL ---
history_aware_retriever_chain = (
    contextualize_q_prompt
    | llm
    | StrOutputParser()
    | retriever
)
print("History-aware retriever chain created.")

rag_chain = (
    RunnablePassthrough.assign(
        context=history_aware_retriever_chain
    )
    | qa_prompt
    | llm
    | StrOutputParser()
)
print("Full RAG chain created using LCEL.")

# --- 6. Set up LangChain Memory for multi-turn conversation ---
memory = ConversationBufferWindowMemory(
    k=5,
    memory_key="chat_history",
    return_messages=True,
    input_key="input"
)
print("LangChain memory (in-memory) initialized.")

# --- Asynchronous Main Chat Loop ---
async def run_rag_chatbot():
    print("\n--- LangChain RAG Chatbot with Multi-Turn Conversation ---")
    print("Type 'exit' to quit at any time.")
    print("Ask questions about Paris, France, family activities, and accommodations.")
    print("---------------------------------------------------------")

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Bot: Goodbye!")
            break

        current_chat_history = memory.load_memory_variables({})["chat_history"]
        chain_input = {"input": user_input, "chat_history": current_chat_history}

        print("Bot (streaming): ", end="")
        full_response_content = ""

        try:
            async for chunk in rag_chain.astream(chain_input):
                print(chunk, end="", flush=True)
                full_response_content += chunk

            print() # Newline after the bot's full response

            memory.chat_memory.add_user_message(user_input)
            memory.chat_memory.add_ai_message(full_response_content)

        except Exception as e:
            print(f"\n[ERROR] An error occurred during response generation: {e}")
            import traceback
            traceback.print_exc()
            print("Please check your Azure OpenAI deployments and network connectivity.")
            memory.chat_memory.add_user_message(user_input)
            memory.chat_memory.add_ai_message(f"Error: {e}. Could not generate response.")

# --- Main execution block ---
if __name__ == "__main__":
    asyncio.run(run_rag_chatbot())