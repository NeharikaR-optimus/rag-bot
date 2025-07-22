import os
import asyncio
from typing import List, Dict, Any
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

# Load environment variables
load_dotenv()

class RAGChatBot:
    def __init__(self):
        self.llm = None
        self.embeddings = None
        self.rag_chain = None
        self.memory = None
        self.vectorstore = None
        self.setup_complete = False
        
    def initialize(self):
        """Initialize all components of the RAG chatbot"""
        try:
            self._load_config()
            self._initialize_models()
            self._setup_rag()
            self._setup_memory()
            self._create_chains()
            self.setup_complete = True
            print("✅ RAG ChatBot initialized successfully!")
            return True
        except Exception as e:
            print(f"❌ Failed to initialize RAG ChatBot: {e}")
            return False
    
    def _load_config(self):
        """Load and validate configuration from environment variables"""
        # Azure OpenAI Configuration
        self.azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.chat_deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")
        self.embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")
        self.api_key = os.getenv("AZURE_OPENAI_CHAT_API_KEY")
        self.api_version = os.getenv("AZURE_OPENAI_CHAT_API_VERSION", "2024-02-01")
        
        # Validate required variables
        required_vars = [
            ("AZURE_OPENAI_ENDPOINT", self.azure_endpoint),
            ("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", self.chat_deployment),
            ("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME", self.embedding_deployment),
            ("AZURE_OPENAI_CHAT_API_KEY", self.api_key)
        ]
        
        for var_name, var_value in required_vars:
            if not var_value:
                raise ValueError(f"Environment variable {var_name} is not set")
        
        print(f"📡 Connecting to Azure OpenAI at: {self.azure_endpoint}")
        print(f"🤖 Chat Model: {self.chat_deployment}")
        print(f"🔍 Embedding Model: {self.embedding_deployment}")
    
    def _initialize_models(self):
        """Initialize Azure OpenAI models"""
        # Initialize Chat Model
        self.llm = AzureChatOpenAI(
            azure_endpoint=self.azure_endpoint,
            azure_deployment=self.chat_deployment,
            api_key=self.api_key,
            api_version=self.api_version,
            temperature=0.7,
            streaming=True
        )
        print("✅ Azure OpenAI Chat Model initialized")
        
        # Initialize Embedding Model
        self.embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=self.azure_endpoint,
            azure_deployment=self.embedding_deployment,
            api_key=self.api_key,
            api_version=self.api_version
        )
        print("✅ Azure OpenAI Embedding Model initialized")
    
    def _setup_rag(self):
        """Setup RAG pipeline with document loading and vector store"""
        print("📚 Setting up RAG pipeline...")
        
        # Load documents
        loader = TextLoader("travel_knowledge.txt", encoding="utf-8")
        docs = loader.load()
        print(f"📄 Loaded {len(docs)} document(s)")
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        splits = text_splitter.split_documents(docs)
        print(f"✂️ Split into {len(splits)} chunks")
        
        # Create vector store
        self.vectorstore = FAISS.from_documents(splits, self.embeddings)
        print("🗄️ Vector store created and populated")
        
        # Create retriever
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )
        print("🔍 Retriever initialized")
    
    def _setup_memory(self):
        """Setup conversation memory"""
        self.memory = ConversationBufferWindowMemory(
            k=5,
            memory_key="chat_history",
            return_messages=True,
            input_key="input"
        )
        print("🧠 Memory initialized")
    
    def _create_chains(self):
        """Create the RAG chain using LCEL"""
        # Contextualize question prompt
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", "Given a chat history and the latest user question "
                       "which might reference context in the chat history, "
                       "formulate a standalone question which can be understood "
                       "without the chat history. Do NOT answer the question, "
                       "just rephrase it if necessary and otherwise return it as is."),
            MessagesPlaceholder("chat_history"),
            ("user", "{input}"),
        ])
        
        # QA prompt
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful travel assistant AI. Answer the user's questions "
                       "based on the following retrieved context about Paris travel information. "
                       "Be friendly, informative, and helpful. If the answer is not in the context, "
                       "say you don't know and suggest they ask about Paris travel topics.\n\n"
                       "Retrieved Context: {context}"),
            MessagesPlaceholder("chat_history"),
            ("user", "{input}")
        ])
        
        # Create history-aware retriever chain
        history_aware_retriever_chain = (
            contextualize_q_prompt
            | self.llm
            | StrOutputParser()
            | self.retriever
        )
        
        # Create the full RAG chain
        self.rag_chain = (
            RunnablePassthrough.assign(
                context=history_aware_retriever_chain
            )
            | qa_prompt
            | self.llm
            | StrOutputParser()
        )
        
        print("🔗 RAG chain created successfully")
    
    async def get_response(self, user_input: str) -> str:
        """Get response from the RAG chatbot"""
        if not self.setup_complete:
            return "❌ Chatbot not initialized. Please check configuration."
        
        try:
            # Get current chat history
            current_chat_history = self.memory.load_memory_variables({})["chat_history"]
            
            # Prepare input for the chain
            chain_input = {
                "input": user_input,
                "chat_history": current_chat_history
            }
            
            # Get response from the chain
            response = await self.rag_chain.ainvoke(chain_input)
            
            # Update memory
            self.memory.chat_memory.add_user_message(user_input)
            self.memory.chat_memory.add_ai_message(response)
            
            return response
            
        except Exception as e:
            error_msg = f"❌ Error generating response: {str(e)}"
            print(error_msg)
            return error_msg
    
    def get_response_sync(self, user_input: str) -> str:
        """Synchronous wrapper for get_response"""
        return asyncio.run(self.get_response(user_input))
    
    def clear_memory(self):
        """Clear conversation memory"""
        if self.memory:
            self.memory.clear()
            print("🧹 Memory cleared")

# Global chatbot instance
chatbot = RAGChatBot()

def initialize_chatbot():
    """Initialize the global chatbot instance"""
    return chatbot.initialize()

async def get_chatbot_response(user_input: str) -> str:
    """Get response from the global chatbot instance"""
    return await chatbot.get_response(user_input)

def get_chatbot_response_sync(user_input: str) -> str:
    """Synchronous version of get_chatbot_response"""
    return chatbot.get_response_sync(user_input)

# Console interface for testing
async def run_console_chat():
    """Run console-based chat interface"""
    print("\n🎯 RAG Travel Chatbot - Console Interface")
    print("=" * 50)
    print("Ask me about Paris travel, attractions, hotels, and activities!")
    print("Type 'exit' to quit, 'clear' to clear memory")
    print("=" * 50)
    
    if not chatbot.initialize():
        print("Failed to initialize chatbot. Exiting.")
        return
    
    while True:
        try:
            user_input = input("\n🙋 You: ").strip()
            
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("🤖 Bot: Goodbye! Have a great trip to Paris! 🗼")
                break
            
            if user_input.lower() == 'clear':
                chatbot.clear_memory()
                print("🤖 Bot: Memory cleared! Feel free to start a new conversation.")
                continue
            
            if not user_input:
                continue
            
            print("🤖 Bot: ", end="", flush=True)
            response = await chatbot.get_response(user_input)
            print(response)
            
        except KeyboardInterrupt:
            print("\n\n🤖 Bot: Goodbye! Have a great trip to Paris! 🗼")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(run_console_chat())
