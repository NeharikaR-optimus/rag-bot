import os
import asyncio
from typing import List, Dict, Any, TypedDict, Annotated, Optional
from datetime import datetime
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential
from cosmos_manager import CosmosDBManager
import json
import uuid

load_dotenv()

class ChatState(TypedDict):
    messages: Annotated[List[Any], add_messages]
    user_query: str
    search_results: List[Dict[str, Any]]
    context: str
    response: str
    session_id: Optional[str]

class DocumentSearchTool:
    
    def __init__(self):
        self.search_client = None
        self.embeddings = None
        self._initialize_search()
    
    def _initialize_search(self):
        try:
            search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
            search_key = os.getenv("AZURE_SEARCH_API_KEY")
            index_name = os.getenv("AZURE_SEARCH_INDEX_NAME", "travel-documents")
            
            if not search_endpoint or not search_key:
                raise ValueError("Azure Search endpoint and API key must be provided")
            
            credential = AzureKeyCredential(search_key)
            self.search_client = SearchClient(
                endpoint=search_endpoint,
                index_name=index_name,
                credential=credential
            )
            
            self.embeddings = AzureOpenAIEmbeddings(
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_EMBEDDING_API_KEY"),
                azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
                api_version=os.getenv("AZURE_OPENAI_EMBEDDING_API_VERSION")
            )
            
            print("Azure AI Search initialized successfully!")
            
        except Exception as e:
            print(f"Failed to initialize Azure AI Search: {e}")
            raise
    
    async def search_documents(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        try:
            query_embedding = await self.embeddings.aembed_query(query)
            
            vector_query = VectorizedQuery(
                vector=query_embedding,
                k_nearest_neighbors=top_k,
                fields="content_vector"
            )
            
            results = self.search_client.search(
                search_text=query,
                vector_queries=[vector_query],
                select=["id", "title", "content", "source"],
                top=top_k
            )
            
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "id": result.get("id", ""),
                    "title": result.get("title", ""),
                    "content": result.get("content", ""),
                    "source": result.get("source", ""),
                    "score": result.get("@search.score", 0)
                })
            
            return formatted_results
            
        except Exception as e:
            print(f"Error searching documents: {e}")
            return []

class RAGLangGraphBot:
    
    def __init__(self):
        self.llm = None
        self.document_search = None
        self.cosmos_manager = None
        self.workflow = None
        self._initialize()
    
    def _initialize(self):
        try:
            self.llm = AzureChatOpenAI(
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_CHAT_API_KEY"),
                azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
                api_version=os.getenv("AZURE_OPENAI_CHAT_API_VERSION"),
                temperature=0.7
            )
            
            self.document_search = DocumentSearchTool()
            self.cosmos_manager = CosmosDBManager()
            self._build_workflow()
            
            print("RAG LangGraph Bot initialized successfully!")
            
        except Exception as e:
            print(f"Failed to initialize RAG LangGraph Bot: {e}")
            raise
    
    def _build_workflow(self):
        workflow = StateGraph(ChatState)
        
        workflow.add_node("extract_query", self._extract_query)
        workflow.add_node("search_documents", self._search_documents)
        workflow.add_node("generate_response", self._generate_response)
        workflow.add_node("save_conversation", self._save_conversation)
        
        workflow.set_entry_point("extract_query")
        workflow.add_edge("extract_query", "search_documents")
        workflow.add_edge("search_documents", "generate_response")
        workflow.add_edge("generate_response", "save_conversation")
        workflow.add_edge("save_conversation", END)
        
        self.workflow = workflow.compile()
    
    async def _extract_query(self, state: ChatState) -> ChatState:
        messages = state.get("messages", [])
        if messages:
            last_message = messages[-1]
            if isinstance(last_message, HumanMessage):
                state["user_query"] = last_message.content
            else:
                state["user_query"] = str(last_message)
        return state
    
    async def _search_documents(self, state: ChatState) -> ChatState:
        query = state.get("user_query", "")
        if query:
            search_results = await self.document_search.search_documents(query)
            state["search_results"] = search_results
            
            context_parts = []
            for result in search_results[:4]:
                title = result.get('title', 'Document')
                content = result.get('content', '')
                source = result.get('source', '')
                
                content_excerpt = content[:800] + ("..." if len(content) > 800 else "")
                context_parts.append(f"Document: {title}\nSource: {source}\nContent: {content_excerpt}")
            
            state["context"] = "\n\n---\n\n".join(context_parts)
        else:
            state["search_results"] = []
            state["context"] = ""
        
        return state
    
    async def _generate_response(self, state: ChatState) -> ChatState:
        query = state.get("user_query", "")
        context = state.get("context", "")
        search_results = state.get("search_results", [])
        
        print(f"Search results for query '{query}':")
        for i, result in enumerate(search_results):
            print(f"  {i+1}. {result.get('title', 'Unknown')} - Score: {result.get('score', 0):.3f}")
        
        greeting_words = ["hello", "hi", "hey", "greetings", "good morning", "good afternoon", "good evening"]
        if any(word in query.lower() for word in greeting_words):
            state["response"] = "Hello! I'm your travel assistant specializing in European destinations, Paris, budget travel tips, and family travel advice. How can I help you plan your next adventure? You can ask me about:\n\n• Budget travel tips for Europe\n• Paris attractions and recommendations\n• Family-friendly travel advice\n• European travel planning"
            state["messages"].append(AIMessage(content=state["response"]))
            return state
        
        if not search_results or len(search_results) == 0:
            state["response"] = f"I'd love to help you with '{query}'! While I don't have specific information about that topic in my current travel guides, I do have great information about European destinations, Paris attractions, budget travel tips, and family travel advice. Could you try asking about one of these areas? For example:\n\n• 'What are budget travel tips for Europe?'\n• 'Tell me about Paris attractions'\n• 'How to plan family travel?'"
            state["messages"].append(AIMessage(content=state["response"]))
            return state
        
        relevant_results = [r for r in search_results if r.get("score", 0) > 0.01]
        if not relevant_results:
            state["response"] = f"I couldn't find very relevant information about '{query}' in my travel guides. However, I have comprehensive information about European travel, Paris, budget tips, and family travel advice. What specific aspect of travel would you like to know about?"
            state["messages"].append(AIMessage(content=state["response"]))
            return state
        
        system_prompt = """You are a friendly and helpful travel assistant. Use the provided travel information to give practical, engaging advice. 

        Guidelines:
        - Be conversational and warm in your tone
        - Provide specific, actionable travel advice when possible
        - If the context has relevant information, use it to give detailed recommendations
        - Organize information in a user-friendly way with bullet points or short paragraphs
        - Be enthusiastic about travel while staying informative
        - If asked about a broad topic like "France" or "Paris," give a well-structured overview
        
        Available travel information:
        {context}
        """
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt.format(context=context)),
            ("human", "{query}")
        ])
        
        chain = prompt | self.llm
        response = await chain.ainvoke({"query": query})
        
        state["response"] = response.content
        state["messages"].append(AIMessage(content=response.content))
        
        return state
    
    async def _save_conversation(self, state: ChatState) -> ChatState:
        try:
            if state.get("session_id"):
                await self.cosmos_manager.save_conversation(
                    session_id=state["session_id"],
                    user_message=state.get("user_query", ""),
                    assistant_response=state.get("response", ""),
                    search_results=state.get("search_results", [])
                )
        except Exception as e:
            print(f"Failed to save conversation: {e}")
        
        return state
    
    async def chat(self, user_input: str, conversation_history: List = None, session_id: str = None) -> str:
        try:
            if not session_id:
                session_id = str(uuid.uuid4())
            
            await self.cosmos_manager.setup_database_and_container()
            
            messages = conversation_history or []
            messages.append(HumanMessage(content=user_input))
            
            initial_state = ChatState(
                messages=messages,
                user_query="",
                search_results=[],
                context="",
                response="",
                session_id=session_id
            )
            
            final_state = await self.workflow.ainvoke(initial_state)
            
            return final_state["response"]
            
        except Exception as e:
            print(f"Error in chat: {e}")
            return "I apologize, but I encountered an error while processing your request. Please try again."
    
    async def get_conversation_history(self, session_id: str, limit: int = 10) -> List[Dict]:
        try:
            await self.cosmos_manager.setup_database_and_container()
            return await self.cosmos_manager.get_conversation_history(session_id, limit)
        except Exception as e:
            print(f"Error retrieving conversation history: {e}")
            return []
    
    async def start_new_session(self, metadata: Dict[str, Any] = None) -> str:
        try:
            session_id = str(uuid.uuid4())
            await self.cosmos_manager.setup_database_and_container()
            
            default_metadata = {
                "start_time": datetime.now().isoformat(),
                "topic": "Travel Assistant"
            }
            if metadata:
                default_metadata.update(metadata)
            
            await self.cosmos_manager.save_session_metadata(session_id, default_metadata)
            return session_id
        except Exception as e:
            print(f"Error starting new session: {e}")
            return str(uuid.uuid4())
    
    async def get_search_results_async(self, query: str) -> List[Dict[str, Any]]:
        try:
            return await self.document_search.search_documents(query)
        except Exception as e:
            print(f"Error in get_search_results_async: {e}")
            return []
    
    def get_search_results(self, query: str) -> List[Dict[str, Any]]:
        try:
            try:
                loop = asyncio.get_running_loop()
                print("Warning: get_search_results called from async context, returning empty results")
                return []
            except RuntimeError:
                return asyncio.run(self.document_search.search_documents(query))
        except Exception as e:
            print(f"Error in get_search_results: {e}")
            return []
    
    async def close(self):
        if self.cosmos_manager:
            await self.cosmos_manager.close()

if __name__ == "__main__":
    async def main():
        bot = None
        try:
            bot = RAGLangGraphBot()
            
            session_id = await bot.start_new_session({
                "user_name": "Test User",
                "topic": "Travel Planning"
            })
            print(f"Started session: {session_id}")
            
            conversation = []
            
            while True:
                try:
                    user_input = input("\nYou: ")
                    if user_input.lower() in ['quit', 'exit', 'bye']:
                        break
                    
                    response = await bot.chat(user_input, conversation, session_id)
                    print(f"Bot: {response}")
                    
                    conversation.append(HumanMessage(content=user_input))
                    conversation.append(AIMessage(content=response))
                    
                except KeyboardInterrupt:
                    print("\nGoodbye!")
                    break
                except EOFError:
                    print("\nGoodbye!")
                    break
                except Exception as e:
                    print(f"Error in conversation: {e}")
                    continue
            
            print("\n" + "="*50)
            print("Conversation History from Cosmos DB:")
            try:
                history = await bot.get_conversation_history(session_id)
                for i, turn in enumerate(history, 1):
                    print(f"\n{i}. User: {turn['user_message']}")
                    print(f"   Bot: {turn['assistant_response']}")
            except Exception as e:
                print(f"Error retrieving history: {e}")
        
        except Exception as e:
            print(f"Error initializing bot: {e}")
        
        finally:
            if bot:
                try:
                    await bot.close()
                except Exception as e:
                    print(f"Error during cleanup: {e}")
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nApplication terminated by user")
    except Exception as e:
        print(f"Application error: {e}")
