import os
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, TypedDict

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_core.messages import AIMessage
from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph, MessagesState, END
from langgraph.prebuilt import ToolNode

from config import Config
from tools.chat_history_tool import ChatHistoryManager
from tools.doc_search_tool import doc_search_tool
from tools.checkpoint_tool import CheckpointManager
from utils.prompts import TRAVEL_ASSISTANT_SYSTEM_PROMPT, DOCUMENT_SEARCH_PROMPT, LOG_MESSAGES

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@tool
def search_documents(query: str) -> str:
    """Search for relevant travel documents in the knowledge base."""
    try:
        result = doc_search_tool(query, top_k=5)
        
        if result["status"] == "success" and result["results"]:
            context_parts = []
            for i, doc in enumerate(result["results"][:5], 1):
                content = doc.get('content', '')
                filename = doc.get('filename', 'Unknown')
                title = doc.get('title', '')
                
                context_parts.append(
                    f"Document {i} ({filename}):\nTitle: {title}\nContent: {content[:500]}..."
                )
            
            return "\n\n".join(context_parts)
        else:
            return "I couldn't find specific information for your query."
            
    except Exception as e:
        return f"Error searching documents: {str(e)}"

class TravelAssistantWorkflow:
    """Main workflow class for travel assistant using LangGraph."""
    
    # Class variable to store session context
    _current_session_id = "default_session"
    
    @classmethod
    def set_session_id(cls, session_id: str):
        """Set the current session ID for conversation storage."""
        cls._current_session_id = session_id
    
    @classmethod
    def filter_messages(cls, messages: list):
        """Filter messages to limit chat history to last 5 conversations."""
        filtered = messages[-5:]
        return filtered
    
    @classmethod
    async def get_workflow_graph(cls):
        """Create and return the uncompiled workflow graph."""
        try:
            config = Config()
            
            llm = AzureChatOpenAI(
                openai_api_key=config.AZURE_OPENAI_API_KEY,
                azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
                deployment_name=config.AZURE_OPENAI_CHAT_MODEL,
                api_version=config.AZURE_OPENAI_API_VERSION,
                temperature=0.7,
                max_tokens=1500,
                timeout=120,
                max_retries=3,
                streaming=True,
                verbose=False
            )
            
            tools = [search_documents]
            tool_node = ToolNode(tools)
            llm_with_tools = llm.bind_tools(tools)
            
            def should_continue(state: dict) -> str:
                """Determine the next node based on LLM tool calls."""
                messages = state["messages"]
                last_message = messages[-1]
                
                if last_message.tool_calls:
                    return "continue"
                else:
                    return "store_history"
            
            async def call_model(state: dict):
                """Call the LLM to decide on the response or tool usage."""
                messages = state["messages"]
                messages = cls.filter_messages(messages)

                system_prompt = TRAVEL_ASSISTANT_SYSTEM_PROMPT

                chain = (
                    ChatPromptTemplate.from_messages([
                        ("system", system_prompt),
                        ("placeholder", "{messages}")
                    ])
                    | llm_with_tools
                )

                try:
                    response = await chain.ainvoke({"messages": messages})
                    return {"messages": [response]}
                except Exception as e:
                    logger.error(f"Error in call_model: {e}")
                    raise e
            
            async def call_final_model(state: dict):
                """Generate final response after tool usage."""
                messages = state["messages"]
                messages = cls.filter_messages(messages)
                
                final_prompt = DOCUMENT_SEARCH_PROMPT
                
                chain = (
                    ChatPromptTemplate.from_messages([
                        ("system", final_prompt),
                        ("placeholder", "{messages}")
                    ])
                    | llm_with_tools
                )
                
                try:
                    response = await chain.ainvoke({"messages": messages})
                    return {"messages": [response]}
                except Exception as e:
                    logger.error(f"Error in call_final_model: {e}")
                    raise e
            
            conversation_context = {'session_id': cls._current_session_id}
            
            async def store_chat_history_tool(state: dict):
                """Store full conversation history in Cosmos DB."""
                try:
                    conversation_id = cls._current_session_id
                    messages = state.get("messages", [])
                    
                    if len(messages) >= 2:
                        chat_history = ChatHistoryManager()
                        user_message = None
                        assistant_message = None
                        
                        for i in range(len(messages) - 1, -1, -1):
                            msg = messages[i]
                            if hasattr(msg, 'content') and msg.content:
                                if hasattr(msg, 'type') and msg.type == 'ai' and assistant_message is None:
                                    assistant_message = msg.content
                                elif hasattr(msg, 'type') and msg.type == 'human' and user_message is None:
                                    user_message = msg.content
                                    
                                if user_message and assistant_message:
                                    break
                        
                        if user_message and assistant_message:
                            await chat_history.store_message(
                                session_id=conversation_id,
                                user_message=user_message,
                                assistant_response=assistant_message,
                                metadata={
                                    "timestamp": datetime.now().isoformat(),
                                    "type": "full_conversation",
                                    "message_count": len(messages)
                                }
                            )
                            
                            logger.info(f"Stored full conversation for session: {conversation_id}")
                            logger.info(f"User: {user_message[:100]}...")
                            logger.info(f"Assistant: {assistant_message[:100]}...")
                    
                    return state
                            
                except Exception as e:
                    logger.warning(f"Failed to store conversation history: {e}")
                    return state
            
            workflow = StateGraph(MessagesState)
            workflow.add_node("brain", call_model)
            workflow.add_node("tools", tool_node)
            workflow.add_node("final_model", call_final_model)
            workflow.add_node("store_history", store_chat_history_tool)
            
            workflow.set_entry_point("brain")
            
            workflow.add_conditional_edges("brain", should_continue, {
                "continue": "tools",
                "store_history": "store_history"
            })
            
            workflow.add_edge("tools", "final_model")
            workflow.add_edge("final_model", "store_history")
            workflow.add_edge("store_history", END)
            
            return workflow
            
        except Exception as ex:
            logger.error(f"Error during travel workflow graph creation: {ex}")
            raise ex
    
    @classmethod
    async def invoke_graph_workflow(cls, request: dict = None, session_id: str = None):
        """Create and return the compiled workflow graph with checkpointer."""
        try:
            if session_id:
                cls.set_session_id(session_id)
                
            workflow = await cls.get_workflow_graph()
            checkpointer = CheckpointManager()
            
            return workflow.compile(checkpointer=checkpointer)
            
        except Exception as ex:
            logger.error(f"Error during travel workflow setup: {ex}")
            raise ex

    @classmethod
    def get_compiled_workflow(cls):
        """Get compiled workflow - static/class method version."""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(cls.invoke_graph_workflow())
        except RuntimeError:
            workflow_coro = cls.invoke_graph_workflow()
            return asyncio.run(workflow_coro)
