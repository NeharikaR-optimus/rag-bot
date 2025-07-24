import os
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, TypedDict

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_core.messages import AIMessage
from langchain_openai import AzureChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, MessagesState, END
from langgraph.prebuilt import ToolNode

from config import Config
from tools.chat_history_tool import ChatHistoryManager
from tools.doc_search_tool import doc_search_tool
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
                    
                    chain = (
                        ChatPromptTemplate.from_messages([
                            ("system", final_prompt),
                            ("placeholder", "{messages}")
                        ])
                        | llm_with_tools
                    )
                    
                    response = await chain.ainvoke({"messages": filtered_messages})
                    return {"messages": [response]}
            
            async def store_chat_history_tool(state: dict):
                """Store chat history using tool-like pattern."""
                try:
                    conversation_id = "default_session"
                    messages = state.get("messages", [])
                    
                    if messages:
                        chat_history = ChatHistoryManager()
                        
                        recent_messages = messages[-3:]
                        summary_parts = []
                        
                        for msg in recent_messages:
                            if hasattr(msg, 'content') and msg.content:
                                role = "User" if hasattr(msg, 'type') and msg.type == 'human' else "Assistant"
                                content = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
                                summary_parts.append(f"{role}: {content}")
                        
                        if summary_parts:
                            summary = " | ".join(summary_parts)
                            
                            await chat_history.store_message(
                                session_id=conversation_id,
                                user_message="Conversation Summary",
                                assistant_response=f"Travel conversation summary: {summary}",
                                metadata={
                                    "timestamp": datetime.now().isoformat(),
                                    "type": "travel_summary",
                                    "message_count": len(messages)
                                }
                            )
                            
                            logger.info(f"Stored travel conversation history for: {conversation_id}")
                    
                    return state
                            
                except Exception as e:
                    logger.warning(f"Failed to store travel conversation history: {e}")
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
    async def invoke_graph_workflow(cls, request: dict = None):
        """Create and return the compiled workflow graph with checkpointer."""
        try:
            workflow = await cls.get_workflow_graph()
            
            checkpointer = MemorySaver()
            
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
