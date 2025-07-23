import os
from typing import List, Dict, Any
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from utils.prompts import LLM_PROMPT_TEMPLATE, HISTORY_FORMAT_TEMPLATE

load_dotenv()

class LLMTool:
    """Tool for generating responses using Azure OpenAI"""
    
    def __init__(self):
        self.llm = None
        self._initialize()
    
    def _initialize(self):
        """Initialize Azure OpenAI client"""
        try:
            self.llm = AzureChatOpenAI(
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_CHAT_API_KEY"),
                azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
                api_version=os.getenv("AZURE_OPENAI_CHAT_API_VERSION"),
                temperature=0.7,
                streaming=True
            )
            
        except Exception as e:
            print(f"Failed to initialize LLM Tool: {e}")
            raise
    
    async def generate_response(self, query: str, context: str, conversation_history: List[Dict] = None) -> str:
        """Generate a response using the LLM with context and conversation history"""
        try:
            history_context = ""
            if conversation_history:
                history_context = "\\n\\nRecent conversation:\\n"
                for i, turn in enumerate(conversation_history[-5:], 1):
                    history_context += HISTORY_FORMAT_TEMPLATE.format(
                        turn_number=i,
                        user_message=turn.get('user_message', ''),
                        assistant_response=turn.get('assistant_response', '')
                    )
            
            system_prompt = LLM_PROMPT_TEMPLATE.format(
                context=context,
                history_context=history_context
            )
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{query}")
            ])
            
            chain = prompt | self.llm
            response = await chain.ainvoke({"query": query})
            
            return response.content
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return "I apologize, but I encountered an error while processing your request. Please try again."
    
    async def generate_streaming_response(self, query: str, context: str, conversation_history: List[Dict] = None):
        """Generate a streaming response using the LLM"""
        try:
            history_context = ""
            if conversation_history:
                history_context = "\\n\\nRecent conversation:\\n"
                for i, turn in enumerate(conversation_history[-5:], 1):
                    history_context += HISTORY_FORMAT_TEMPLATE.format(
                        turn_number=i,
                        user_message=turn.get('user_message', ''),
                        assistant_response=turn.get('assistant_response', '')
                    )
            
            system_prompt = LLM_PROMPT_TEMPLATE.format(
                context=context,
                history_context=history_context
            )
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{query}")
            ])
            
            chain = prompt | self.llm
            
            async for chunk in chain.astream({"query": query}):
                content = getattr(chunk, 'content', '')
                if content:
                    yield content
            
        except Exception as e:
            print(f"Error generating streaming response: {e}")
            yield "I apologize, but I encountered an error while processing your request. Please try again."
    
    def generate_greeting(self) -> str:
        """Generate a greeting message"""
        return ("Hello! I'm your travel assistant specializing in European destinations, Paris, "
                "budget travel tips, and family travel advice. How can I help you plan your next adventure?\\n\\n"
                "You can ask me about:\\n"
                "• Budget travel tips for Europe\\n"
                "• Paris attractions and recommendations\\n"
                "• Family-friendly travel advice\\n"
                "• European travel planning")
