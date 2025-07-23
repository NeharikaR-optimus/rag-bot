import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Configuration class for all application settings."""
    
    # Azure OpenAI Configuration
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_CHAT_API_KEY")
    AZURE_OPENAI_CHAT_MODEL = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", "gpt-4.1")
    AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_CHAT_API_VERSION", "2025-01-01-preview")
    AZURE_OPENAI_EMBEDDING_MODEL = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME", "text-embedding-3-small")
    AZURE_OPENAI_EMBEDDING_API_KEY = os.getenv("AZURE_OPENAI_EMBEDDING_API_KEY")
    AZURE_OPENAI_EMBEDDING_API_VERSION = os.getenv("AZURE_OPENAI_EMBEDDING_API_VERSION", "2025-01-01-preview")
    
    # Azure AI Search Configuration
    AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
    AZURE_SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY")
    AZURE_SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME", "travel-documents")
    
    # Cosmos DB Configuration
    COSMOS_DB_ENDPOINT = os.getenv("COSMOS_DB_ENDPOINT")
    COSMOS_DB_KEY = os.getenv("COSMOS_DB_KEY")
    COSMOS_DB_DATABASE_NAME = os.getenv("COSMOS_DB_DATABASE_NAME", "travel_knowledge")
    COSMOS_DB_CONTAINER_NAME = os.getenv("COSMOS_DB_CONTAINER_NAME", "conversations")
    
    # Backend Configuration
    BACKEND_HOST = os.getenv("BACKEND_HOST", "localhost")
    BACKEND_PORT = int(os.getenv("BACKEND_PORT", 8000))
    BACKEND_URL = os.getenv("BACKEND_URL", f"http://{BACKEND_HOST}:{BACKEND_PORT}")
    
    # Frontend Configuration
    FRONTEND_HOST = os.getenv("FRONTEND_HOST", "localhost")
    FRONTEND_PORT = int(os.getenv("FRONTEND_PORT", 8501))
    
    # Application Settings
    DEBUG = os.getenv("DEBUG", "true").lower() == "true"
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    # Chat and Search Settings
    TEMPERATURE = 0.7
    MAX_COMPLETION_TOKENS = 2000
    VECTOR_SEARCH_TOP_K = 5
    SEARCH_RESULTS_LIMIT = 3
    CHAT_HISTORY_FILTERING_LIMIT = 10
    
    @classmethod
    def validate(cls):
        """Validate that all required configuration is present."""
        required_vars = [
            ("AZURE_OPENAI_ENDPOINT", cls.AZURE_OPENAI_ENDPOINT),
            ("AZURE_OPENAI_API_KEY", cls.AZURE_OPENAI_API_KEY),
            ("AZURE_SEARCH_ENDPOINT", cls.AZURE_SEARCH_ENDPOINT),
            ("AZURE_SEARCH_API_KEY", cls.AZURE_SEARCH_API_KEY),
            ("COSMOS_DB_ENDPOINT", cls.COSMOS_DB_ENDPOINT),
            ("COSMOS_DB_KEY", cls.COSMOS_DB_KEY),
        ]
        
        missing_vars = []
        for var_name, var_value in required_vars:
            if not var_value:
                missing_vars.append(var_name)
        
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
        
        return True
