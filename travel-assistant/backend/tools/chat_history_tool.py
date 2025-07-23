import logging
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional
from azure.cosmos.aio import CosmosClient
from azure.cosmos import exceptions
from config import Config

logger = logging.getLogger(__name__)

class ChatHistoryManager:
    """Manages chat history storage and retrieval in Cosmos DB."""
    
    def __init__(self):
        self.client = None
        self.database = None
        self.container = None
        self._initialize()
    
    def _initialize(self):
        """Initialize Cosmos DB client."""
        try:
            self.client = CosmosClient(Config.COSMOS_DB_ENDPOINT, Config.COSMOS_DB_KEY)
            self.database_name = Config.COSMOS_DB_DATABASE_NAME
            self.container_name = Config.COSMOS_DB_CONTAINER_NAME
        except Exception as e:
            logger.error(f"Failed to initialize Chat History Manager: {e}")
            raise
    
    async def setup_database_and_container(self):
        """Setup database and container if they don't exist."""
        try:
            database = await self.client.create_database_if_not_exists(
                id=self.database_name,
                offer_throughput=400
            )
            self.database = database
            
            from azure.cosmos import PartitionKey
            container = await database.create_container_if_not_exists(
                id=self.container_name,
                partition_key=PartitionKey(path="/session_id"),
                offer_throughput=400
            )
            self.container = container
            
        except exceptions.CosmosHttpResponseError as e:
            logger.error(f"Failed to setup Cosmos DB: {e}")
            raise
    
    async def store_message(self, session_id: str, user_message: str, assistant_response: str, metadata: Dict = None) -> str:
        """Store a conversation turn."""
        try:
            if not self.container:
                await self.setup_database_and_container()
            
            message_doc = {
                "id": str(uuid.uuid4()),
                "session_id": session_id,
                "timestamp": datetime.now().isoformat(),
                "user_message": user_message,
                "assistant_response": assistant_response,
                "metadata": metadata or {},
                "type": "conversation"
            }
            
            created_item = await self.container.create_item(body=message_doc)
            return created_item['id']
            
        except exceptions.CosmosHttpResponseError as e:
            logger.error(f"Failed to store conversation: {e}")
            raise
    
    async def get_conversation_history(self, session_id: str, limit: int = 5) -> List[Dict]:
        """Get conversation history for a session."""
        try:
            if not self.container:
                await self.setup_database_and_container()
            
            query = """
                SELECT * FROM c 
                WHERE c.session_id = @session_id 
                AND c.type = 'conversation'
                ORDER BY c.timestamp DESC
                OFFSET 0 LIMIT @limit
            """
            
            parameters = [
                {"name": "@session_id", "value": session_id},
                {"name": "@limit", "value": limit}
            ]
            
            items = []
            async for item in self.container.query_items(
                query=query,
                parameters=parameters,
                partition_key=session_id
            ):
                items.append(item)
            
            items.reverse()
            return items
            
        except exceptions.CosmosHttpResponseError as e:
            logger.error(f"Failed to retrieve conversation history: {e}")
            return []
    
    async def close(self):
        """Close the Cosmos DB client."""
        if self.client:
            await self.client.close()
