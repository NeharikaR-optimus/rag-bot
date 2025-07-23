import os
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
from dotenv import load_dotenv
from azure.cosmos.aio import CosmosClient
from azure.cosmos import exceptions
import uuid
import json

load_dotenv()

class CosmosDBManager:
    
    def __init__(self):
        self.client = None
        self.database = None
        self.container = None
        self._initialize()
    
    def _initialize(self):
        try:
            endpoint = os.getenv("COSMOS_DB_ENDPOINT")
            key = os.getenv("COSMOS_DB_KEY")
            database_name = os.getenv("COSMOS_DB_DATABASE_NAME", "travel_knowledge")
            container_name = os.getenv("COSMOS_DB_CONTAINER_NAME", "conversations")
            
            if not endpoint or not key:
                raise ValueError("Cosmos DB endpoint and key must be provided")
            
            self.client = CosmosClient(endpoint, key)
            self.database_name = database_name
            self.container_name = container_name
            
            print("Cosmos DB Manager initialized successfully!")
            
        except Exception as e:
            print(f"Failed to initialize Cosmos DB Manager: {e}")
            raise
    
    async def setup_database_and_container(self):
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
            
            print(f"Database '{self.database_name}' and container '{self.container_name}' ready!")
            
        except exceptions.CosmosHttpResponseError as e:
            print(f"Failed to setup Cosmos DB: {e}")
            raise
    
    async def save_conversation(self, session_id: str, user_message: str, 
                              assistant_response: str, search_results: List[Dict] = None) -> str:
        try:
            if not self.container:
                await self.setup_database_and_container()
            
            conversation_doc = {
                "id": str(uuid.uuid4()),
                "session_id": session_id,
                "timestamp": datetime.now().isoformat(),
                "user_message": user_message,
                "assistant_response": assistant_response,
                "search_results": search_results or [],
                "type": "conversation_turn"
            }
            
            created_item = await self.container.create_item(body=conversation_doc)
            
            print(f"Saved conversation turn: {created_item['id']}")
            return created_item['id']
            
        except exceptions.CosmosHttpResponseError as e:
            print(f"Failed to save conversation: {e}")
            raise
    
    async def get_conversation_history(self, session_id: str, limit: int = 10) -> List[Dict]:
        try:
            if not self.container:
                await self.setup_database_and_container()
            
            query = """
                SELECT * FROM c 
                WHERE c.session_id = @session_id 
                AND c.type = 'conversation_turn'
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
            
            print(f"Retrieved {len(items)} conversation turns for session {session_id}")
            return items
            
        except exceptions.CosmosHttpResponseError as e:
            print(f"Failed to retrieve conversation history: {e}")
            return []
    
    async def save_session_metadata(self, session_id: str, metadata: Dict[str, Any]) -> str:
        try:
            if not self.container:
                await self.setup_database_and_container()
            
            # Create session metadata document
            session_doc = {
                "id": f"session_{session_id}",
                "session_id": session_id,
                "timestamp": datetime.now().isoformat(),
                "metadata": metadata,
                "type": "session_metadata"
            }
            
            # Upsert (create or update) the document
            created_item = await self.container.upsert_item(body=session_doc)
            
            print(f"Saved session metadata: {created_item['id']}")
            return created_item['id']
            
        except exceptions.CosmosHttpResponseError as e:
            print(f"Failed to save session metadata: {e}")
            raise
    
    async def get_session_metadata(self, session_id: str) -> Optional[Dict]:
        try:
            if not self.container:
                await self.setup_database_and_container()
            
            try:
                item = await self.container.read_item(
                    item=f"session_{session_id}",
                    partition_key=session_id
                )
                return item.get("metadata", {})
            except exceptions.CosmosResourceNotFoundError:
                return None
                
        except exceptions.CosmosHttpResponseError as e:
            print(f"Failed to retrieve session metadata: {e}")
            return None
    
    async def delete_conversation(self, session_id: str) -> bool:
        try:
            if not self.container:
                await self.setup_database_and_container()
            
            # Query for all items in the session
            query = "SELECT c.id FROM c WHERE c.session_id = @session_id"
            parameters = [{"name": "@session_id", "value": session_id}]
            
            # Get all item IDs
            item_ids = []
            async for item in self.container.query_items(
                query=query,
                parameters=parameters,
                partition_key=session_id
            ):
                item_ids.append(item['id'])
            
            # Delete each item
            deleted_count = 0
            for item_id in item_ids:
                try:
                    await self.container.delete_item(
                        item=item_id,
                        partition_key=session_id
                    )
                    deleted_count += 1
                except exceptions.CosmosResourceNotFoundError:
                    pass  # Item already deleted
            
            print(f"Deleted {deleted_count} items for session {session_id}")
            return True
            
        except exceptions.CosmosHttpResponseError as e:
            print(f"Failed to delete conversation: {e}")
            return False
    
    async def get_all_sessions(self, limit: int = 50) -> List[Dict]:
        try:
            if not self.container:
                await self.setup_database_and_container()
            
            query = """
                SELECT * FROM c 
                WHERE c.type = 'session_metadata'
                ORDER BY c.timestamp DESC
                OFFSET 0 LIMIT @limit
            """
            
            parameters = [{"name": "@limit", "value": limit}]
            
            sessions = []
            async for item in self.container.query_items(
                query=query,
                parameters=parameters,
                partition_key="metadata"
            ):
                sessions.append(item)
            
            print(f"Retrieved {len(sessions)} sessions")
            return sessions
            
        except exceptions.CosmosHttpResponseError as e:
            print(f"Failed to retrieve sessions: {e}")
            return []
    
    async def close(self):
        if self.client:
            await self.client.close()
    
async def main():
    cosmos_manager = CosmosDBManager()
    
    try:
        await cosmos_manager.setup_database_and_container()
        
        session_id = str(uuid.uuid4())
        
        await cosmos_manager.save_session_metadata(
            session_id,
            {
                "user_name": "Test User",
                "start_time": datetime.utcnow().isoformat(),
                "topic": "Travel Planning"
            }
        )
        
        await cosmos_manager.save_conversation(
            session_id,
            "What are the best places to visit in Paris?",
            "Paris offers many wonderful attractions including the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral.",
            [{"title": "Paris Travel Guide", "score": 0.95}]
        )
        
        history = await cosmos_manager.get_conversation_history(session_id)
        print(f"Conversation history: {len(history)} turns")
        
        metadata = await cosmos_manager.get_session_metadata(session_id)
        print(f"Session metadata: {metadata}")
        
    finally:
        await cosmos_manager.close()

if __name__ == "__main__":
    asyncio.run(main())
