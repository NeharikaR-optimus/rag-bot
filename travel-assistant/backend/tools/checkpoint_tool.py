import json
import logging
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4
from datetime import datetime

from langgraph.checkpoint.base import BaseCheckpointSaver, Checkpoint, CheckpointMetadata, CheckpointTuple
from azure.cosmos.aio import CosmosClient
from azure.cosmos import exceptions, PartitionKey

from config import Config

logger = logging.getLogger(__name__)


class CheckpointManager(BaseCheckpointSaver):
    """
    Manages LangGraph workflow checkpoints in Cosmos DB.
    Integrated with the tools framework for consistent data management.
    """
    
    def __init__(self):
        """Initialize the Cosmos DB checkpoint manager."""
        self.config = Config()
        self.client = None
        self.database = None
        self.container = None
        self._initialized = False
    
    async def _ensure_initialized(self):
        """Ensure Cosmos DB client and containers are initialized."""
        if self._initialized:
            return
            
        try:
            # Initialize Cosmos client
            self.client = CosmosClient(
                url=self.config.COSMOS_ENDPOINT,
                credential=self.config.COSMOS_KEY
            )
            
            # Get or create database
            try:
                self.database = await self.client.create_database_if_not_exists(
                    id=self.config.COSMOS_DATABASE_NAME
                )
            except exceptions.CosmosResourceExistsError:
                self.database = self.client.get_database_client(self.config.COSMOS_DATABASE_NAME)
            
            try:
                self.container = await self.database.create_container_if_not_exists(
                    id="checkpoints",
                    partition_key=PartitionKey(path="/thread_id")
                )
            except exceptions.CosmosResourceExistsError:
                self.container = self.database.get_container_client("checkpoints")
            
            self._initialized = True
            logger.info("Cosmos DB checkpoint manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Cosmos DB checkpoint manager: {e}")
            raise
    
    async def aget(self, config: Dict[str, Any]) -> Optional[Checkpoint]:
        """Get the latest checkpoint for a thread."""
        await self._ensure_initialized()
        
        thread_id = config.get("configurable", {}).get("thread_id")
        if not thread_id:
            return None
        
        try:
            # Query for the latest checkpoint for this thread
            query = """
                SELECT * FROM c 
                WHERE c.thread_id = @thread_id 
                ORDER BY c.checkpoint_id DESC 
                OFFSET 0 LIMIT 1
            """
            
            items = []
            async for item in self.container.query_items(
                query=query,
                parameters=[{"name": "@thread_id", "value": thread_id}],
                enable_cross_partition_query=True
            ):
                items.append(item)
            
            if not items:
                return None
            
            item = items[0]
            
            # Reconstruct checkpoint from stored data
            checkpoint = Checkpoint(
                v=item["checkpoint_data"]["v"],
                ts=item["checkpoint_data"]["ts"],
                id=item["checkpoint_data"]["id"],
                channel_values=self._deserialize_values(item["checkpoint_data"]["channel_values"]),
                channel_versions=item["checkpoint_data"]["channel_versions"],
                versions_seen=item["checkpoint_data"]["versions_seen"],
                pending_sends=item["checkpoint_data"].get("pending_sends", [])
            )
            
            return checkpoint
            
        except Exception as e:
            logger.error(f"Error getting checkpoint for thread {thread_id}: {e}")
            return None
    
    async def aget_tuple(self, config: Dict[str, Any]) -> Optional[Tuple[Dict[str, Any], Checkpoint, CheckpointMetadata]]:
        """Get the latest checkpoint tuple for a thread (required by LangGraph)."""
        await self._ensure_initialized()
        
        thread_id = config.get("configurable", {}).get("thread_id")
        if not thread_id:
            return None
        
        try:
            # Query for the latest checkpoint for this thread
            query = """
                SELECT * FROM c 
                WHERE c.thread_id = @thread_id 
                ORDER BY c.checkpoint_id DESC 
                OFFSET 0 LIMIT 1
            """
            
            items = []
            async for item in self.container.query_items(
                query=query,
                parameters=[{"name": "@thread_id", "value": thread_id}],
                enable_cross_partition_query=True
            ):
                items.append(item)
            
            if not items:
                return None
            
            item = items[0]
            
            # Reconstruct checkpoint from stored data
            checkpoint = Checkpoint(
                v=item["checkpoint_data"]["v"],
                ts=item["checkpoint_data"]["ts"],
                id=item["checkpoint_data"]["id"],
                channel_values=self._deserialize_values(item["checkpoint_data"]["channel_values"]),
                channel_versions=item["checkpoint_data"]["channel_versions"],
                versions_seen=item["checkpoint_data"]["versions_seen"],
                pending_sends=item["checkpoint_data"].get("pending_sends", [])
            )
            
            config_dict = {
                "configurable": {
                    "thread_id": item["thread_id"],
                    "checkpoint_id": item["checkpoint_id"]
                }
            }
            
            return (config_dict, checkpoint, item["metadata"])
            
        except Exception as e:
            logger.error(f"Error getting checkpoint tuple for thread {thread_id}: {e}")
            return None
    
    async def aput(
        self,
        config: Dict[str, Any],
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Save a checkpoint to Cosmos DB."""
        await self._ensure_initialized()
        
        thread_id = config.get("configurable", {}).get("thread_id")
        if not thread_id:
            thread_id = str(uuid4())
        
        try:
            # Handle checkpoint as dict or object
            if hasattr(checkpoint, 'id'):
                checkpoint_id = checkpoint.id
                checkpoint_v = checkpoint.v
                checkpoint_ts = checkpoint.ts
                channel_values = checkpoint.channel_values
                channel_versions = checkpoint.channel_versions
                versions_seen = checkpoint.versions_seen
                pending_sends = checkpoint.pending_sends
            else:
                # Handle checkpoint as dictionary
                checkpoint_id = checkpoint.get('id', str(uuid4()))
                checkpoint_v = checkpoint.get('v', 1)
                checkpoint_ts = checkpoint.get('ts', datetime.utcnow().isoformat())
                channel_values = checkpoint.get('channel_values', {})
                channel_versions = checkpoint.get('channel_versions', {})
                versions_seen = checkpoint.get('versions_seen', {})
                pending_sends = checkpoint.get('pending_sends', [])
            
            # Create document for Cosmos DB
            checkpoint_doc = {
                "id": str(uuid4()),
                "thread_id": thread_id,
                "checkpoint_id": checkpoint_id,
                "timestamp": datetime.utcnow().isoformat(),
                "checkpoint_data": {
                    "v": checkpoint_v,
                    "ts": checkpoint_ts,
                    "id": checkpoint_id,
                    "channel_values": self._serialize_values(channel_values),
                    "channel_versions": channel_versions,
                    "versions_seen": versions_seen,
                    "pending_sends": pending_sends
                },
                "metadata": metadata,
                "new_versions": new_versions
            }
            
            # Save to Cosmos DB
            await self.container.create_item(body=checkpoint_doc)
            
            logger.info(f"Saved checkpoint {checkpoint_id} for thread {thread_id}")
            
            return {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_id": checkpoint_id
                }
            }
            
        except Exception as e:
            logger.error(f"Error saving checkpoint for thread {thread_id}: {e}")
            raise
    
    async def alist(
        self,
        config: Dict[str, Any],
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None
    ) -> List[Tuple[Dict[str, Any], Checkpoint, CheckpointMetadata]]:
        """List checkpoints for a thread."""
        await self._ensure_initialized()
        
        thread_id = config.get("configurable", {}).get("thread_id")
        if not thread_id:
            return []
        
        try:
            query = "SELECT * FROM c WHERE c.thread_id = @thread_id ORDER BY c.timestamp DESC"
            
            items = []
            async for item in self.container.query_items(
                query=query,
                parameters=[{"name": "@thread_id", "value": thread_id}],
                enable_cross_partition_query=True
            ):
                # Reconstruct checkpoint
                checkpoint = Checkpoint(
                    v=item["checkpoint_data"]["v"],
                    ts=item["checkpoint_data"]["ts"],
                    id=item["checkpoint_data"]["id"],
                    channel_values=self._deserialize_values(item["checkpoint_data"]["channel_values"]),
                    channel_versions=item["checkpoint_data"]["channel_versions"],
                    versions_seen=item["checkpoint_data"]["versions_seen"],
                    pending_sends=item["checkpoint_data"].get("pending_sends", [])
                )
                
                config_dict = {
                    "configurable": {
                        "thread_id": item["thread_id"],
                        "checkpoint_id": item["checkpoint_id"]
                    }
                }
                
                items.append((config_dict, checkpoint, item["metadata"]))
                
                if limit and len(items) >= limit:
                    break
            
            return items
            
        except Exception as e:
            logger.error(f"Error listing checkpoints for thread {thread_id}: {e}")
            return []
    
    def _serialize_values(self, values: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize complex values for storage."""
        serialized = {}
        for key, value in values.items():
            try:
                # Try to serialize as JSON
                serialized[key] = json.loads(json.dumps(value, default=str))
            except (TypeError, ValueError):
                # If serialization fails, convert to string
                serialized[key] = str(value)
        return serialized
    
    def _deserialize_values(self, values: Dict[str, Any]) -> Dict[str, Any]:
        """Deserialize values from storage."""
        return values
    
    async def cleanup_old_checkpoints(self, thread_id: str, keep_latest: int = 10):
        """Clean up old checkpoints, keeping only the latest N checkpoints."""
        await self._ensure_initialized()
        
        try:
            # Get all checkpoints for thread ordered by timestamp
            query = """
                SELECT c.id, c.timestamp FROM c 
                WHERE c.thread_id = @thread_id 
                ORDER BY c.timestamp DESC 
                OFFSET @offset LIMIT 1000
            """
            
            items_to_delete = []
            async for item in self.container.query_items(
                query=query,
                parameters=[
                    {"name": "@thread_id", "value": thread_id},
                    {"name": "@offset", "value": keep_latest}
                ],
                enable_cross_partition_query=True
            ):
                items_to_delete.append(item["id"])
            
            # Delete old checkpoints
            for item_id in items_to_delete:
                await self.container.delete_item(
                    item=item_id,
                    partition_key=thread_id
                )
            
            if items_to_delete:
                logger.info(f"Cleaned up {len(items_to_delete)} old checkpoints for thread {thread_id}")
                
        except Exception as e:
            logger.error(f"Error cleaning up checkpoints for thread {thread_id}: {e}")
    
    async def close(self):
        """Close the Cosmos DB client."""
        if self.client:
            await self.client.close()
