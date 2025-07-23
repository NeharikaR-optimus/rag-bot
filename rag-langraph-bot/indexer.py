import os
import asyncio
from typing import List, Dict, Any
from dotenv import load_dotenv
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
    SemanticConfiguration,
    SemanticSearch,
    SemanticPrioritizedFields,
    SemanticField
)
from azure.core.credentials import AzureKeyCredential
from langchain_openai import AzureOpenAIEmbeddings
import json
import uuid
from datetime import datetime

# Load environment variables
load_dotenv()

class AzureSearchDocumentIndexer:
    
    def __init__(self):
        load_dotenv()
        self.search_client = None
        self.index_client = None
        self.embeddings = None
        self._initialize()
    
    def _initialize(self):
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
            self.index_client = SearchIndexClient(
                endpoint=search_endpoint,
                credential=credential
            )
            
            self.embeddings = AzureOpenAIEmbeddings(
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_EMBEDDING_API_KEY"),
                azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
                api_version=os.getenv("AZURE_OPENAI_EMBEDDING_API_VERSION")
            )
            
            print("✅ Azure Search Document Indexer initialized successfully!")
            
        except Exception as e:
            print(f"❌ Failed to initialize Azure Search Document Indexer: {e}")
            raise
    
    def create_search_index(self):
        try:
            # Define search fields
            fields = [
                SimpleField(name="id", type=SearchFieldDataType.String, key=True),
                SearchableField(name="title", type=SearchFieldDataType.String),
                SearchableField(name="content", type=SearchFieldDataType.String),
                SimpleField(name="source", type=SearchFieldDataType.String, filterable=True),
                SimpleField(name="filename", type=SearchFieldDataType.String, filterable=True),
                SimpleField(name="file_size", type=SearchFieldDataType.Int32),
                SimpleField(name="indexed_date", type=SearchFieldDataType.DateTimeOffset),
                SearchField(
                    name="content_vector",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    searchable=True,
                    vector_search_dimensions=1536,  # text-embedding-3-small dimension
                    vector_search_profile_name="myHnswProfile"
                )
            ]
            
            # Configure vector search
            vector_search = VectorSearch(
                algorithms=[
                    HnswAlgorithmConfiguration(
                        name="myHnsw",
                        parameters={
                            "m": 4,
                            "ef_construction": 400,
                            "ef_search": 500,
                            "metric": "cosine"
                        }
                    )
                ],
                profiles=[
                    VectorSearchProfile(
                        name="myHnswProfile",
                        algorithm_configuration_name="myHnsw"
                    )
                ]
            )
            
            # Configure semantic search
            semantic_config = SemanticConfiguration(
                name="my-semantic-config",
                prioritized_fields=SemanticPrioritizedFields(
                    content_fields=[SemanticField(field_name="content")],
                    keywords_fields=[SemanticField(field_name="title")]
                )
            )
            
            semantic_search = SemanticSearch(configurations=[semantic_config])
            
            # Create the search index
            index = SearchIndex(
                name=os.getenv("AZURE_SEARCH_INDEX_NAME", "travel-documents"),
                fields=fields,
                vector_search=vector_search,
                semantic_search=semantic_search
            )
            
            result = self.index_client.create_or_update_index(index)
            print(f"✅ Search index '{result.name}' created/updated successfully!")
            
        except Exception as e:
            print(f"❌ Failed to create search index: {e}")
            raise
    
    async def process_document(self, file_path: str, title: str = None) -> Dict:
        """Process a complete document for indexing (no text splitting)"""
        try:
            # Read the entire document
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # Get file information
            file_size = os.path.getsize(file_path)
            filename = os.path.basename(file_path)
            
            # Generate embedding for the entire document
            print(f"📝 Generating embedding for {filename}...")
            embedding = await self.embeddings.aembed_query(content[:8000])  # Limit to first 8000 chars for embedding
            
            # Create document for indexing
            doc = {
                "id": str(uuid.uuid4()),
                "title": title or filename.replace('.txt', '').replace('_', ' ').title(),
                "content": content,
                "source": file_path,  # Use local file path as source
                "filename": filename,
                "file_size": file_size,
                "indexed_date": datetime.now().isoformat() + "Z",
                "content_vector": embedding
            }
            
            print(f"✅ Processed {filename} as a complete document")
            return doc
            
        except Exception as e:
            print(f"❌ Failed to process document {file_path}: {e}")
            return None
    
    def index_document(self, document: Dict):
        """Index a single document in Azure AI Search"""
        try:
            if not document:
                print("⚠️ No document to index")
                return False
            
            # Upload document directly
            result = self.search_client.upload_documents([document])
            
            if result[0].succeeded:
                print(f"✅ Successfully indexed document: {document['title']}")
                return True
            else:
                print(f"❌ Failed to index document: {result[0].error_message}")
                return False
                    
        except Exception as e:
            print(f"❌ Failed to index document: {e}")
            return False
    
    async def index_documents_from_folder(self, folder_path: str):
        """Index all text files from a folder as complete documents"""
        try:
            if not os.path.exists(folder_path):
                print(f"❌ Folder not found: {folder_path}")
                return
            
            success_count = 0
            total_count = 0
            
            # Process all .txt files in the folder
            for filename in os.listdir(folder_path):
                if filename.endswith('.txt'):
                    total_count += 1
                    file_path = os.path.join(folder_path, filename)
                    print(f"\n📄 Processing {filename}...")
                    
                    # Extract title from filename
                    title = filename.replace('.txt', '').replace('_', ' ').title()
                    
                    # Process document
                    document = await self.process_document(file_path, title)
                    
                    # Index document
                    if document and self.index_document(document):
                        success_count += 1
            
            print(f"\n✅ Successfully indexed {success_count}/{total_count} documents from {folder_path}")
                
        except Exception as e:
            print(f"❌ Failed to index documents from folder: {e}")
            raise
    
    async def search_documents(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search documents in Azure AI Search"""
        try:
            # Generate query embedding for vector search
            query_embedding = await self.embeddings.aembed_query(query)
            
            # Create vectorized query
            from azure.search.documents.models import VectorizedQuery
            vector_query = VectorizedQuery(
                vector=query_embedding,
                k_nearest_neighbors=top_k,
                fields="content_vector"
            )
            
            # Perform hybrid search
            results = self.search_client.search(
                search_text=query,
                vector_queries=[vector_query],
                select=["id", "title", "content", "source", "filename"],
                top=top_k
            )
            
            # Format results
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "id": result.get("id", ""),
                    "title": result.get("title", ""),
                    "content": result.get("content", "")[:1000] + "..." if len(result.get("content", "")) > 1000 else result.get("content", ""),
                    "source": result.get("source", ""),
                    "filename": result.get("filename", ""),
                    "score": result.get("@search.score", 0)
                })
            
            return formatted_results
            
        except Exception as e:
            print(f"❌ Error searching documents: {e}")
            return []

# Example usage
async def main():
    """Main function to demonstrate document indexing"""
    indexer = AzureSearchDocumentIndexer()
    
    # Create search index
    print("Creating search index...")
    indexer.create_search_index()
    
    # Index documents from the documents folder
    documents_folder = "documents"
    if os.path.exists(documents_folder):
        print(f"\nIndexing documents from {documents_folder}...")
        await indexer.index_documents_from_folder(documents_folder)
        
        # Test search
        print("\n" + "="*50)
        print("Testing search functionality...")
        results = await indexer.search_documents("Paris travel tips")
        for i, result in enumerate(results[:3], 1):
            print(f"\n{i}. {result['title']}")
            print(f"   Score: {result['score']:.3f}")
            print(f"   Content: {result['content'][:200]}...")
    else:
        print(f"Documents folder not found: {documents_folder}")
        print("Please create the folder and add your .txt files")

if __name__ == "__main__":
    asyncio.run(main())
