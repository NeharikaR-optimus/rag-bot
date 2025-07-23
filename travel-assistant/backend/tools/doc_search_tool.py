import os
import sys
from pathlib import Path
from typing import List, Dict, Any
import json
import requests
import logging
from datetime import datetime

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from config import Config

logger = logging.getLogger(__name__)

def doc_search_tool(query: str, top_k: int = 5) -> Dict[str, Any]:
    """Tool function for document search compatible with workflow."""
    try:
        doc_tool = DocumentManagementTool()
        
        search_url = f"{doc_tool.azure_search_endpoint}/indexes/{doc_tool.azure_search_index_name}/docs/search"
        
        search_body = {
            "search": query,
            "queryType": "semantic",
            "semanticConfiguration": "default",
            "count": True,
            "top": top_k,
            "select": "content,title,source,filename"
        }
        
        response = requests.post(
            search_url,
            headers=doc_tool.headers,
            params=doc_tool.params,
            data=json.dumps(search_body),
            timeout=30
        )
        
        if response.status_code == 200:
            results = response.json()
            search_results = results.get('value', [])
            
            return {
                "status": "success",
                "results": search_results,
                "count": len(search_results)
            }
        else:
            return {
                "status": "error",
                "results": [],
                "count": 0,
                "error": f"Search failed: {response.status_code}"
            }
            
    except Exception as e:
        return {
            "status": "error",
            "results": [],
            "count": 0,
            "error": str(e)
        }

class DocumentManagementTool:
    """Tool for document indexing and semantic search in Azure AI Search."""
    
    def __init__(self):
        """Initialize the document management tool."""
        self.azure_search_endpoint = Config.AZURE_SEARCH_ENDPOINT
        self.azure_search_api_key = Config.AZURE_SEARCH_API_KEY
        self.azure_search_api_version = "2024-07-01"
        self.azure_search_index_name = Config.AZURE_SEARCH_INDEX_NAME
        
        self.headers = {
            'Content-Type': 'application/json',
            'api-key': self.azure_search_api_key
        }
        
        self.params = {
            'api-version': self.azure_search_api_version
        }
        
        self.documents_dir = Path("documents")
    
    def semantic_search(self, query_text: str, top_k: int = 5) -> str:
        """Perform semantic search using Azure AI Search."""
        try:
            search_url = f"{self.azure_search_endpoint}/indexes/{self.azure_search_index_name}/docs/search"
            
            search_body = {
                "search": query_text,
                "queryType": "semantic",
                "semanticConfiguration": "default",
                "count": True,
                "top": top_k,
                "select": "content,title,source"
            }
            
            response = requests.post(
                search_url,
                headers=self.headers,
                params=self.params,
                data=json.dumps(search_body),
                timeout=30
            )
            
            if response.status_code == 200:
                results = response.json()
                search_results = results.get('value', [])
                
                if search_results:
                    combined_content = ""
                    for result in search_results:
                        content = result.get('content', '')
                        title = result.get('title', 'Untitled')
                        combined_content += f"Title: {title}\\n{content}\\n\\n"
                    
                    return combined_content.strip()
                else:
                    return ""
            else:
                print(f"Search failed: {response.status_code}")
                return ""
                
        except Exception as e:
            print(f"Error during search: {e}")
            return ""
    
    def load_and_prepare_documents(self) -> List[Dict[str, Any]]:
        """Load documents and prepare them for indexing."""
        documents = []
        
        if not self.documents_dir.exists():
            return documents
        
        for file_path in self.documents_dir.glob("*.txt"):
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read().strip()
                    
                    if content:
                        doc_id = file_path.stem
                        
                        document = {
                            "id": doc_id,
                            "title": file_path.stem.replace('_', ' ').title(),
                            "content": content,
                            "source": str(file_path),
                            "filename": file_path.name,
                            "file_size": len(content),
                            "indexed_date": datetime.now().isoformat() + "Z",
                            "@search.action": "upload"
                        }
                        
                        documents.append(document)
                    
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
        
        return documents
    
    def upload_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """Upload documents to Azure AI Search."""
        try:
            if not documents:
                return False
            
            upload_url = f"{self.azure_search_endpoint}/indexes/{self.azure_search_index_name}/docs/index"
            
            upload_body = {
                "value": documents
            }
            
            response = requests.post(
                upload_url,
                headers=self.headers,
                params=self.params,
                data=json.dumps(upload_body),
                timeout=60
            )
            
            if response.status_code in [200, 201]:
                return True
            else:
                print(f"Upload failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"Error uploading documents: {e}")
            return False
    
    def index_documents(self):
        """Index all travel documents."""
        try:
            documents = self.load_and_prepare_documents()
            
            if not documents:
                return False
            
            success = self.upload_documents(documents)
            return success
            
        except Exception as e:
            print(f"Error during indexing: {e}")
            return False
