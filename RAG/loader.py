import json
from langchain_core.documents import Document

def load_json(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    documents = []
    
    for item in data:
        text = f""" {item['question']}
{item['answer']}"""
        
        documents.append(
            Document(
                page_content=text,
                metadata={"lang": "my"}
            )
        )
        
    return documents