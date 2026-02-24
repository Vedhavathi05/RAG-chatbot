import json
import os
from datetime import datetime
from typing import Optional, List, Dict
import uuid
from app.models import Conversation, Message

class ConversationDB:
    """Simple JSON-based conversation storage"""
    
    def __init__(self, db_path: str = "storage/conversations"):
        self.db_path = db_path
        os.makedirs(db_path, exist_ok=True)
    
    def _get_conversation_file(self, conversation_id: str) -> str:
        return os.path.join(self.db_path, f"{conversation_id}.json")
    
    def create_conversation(self, title: str = "New Conversation") -> Conversation:
        """Create a new conversation"""
        conversation_id = str(uuid.uuid4())
        conversation = Conversation(
            id=conversation_id,
            title=title,
            messages=[],
            context=""
        )
        self.save_conversation(conversation)
        return conversation
    
    def save_conversation(self, conversation: Conversation):
        """Save conversation to disk"""
        path = self._get_conversation_file(conversation.id)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                'id': conversation.id,
                'title': conversation.title,
                'created_at': conversation.created_at.isoformat(),
                'updated_at': datetime.now().isoformat(),
                'context': conversation.context,
                'messages': [
                    {
                        'role': msg.role,
                        'content': msg.content,
                        'timestamp': msg.timestamp.isoformat(),
                        'citations': msg.citations
                    }
                    for msg in conversation.messages
                ]
            }, f, ensure_ascii=False, indent=2)
    
    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """Load conversation from disk"""
        path = self._get_conversation_file(conversation_id)
        if not os.path.exists(path):
            return None
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        messages = [
            Message(
                role=msg['role'],
                content=msg['content'],
                timestamp=datetime.fromisoformat(msg['timestamp']),
                citations=msg.get('citations')
            )
            for msg in data['messages']
        ]
        
        return Conversation(
            id=data['id'],
            title=data['title'],
            messages=messages,
            created_at=datetime.fromisoformat(data['created_at']),
            updated_at=datetime.fromisoformat(data['updated_at']),
            context=data.get('context', '')
        )
    
    def list_conversations(self) -> List[Dict]:
        """List all conversations"""
        conversations = []
        for filename in os.listdir(self.db_path):
            if filename.endswith('.json'):
                conversation_id = filename[:-5]
                conv = self.get_conversation(conversation_id)
                if conv:
                    preview = ""
                    if conv.messages:
                        last_msg = next((m for m in reversed(conv.messages) if m.role == "assistant"), None)
                        if last_msg:
                            preview = last_msg.content[:100]
                    
                    conversations.append({
                        'id': conv.id,
                        'title': conv.title,
                        'updated_at': conv.updated_at.isoformat(),
                        'preview': preview
                    })
        
        # Sort by updated_at descending
        conversations.sort(key=lambda x: x['updated_at'], reverse=True)
        return conversations
    
    def add_message(self, conversation_id: str, role: str, content: str, citations: Optional[List] = None):
        """Add message to conversation"""
        conv = self.get_conversation(conversation_id)
        if not conv:
            return None
        
        message = Message(role=role, content=content, citations=citations)
        conv.messages.append(message)
        self.save_conversation(conv)
        return message
    
    def update_context(self, conversation_id: str, context: str):
        """Update accumulated context for follow-up questions"""
        conv = self.get_conversation(conversation_id)
        if conv:
            conv.context = context
            self.save_conversation(conv)
    
    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation"""
        path = self._get_conversation_file(conversation_id)
        if os.path.exists(path):
            os.remove(path)
            return True
        return False

# Global instance
db = ConversationDB()
