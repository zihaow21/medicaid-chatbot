"""
Conversation Memory - Conceptual Framework

Demonstrates conversation state management patterns for multi-turn
AI chatbot interactions. Pure architectural concepts.
"""

from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime
from enum import Enum


class ConversationState(Enum):
    """Conversation state enumeration"""
    ACTIVE = "active"
    WAITING_CLARIFICATION = "waiting_clarification"
    COMPLETED = "completed"


@dataclass
class ConversationTurn:
    """
    Single conversation turn structure
    Concept: User input + Bot response + Context tracking
    """
    user_message: str
    bot_response: str
    timestamp: datetime
    context_used: Optional[List[str]] = None


@dataclass
class ConversationContext:
    """
    Conversation context aggregation
    Concept: Topic tracking + Entity persistence + State management
    """
    current_topic: Optional[str] = None
    mentioned_entities: List[str] = None
    clarification_needed: bool = False
    
    def __post_init__(self):
        if self.mentioned_entities is None:
            self.mentioned_entities = []


class ConversationMemory:
    """
    Conversation Memory Manager
    Architectural Pattern: State persistence + Context management
    Concept: Multi-turn conversation continuity
    """
    
    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        self.sessions = {}  # session_id -> List[ConversationTurn]
        self.contexts = {}  # session_id -> ConversationContext
    
    def create_session(self, session_id: str) -> None:
        """Initialize new conversation session"""
        self.sessions[session_id] = []
        self.contexts[session_id] = ConversationContext()
    
    def add_turn(self, session_id: str, user_message: str, bot_response: str) -> None:
        """
        Add conversation turn
        Concept: Turn persistence + Context update + History management
        """
        if session_id not in self.sessions:
            self.create_session(session_id)
        
        # Create turn
        turn = ConversationTurn(
            user_message=user_message,
            bot_response=bot_response,
            timestamp=datetime.now()
        )
        
        # Add to session history
        self.sessions[session_id].append(turn)
        
        # Maintain turn limit
        if len(self.sessions[session_id]) > self.max_turns:
            self.sessions[session_id] = self.sessions[session_id][-self.max_turns:]
        
        # Update context
        self._update_context(session_id, user_message)
    
    def get_history(self, session_id: str, limit: int = 5) -> List[ConversationTurn]:
        """Retrieve conversation history"""
        if session_id not in self.sessions:
            return []
        return self.sessions[session_id][-limit:]
    
    def get_context(self, session_id: str) -> Optional[ConversationContext]:
        """Get conversation context"""
        return self.contexts.get(session_id)
    
    def _update_context(self, session_id: str, user_message: str) -> None:
        """
        Context update concept
        Concept: Entity extraction + Topic tracking + State management
        """
        context = self.contexts[session_id]
        
        # Simple topic detection concept examples
        if "dental" in user_message.lower():
            context.current_topic = "dental_benefits"
        elif "vision" in user_message.lower():
            context.current_topic = "vision_benefits"
        
        # Entity extraction concept
        if "doctor" in user_message.lower():
            if "doctor" not in context.mentioned_entities:
                context.mentioned_entities.append("doctor") 