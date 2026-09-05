# app/models.py
from datetime import datetime
from typing import List, Dict, Optional
from pydantic import BaseModel

class ChatSession:
    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        self.history: List[Dict[str, str]] = []
        self.created_at = datetime.now()
        self.last_updated = datetime.now()
    
    def add_interaction(self, user_message: str, ai_response: str):
        self.history.append({"trigger": user_message, "reply": ai_response})
        self.last_updated = datetime.now()
    
    def to_json(self) -> Dict:
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "total_interactions": len(self.history),
            "history": self.history
        }
    
    def to_text(self) -> str:
        lines = [
            "=" * 60,
            f"Chat Session: {self.session_id}",
            f"Created: {self.created_at.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Last Updated: {self.last_updated.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Interactions: {len(self.history)}",
            "=" * 60, ""
        ]
        for i, h in enumerate(self.history, 1):
            lines += [
                f"[{i}] You:    {h['trigger']}",
                f"[{i}] IvieAI: {h['reply']}",
                "-" * 40
            ]
        return "\n".join(lines)

class ModelStatus(BaseModel):
    ready: bool = False
    status: str = "not_started"
    stage: str = "waiting"
    progress: int = 0
    detail: Optional[str] = None