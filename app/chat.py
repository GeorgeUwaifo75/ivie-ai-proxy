# app/chat.py
import re
import torch
import threading
import logging
from typing import Optional, Dict, List
from datetime import datetime
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os
from dotenv import load_dotenv

from app.models import ChatSession
from app.utils import (
    classify_query, 
    clean_response_text, 
    polish_response,
    ensure_natural_ending,
    remove_consecutive_word_repeats,
    get_model_status
)

load_dotenv()
logger = logging.getLogger(__name__)

class ChatManager:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_loading = False
        self.model_load_error = None
        self.model_load_stage = "not_started"
        self.model_load_progress = 0
        self.chat_sessions: Dict[str, ChatSession] = {}
        self.current_session_id: Optional[str] = None
        
        # Model configuration
        self.MODEL_ID = "GeorgeUwaifo/ivie_gpt2_new01c_results"
        self.BASE_TOKENIZER = "gpt2"
        self.TOKEN = os.getenv('HF_TOKEN')
        self.MAX_TOKENS_HARD_CAP = 450
        
    async def initialize(self):
        """Initialize the model in a background thread."""
        self.model_loading = True
        thread = threading.Thread(target=self._load_model, daemon=True)
        thread.start()
        logger.info("Background model loading started")
    
    def _load_model(self):
        """Load tokenizer and model weights."""
        try:
            logger.info("=" * 60)
            logger.info("Loading IvieAI model…")
            logger.info("=" * 60)
            
            # Stage 1: Load tokenizer
            self.model_load_stage = "Loading tokenizer from base GPT-2"
            self.model_load_progress = 20
            logger.info("Stage 1/3: Loading tokenizer…")
            
            self.tokenizer = GPT2Tokenizer.from_pretrained(self.BASE_TOKENIZER)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model_load_progress = 40
            logger.info("✅ Tokenizer loaded")
            
            # Stage 2: Load model
            self.model_load_stage = "Downloading model weights"
            self.model_load_progress = 50
            logger.info("Stage 2/3: Loading model weights…")
            
            _model = GPT2LMHeadModel.from_pretrained(
                self.MODEL_ID,
                token=self.TOKEN if self.TOKEN else None,
                low_cpu_mem_usage=True
            )
            
            self.model_load_progress = 80
            logger.info("✅ Model weights loaded")
            
            # Stage 3: Finalize
            self.model_load_stage = "Finalising model"
            self.model_load_progress = 90
            logger.info("Stage 3/3: Finalising…")
            
            if torch.cuda.is_available():
                _model = _model.cuda()
                logger.info("Moved to GPU ✅")
            _model.eval()
            
            self.model = _model
            self.model_load_stage = "ready"
            self.model_load_progress = 100
            
            logger.info("✅ Model loaded and ready!")
            logger.info("=" * 60)
            
            # Quick smoke test
            test = self.generate_response_sync("Hello", None)
            logger.info(f"Smoke test OK — response: {test[:100]}")
            
        except Exception as e:
            self.model_load_error = str(e)
            self.model_load_stage = "error"
            logger.error(f"❌ Model load failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
        self.model_loading = False
    
    async def generate_response(self, user_message: str, session: Optional[ChatSession] = None) -> str:
        """Generate a response asynchronously."""
        # Run generation in thread pool to avoid blocking
        import asyncio
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, 
            self.generate_response_sync, 
            user_message, 
            session
        )
        return response
    
    def generate_response_sync(self, user_message: str, session: Optional[ChatSession] = None) -> str:
        """Generate a response synchronously."""
        if self.model is None or self.tokenizer is None:
            return "I'm still loading. Please give me just a moment!"
        
        # Classify query
        params = classify_query(user_message, self.MAX_TOKENS_HARD_CAP)
        logger.info(
            f"Generating | max_new={params['max_new_tokens']} "
            f"min_new={params['min_new_tokens']} "
            f"temp={params['temperature']}"
        )
        
        try:
            # Format prompt
            formatted_prompt = f"User: {user_message}\nAssistant:"
            
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            
            if torch.cuda.is_available() and hasattr(self.model, 'device'):
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Set EOS token
            eos_token_id = self.tokenizer.eos_token_id
            if eos_token_id is None:
                eos_token_id = self.tokenizer.pad_token_id
            
            prompt_len = inputs['input_ids'].shape[-1]
            
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_new_tokens=params['max_new_tokens'],
                    min_new_tokens=params['min_new_tokens'],
                    temperature=params['temperature'],
                    top_p=params['top_p'],
                    top_k=params['top_k'],
                    do_sample=params['do_sample'],
                    repetition_penalty=params['repetition_penalty'],
                    no_repeat_ngram_size=params['no_repeat_ngram_size'],
                    early_stopping=params['early_stopping'],
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=eos_token_id,
                    forced_eos_token_id=eos_token_id,
                )
            
            # Decode only new tokens
            new_tokens = outputs[0][prompt_len:]
            response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            
            # Clean response
            response = re.sub(r'^Assistant:\s*', '', response.strip())
            if response.lower().startswith(user_message.lower()):
                response = response[len(user_message):].strip()
            
            response = clean_response_text(response)
            response = remove_consecutive_word_repeats(response)
            response = ensure_natural_ending(response)
            response = polish_response(response, char_limit=params['char_limit'])
            
            # Final safety check
            if len(response) < 10:
                response = "I understand your query. Could you please provide more details so I can give you a better response?"
            
            logger.info(f"Response length: {len(response)} chars")
            return response
            
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return "I encountered a small issue generating a response. Could you please try again?"
    
    def get_or_create_session(self, session_id: Optional[str] = None) -> ChatSession:
        """Get existing session or create new one."""
        if session_id and session_id in self.chat_sessions:
            return self.chat_sessions[session_id]
        
        session = ChatSession()
        self.chat_sessions[session.session_id] = session
        self.current_session_id = session.session_id
        return session
    
    def create_session(self) -> ChatSession:
        """Create a new session."""
        session = ChatSession()
        self.chat_sessions[session.session_id] = session
        self.current_session_id = session.session_id
        return session
    
    def get_session(self, session_id: str) -> Optional[ChatSession]:
        """Get a specific session."""
        return self.chat_sessions.get(session_id)
    
    def get_current_session(self) -> Optional[ChatSession]:
        """Get the current session."""
        if self.current_session_id and self.current_session_id in self.chat_sessions:
            return self.chat_sessions[self.current_session_id]
        return None
    
    def get_all_sessions(self) -> List[Dict]:
        """Get all sessions as list of dicts."""
        sessions = []
        for s in self.chat_sessions.values():
            sessions.append({
                'session_id': s.session_id,
                'created_at': s.created_at.isoformat(),
                'last_updated': s.last_updated.isoformat(),
                'total_interactions': len(s.history)
            })
        sessions.sort(key=lambda x: x['last_updated'], reverse=True)
        return sessions
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        if session_id in self.chat_sessions:
            if self.current_session_id == session_id:
                self.current_session_id = None
            del self.chat_sessions[session_id]
            return True
        return False