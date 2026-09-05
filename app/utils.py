# app/utils.py
import re
import logging
from typing import Dict

logger = logging.getLogger(__name__)

# Simple patterns for classification
SIMPLE_PATTERNS = [
    r"^(hi|hello|hey|howdy|hiya|yo)\b",
    r"^(good\s)?(morning|afternoon|evening|night)\b",
    r"^how are you",
    r"^how('s| is) it going",
    r"^what('s| is) up",
    r"^(thanks|thank you|cheers|ok|okay|sure|great|cool|nice|wow|lol)\b",
    r"^(yes|no|yeah|nope|yep|nah)\b",
    r"^(bye|goodbye|see you|take care|later)\b",
    r"^(who are you|what are you|what is your name)\??$",
]

ELABORATE_KEYWORDS = [
    "explain", "describe", "how does", "how do", "why does", "why do",
    "what is the difference", "compare", "elaborate", "detail", "in depth",
    "step by step", "walk me through", "tell me about", "give me a summary",
    "what are the", "list", "pros and cons", "advantages", "disadvantages",
    "history of", "meaning of", "definition of", "what causes", "how to",
    "can you help me", "i need help with", "write a", "create a",
    "give me an example", "examples of",
]

def classify_query(message: str, hard_cap: int = 450) -> Dict:
    """Classify query and return generation parameters."""
    params = _classify_query_raw(message)
    params['max_new_tokens'] = min(params['max_new_tokens'], hard_cap)
    return params

def _classify_query_raw(message: str) -> Dict:
    msg_lower = message.lower().strip()
    word_count = len(msg_lower.split())
    
    # Simple patterns
    for pattern in SIMPLE_PATTERNS:
        if re.search(pattern, msg_lower):
            logger.info(f"Query classified: SIMPLE ({word_count} words)")
            return {
                "max_new_tokens": 50,
                "min_new_tokens": 5,
                "temperature": 0.25,
                "top_p": 0.80,
                "top_k": 30,
                "char_limit": 200,
                "repetition_penalty": 1.20,
                "no_repeat_ngram_size": 2,
                "early_stopping": False,
                "do_sample": True,
            }
    
    # Short messages
    if word_count <= 4:
        logger.info(f"Query classified: SIMPLE — short message ({word_count} words)")
        return {
            "max_new_tokens": 60,
            "min_new_tokens": 5,
            "temperature": 0.22,
            "top_p": 0.80,
            "top_k": 30,
            "char_limit": 250,
            "repetition_penalty": 1.20,
            "no_repeat_ngram_size": 2,
            "early_stopping": False,
            "do_sample": True,
        }
    
    # Elaborate queries
    for kw in ELABORATE_KEYWORDS:
        if kw in msg_lower:
            logger.info(f"Query classified: ELABORATE — keyword '{kw}' found")
            return {
                "max_new_tokens": 450,
                "min_new_tokens": 100,
                "temperature": 0.30,
                "top_p": 0.80,
                "top_k": 35,
                "char_limit": 1800,
                "repetition_penalty": 1.25,
                "no_repeat_ngram_size": 2,
                "early_stopping": False,
                "do_sample": True,
            }
    
    # Long messages
    if word_count > 12:
        logger.info(f"Query classified: ELABORATE — long message ({word_count} words)")
        return {
            "max_new_tokens": 450,
            "min_new_tokens": 80,
            "temperature": 0.28,
            "top_p": 0.80,
            "top_k": 35,
            "char_limit": 1800,
            "repetition_penalty": 1.25,
            "no_repeat_ngram_size": 2,
            "early_stopping": False,
            "do_sample": True,
        }
    
    # Moderate
    logger.info(f"Query classified: MODERATE ({word_count} words)")
    return {
        "max_new_tokens": 200,
        "min_new_tokens": 20,
        "temperature": 0.30,
        "top_p": 0.80,
        "top_k": 35,
        "char_limit": 900,
        "repetition_penalty": 1.20,
        "no_repeat_ngram_size": 2,
        "early_stopping": False,
        "do_sample": True,
    }

def clean_response_text(text: str) -> str:
    """Remove model artefacts and fix spacing."""
    if not text:
        return text
    
    prefixes_to_remove = [
        "### Response:", "### Response", "Response:", "RESPONSE:",
        "AI:", "Assistant:", "Bot:", "Answer:", "### Answer:", "### Answer",
        "\nResponse:", "\n### Response:", "IvieAI:", "🤖:", "<|response|>",
        "<|assistant|>", "Instruction:", "### Instruction:", "Human:", "User:",
    ]
    
    cleaned = text
    for prefix in prefixes_to_remove:
        if cleaned.lower().startswith(prefix.lower()):
            cleaned = cleaned[len(prefix):].strip()
        if f"\n{prefix}" in cleaned:
            cleaned = cleaned.replace(f"\n{prefix}", "")
    
    cleaned = re.sub(r'###\s*\w*:?', '', cleaned)
    cleaned = re.sub(r'<[^>]+>', '', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned)
    cleaned = re.sub(r'\s+([.!?,;:])', r'\1', cleaned)
    cleaned = re.sub(r'([.!?])\s*([a-zA-Z])', lambda m: m.group(1) + ' ' + m.group(2).upper(), cleaned)
    cleaned = re.sub(r'([,;:])\s*(\w)', r'\1 \2', cleaned)
    cleaned = cleaned.strip()
    return cleaned

def polish_response(text: str, char_limit: int) -> str:
    """Trim to sentence boundaries and ensure proper formatting."""
    if not text:
        return "I'm not sure how to respond to that. Could you rephrase?"
    
    if len(text) > char_limit:
        truncated = text[:char_limit]
        last_period = truncated.rfind('.')
        last_question = truncated.rfind('?')
        last_exclaim = truncated.rfind('!')
        last_end = max(last_period, last_question, last_exclaim)
        
        if last_end > 0:
            text = text[:last_end + 1]
        else:
            text = truncated.rstrip() + '...'
            for sep in ['. ', '? ', '! ', ', ', '; ']:
                if sep in truncated:
                    text = truncated.split(sep)[0] + '.'
                    break
    
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    
    polished = []
    for s in sentences:
        for prefix in ["### Response:", "Response:", "AI:", "Assistant:", "IvieAI:", "User:", "Human:"]:
            if s.lower().startswith(prefix.lower()):
                s = s[len(prefix):].strip()
        if s and s[0].islower():
            s = s[0].upper() + s[1:]
        if s and s[-1] not in '.!?':
            s += '.'
        s = re.sub(r'\.\.+', '.', s)
        if len(s) >= 5:
            polished.append(s)
    
    if not polished:
        return "I understand your question. Could you please rephrase it or provide more context?"
    
    result = ' '.join(polished)
    if result and result[-1] not in '.!?':
        result += '.'
    return result

def ensure_natural_ending(text: str) -> str:
    """Ensure the response ends with proper punctuation."""
    if not text:
        return text
    
    text = text.strip()
    if text[-1] in '.!?':
        return text
    
    sentence_endings = [m.end() for m in re.finditer(r'[.!?]\s+', text)]
    if sentence_endings:
        last_end = sentence_endings[-1]
        after_last = text[last_end:].strip()
        if after_last:
            text = text[:last_end].strip()
            if text and text[-1] not in '.!?':
                text += '.'
            return text
    
    if text:
        text = text.rstrip()
        if text[-1] not in '.!?':
            text += '.'
    return text

def remove_consecutive_word_repeats(text: str) -> str:
    """Remove consecutive repeated words."""
    if not text:
        return text
    
    words = text.split()
    if not words:
        return text
    
    deduped = [words[0]]
    for word in words[1:]:
        prev_core = re.sub(r'[.!?,;:]+$', '', deduped[-1]).lower()
        curr_core = re.sub(r'[.!?,;:]+$', '', word).lower()
        if curr_core != prev_core:
            deduped.append(word)
    
    result = ' '.join(deduped)
    result = re.sub(r'\s+([.!?,;:])', r'\1', result)
    return result.strip()

async def get_model_status(chat_manager):
    """Get model status."""
    if chat_manager.model is not None:
        return {"ready": True, "status": "ready", "stage": "ready", "progress": 100}
    if chat_manager.model_loading:
        return {"ready": False, "status": "loading", "stage": chat_manager.model_load_stage, "progress": chat_manager.model_load_progress}
    if chat_manager.model_load_error:
        return {"ready": False, "status": "error", "stage": "error", "detail": chat_manager.model_load_error, "progress": 0}
    return {"ready": False, "status": "not_started", "stage": "waiting", "progress": 0}