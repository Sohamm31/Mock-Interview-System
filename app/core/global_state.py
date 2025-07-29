# app/core/global_state.py
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Global dictionary for in-memory active interview session data
# This is the ONLY place this global state is defined.
active_sessions: Dict[str, Dict[str, Any]] = {}
logger.info("GlobalState: active_sessions dictionary initialized.")
