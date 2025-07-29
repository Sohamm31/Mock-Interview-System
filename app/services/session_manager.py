# app/services/session_manager.py
import logging
from typing import Dict, Any

# Import active_sessions from the dedicated global_state module
from app.core.global_state import active_sessions

logger = logging.getLogger(__name__)

# The active_sessions dictionary is now managed by app.core.global_state
# active_sessions: Dict[str, Dict[str, Any]] = {} # REMOVED: Defined in global_state.py
# logger.info("SessionManager: Active sessions dictionary initialized.") # REMOVED: Logged in global_state.py

def _determine_next_question_type(session_data: Dict[str, Any]) -> str:
    """Determines the next section for questions based on coverage."""
    sections_covered = session_data['sections_covered']
    section_questions_asked = session_data['section_questions_asked']
    max_questions_per_section = session_data['max_questions_per_section']

    if not sections_covered['introduction']:
        # Initial introduction question + max_questions_per_section follow-ups
        if section_questions_asked['introduction'] <= max_questions_per_section:
            return 'introduction'
        else:
            sections_covered['introduction'] = True # Mark as covered if max questions asked

    if not sections_covered['skills']:
        if section_questions_asked['skills'] < max_questions_per_section:
            return 'skills'
        else:
            sections_covered['skills'] = True

    if not sections_covered['projects']:
        if section_questions_asked['projects'] < max_questions_per_section:
            return 'projects'
        else:
            sections_covered['projects'] = True

    if not sections_covered['experience']:
        if section_questions_asked['experience'] < max_questions_per_section:
            return 'experience'
        else:
            sections_covered['experience'] = True
            
    logger.info("SessionManager: Determined next stage: feedback_stage (all question sections covered).")
    return 'feedback_stage'

