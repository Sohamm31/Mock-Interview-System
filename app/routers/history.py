# app/routers/history.py
import logging
import json
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import get_current_user
from app.core.models import User, InterviewSession

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/")
async def get_history(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Retrieves interview history for the current user."""
    logger.info(f"History Router: User {current_user.username} requesting interview history.")
    history_records = db.query(InterviewSession).filter(InterviewSession.user_id == current_user.id).all()

    response_data = []
    for session in history_records:
        conversations = []
        # Order conversations by timestamp to ensure correct sequence
        for conv in sorted(session.conversations, key=lambda c: c.timestamp):
            conversations.append({"role": conv.role, "text": conv.text})

        feedback_data = None
        hr_feedback_data = None
        if session.feedback:
            feedback_data = {
                "rating": session.feedback.technical_rating,
                "tips": json.loads(session.feedback.technical_tips) if session.feedback.technical_tips else []
            }
            hr_feedback_data = {
                "rating": session.feedback.hr_rating,
                "tips": json.loads(session.feedback.hr_tips) if session.feedback.hr_tips else []
            }

        response_data.append({
            "interview_session_id": session.id, # Use DB ID for history
            "start_time": session.start_time.isoformat(),
            "end_time": session.end_time.isoformat() if session.end_time else None,
            "resume_snippet": session.resume_text_snippet,
            "github_summary": json.loads(session.github_knowledge_summary) if session.github_knowledge_summary else [],
            "conversation": conversations,
            "technical_feedback": feedback_data,
            "hr_feedback": hr_feedback_data
        })
    # Sort by start_time descending
    response_data.sort(key=lambda x: x['start_time'], reverse=True)
    logger.info(f"History Router: Retrieved {len(response_data)} history records for user {current_user.username}.")
    return JSONResponse(content=response_data)
