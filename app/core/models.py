# app/core/models.py
import logging
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from pydantic import BaseModel as LangchainBaseModel, Field as LangchainField
from langchain_core.output_parsers import PydanticOutputParser
from typing import List

from app.core.database import Base # Import Base from database.py

logger = logging.getLogger(__name__)

# --- SQLAlchemy ORM Models ---
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=func.now())

    sessions = relationship("InterviewSession", back_populates="user")
    logger.debug("ORM Model: User defined.")

class InterviewSession(Base):
    __tablename__ = "interview_sessions"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    start_time = Column(DateTime, default=func.now(), nullable=False)
    end_time = Column(DateTime, nullable=True)
    resume_text_snippet = Column(Text, nullable=True) # Store first N chars of resume
    github_knowledge_summary = Column(Text, nullable=True) # Store summary of processed github repos

    user = relationship("User", back_populates="sessions")
    conversations = relationship("InterviewConversation", back_populates="session", cascade="all, delete-orphan")
    feedback = relationship("InterviewFeedback", back_populates="session", uselist=False, cascade="all, delete-orphan")
    logger.debug("ORM Model: InterviewSession defined.")

class InterviewConversation(Base):
    __tablename__ = "interview_conversations"
    id = Column(Integer, primary_key=True, index=True)
    interview_session_id = Column(Integer, ForeignKey("interview_sessions.id"), nullable=False)
    role = Column(String(50), nullable=False) # 'ai' or 'user'
    text = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=func.now(), nullable=False)

    session = relationship("InterviewSession", back_populates="conversations")
    logger.debug("ORM Model: InterviewConversation defined.")

class InterviewFeedback(Base):
    __tablename__ = "interview_feedback"
    id = Column(Integer, primary_key=True, index=True)
    interview_session_id = Column(Integer, ForeignKey("interview_sessions.id"), unique=True, nullable=False)
    technical_rating = Column(Integer, nullable=True)
    technical_tips = Column(Text, nullable=True) # Stored as JSON string
    hr_rating = Column(Integer, nullable=True)
    hr_tips = Column(Text, nullable=True) # Stored as JSON string

    session = relationship("InterviewSession", back_populates="feedback")
    logger.debug("ORM Model: InterviewFeedback defined.")


# --- Pydantic Models for Structured Output (LangChain) ---
class TechnicalFeedback(LangchainBaseModel):
    """Feedback from the technical agent."""
    technical_knowledge_rating: int = LangchainField(description="Rating of technical knowledge out of 5.")
    technical_tips: List[str] = LangchainField(description="List of actionable tips to improve technical answers.")
    logger.debug("Pydantic Model: TechnicalFeedback defined.")

class HRFeedback(LangchainBaseModel):
    """Feedback from the HR agent."""
    communication_skills_rating: int = LangchainField(description="Rating of communication skills out of 5.")
    communication_tips: List[str] = LangchainField(description="List of actionable tips to improve communication and soft skills.")
    logger.debug("Pydantic Model: HRFeedback defined.")

class InterviewQuestion(LangchainBaseModel):
    """A single, direct interview question for the candidate."""
    question: str = LangchainField(description="A single, direct interview question for the candidate.")
    logger.debug("Pydantic Model: InterviewQuestion defined.")

# Output parser for the interview question
question_parser = PydanticOutputParser(pydantic_object=InterviewQuestion)
question_format_instructions = question_parser.get_format_instructions()
logger.info("Pydantic Output Parser for InterviewQuestion initialized.")
