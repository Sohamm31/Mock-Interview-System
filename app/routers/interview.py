# app/routers/interview.py
import logging
import os
import shutil
import tempfile
import json
from datetime import datetime
from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException, status
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import get_current_user
from app.core.models import User, InterviewSession, InterviewConversation, InterviewFeedback
from app.services.resume_processor import extract_text_from_pdf, extract_text_from_docx, extract_github_links, clone_and_embed_github_repo
from app.services.llm_agent import generate_question, get_feedback_agents

# CORRECTED: Import active_sessions and _determine_next_question_type directly from session_manager
from app.services.session_manager import active_sessions, _determine_next_question_type 

from langchain.text_splitter import RecursiveCharacterTextSplitter # Required for text_splitter
from langchain_community.vectorstores import Chroma # Required for Chroma
from langchain_core.documents import Document # Required for Document

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/upload_resume")
async def upload_resume_endpoint(
    resume_file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Receives resume file, extracts text, finds GitHub links, clones repos,
    and creates ChromaDB vector store for the session.
    """
    session_id = str(current_user.id) + "_" + datetime.now().strftime("%Y%m%d%H%M%S%f")
    logger.info(f"Interview Router: User {current_user.username} (ID: {current_user.id}) uploading resume. Session ID: {session_id}")

    temp_dir = tempfile.mkdtemp(prefix=f"resume_upload_{session_id}_")
    file_path = os.path.join(temp_dir, resume_file.filename)
    logger.info(f"Interview Router: Saving uploaded file to temporary path: {file_path}")

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(resume_file.file, buffer)

        resume_text = ""
        file_extension = os.path.splitext(resume_file.filename)[1].lower()
        if file_extension == '.pdf':
            resume_text = extract_text_from_pdf(file_path)
        elif file_extension == '.docx':
            resume_text = extract_text_from_docx(file_path)
        elif file_extension == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                resume_text = f.read()
        else:
            logger.error(f"Interview Router: Unsupported file type uploaded: {file_extension}")
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unsupported file type. Please upload a PDF, DOCX, or TXT file.")

        if not resume_text:
            logger.error("Interview Router: Failed to extract text from uploaded resume.")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to extract text from resume.")
        logger.info(f"Interview Router: Resume text extracted. Length: {len(resume_text)} characters.")

        chroma_db_path = os.path.join(tempfile.gettempdir(), f"chroma_db_{session_id}")
        os.makedirs(chroma_db_path, exist_ok=True)
        logger.info(f"Interview Router: Created ChromaDB directory: {chroma_db_path}")

        # Embed resume text
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        resume_chunks = text_splitter.split_text(resume_text)
        resume_docs = [Document(page_content=chunk, metadata={"source": "resume"}) for chunk in resume_chunks]
        logger.info(f"Interview Router: Split resume into {len(resume_chunks)} chunks.")

        from app.core.config import embeddings # Import embeddings here to avoid circular dependency
        Chroma.from_documents(
            documents=resume_docs,
            embedding=embeddings,
            persist_directory=chroma_db_path
        )
        logger.info(f"Interview Router: Resume text embedded into ChromaDB at {chroma_db_path}")

        github_links = extract_github_links(resume_text)
        github_knowledge_summary = []
        for link in github_links:
            logger.info(f"Interview Router: Processing GitHub link: {link}")
            # clone_and_embed_github_repo returns a list of summaries of processed content
            repo_summaries = await clone_and_embed_github_repo(link, session_id, chroma_db_path)
            github_knowledge_summary.extend(repo_summaries) # Extend the list

        # Create a new interview session record in DB
        new_interview_session = InterviewSession(
            user_id=current_user.id,
            resume_text_snippet=resume_text[:500] + "..." if len(resume_text) > 500 else resume_text,
            github_knowledge_summary=json.dumps(github_knowledge_summary) if github_knowledge_summary else None,
            start_time=datetime.now()
        )
        db.add(new_interview_session)
        db.commit()
        db.refresh(new_interview_session)
        logger.info(f"Interview Router: New interview session DB record created with ID: {new_interview_session.id}")

        # Initialize active session data
        active_sessions[session_id] = {
            "db_session_id": new_interview_session.id, # Link to DB record
            "resume_text": resume_text,
            "github_project_knowledge": github_knowledge_summary, # Use the summaries
            "chroma_db_path": chroma_db_path,
            "conversation_history": [],
            "current_question_count": 0,
            "sections_covered": {
                'introduction': False,
                'skills': False,
                'projects': False,
                'experience': False
            },
            "section_questions_asked": {
                'introduction': 0,
                'skills': 0,
                'projects': 0,
                'experience': 0
            },
            "max_questions_per_section": 2
        }
        logger.info(f"Interview Router: Active session data initialized for session ID: {session_id}")

        return JSONResponse(content={"message": "Resume processed and embeddings created.", "interview_session_id": session_id})

    except HTTPException: # Re-raise FastAPI HTTPExceptions directly
        raise
    except Exception as e:
        logger.exception(f"Interview Router: Critical error during resume upload for user {current_user.username}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to process resume: {str(e)}")
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            logger.info(f"Interview Router: Cleaned up temporary resume directory: {temp_dir}")


@router.post("/start_interview")
async def start_interview_endpoint(
    interview_session_id: str = Form(...),
    current_user: User = Depends(get_current_user) # Ensure user is authenticated
):
    """Starts the mock interview for a given session."""
    logger.info(f"Interview Router: User {current_user.username} starting interview for session ID: {interview_session_id}")
    session_data = active_sessions.get(interview_session_id)
    if not session_data:
        logger.error(f"Interview Router: Attempted to start interview for non-existent active session: {interview_session_id}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Active interview session not found. Please upload resume first.")

    # Ensure this session belongs to the current user (basic check)
    if not interview_session_id.startswith(str(current_user.id) + "_"):
        logger.error(f"Interview Router: Unauthorized access attempt to session {interview_session_id} by user {current_user.username}")
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Unauthorized access to session.")


    # Reset interview state for a fresh start if already exists in active_sessions
    session_data["conversation_history"] = []
    session_data["current_question_count"] = 0
    session_data["sections_covered"] = {
        'introduction': False,
        'skills': False,
        'projects': False,
        'experience': False
    }
    session_data["section_questions_asked"] = {
        'introduction': 0,
        'skills': 0,
        'projects': 0,
        'experience': 0
    }
    logger.info(f"Interview Router: Interview state reset for session ID: {interview_session_id}")

    try:
        initial_question = await generate_question(session_data, 'introduction')
        logger.info(f"Interview Router: Initial question generated for session {interview_session_id}.")
        return JSONResponse(content={"question": initial_question})
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Interview Router: Error starting interview for session {interview_session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to start interview: {str(e)}")

@router.post("/submit_answer")
async def submit_answer_endpoint(
    interview_session_id: str = Form(...),
    user_answer: str = Form(...),
    current_user: User = Depends(get_current_user), # Ensure user is authenticated
    db: Session = Depends(get_db)
):
    """Submits user's answer and generates the next question."""
    logger.info(f"Interview Router: User {current_user.username} submitting answer for session {interview_session_id}.")
    session_data = active_sessions.get(interview_session_id)
    if not session_data:
        logger.error(f"Interview Router: Attempted to submit answer for non-existent active session: {interview_session_id}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Active interview session not found. Please upload resume first.")

    if not user_answer.strip():
        logger.warning(f"Interview Router: Empty answer submitted for session {interview_session_id}.")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="User answer cannot be empty.")

    # Save user answer to DB
    new_conversation_entry = InterviewConversation(
        interview_session_id=session_data["db_session_id"],
        role='user',
        text=user_answer,
        timestamp=datetime.now()
    )
    db.add(new_conversation_entry)
    db.commit() # Commit immediately to ensure it's saved
    logger.info(f"Interview Router: User answer saved to DB for session {interview_session_id}.")

    session_data['conversation_history'].append(('user', user_answer))

    next_question_type = _determine_next_question_type(session_data)
    if next_question_type == 'feedback_stage':
        logger.info(f"Interview Router: Interview questions completed for session {interview_session_id}. Moving to feedback stage.")
        return JSONResponse(content={"status": "interview_finished"})

    try:
        next_ai_question = await generate_question(session_data, next_question_type, user_answer)

        # Save AI question to DB
        new_conversation_entry = InterviewConversation(
            interview_session_id=session_data["db_session_id"],
            role='ai',
            text=next_ai_question,
            timestamp=datetime.now()
        )
        db.add(new_conversation_entry)
        db.commit() # Commit immediately
        logger.info(f"Interview Router: AI question saved to DB for session {interview_session_id}.")

        if next_ai_question == "INTERVIEW_FINISHED":
             logger.info(f"Interview Router: AI explicitly signalled INTERVIEW_FINISHED for session {interview_session_id}.")
             return JSONResponse(content={"status": "interview_finished"})
        return JSONResponse(content={"question": next_ai_question})
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Interview Router: Error submitting answer or generating next question for session {interview_session_id}: {e}", exc_info=True)
        db.rollback() # Rollback if error
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to generate next question: {str(e)}")

@router.post("/get_feedback")
async def get_interview_feedback_endpoint(
    interview_session_id: str = Form(...),
    current_user: User = Depends(get_current_user), # Ensure user is authenticated
    db: Session = Depends(get_db)
):
    """Generates and returns technical and HR feedback."""
    logger.info(f"Interview Router: User {current_user.username} requesting feedback for session {interview_session_id}.")
    session_data = active_sessions.get(interview_session_id)
    if not session_data:
        logger.error(f"Interview Router: Attempted to get feedback for non-existent active session: {interview_session_id}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Active interview session not found. Please upload resume first.")

    full_conversation_str = "\n".join([f"{role.upper()}: {text}" for role, text in session_data['conversation_history']])
    logger.info("Interview Router: Generating feedback from technical and HR agents.")

    try:
        technical_feedback, hr_feedback = await get_feedback_agents(session_data)

        # Save feedback to DB
        new_feedback_entry = InterviewFeedback(
            interview_session_id=session_data["db_session_id"],
            technical_rating=technical_feedback.technical_knowledge_rating,
            technical_tips=json.dumps(technical_feedback.technical_tips), # Store as JSON string
            hr_rating=hr_feedback.communication_skills_rating,
            hr_tips=json.dumps(hr_feedback.communication_tips) # Store as JSON string
        )
        db.add(new_feedback_entry)

        # Update interview session end time
        interview_session_db = db.query(InterviewSession).filter(InterviewSession.id == session_data["db_session_id"]).first()
        if interview_session_db:
            interview_session_db.end_time = datetime.now()
        db.commit()
        logger.info(f"Interview Router: Feedback and session end time saved to DB for session {interview_session_id}.")

        # Clean up ChromaDB directory after feedback is generated
        if os.path.exists(session_data['chroma_db_path']):
            shutil.rmtree(session_data['chroma_db_path'])
            logger.info(f"Interview Router: Cleaned up ChromaDB directory: {session_data['chroma_db_path']}")
        del active_sessions[interview_session_id] # Remove session from memory
        logger.info(f"Interview Router: Session {interview_session_id} removed from active sessions.")

        return JSONResponse(content={
            "technical_feedback": technical_feedback.dict(),
            "hr_feedback": hr_feedback.dict()
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Interview Router: Error getting feedback for session {interview_session_id}: {e}", exc_info=True)
        # Log raw LLM outputs if parsing failed to help debug
        # tech_response_obj and hr_response_obj might not be defined if error occurs early
        # in get_feedback_agents, so we pass session_data to get_feedback_agents
        # and let it handle logging raw outputs.
        db.rollback() # Rollback if error
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to generate feedback: {str(e)}")
