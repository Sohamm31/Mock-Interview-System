import os
import uuid
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# --- LLM and LangChain Imports ---
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

# --- 1. Initial Setup and Configuration ---
load_dotenv()
api_key = os.getenv('OPENROUTESERVICE_API_KEY')

# Vercel expects the FastAPI object to be named 'app'
app = FastAPI()

# In-memory storage for interview sessions
sessions = {}
app.mount("/static", StaticFiles(directory="../static"), name="static")
# --- 2. LLM and Chain Definitions ---

llm = ChatOpenAI(
    model_name="deepseek/deepseek-r1-0528-qwen3-8b:free",
    openai_api_key=api_key,
    openai_api_base="https://openrouter.ai/api/v1",
)

# --- Dynamic Question Generation Chain ---
dynamic_question_prompt = PromptTemplate(
    template="""You are an expert interviewer. Your goal is to conduct a natural, flowing interview.
    Based on the candidate's resume and the conversation so far, ask the NEXT most relevant interview question.
    Avoid repeating questions. Make the conversation feel adaptive. The questions should be insightful and probing.

    CANDIDATE'S RESUME:
    {resume}

    CONVERSATION HISTORY (Interviewer and Candidate):
    {history}

    Generate only ONE new question based on the context.""",
    input_variables=["resume", "history"]
)
dynamic_question_chain = dynamic_question_prompt | llm | StrOutputParser()

# --- HR Feedback Agent Chain ---
hr_feedback_schema = [
    ResponseSchema(name="Communication Skills", description="Evaluate clarity, articulation, and listening. Rate out of 5 and justify."),
    ResponseSchema(name="Confidence and Professionalism", description="Assess confidence, tone, and professional demeanor. Rate out of 5 and justify."),
    ResponseSchema(name="Behavioral Competencies", description="Comment on teamwork, problem-solving, and attitude based on answers.")
]
hr_parser = StructuredOutputParser.from_response_schemas(hr_feedback_schema)
hr_prompt = PromptTemplate(
    template="""You are an HR Manager. Analyze the following interview transcript from a behavioral and communication perspective.
    Do NOT judge technical skills. Focus on soft skills and culture fit.

    INTERVIEW TRANSCRIPT:
    {interview_summary}

    {format_instruction}""",
    input_variables=["interview_summary"],
    partial_variables={"format_instruction": hr_parser.get_format_instructions()}
)
hr_feedback_chain = hr_prompt | llm | hr_parser

# --- Technical Feedback Agent Chain ---
tech_feedback_schema = [
    ResponseSchema(name="Technical Knowledge", description="Evaluate understanding of concepts from their resume and answers. Rate out of 5 and justify."),
    ResponseSchema(name="Project Understanding", description="Assess how well they explained their projects, role, and technologies. Rate out of 5 and justify."),
    ResponseSchema(name="Problem-Solving Approach", description="Comment on their approach to technical questions and articulating solutions.")
]
tech_parser = StructuredOutputParser.from_response_schemas(tech_feedback_schema)
tech_prompt = PromptTemplate(
    template="""You are a Senior Technical Lead. Analyze the following interview transcript from a purely technical standpoint.
    Do NOT judge soft skills. Focus on technical accuracy, depth, and problem-solving skills.

    INTERVIEW TRANSCRIPT:
    {interview_summary}

    {format_instruction}""",
    input_variables=["interview_summary"],
    partial_variables={"format_instruction": tech_parser.get_format_instructions()}
)
tech_feedback_chain = tech_prompt | llm | tech_parser

# --- 3. Pydantic Models ---
class StartRequest(BaseModel):
    resume_text: str

class AnswerRequest(BaseModel):
    session_id: str
    answer: str

# --- 4. API Endpoints ---

@app.post("/api//start_interview")
async def start_interview(request: StartRequest):
    """
    Creates a session with the provided resume text and returns the first question.
    """
    session_id = str(uuid.uuid4())
    intro_question = "To start, could you please tell me a little bit about yourself?"
    
    sessions[session_id] = {
        "resume_content": request.resume_text,
        "history": [f"Interviewer: {intro_question}"],
        "question_count": 0,
        "max_questions": 5
    }
    
    return {"session_id": session_id, "question": intro_question}

@app.post("/api/submit_answer")
async def submit_answer(request: AnswerRequest):
    """Submits an answer and gets the next question or final feedback."""
    session_id = request.session_id
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Invalid session ID")

    session = sessions[session_id]
    session["history"].append(f"Candidate: {request.answer}")
    session["question_count"] += 1

    if session["question_count"] >= session["max_questions"]:
        final_summary = "\n".join(session["history"])
        hr_feedback = hr_feedback_chain.invoke({"interview_summary": final_summary})
        tech_feedback = tech_feedback_chain.invoke({"interview_summary": final_summary})
        if session_id in sessions:
            del sessions[session_id]
        
        return { "interview_over": True, "feedback": { "hr_feedback": hr_feedback, "tech_feedback": tech_feedback } }

    history_str = "\n".join(session["history"])
    next_question = dynamic_question_chain.invoke({
        "resume": session["resume_content"],
        "history": history_str
    })
    
    session["history"].append(f"Interviewer: {next_question}")
    return {"interview_over": False, "question": next_question}
@app.get("/")
async def serve_frontend():
    return FileResponse("../static/index.html")