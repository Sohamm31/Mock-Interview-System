# app/services/llm_agent.py
import logging
import json
from typing import List, Dict, Any, Tuple

from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import PydanticOutputParser

from app.core import config # Import config for llm and embeddings
from app.core.models import TechnicalFeedback, HRFeedback, InterviewQuestion, question_parser, question_format_instructions # Import Pydantic models and parser
from langchain_community.vectorstores import Chroma # For retrieving context

logger = logging.getLogger(__name__)

async def _get_relevant_context(chroma_db_path: str, query: str) -> str:
    """Retrieves relevant chunks from the ChromaDB vector store."""
    if not chroma_db_path:
        logger.warning("LLMAgent: ChromaDB path not provided for context retrieval.")
        return ""
    try:
        vector_store = Chroma(persist_directory=chroma_db_path, embedding_function=config.embeddings)
        retrieved_docs = vector_store.similarity_search(query, k=3)
        context = "\n---\n".join([doc.page_content for doc in retrieved_docs])
        logger.info(f"LLMAgent: Retrieved {len(retrieved_docs)} documents from ChromaDB for query: '{query[:50]}...'")
        return context
    except Exception as e:
        logger.error(f"LLMAgent: Error retrieving from ChromaDB at {chroma_db_path} for query '{query[:50]}...': {e}", exc_info=True)
        return ""

async def generate_question(session_data: Dict[str, Any], question_type: str, user_answer_prev: str = "") -> str:
    """Generates a new interview question based on type and context."""
    prompt_template_str = ""
    context_query = ""
    context_data = ""
    github_context = ""

    sections_covered = session_data['sections_covered']
    section_questions_asked = session_data['section_questions_asked']
    max_questions_per_section = session_data['max_questions_per_section']

    logger.info(f"LLMAgent: Generating question for type: {question_type}, asked: {section_questions_asked[question_type]}")

    if question_type == 'introduction' and not sections_covered['introduction']:
        if section_questions_asked['introduction'] == 0:
            prompt_template_str = "You are an AI interviewer. Begin the mock interview by asking the candidate to introduce themselves. Be polite and professional. Question:"
        else:
            prompt_template_str = "The candidate just introduced themselves: '{user_answer_prev}'. Based on this, ask a follow-up question to probe deeper into their background or motivations. Question:"
            context_data = user_answer_prev
        section_questions_asked['introduction'] += 1
        if section_questions_asked['introduction'] >= max_questions_per_section + 1:
            sections_covered['introduction'] = True

    elif question_type == 'skills' and not sections_covered['skills']:
        context_query = "candidate's skills, technical proficiencies, software, tools, languages"
        relevant_resume_skills = await _get_relevant_context(session_data['chroma_db_path'], context_query)
        if section_questions_asked['skills'] == 0:
            prompt_template_str = """You are an AI interviewer. Based on the candidate's resume, specifically their skills section, ask a question that assesses their proficiency or experience with a key skill.
            Resume Skills Context:
            {context}
            Question:"""
        else:
            prompt_template_str = """You are an AI interviewer. The candidate just answered a question about their skills. Based on their resume and previous answer, ask another question that probes deeper into a specific skill or how they applied it.
            Resume Skills Context:
            {context}
            Previous Answer: {user_answer_prev}
            Question:"""
        context_data = relevant_resume_skills
        section_questions_asked['skills'] += 1
        if section_questions_asked['skills'] >= max_questions_per_section:
            sections_covered['skills'] = True

    elif question_type == 'projects' and not sections_covered['projects']:
        context_query = "candidate's projects, personal projects, portfolio, github repositories, code examples"
        relevant_resume_projects = await _get_relevant_context(session_data['chroma_db_path'], context_query)
        github_knowledge_str = "\n".join(session_data['github_project_knowledge'])
        if section_questions_asked['projects'] == 0:
            prompt_template_str = """You are an AI interviewer. Based on the candidate's resume projects and any provided GitHub project knowledge (code snippets/READMEs), ask a question about a specific project, their role, or the technologies used.
            Resume Projects Context:
            {context}
            GitHub Project Knowledge:
            {github_knowledge}
            Question:"""
        else:
            prompt_template_str = """You are an AI interviewer. The candidate just answered a question about their projects. Based on their resume, GitHub project knowledge, and previous answer, ask a follow-up question about a specific project, challenges faced, or impact achieved.
            Resume Projects Context:
            {context}
            GitHub Project Knowledge:
            {github_knowledge}
            Previous Answer: {user_answer_prev}
            Question:"""
        context_data = relevant_resume_projects
        github_context = github_knowledge_str
        section_questions_asked['projects'] += 1
        if section_questions_asked['projects'] >= max_questions_per_section:
            sections_covered['projects'] = True

    elif question_type == 'experience' and not sections_covered['experience']:
        context_query = "candidate's work experience, professional roles, job history, responsibilities, achievements"
        relevant_resume_experience = await _get_relevant_context(session_data['chroma_db_path'], context_query)
        if section_questions_asked['experience'] == 0:
            prompt_template_str = """You are an AI interviewer. Based on the candidate's work experience in their resume, ask a question about a past role, responsibility, or achievement.
            Resume Experience Context:
            {context}
            Question:"""
        else:
            prompt_template_str = """You are an AI interviewer. The candidate just answered a question about their experience. Based on their resume and previous answer, ask a follow-up question to delve deeper into a specific experience, challenge, or learning.
            Resume Experience Context:
            {context}
            Previous Answer: {user_answer_prev}
            Question:"""
        context_data = relevant_resume_experience
        section_questions_asked['experience'] += 1
        if section_questions_asked['experience'] >= max_questions_per_section:
            sections_covered['experience'] = True
    else:
        logger.info("LLMAgent: All question sections covered. Signalling INTERVIEW_FINISHED.")
        return "INTERVIEW_FINISHED"

    if not prompt_template_str:
        logger.error("LLMAgent: No prompt template generated for current question type. Signalling INTERVIEW_FINISHED.")
        return "INTERVIEW_FINISHED"

    prompt_obj = PromptTemplate.from_template(prompt_template_str)
    
    formatted_inputs = {
        "context": context_data,
        "github_knowledge": github_context,
        "user_answer_prev": user_answer_prev,
        "introduction": user_answer_prev
    }
    
    final_inputs = {k: v for k, v in formatted_inputs.items() if "{" + k + "}" in prompt_obj.template}

    full_prompt_template_str = prompt_template_str + "\n\n{format_instructions}"
    prompt_obj_with_parser = PromptTemplate(
        template=full_prompt_template_str,
        input_variables=list(final_inputs.keys()) + ["format_instructions"],
        partial_variables={"format_instructions": question_format_instructions}
    )

    prompt_messages = prompt_obj_with_parser.format_prompt(**final_inputs).to_messages()

    ai_response_content = "" # Initialize here for scope in except block
    try:
        ai_response_obj = await config.llm.ainvoke(prompt_messages)
        ai_response_content = ai_response_obj.content
        
        parsed_question = question_parser.parse(ai_response_content)
        question_text = parsed_question.question
        
        logger.info(f"LLMAgent: AI generated question: {question_text[:100]}...")
        return question_text
    except Exception as e:
        logger.error(f"LLMAgent: Error invoking LLM for question generation or parsing: {e}", exc_info=True)
        if isinstance(e, json.JSONDecodeError):
            logger.error(f"LLMAgent: LLM did not return valid JSON for question. Raw content: {ai_response_content[:500]}...")
            return "I'm sorry, I encountered an error generating a structured question. Please try again."
        return "I'm sorry, I encountered an error generating the question. Let's move to the next section or end the interview."

async def get_feedback_agents(session_data: Dict[str, Any]) -> Tuple[TechnicalFeedback, HRFeedback]:
    """Generates technical and HR feedback based on the full conversation."""
    full_conversation_str = "\n".join([f"{role.upper()}: {text}" for role, text in session_data['conversation_history']])
    logger.info("LLMAgent: Generating feedback from technical and HR agents.")

    # Initialize response objects to None for safe access in except block
    tech_response_obj = None
    hr_response_obj = None

    # Technical Agent Prompt
    technical_parser = PydanticOutputParser(pydantic_object=TechnicalFeedback)
    technical_prompt = PromptTemplate(
        template="""You are a technical interviewer and expert. Analyze the following mock interview conversation.
        Based on the candidate's answers, especially those related to skills, projects, and experience,
        rate their technical knowledge on a scale of 1 to 5 (1=Poor, 5=Excellent).
        Then, provide 3-5 specific, actionable tips to help them improve their technical responses and knowledge.

        Conversation:
        {conversation}

        {format_instructions}
        """,
        input_variables=["conversation"],
        partial_variables={"format_instructions": technical_parser.get_format_instructions()},
    )
    technical_chain = technical_prompt | config.llm # Use config.llm

    # HR Agent Prompt
    hr_parser = PydanticOutputParser(pydantic_object=HRFeedback)
    hr_prompt = PromptTemplate(
        template="""You are an HR interviewer and communication expert. Analyze the following mock interview conversation.
        Based on the candidate's overall communication, clarity, confidence, and how they structured their answers (e.g., STAR method for behavioral questions),
        rate their communication skills on a scale of 1 to 5 (1=Poor, 5=Excellent).
        Then, provide 3-5 specific, actionable tips to help them improve their communication and soft skills.

        Conversation:
        {conversation}

        {format_instructions}
        """,
        input_variables=["conversation"],
        partial_variables={"format_instructions": hr_parser.get_format_instructions()},
    )
    hr_chain = hr_prompt | config.llm # Use config.llm

    try:
        # Invoke the chains and get the raw AIMessage objects
        tech_response_obj = await technical_chain.ainvoke({"conversation": full_conversation_str})
        hr_response_obj = await hr_chain.ainvoke({"conversation": full_conversation_str})

        # Explicitly parse the content using the respective parsers
        technical_feedback = technical_parser.parse(tech_response_obj.content)
        hr_feedback = hr_parser.parse(hr_response_obj.content)

        logger.info(f"LLMAgent: Technical feedback generated. Rating: {technical_feedback.technical_knowledge_rating}/5")
        logger.info(f"LLMAgent: HR feedback generated. Rating: {hr_feedback.communication_skills_rating}/5")
        return technical_feedback, hr_feedback
    except Exception as e:
        logger.error(f"LLMAgent: Error generating feedback: {e}", exc_info=True)
        # Log raw LLM outputs if parsing failed to help debug
        if tech_response_obj: # Check if defined and not None
            logger.error(f"LLMAgent: Raw LLM output for technical feedback (if available): {tech_response_obj.content[:500]}...")
        if hr_response_obj: # Check if defined and not None
            logger.error(f"LLMAgent: Raw LLM output for HR feedback (if available): {hr_response_obj.content[:500]}...")
        
        # Return default/empty feedback to prevent a full crash
        return TechnicalFeedback(technical_knowledge_rating=0, technical_tips=["Error generating feedback."]), \
               HRFeedback(communication_skills_rating=0, communication_tips=["Error generating feedback."])
