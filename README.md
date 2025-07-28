AI Mock Interview System
Project Overview
The AI Mock Interview System is a comprehensive web application designed to help users prepare for job interviews. It leverages Artificial Intelligence to simulate realistic interview scenarios, provide personalized feedback based on the user's resume and conversation, and track interview history.

The system features user authentication, a voice-only interview experience, dynamic question generation, and detailed feedback from specialized AI agents.

Features
User Authentication: Secure registration and login system.

Resume-Powered Interviews: Upload your resume (PDF, DOCX, TXT) to tailor interview questions specifically to your skills, projects, and experience.

GitHub Domain Knowledge Integration: If your resume contains public GitHub repository links, the system will attempt to clone and analyze the code/READMEs from those repositories to generate more in-depth, project-specific technical questions.

Voice-Only Interview: Experience a natural, hands-free interview. Speak your answers, and the AI will respond verbally.

Live Transcription: See a live transcription of your speech during the interview.

Pause Tolerance: The speech recognition is configured to allow for natural pauses in your speech (e.g., 2 seconds of silence) before considering your answer complete.

Fullscreen Mode: The interview enters fullscreen mode to minimize distractions and prevent cheating. Attempts to exit fullscreen during the interview will prompt a warning.

Dynamic Question Generation: AI dynamically generates questions based on different sections of your resume (introduction, skills, projects, experience).

AI Feedback Agents: After the interview, receive detailed, personalized feedback from two AI agents:

Technical Agent: Rates your technical knowledge (out of 5) and provides tips for improvement.

HR Agent: Rates your communication skills (out of 5) and offers advice on improving your delivery and behavioral responses.

Interview History: All your past interview sessions, including conversations and feedback, are securely stored in a MySQL database and accessible via the history dashboard.

Clean and Responsive UI: A modern, centered, and responsive user interface built with Tailwind CSS.

Technologies Used
Backend (FastAPI - Python)
FastAPI: High-performance web framework.

Uvicorn: ASGI server for running FastAPI.

SQLAlchemy: Python SQL Toolkit and Object Relational Mapper for database interactions.

PyMySQL: MySQL database connector for Python.

Passlib: For secure password hashing (bcrypt).

Python-Jose: For JWT (JSON Web Token) authentication.

LangChain: Framework for developing applications powered by language models.

langchain-openai: For connecting to OpenRouter's API (DeepSeek model).

langchain-community: For HuggingFaceEmbeddings, Chroma vector store, GitLoader, PydanticOutputParser.

langchain-core: For PromptTemplate, AIMessage, HumanMessage.

ChromaDB: Lightweight, in-memory (for session) vector database for storing resume and code embeddings.

Sentence-Transformers: For generating embeddings (all-MiniLM-L6-v2).

GitPython: Python library to interact with Git repositories (used by GitLoader).

PyPDF2, python-docx: For parsing text from PDF and DOCX resume files.

python-dotenv: For managing environment variables.

Logging: Built-in Python logging for debugging and monitoring.

Frontend (HTML, CSS, JavaScript)
HTML5: Structure of the web pages.

Tailwind CSS: Utility-first CSS framework for rapid UI development and responsive design.

JavaScript: Client-side logic for UI interaction, API calls, and Web Speech API integration.

Web Speech API: For Speech-to-Text (STT) and Text-to-Speech (TTS) functionalities.

Fetch API: For asynchronous communication with the FastAPI backend.

localStorage: For storing the JWT access token.

Architecture
The application follows a client-server architecture:

Frontend: A single-page application (index.html) served directly by the FastAPI backend. It handles all user interactions, microphone input, and AI voice output.

Backend: A FastAPI server that exposes RESTful APIs. It manages:

User authentication and session management.

Resume parsing and GitHub repository cloning/analysis.

Creation and management of a temporary ChromaDB vector store for each interview session.

Orchestration of LLM calls (via OpenRouter) for dynamic question generation and structured feedback.

Persistence of user data, interview conversations, and feedback in a MySQL database.

