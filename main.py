import os
import logging
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Disable tokenizers parallelism warning (often seen with multiprocessing/Flask reloader)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import routers from the new app structure
from app.routers import auth, interview, history
from app.core import database # Import to ensure models are loaded and tables created

# --- FastAPI App Setup ---
app = FastAPI(
    title="AI Mock Interview System",
    description="An AI-powered web application for personalized mock interview practice.",
    version="1.0.0",
)

# Allow CORS for frontend interaction (important for local development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Adjust in production to specific frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
logger.info("CORS middleware configured.")

# Mount static files to serve frontend assets (e.g., /static/style.css, /static/images/logo.png)
app.mount("/static", StaticFiles(directory="static"), name="static_assets")
logger.info("Static files mounted at /static.")

# Include API routers
app.include_router(auth.router, prefix="/api/auth", tags=["Authentication"])
app.include_router(interview.router, prefix="/api/interview", tags=["Interview"])
app.include_router(history.router, prefix="/api/history", tags=["History"])
logger.info("API routers included.")

# --- Root Endpoint to Serve Frontend ---
@app.get("/")
async def serve_frontend():
    """Serves the main HTML frontend file."""
    frontend_path = os.path.join(os.getcwd(), "static", "index.html")
    logger.info(f"Attempting to serve frontend from: {frontend_path}")
    if not os.path.exists(frontend_path):
        logger.error(f"Frontend file not found at: {frontend_path}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Frontend file not found.")
    return FileResponse(frontend_path)

# --- Health Check Endpoint ---
@app.get("/health")
async def health_check():
    logger.info("Health check endpoint accessed.")
    return {"status": "ok", "message": "AI Mock Interview System is running."}

# Ensure database tables are created on startup
# This is handled by importing app.core.database at the top level
# which implicitly calls Base.metadata.create_all(bind=engine)
# However, for robustness, you might consider a separate startup event in a production app.
