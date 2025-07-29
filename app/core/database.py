# app/core/database.py
import logging
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from app.core import config # Import config

logger = logging.getLogger(__name__)

# Database engine
engine = create_engine(config.DATABASE_URL)

# Base class for declarative models
Base = declarative_base()

# SessionLocal for creating database sessions
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Create database tables (called implicitly when models are imported and Base.metadata.create_all is run)
# For robustness, you might explicitly call this in main.py's startup event
try:
    # Import models here to ensure they are registered with Base before creating tables
    from app.core import models
    Base.metadata.create_all(bind=engine)
    logger.info("Database: Tables checked/created successfully.")
except Exception as e:
    logger.error(f"Database: Error creating database tables: {e}", exc_info=True)
    # Re-raise or handle as critical error for application startup
