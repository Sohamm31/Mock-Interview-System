# app/__init__.py
# This file makes the 'app' directory a Python package.

# Import specific modules or variables that should be directly accessible
# from the 'app' package, or that need to be initialized early.
# For example, to ensure SQLAlchemy models are registered, you might import them.
# from .core import models, database, config

# Removed: Direct import of active_sessions and _determine_next_question_type
# from .services.session_manager import active_sessions, _determine_next_question_type
# These should be imported directly in the modules that use them (e.g., app/routers/interview.py)
# to avoid potential circular dependencies or state issues.
