# app/routers/auth.py
import logging
from fastapi import APIRouter, Depends, HTTPException, status, Form
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.models import User
from app.core.security import get_password_hash, verify_password, create_access_token

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/register")
async def register_user(username: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    logger.info(f"Auth Router: Attempting to register user: {username}")
    user = db.query(User).filter(User.username == username).first()
    if user:
        logger.warning(f"Auth Router: Registration failed: Username '{username}' already exists.")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Username already registered")
    
    hashed_password = get_password_hash(password)
    new_user = User(username=username, hashed_password=hashed_password)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    logger.info(f"Auth Router: User '{username}' registered successfully with ID: {new_user.id}")
    return JSONResponse(content={"message": "User registered successfully"})

@router.post("/token")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    logger.info(f"Auth Router: Attempting to log in user: {form_data.username}")
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        logger.warning(f"Auth Router: Login failed for user '{form_data.username}': Incorrect credentials.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(data={"sub": user.username})
    logger.info(f"Auth Router: User '{form_data.username}' logged in successfully.")
    return {"access_token": access_token, "token_type": "bearer"}
