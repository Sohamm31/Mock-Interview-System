# app/core/security.py
import logging
from datetime import datetime, timedelta
from typing import Optional
from fastapi import HTTPException, status, Depends
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from app.core import config # Import config
from app.core.database import get_db # Import get_db
from app.core.models import User # Import User ORM model

logger = logging.getLogger(__name__)

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme for authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/auth/token") # Updated tokenUrl to match new router prefix

# --- Password Utilities ---
def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies a plain password against a hashed password."""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Hashes a plain password."""
    return pwd_context.hash(password)

# --- JWT Utilities ---
def create_access_token(data: dict) -> str:
    """Creates a JWT access token."""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=config.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, config.SECRET_KEY, algorithm=config.ALGORITHM)
    logger.info(f"Security: Access token created for sub: {data.get('sub')}")
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    """Dependency to get the current authenticated user from a JWT."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, config.SECRET_KEY, algorithms=[config.ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            logger.warning("Security: JWT payload missing 'sub' (username).")
            raise credentials_exception
    except JWTError as e:
        logger.error(f"Security: JWT decoding error: {e}", exc_info=True)
        raise credentials_exception
    user = db.query(User).filter(User.username == username).first()
    if user is None:
        logger.warning(f"Security: User '{username}' not found in DB during token validation.")
        raise credentials_exception
    logger.info(f"Security: User '{username}' authenticated successfully.")
    return user
