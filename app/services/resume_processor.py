# app/services/resume_processor.py
import os
import re
import shutil
import tempfile
import logging
from typing import List
import PyPDF2
import docx
import requests

from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
from langchain_community.document_loaders import GitLoader
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

from app.core import config # Import config for embeddings

logger = logging.getLogger(__name__)

def extract_text_from_pdf(file_path: str) -> str:
    """Extracts text from a PDF file."""
    text = ""
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page_num in range(len(reader.pages)):
                text += reader.pages[page_num].extract_text()
        logger.info(f"ResumeProcessor: Text extracted from PDF: {file_path}")
        return text
    except Exception as e:
        logger.error(f"ResumeProcessor: Error reading PDF {file_path}: {e}", exc_info=True)
        return ""

def extract_text_from_docx(file_path: str) -> str:
    """Extracts text from a DOCX file."""
    text = ""
    try:
        doc = docx.Document(file_path)
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        logger.info(f"ResumeProcessor: Text extracted from DOCX: {file_path}")
        return text
    except Exception as e:
        logger.error(f"ResumeProcessor: Error reading DOCX {file_path}: {e}", exc_info=True)
        return ""

def extract_github_links(text: str) -> List[str]:
    """Extracts GitHub repository URLs from text."""
    github_patterns = [
        r'https?://github\.com/([a-zA-Z0-9_-]+/[a-zA-Z0-9_.-]+)(?:/tree/[a-zA-Z0-9_.-]+)?(?:/blob/[a-zA-Z0-9_.-]+)?',
        r'git@github\.com:([a-zA-Z0-9_-]+/[a-zA-Z0-9_.-]+)\.git'
    ]
    links = []
    for pattern in github_patterns:
        found_repos = re.findall(pattern, text)
        for repo_path in found_repos:
            links.append(f"https://github.com/{repo_path.replace('.git', '')}")
    unique_links = list(set(links))
    logger.info(f"ResumeProcessor: Extracted GitHub links: {unique_links}")
    return unique_links

async def clone_and_embed_github_repo(repo_url: str, session_id: str, chroma_db_path: str) -> List[str]:
    """
    Clones a GitHub repository, processes its code files for embedding,
    and adds them to the specified ChromaDB.
    Returns a list of summaries of processed content.
    """
    temp_dir = tempfile.mkdtemp(prefix=f"github_clone_{session_id}_")
    logger.info(f"ResumeProcessor: Cloning {repo_url} into {temp_dir}")
    github_knowledge_summary = []
    try:
        loader = GitLoader(repo_url=repo_url, branch="main", clone_dir=temp_dir)
        all_docs = loader.load()
        logger.info(f"ResumeProcessor: Loaded {len(all_docs)} documents from {repo_url}")

        processed_docs = []
        
        python_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON, chunk_size=1000, chunk_overlap=200
        )
        js_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.JS, chunk_size=1000, chunk_overlap=200
        )
        markdown_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.MARKDOWN, chunk_size=1000, chunk_overlap=200
        )
        generic_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

        for doc in all_docs:
            file_path = doc.metadata.get('file_path', '')
            file_extension = os.path.splitext(file_path)[1].lower()
            
            if file_extension == '.py':
                chunks = python_splitter.split_documents([doc])
                logger.debug(f"ResumeProcessor: Split {file_path} as Python.")
            elif file_extension == '.js' or file_extension == '.ts':
                chunks = js_splitter.split_documents([doc])
                logger.debug(f"ResumeProcessor: Split {file_path} as JavaScript/TypeScript.")
            elif file_extension == '.md':
                chunks = markdown_splitter.split_documents([doc])
                logger.debug(f"ResumeProcessor: Split {file_path} as Markdown.")
            elif file_extension in ['.txt', '.json', '.xml', '.yml', '.yaml', '.csv']:
                chunks = generic_splitter.split_documents([doc])
                logger.debug(f"ResumeProcessor: Split {file_path} as generic text.")
            else:
                logger.debug(f"ResumeProcessor: Skipping file {file_path} (unhandled type: {file_extension}).")
                continue
            
            processed_docs.extend(chunks)
        
        if processed_docs:
            # Add GitHub code documents to the specified ChromaDB
            Chroma.from_documents(
                documents=processed_docs,
                embedding=config.embeddings, # Use embeddings from config
                persist_directory=chroma_db_path
            )
            logger.info(f"ResumeProcessor: Embedded {len(processed_docs)} chunks from {repo_url} into ChromaDB.")
            github_knowledge_summary.append(f"Processed code from {repo_url}")
        else:
            logger.info(f"ResumeProcessor: No processable code documents found from {repo_url}.")

        return github_knowledge_summary

    except Exception as e:
        logger.error(f"ResumeProcessor: Error cloning or processing GitHub repo {repo_url}: {e}", exc_info=True)
        return []
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            logger.info(f"ResumeProcessor: Cleaned up temporary directory: {temp_dir}")
