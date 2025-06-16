from fastapi import HTTPException, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import Column, String, Integer
from sqlalchemy.ext.declarative import declarative_base
from .database import get_db
from sqlalchemy import select

Base = declarative_base()

class ClientKey(Base):
    __tablename__ = "client_keys"
    
    id = Column(Integer, primary_key=True)
    client_id = Column(String, unique=True, index=True)
    api_key = Column(String)

async def store_api_key(db: AsyncSession, client_id: str, api_key: str):
    """Store a client's API key"""
    client_key = ClientKey(
        client_id=client_id,
        api_key=api_key
    )
    db.add(client_key)
    await db.commit()
    return client_key

async def get_api_key(db: AsyncSession, client_id: str) -> str:
    """Retrieve a client's API key"""
    result = await db.execute(
        select(ClientKey).filter(ClientKey.client_id == client_id)
    )
    client_key = result.scalar_one_or_none()
    return client_key.api_key if client_key else None

async def verify_client(db: AsyncSession = Depends(get_db), client_id: str = None) -> str:
    """Verify the client using their client ID"""
    if not client_id:
        raise HTTPException(status_code=401, detail="Missing client ID")
    
    # Get API key from database
    api_key = await get_api_key(db, client_id)
    if not api_key:
        raise HTTPException(status_code=403, detail="Invalid client ID")
    
    return client_id 