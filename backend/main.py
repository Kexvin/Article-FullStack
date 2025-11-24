# main.py - Complete FastAPI Backend
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime
from typing import List, Optional
import json
import re
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# Database Configuration
DATABASE_URL = "postgresql://postgres:Jumpman23@localhost:5432/sentiment"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Database Model
class ArticleDB(Base):
    __tablename__ = "articles"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=True)
    text = Column(Text, nullable=False)
    overall_sentiment = Column(String, nullable=False)
    sentiment_score = Column(Float, nullable=False)
    positive_count = Column(Integer, default=0)
    neutral_count = Column(Integer, default=0)
    negative_count = Column(Integer, default=0)
    paragraph_details = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

# Create tables
Base.metadata.create_all(bind=engine)

# FastAPI App
app = FastAPI(title="Article Sentiment Analysis API")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic Models
class ArticleInput(BaseModel):
    title: Optional[str] = None
    text: str

class ArticleResponse(BaseModel):
    id: int
    title: Optional[str]
    text: str
    overall_sentiment: str
    sentiment_score: float
    positive_count: int
    neutral_count: int
    negative_count: int
    paragraph_details: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True

class ArticleSummary(BaseModel):
    id: int
    title: Optional[str]
    overall_sentiment: str
    sentiment_score: float
    created_at: datetime

    class Config:
        from_attributes = True

# Database Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Simplified sentiment analysis (you'll integrate your pipeline here)
def analyze_article_sentiment(text: str) -> dict:
    """
    Run your actual sentiment pipeline.
    """
    try:
        # Import your pipeline
        from sentiment_pipeline import article_pipeline
        
        # Run your pipeline
        result = article_pipeline.run({"text": text})
        
        # Parse the output from your pipeline
        overall_exp = result.get("overall_explanation", "")
        
        # Extract sentiment label
        if "POSITIVE" in overall_exp:
            sentiment_label = "POSITIVE"
        elif "NEGATIVE" in overall_exp:
            sentiment_label = "NEGATIVE"
        else:
            sentiment_label = "NEUTRAL"
        
        # Extract signed score using regex
        import re
        score_match = re.search(r'signed score=([+-]?\d+\.\d+)', overall_exp)
        sentiment_score = float(score_match.group(1)) if score_match else 0.0
        
        # Extract counts
        pos_match = re.search(r'Positive: (\d+)', overall_exp)
        neu_match = re.search(r'Neutral: (\d+)', overall_exp)
        neg_match = re.search(r'Negative: (\d+)', overall_exp)
        
        positive_count = int(pos_match.group(1)) if pos_match else 0
        neutral_count = int(neu_match.group(1)) if neu_match else 0
        negative_count = int(neg_match.group(1)) if neg_match else 0
        
        return {
            "sentiment_label": sentiment_label,
            "sentiment_score": sentiment_score,
            "positive_count": positive_count,
            "neutral_count": neutral_count,
            "negative_count": negative_count,
            "paragraph_details": json.dumps(result.get("paragraph_explanations", []))
        }
    except Exception as e:
        # Fallback to simple analysis if pipeline fails
        logger.error(f"Pipeline error: {e}")
        # Use the simple keyword-based analysis as fallback
        text_lower = text.lower()
        positive_words = ['growth', 'profit', 'increase', 'success', 'gain', 'surge', 'rise']
        negative_words = ['loss', 'decline', 'fall', 'decrease', 'drop', 'crisis', 'risk']
        
        pos_count = sum(word in text_lower for word in positive_words)
        neg_count = sum(word in text_lower for word in negative_words)
        
        if pos_count > neg_count:
            sentiment_label = "POSITIVE"
            sentiment_score = 0.3 + (pos_count * 0.1)
        elif neg_count > pos_count:
            sentiment_label = "NEGATIVE"
            sentiment_score = -0.3 - (neg_count * 0.1)
        else:
            sentiment_label = "NEUTRAL"
            sentiment_score = 0.0
        
        sentiment_score = max(-1.0, min(1.0, sentiment_score))
        
        return {
            "sentiment_label": sentiment_label,
            "sentiment_score": sentiment_score,
            "positive_count": pos_count,
            "neutral_count": 1,
            "negative_count": neg_count,
            "paragraph_details": json.dumps(["Fallback analysis used"])
        }

# API Endpoints
@app.get("/")
async def root():
    return {
        "message": "Article Sentiment Analysis API",
        "version": "1.0",
        "endpoints": {
            "POST /articles": "Create and analyze article",
            "GET /articles": "Get all articles",
            "GET /articles/{id}": "Get specific article",
            "DELETE /articles/{id}": "Delete article",
            "GET /stats": "Get statistics"
        }
    }

@app.post("/articles", response_model=ArticleResponse, status_code=201)
async def create_article(article: ArticleInput, db: Session = Depends(get_db)):
    """Analyze and store a new article."""
    sentiment_data = analyze_article_sentiment(article.text)
    
    db_article = ArticleDB(
        title=article.title,
        text=article.text,
        overall_sentiment=sentiment_data["sentiment_label"],
        sentiment_score=sentiment_data["sentiment_score"],
        positive_count=sentiment_data["positive_count"],
        neutral_count=sentiment_data["neutral_count"],
        negative_count=sentiment_data["negative_count"],
        paragraph_details=sentiment_data["paragraph_details"]
    )
    
    db.add(db_article)
    db.commit()
    db.refresh(db_article)
    
    return db_article

@app.get("/articles", response_model=List[ArticleSummary])
async def get_all_articles(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """Retrieve all articles."""
    articles = db.query(ArticleDB).order_by(ArticleDB.created_at.desc()).offset(skip).limit(limit).all()
    return articles

@app.get("/articles/{article_id}", response_model=ArticleResponse)
async def get_article(article_id: int, db: Session = Depends(get_db)):
    """Retrieve specific article."""
    article = db.query(ArticleDB).filter(ArticleDB.id == article_id).first()
    if not article:
        raise HTTPException(status_code=404, detail="Article not found")
    return article

@app.delete("/articles/{article_id}")
async def delete_article(article_id: int, db: Session = Depends(get_db)):
    """Delete an article."""
    article = db.query(ArticleDB).filter(ArticleDB.id == article_id).first()
    if not article:
        raise HTTPException(status_code=404, detail="Article not found")
    
    db.delete(article)
    db.commit()
    
    return {"message": f"Article {article_id} deleted successfully"}

@app.get("/stats")
async def get_statistics(db: Session = Depends(get_db)):
    """Get overall statistics."""
    total_articles = db.query(ArticleDB).count()
    positive_articles = db.query(ArticleDB).filter(ArticleDB.overall_sentiment == "POSITIVE").count()
    negative_articles = db.query(ArticleDB).filter(ArticleDB.overall_sentiment == "NEGATIVE").count()
    neutral_articles = db.query(ArticleDB).filter(ArticleDB.overall_sentiment == "NEUTRAL").count()
    
    return {
        "total_articles": total_articles,
        "positive": positive_articles,
        "negative": negative_articles,
        "neutral": neutral_articles
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)