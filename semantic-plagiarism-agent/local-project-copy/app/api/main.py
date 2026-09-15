import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from app.pipeline.orchestrator import PipelineOrchestrator
from app.db.vector_store import VectorStore

app = FastAPI(title="Semantic Plagiarism Detection API", version="1.0")

orchestrator = PipelineOrchestrator()
db = VectorStore()

# Ensure temp directory exists for file uploads
os.makedirs("temp_uploads", exist_ok=True)

@app.post("/api/v1/ingest")
async def ingest_document(file: UploadFile = File(...)):
    """Uploads a document, generates embeddings, and saves them to the Vector DB."""
    temp_path = f"temp_uploads/{file.filename}"
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Run Phase 1 Engine
        embeddings = orchestrator.process_file(temp_path)
        
        # Run Phase 2 DB Storage
        db.store_embeddings(embeddings)
        
        return {"message": "Successfully ingested document", "chunks_stored": len(embeddings)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/api/v1/check-plagiarism")
async def check_plagiarism(file: UploadFile = File(...)):
    """Uploads a document and checks it against the Vector DB for plagiarism."""
    temp_path = f"temp_uploads/{file.filename}"
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Get embeddings for the new file
        query_embeddings = orchestrator.process_file(temp_path)
        
        all_matches = []
        for q_emb in query_embeddings:
            # Search the database for each chunk
            hits = db.search_similar(q_emb.embedding)
            
            for hit in hits:
                all_matches.append({
                    "query_text": q_emb.chunk.cleaned_text,
                    "matched_text": hit.payload["cleaned_text"],
                    "similarity_score": round(hit.score, 4),
                    "matched_document_id": hit.payload["document_id"]
                })
                
        # Sort matches by highest score first
        all_matches.sort(key=lambda x: x["similarity_score"], reverse=True)
        
        return JSONResponse(content={
            "status": "success",
            "total_matches_found": len(all_matches),
            "matches": all_matches
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)