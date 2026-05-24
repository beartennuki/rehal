# app.py
import os

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse

from celery.result import AsyncResult
from celery_app import celery_app

from src.job.submit import submit_job
from src.job.retrieve import retrieve_doc
from config import Config
from env import load_rehal_env

load_rehal_env()

app = FastAPI()


@app.get("/")
def ok():
    return {"status": "up"}

# ==> Submit
@app.post("/job/submit")
async def submit(request: Request):
    # Check if the request has JSON
    if request.headers.get("Content-Type") != "application/json":
        raise HTTPException(status_code=400, detail="Bad Request - JSON data required")
    data = await request.json()
    #task = submit_job.apply_async(args=[data])
    task = submit_job.delay(data)
    return JSONResponse(content={"task_id": task.id, "status": "submitted"}, status_code=202)

# ==> Job status
@app.get("/job/status/{task_id}")
async def job_status(task_id: str):
    """
    Check the status of the job with the given task_id.
    """
    task = AsyncResult(task_id, app=celery_app)
    # task.result might be None if the task hasn’t produced a result yet.
    result = task.result if isinstance(task.result, dict) else {}
    response = {
        "task_id": task_id,
        "state": task.state,
        "status": result.get("status"),
        "msg": result.get("message")
    }
    return JSONResponse(content=response, status_code=200)

@app.post("/job/load")
async def job_result(request: Request):
    """
    Retrieve document.
    """
    content_type = request.headers.get("Content-Type", "")
    if "application/json" not in content_type:
        raise HTTPException(status_code=400, detail="Bad Request - JSON data required")

    data = await request.json()
    try:
        # Run the task synchronously
        task_result = retrieve_doc.apply(args=[data])

        if task_result.successful():
            result_data = task_result.result  # Get the actual result

            if isinstance(result_data, dict):  # Ensure it is a dictionary
                return JSONResponse(content=result_data, status_code=200)
            else:
                return JSONResponse(content={"error": "Unexpected response format"}, status_code=500)
        else:
            return JSONResponse(content={"error": str(task_result.info)}, status_code=400)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/db-config")
async def get_db_config():
    cfg = Config()
    db_config = {
        'uri': cfg.mongo_uri,
        'eval_db_name': cfg.eval_mongo_db_name,
        'assess_db_name': cfg.assess_mongo_db_name,
        'auth_db_name': cfg.auth_mongo_db_name,
        'user_db_name': cfg.user_mongo_db_name,
        'mcq_collection_name': cfg.mongo_collection_mcq_name,
    }

    return JSONResponse(content=db_config)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app_fast:app",
        host=os.getenv("REHAL_HOST", "0.0.0.0"),
        port=int(os.getenv("REHAL_PORT", "5500")),
        reload=os.getenv("REHAL_RELOAD", "false").strip().lower() in {"1", "true", "yes", "on"}
    )
