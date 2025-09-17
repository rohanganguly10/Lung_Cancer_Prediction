from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from data_api import router as data_router
from predict_api import router as predict_router

app = FastAPI()

# Static files
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Include routers
app.include_router(data_router, prefix="/api/data", tags=["Data"])
app.include_router(predict_router, prefix="/api/predict", tags=["Prediction"])

# Single-page app
@app.get("/", response_class=HTMLResponse)
async def main_page(request: Request):
    return templates.TemplateResponse("main.html", {"request": request})
