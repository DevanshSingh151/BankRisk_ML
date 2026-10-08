from fastapi import FastAPI
from backend.api.router import router
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="BankRisk360 API",
    description="Intelligent Banking Decision & Risk Analytics Platform",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")

@app.get("/")
def root():
    return {"message": "Welcome to BankRisk360 API"}
