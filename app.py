from fastapi import FastAPI
from pydantic import BaseModel
from trainer import train_and_save
from classifier import Classifier
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
classifier = None

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

class TrainRequest(BaseModel):
    phrases: list[str]

class ClassifyRequest(BaseModel):
    text: str

@app.post("/train")
def train(req: TrainRequest):
    images = train_and_save(req.phrases)
    global classifier
    classifier = Classifier()
    return {"message": "Training complete","images":images}

@app.post("/classify")
def classify(req: ClassifyRequest):
    if classifier is None:
        return {"error": "Model not trained yet"}
    return classifier.classify(req.text)[0]