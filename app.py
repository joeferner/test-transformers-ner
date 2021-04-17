from typing import Optional, List
from fastapi import FastAPI, Depends
from pydantic import BaseModel
from ner_model import NerModel, get_ner_model

app = FastAPI()


class NerRequest(BaseModel):
    text: str


class NerEntry(BaseModel):
    word: str
    score: float
    entity: str
    index: int
    start: int
    end: int


class NerResponse(BaseModel):
    entities: List[NerEntry]


@app.post("/ner", response_model=NerResponse)
def ner(request: NerRequest, model: NerModel = Depends(get_ner_model)):
    entities = model.extract(request.text)
    return NerResponse(entities=entities)
