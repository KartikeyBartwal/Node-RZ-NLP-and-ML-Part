import uvicorn
from fastapi import FastAPI
import numpy as np 
import pandas as pd
import os
from langchain_community.llms import Replicate
from model import LoadModel

app = FastAPI()
load_model = None 


@app.get('/')
def index():
    return {"message" : "API Loaded "}


@app.get('/{name}')
def get_name(name: str):
    return {"Get ready to use this thing" : f'{name}'}


@app.post('/output')
def output_json(prompt  : str):
    prompt = prompt 
    print(prompt)

    llm = LoadModel.use_hosting()    
    output = LoadModel.use_model(llm , prompt)

    return output


if __name__ == '__main__':
    uvicorn.run(app , host = '127.0.0.1' , port = 8000)

