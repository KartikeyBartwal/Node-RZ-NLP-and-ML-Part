import uvicorn
from fastapi import FastAPI
import numpy as np 
import pandas as pd
import os
from langchain_community.llms import Replicate
from model import LoadModel
from vector_store import VectorStore
from NLP_model import NLP_Model
from NLP_model import NLP_Model_BERT
from NLP_model import remove_stopwords
from NLP_model import stemming
import json

app = FastAPI()
load_model = None 


def fix_json(json_string):
    try:
        # Remove unnecessary double quotes at the beginning and end of the string
        json_string = json_string.strip('"')
        # Replace escaped newlines with actual newlines
        json_string = json_string.replace('\\n', '\n')
        # Parse the JSON string
        json_obj = json.loads(json_string)

        return json_obj
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return None
    

@app.get('/')
def index():
    return {"message" : "API Loaded "}


@app.get('/output/')
def output_json(prompt  : str):

    prompt = prompt 
    print(prompt)
    
    # print("here")
    # print("*" * 50)

    prompt = remove_stopwords(prompt)
    prompt = stemming(prompt)


#    ################ Reserved for the fine tuning and instruction tuning of the large language models
    # llm = LoadModel.use_hosting()    
    # output = LoadModel.use_model(llm , prompt)
#     ################ Reserved for the fine tuning and instruction tuning of the large language models
    # print("*" * 50)

    nlp_model = NLP_Model_BERT()
    output = nlp_model.GetJSON(prompt)
    
#     print(json.load(output))

    return json.loads(output)

    # return {"prompt: " : prompt}

if __name__ == '__main__':
    uvicorn.run(app , host = '127.0.0.1' , port = 8000)

