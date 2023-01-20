# pip install uvicorn gunicorn fastapi pydantic sklearn pandas tensorflow-hub tensorflow-text

import json
import tensorflow_hub as hub
import tensorflow_text as text
from fastapi import FastAPI
from pydantic import BaseModel

# cd into directory where this files lives and run this code to start webserver: uvicorn ml-api-bert-endpoint:app --reload

preprocess_model = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'
bert_model_url = 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/1'
bert_preprocess_model = hub.KerasLayer(preprocess_model)
bert_model = hub.KerasLayer(bert_model_url)

app = FastAPI()

class InputModel(BaseModel):
    text_for_bert:str

@app.post('/')
async def get_semantic_vector(item:InputModel):
    text_preprocessed = bert_preprocess_model([item.text_for_bert])
    bert_results = bert_model(text_preprocessed)
    return json.dumps(
        bert_results['pooled_output']\
            .numpy()\
            .tolist())