from fastapi import FastAPI
import pandas as pd
import numpy as np
from pydantic import BaseModel
from CARTE.carte_inference import get_preprocessor_and_model, get_prediction
from typing import List, Dict, Any

app = FastAPI()

# CARTE model
preprocessor, model = get_preprocessor_and_model()

class PredictRequest(BaseModel):
    data: List[Dict[str, Any]] 
    min_value: int                
    max_value: int
    integer_grades_flag: bool  


@app.post("/predict_carte_model")
def predict_model(input_data: PredictRequest):
    data = pd.DataFrame(input_data.data)
    data = data.replace({None: np.nan})
    get_prediction(preprocessor, model, data, input_data.min_value, input_data.max_value, input_data.integer_grades_flag, clipping=False)
    data = data.replace({np.nan: None})
    return data.to_dict(orient="records")

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)