from fastapi import FastAPI, Query
from pydantic import BaseModel
import pandas as pd
import cloudpickle
from typing import Literal
import uvicorn
from MLOps.utils.common import read_yaml
from pathlib import Path
app = FastAPI(
    title="🌶️ Green Chilly Price Forecast API",
    version="1.0.0",
    description="""
This API provides **price predictions for green chillies** at the **Dambulla market** using a seasonality-aware forecasting model built with **Facebook Prophet**.

The underlying model uses historical market data to learn **trends**, **seasonality**, and **cyclic behavior**, enabling it to forecast future prices with reasonable confidence intervals.

### Features:
- 🗕️ Supports forecasting by **day**, **week**, or **month**
- 🔮 Uses Prophet for **robust seasonal modeling**
- 💱 Adjusted with real-time **dollar rate** if needed
- 📊 Returns predicted values with **upper and lower confidence bounds**

Whether you're a **farmer**, **trader**, or **analyst**, this API can help you make **informed decisions** about buying, selling, or storing green chillies based on projected market prices at Sri Lanka’s largest vegetable distribution hub — **Dambulla Economic Centre**.

---

### Example:
```
POST /forecast?periods=8&freq=W&&include_history=True&&start_from=2024-10-01
```

Returns predicted weekly prices for the next 8 weeks.
""",
    docs_url="/docs",
    redoc_url="/redoc"
)

config = read_yaml(Path('config/config.yaml'))
with open(config['model_evaluation']['model_path'], 'rb') as f:
    predictor = cloudpickle.load(f)
    
class PredictionRequest(BaseModel):
    periods: int = Query(10, description="Number of periods to predict")
    freq: Literal['D', 'M','W'] = Query(..., description="Frequency of the prediction: 'D' for daily, 'M' for monthly, 'W' for weekly")
    include_history: bool = Query(False, description="Whether to include historical data in the prediction")
    start_from: str = Query("2025-01-01", description="Start date for the prediction in 'YYYY-MM-DD' format. Required if include_history is True.")
    
@app.post("/predict")
async def predict(request: PredictionRequest):
    predictions = predictor.predict(periods=request.periods, freq=request.freq, include_history=request.include_history, start_from= request.start_from if request.include_history else None)
    predictions_df = predictions[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
    predictions_df['ds'] = predictions_df['ds'].dt.strftime('%Y-%m-%d')

    return predictions_df.to_dict(orient='records')

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
# To run the server, use the command: uvicorn app:app --reload