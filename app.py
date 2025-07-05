from flask import Flask, request, jsonify,render_template
import os
import numpy as np
import pandas as pd
from MLOps.pipeline.prediction import PredictionPipeline

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')