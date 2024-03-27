from flask import Flask,render_template,jsonify,request,send_file
import os,sys

from src.exception import CustomException
from src.logger import logging
from src.pipeline import train_pipeline,predict_pipeline

app = Flask(__name__)

@app.route('/')
def home():
    return jsonify('home')

@app.route('/train')
def train_route():
    train_pipeline = TrainingPipeline()
    train_pipeline.run_pipeline()
    return 'Training Completed'

@app.route('/predict')
def predict_route(method=["GET","POST"]):
    try:
        if request.method == "POST":
            predict_pipeline = PredictionPipeline()
            predict_pipeline.run_pipeline()
            logging.info("prediction done and downloading csv")
            return send_file()

        else:
            return render_template('upload_file.html')

    except Exception as e:
        raise CustomException(e,sys)

if __name__=="__main__":
    app.run(host="0.0.0.0",port=5000,debug=True)

