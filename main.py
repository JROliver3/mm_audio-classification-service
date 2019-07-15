import os
from predict import build_single_prediction
from cfg import Config
from flask import Flask, escape, request

STATIC_PRED_AUDIO_DIR = 'audio'

app = Flask(__name__)

@app.route('/')
def ping():
    pong = request.args.get('pinger', 'pong2')
    return f'Hello {escape(pong)}!'

@app.route('/inference/<filename>')
def predict_audio_class(filename):
    prediction = build_single_prediction(filename, STATIC_PRED_AUDIO_DIR)
    return prediction