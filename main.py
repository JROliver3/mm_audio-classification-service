import os
from predict import build_single_prediction
from cfg import Config
from flask import Flask, escape, request

STATIC_PRED_AUDIO_DIR = 'asc_data/audio'

app = Flask(__name__)

@app.route('/')
def ping():
    pong = request.args.get('pinger', 'pong')
    return pong

@app.route('/inference/<filename>')
def predict_audio_class(filename):
    try:
        response = build_single_prediction(filename, STATIC_PRED_AUDIO_DIR)
        return response
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80)