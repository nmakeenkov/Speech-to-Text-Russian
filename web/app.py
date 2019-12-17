#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import base64
import sys
import sox
import pandas as pd
import matplotlib.pyplot as plt
import librosa
from librosa import display
from time import time, gmtime, strftime
from pathlib import Path
from flask import Flask, render_template, request, redirect, flash, jsonify
import uuid

sys.path.append('..')
from tools import data_preparator, segmenter, recognizer, transcriptions_parser
from tools.utils import make_ass, make_wav_scp, delete_folder, ThreadPool

app = Flask(__name__)
app.config['SECRET_KEY'] = '8dgn89vdf8vff8v9df99f'
app.config['ALLOWED_EXTENSIONS'] = ['ogg']
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = Path('/archive')

tp = ThreadPool(queue_threads=16)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def recognize(temp, wav):
    wav_scp = str(Path(temp) / 'wav.scp')
    make_wav_scp(wav, wav_scp)
    segm = segmenter.Segmenter(wav_scp, '../model/final.raw', '../model/conf/post_output.vec', '../model/conf/mfcc_hires.conf', temp)
    segments = segm.segment()
    wav_segments_scp, utt2spk, spk2utt = segm.extract_segments(segments)
    rec = recognizer.Recognizer(wav_segments_scp, '../model/final.mdl', '../model/HCLG.fst', '../model/words.txt', 
                                '../model/conf/mfcc.conf', '../model/conf/ivector_extractor.conf', spk2utt, temp)
    transcriptions = rec.recognize(Path(wav).stem)
    ass = str(Path(temp) / 'wav.ass')
    make_ass(Path(wav).name, segments, transcriptions, utt2spk, ass)
    pars = transcriptions_parser.TranscriptionsParser('', '', '', 0, 0, 'wav.csv')
    transcriptions_df = pars.process_file(ass)
    return transcriptions_df

def plot_waveform(temp, wav, channels):
    y, sr = librosa.load(wav, mono=False)
    if channels == 1:
        y = y.reshape(-1, len(y))
    DPI = 72
    plt.figure(1, figsize=(16, 9), dpi=DPI) 
    plt.subplots_adjust(wspace=0, hspace=0) 
    for n in range(channels): 
        plt.subplot(2, 1, n + 1, facecolor='200')
        display.waveplot(y[n], sr)
        plt.grid(True, color='w') 
    waveform = str(Path(temp) / 'waveform.png')
    plt.savefig(waveform, dpi=DPI)
    waveform = str(base64.b64encode(open(waveform, 'rb').read()))[2: -1]
    return waveform

def perform_conversion(file_id):
    res = str(app.config['UPLOAD_FOLDER'] / (file_id + ".res"))
    try:
        filename = file_id + ".ogg"
        ogg = str(app.config['UPLOAD_FOLDER'] / filename)
        os.system('ffmpeg -i {0}.ogg {0}.wav'.format(str(app.config['UPLOAD_FOLDER'] / file_id)))
        wav = ogg[:-3] + "wav"
        temp = str(app.config['UPLOAD_FOLDER'] / Path(wav).stem)
        os.makedirs(temp, exist_ok=True)
        transcriptions = recognize(temp, wav)
        delete_folder(temp)
        os.remove(wav)
        os.remove(ogg)
        transcriptions = transcriptions[['Text']]
        result = transcriptions.values[0][0]
        with open(res, 'w') as f:
            f.write(result)
    except Exception as e:
        print(e)
        with open(res, 'w') as f:
            f.write('xxxFAILxxx')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_post', methods=['POST'])
def upload_post():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('Отсутствует файл')
            return jsonify({'error': 'Отсутствует файл'})
        file = request.files['file']
        if file.filename == '':
            flash('Не выбран файл для загрузки')
            return jsonify({'error': 'Не выбран файл для загрузки'})
        if not allowed_file(file.filename):
            flash('Файл должен иметь расширение .ogg')
            return jsonify({'error': 'Файл должен иметь расширение .ogg'})
        file_id = str(uuid.uuid4())
        filename = file_id + ".ogg"
        wav = str(app.config['UPLOAD_FOLDER'] / filename)
        file.save(wav)
        tp.queue.put({'call': perform_conversion, 'args': (file_id,)})
        return jsonify({'id': file_id})
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        file_id = str(uuid.uuid4())
        filename = file_id + ".ogg"
        wav = str(app.config['UPLOAD_FOLDER'] / filename)
        file = request.get_data()
        with open(wav, 'wb') as f:
            f.write(file)
        tp.queue.put({'call': perform_conversion, 'args': (file_id,)})
        return jsonify({'id': file_id})

@app.route('/get/<file_id>', methods=['GET'])
def get_status(file_id):
    res = str(app.config['UPLOAD_FOLDER'] / (file_id + ".res"))
    if not os.path.exists(res):
        return jsonify({"status": "in_progress", "id": file_id})
    with open(res, 'r') as f:
        text = f.read()
    if text == 'xxxFAILxxx':
        return jsonify({'status': 'error', "id": file_id})
    return jsonify({'status': 'completed', 'text': text})

@app.errorhandler(413)
def request_entity_too_large(e):
        flash('Размер файла не должен превышать 20 МБ')
        return redirect('/')

if __name__ == '__main__':
    app.run(host='0.0.0.0')
