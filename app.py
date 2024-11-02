from flask import Flask, render_template, request
import pickle
import librosa
import os
import numpy as np
import pandas as pd
import json
from werkzeug.utils import secure_filename


app = Flask(__name__)
app.config['UPLOAD_DIR'] = "UPLOAD_FOLDER"

if not os.path.exists(app.config['UPLOAD_DIR']):
    os.makedirs(app.config['UPLOAD_DIR'])


# Load the trained model
with open('deployed_models/label-encoder-model.pk1', 'rb') as f:
    encoder = pickle.load(f)

with open('deployed_models/scaler-model.pk1', 'rb') as f:
    scaler = pickle.load(f)

with open('deployed_models/classifier-model.pk1', 'rb') as f:
    model = pickle.load(f)

ALLOWED_EXTENSIONS = set(['wav', 'mp3'])

EMAILS_FILE = 'emails.json'



@app.route('/email', methods=['POST'])
def submit_email():
    if request.method == 'POST':
        email = request.form['email']
        try:
            # Open the JSON file and load existing data
            with open(EMAILS_FILE, 'r') as f:
                emails = json.load(f)
        except FileNotFoundError:
            emails = []

        emails.append(email)

        # Write the updated list of emails back to the JSON file
        with open(EMAILS_FILE, 'w') as f:
            json.dump(emails, f, indent=4)

        return "Email registered successfully!"


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


def extract_features(file):

    header = [
        'tempo', 'mean_y_harmonic', 'var_y_harmonic', 'mean_y_perceptr','var_y_perceptr',
        'mean_rms', 'var_rms', 'mean_zcr', 'var_zcr', 'mean_spectral_cent',
        'var_spectral_cent', 'mean_spectral_rolloff', 'var_spectral_rolloff',
        'mean_spectral_bandwidth', 'var_spectral_bandwidth', 'mean_chroma_stft',
        'var_chroma_stft'
    ]

    for i in range(1, 21):
        header.append(f'mfcc_mean{i}')

    for i in range(1, 21):
        header.append(f'mfcc_var{i}')


    y, sr = librosa.load(file)
    y_harmonic, y_perceptr = librosa.effects.hpss(y)
    tempo, _ = librosa.beat.beat_track(y=y_harmonic, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y=y)
    rms = librosa.feature.rms(y=y)
    spectral_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)

    
    features_list = [
            tempo, np.mean(y_harmonic), np.var(y_harmonic), np.mean(y_perceptr),
            np.var(y_perceptr), np.mean(rms), np.var(rms), np.mean(zcr), np.var(zcr),
            np.mean(spectral_cent), np.var(spectral_cent), np.mean(spectral_rolloff),
            np.var(spectral_rolloff), np.mean(spectral_bandwidth),
            np.var(spectral_bandwidth), np.mean(chroma_stft), np.var(chroma_stft)
        ]
    
    for e_mean in mfcc:
        features_list.append(np.mean(e_mean))

    for e_var in mfcc:
        features_list.append(np.var(e_var))

    
    features_dict = dict(zip(header, features_list))
    features = pd.DataFrame([features_dict])

    return features


def predict_genre(data):
    features = extract_features(data)
    print("d")
    features_scaled = scaler.transform(features)
    print("done")
    prediction = model.predict(features_scaled)
    print("donedone")
    genre = encoder.inverse_transform(prediction)
    print("donedonedone")
    genre = str(genre).strip("[]").replace("'", "")

    return genre

# Define routes
@app.route('/')
def index():
    return render_template('index.html')



            

@app.route('/upload', methods=['POST','GET'])
def upload():
    try: 
        if 'file' not in request.files:
            return render_template('index.html', errors=['No file part'])

        file = request.files['file']

        if file.filename == '':
           return render_template('index.html', errors=['No file selected for uploading'])
            
        if file and allowed_file(file.filename):  
            try:
                file_path = os.path.join(app.config['UPLOAD_DIR'], file.filename)
                file.save(file_path)
                genre = predict_genre(file_path)

                return render_template('index.html', prediction=genre)
            except Exception as e:
                print(f'Error during prediction: {e}')  # Log the error to the console
                return render_template('index.html', errors=['An error occurred while predicting the genre']) 

        return render_template('index.html', errors=['Allowed file types are .wav, .mp3'])

    except Exception as e:
        print(f'Internal server error: {e}')  # Log the error to the console
        return render_template('index.html', errors=['Internal server error'])
  


if __name__ == '__main__':
    app.run(debug=True)

