# HARMONIC SOUNDS ANALYSIS : Music Genre Classification

**Muzec** is a project focused on classifying music genres using machine learning. The goal is to take an audio file, analyze its properties, and categorize it into one of several music genres. Here’s a detailed breakdown of how to set up, develop, and deploy this project.

## Project Workflow

### 1. Preparing the Dataset
   Gather and prepare data for training a machine learning model that can distinguish between music genres.
   - **Dataset**: We use the [GTZAN music genre dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification), which contains 1,000 audio tracks (30 seconds each) across 10 genres like rock, jazz, blues, and classical.
   - **Preprocessing**: Each 30-second track is segmented into smaller 3-second clips, creating more data samples. This segmentation provides the model with more varied data, improving genre recognition accuracy.

### 2. Extracting Features from Audio Files
   Audio files are complex, and the raw waveforms don’t provide direct insight into genre. Feature extraction translates these files into data the model can interpret.
   - **Tool**: We use the `Librosa` library in Python for this step.
   - **Features Extracted**:
**Tempo**: Speed of the music in beats per minute (BPM).
**RMS (Root Mean Square Energy)**: Average energy level of the audio signal, indicating loudness.
**ZCR (ZeroCrossing Rate)**: Frequency with which the signal changes from positive to negative, often used to detect percussive sounds.
**Harmonic Component**: Part of the signal with consistent, harmonic frequencies (melody).
**Percussive Component**: Part of the signal with abrupt changes, often representing drum or beat sounds.
**Spectral Centroid**: The "center of mass" of the spectrum, associated with the perceived brightness of a sound.
**Spectral Bandwidth**: Range of frequencies present in the sound, often associated with texture or timbre.
**Spectral Roll-off**: Frequency below which a set percentage (usually 85%) of the signal energy is contained, indicating the "edge" of the spectrum.
**Chroma**: Representation of the energy in each of the 12 musical pitches, disregarding octave, capturing harmonic elements.
**STFT (Short-Time Fourier Transform)**: Time-frequency representation of the signal, analyzing how frequency components evolve over time.
**MFCC (Mel-Frequency Cepstral Coefficients)**: Encodes the timbral qualities of the sound, useful for identifying vocal and instrument characteristics.
   - **Storing Features**: After extraction, features from each clip are saved in a structured CSV file. This file is then used as the input data for the model.

### 3. Building and Training the Model
   - **Goal**: Train a model to classify music genres based on the extracted features.
   - **Algorithm Choices**: Three algorithms were evaluated: `Logistic Regression`, `Random Forest`, and `XGBoost`. Each model was trained and tested on the dataset.
   - **Model Selection**: `XGBoost` outperformed the others, showing the highest accuracy. This model uses boosting, where multiple weak learners are combined to form a strong predictive model.
   - **Hyperparameter Tuning**: Using Grid Search, we optimized the hyperparameters (like `learning rate`, `maximum depth`, and `estimators` for XGBoost) to maximize accuracy.
Here’s a model comparison table based on the project information provided:

| **Model**               | **Algorithm**               | **Accuracy (Before Tuning)** | **Accuracy (After Tuning)** | **Notes**                                          |
|-------------------------|-----------------------------|------------------------------|-----------------------------|---------------------------------------------------|
| **Logistic Regression** | Linear classifier (One-vs-Rest) | 70.3%                        | 72.6%                   | Simple, interpretable, lower accuracy than others.|
| **Random Forest**       | Ensemble of Decision Trees  | 80.6%                        | 86.0%                       | Uses bagging, better accuracy due to ensemble learning. |
| **XGBoost**             | Gradient Boosted Trees      | 82.9%                        | 89.2%                       | Best performance, strong in capturing feature patterns.|


### 4. Saving the Model for Deployment
   Once the best-performing model is identified, it’s saved to be loaded later by the web application for predictions.
   - **Library Used**: `pickle` in Python, which serializes the model, allowing it to be reloaded without retraining.

### 5. Building the Web Application
   **Framework**: Flask (Python) serves as the backend to handle requests and load the machine learning model.
    **Frontend**: HTML, CSS, and JavaScript are used to create an interactive web interface.
    **Structure**:
      **Home**: Introduction to Muzec
      
   ![image](https://github.com/houda-moudni/Music-Genre-Classification/blob/main/static/images/Screens/home_page.png)
      **About**: An overview of how the project works.
     
   ![image](https://github.com/houd-amoudni/MusicGenre-Classification/blob/main/static/images/Screens/about_page.png)
      **Service**: Where users can upload an audio file or input a YouTube URL for analysis. This section extracts the audio features and uses the model to predict the genre
     
   ![image](https://github.com/houda-moudni/Music-Genre-Classification/blob/main/static/images/Screens/service_page.png)
      **Contact**: Developer contact information
     
   ![image](https://github.com/houda-moudni/Music-Genre-Classification/blob/main/static/images/Screens/contact_page.png)
    
   - **Data Flow**:
     - The user uploads an audio file.
     - The backend processes the file, extracts audio features, and applies the trained model.
     - The genre prediction is returned to the user and displayed on the page.

