# Import essential libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from glob import glob

import librosa
import librosa.display
import IPython.display as ipd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score, accuracy_score, confusion_matrix, classification_report

from LogisticRegression import LogisticRegression, LogisticRegressionSimplified
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import matplotlib.ticker as mticker
from sklearn.metrics import ConfusionMatrixDisplay



# Custom Normalization Tool (I wasn't sure if we could use sklearn for this)
class Normalize:
    def fit(self, X):
        self.means = X.mean()
        self.stds = X.std()
        return self
    def transform(self, X):
        return (X - self.means) / self.stds
    def fit_transform(self, X):
        return self.fit(X).transform(X)

# Custom Scaling Tool (I wasn't sure if we could use sklearn for this)
class Scaler:
    def fit(self, X):
        self.mins = X.min()
        self.maxs = X.max()
        return self
    def transform(self, X):
        return (X - self.mins) / (self.maxs - self.mins)
    def fit_transform(self, X):
        return self.fit(X).transform(X)



# DATA PROCESSING
# PATH TO TRAINING DATASET IS DEFINED HERE
# Loads files for each genre
# UPDATE THIS BASE PATH TO WHERE YOUR DATA FOLDER IS
BASE_PATH = 'C:/Users/mendo/CS-529-IntroToMachineLearning-Project2-LogisticRegression'

blues_audio = glob(f'{BASE_PATH}/data/train/blues/*.au')
classical_audio = glob(f'{BASE_PATH}/data/train/classical/*.au')
country_audio = glob(f'{BASE_PATH}/data/train/country/*.au')
disco_audio = glob(f'{BASE_PATH}/data/train/disco/*.au')
hiphop_audio = glob(f'{BASE_PATH}/data/train/hiphop/*.au')
jazz_audio = glob(f'{BASE_PATH}/data/train/jazz/*.au')
metal_audio = glob(f'{BASE_PATH}/data/train/metal/*.au')
pop_audio = glob(f'{BASE_PATH}/data/train/pop/*.au')
reggae_audio = glob(f'{BASE_PATH}/data/train/reggae/*.au')
rock_audio = glob(f'{BASE_PATH}/data/train/rock/*.au')

# Verify files are found
print("Blues files found:", len(blues_audio))
print("Classical files found:", len(classical_audio))
print("Country files found:", len(country_audio))
print("Disco files found:", len(disco_audio))
print("Hiphop files found:", len(hiphop_audio))
print("Jazz files found:", len(jazz_audio))
print("Metal files found:", len(metal_audio))
print("Pop files found:", len(pop_audio))
print("Reggae files found:", len(reggae_audio))
print("Rock files found:", len(rock_audio))


# Create a features list
features = []

# Defines number of Mel-Frequency Cepstral Coefficients to extract
num_mfcc = 40


# Load 'Blues' into Librosa and Extract Features
for file in blues_audio:
    y, sr = librosa.load(file, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc = num_mfcc)
    mfcc_mean = mfcc.mean(axis=1)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = chroma.mean(axis=1)
    chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_cqt_mean = chroma_cqt.mean(axis=1)
    chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr)
    chroma_cens_mean = chroma_cens.mean(axis=1)
    spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    spec_contrast_mean = spec_contrast.mean(axis=1)
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean = zcr.mean()
    rms = librosa.feature.rms(y=y)
    rms_mean = rms.mean()
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    spectral_rolloff_mean = spectral_rolloff.mean()
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_centroid_mean = spectral_centroid.mean()
    features.append({
                    'genre': 'blues',
                     **{f'mfcc_{i+1}': mfcc_mean[i] for i in range(len(mfcc_mean))},
                     **{f'chroma_{i+1}': chroma_mean[i] for i in range(len(chroma_mean))},
                     **{f'chromaCQT_{i+1}': chroma_cqt_mean[i] for i in range(len(chroma_cqt_mean))},
                     **{f'chromaCENS_{i+1}': chroma_cens_mean[i] for i in range(len(chroma_cens_mean))},
                     **{f'spec_contrast_{i+1}': spec_contrast_mean[i] for i in range(len(spec_contrast_mean))},
                     'zcr': zcr_mean,
                     'rms': rms_mean,
                     'tempo': float(np.squeeze(tempo)),
                     'spectral_rolloff': spectral_rolloff_mean,
                     'spectral_centroid': spectral_centroid_mean
                     })

# Load 'Classical' into Librosa and Extract Features
for file in classical_audio:
    y, sr = librosa.load(file, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc = num_mfcc)
    mfcc_mean = mfcc.mean(axis=1)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = chroma.mean(axis=1)
    chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_cqt_mean = chroma_cqt.mean(axis=1)
    chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr)
    chroma_cens_mean = chroma_cens.mean(axis=1)
    spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    spec_contrast_mean = spec_contrast.mean(axis=1)
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean = zcr.mean()
    rms = librosa.feature.rms(y=y)
    rms_mean = rms.mean()
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    spectral_rolloff_mean = spectral_rolloff.mean()
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_centroid_mean = spectral_centroid.mean()
    features.append({
                    'genre': 'classical',
                     **{f'mfcc_{i+1}': mfcc_mean[i] for i in range(len(mfcc_mean))},
                     **{f'chroma_{i+1}': chroma_mean[i] for i in range(len(chroma_mean))},
                     **{f'chromaCQT_{i+1}': chroma_cqt_mean[i] for i in range(len(chroma_cqt_mean))},
                     **{f'chromaCENS_{i+1}': chroma_cens_mean[i] for i in range(len(chroma_cens_mean))},
                     **{f'spec_contrast_{i + 1}': spec_contrast_mean[i] for i in range(len(spec_contrast_mean))},
                     'zcr': zcr_mean,
                     'rms': rms_mean,
                     'tempo': float(np.squeeze(tempo)),
                     'spectral_rolloff': spectral_rolloff_mean,
                     'spectral_centroid': spectral_centroid_mean
                     })

# Load 'Country' into Librosa and Extract Features
for file in country_audio:
    y, sr = librosa.load(file, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc = num_mfcc)
    mfcc_mean = mfcc.mean(axis=1)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = chroma.mean(axis=1)
    chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_cqt_mean = chroma_cqt.mean(axis=1)
    chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr)
    chroma_cens_mean = chroma_cens.mean(axis=1)
    spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    spec_contrast_mean = spec_contrast.mean(axis=1)
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean = zcr.mean()
    rms = librosa.feature.rms(y=y)
    rms_mean = rms.mean()
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    spectral_rolloff_mean = spectral_rolloff.mean()
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_centroid_mean = spectral_centroid.mean()
    features.append({
                    'genre': 'country',
                     **{f'mfcc_{i+1}': mfcc_mean[i] for i in range(len(mfcc_mean))},
                     **{f'chroma_{i+1}': chroma_mean[i] for i in range(len(chroma_mean))},
                     **{f'chromaCQT_{i+1}': chroma_cqt_mean[i] for i in range(len(chroma_cqt_mean))},
                     **{f'chromaCENS_{i+1}': chroma_cens_mean[i] for i in range(len(chroma_cens_mean))},
                     **{f'spec_contrast_{i+1}': spec_contrast_mean[i] for i in range(len(spec_contrast_mean))},
                     'zcr': zcr_mean,
                     'rms': rms_mean,
                     'tempo': float(np.squeeze(tempo)),
                     'spectral_rolloff': spectral_rolloff_mean,
                     'spectral_centroid': spectral_centroid_mean
                     })

# Load 'Disco' into Librosa and Extract Features
for file in disco_audio:
    y, sr = librosa.load(file, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc = num_mfcc)
    mfcc_mean = mfcc.mean(axis=1)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = chroma.mean(axis=1)
    chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_cqt_mean = chroma_cqt.mean(axis=1)
    chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr)
    chroma_cens_mean = chroma_cens.mean(axis=1)
    spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    spec_contrast_mean = spec_contrast.mean(axis=1)
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean = zcr.mean()
    rms = librosa.feature.rms(y=y)
    rms_mean = rms.mean()
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    spectral_rolloff_mean = spectral_rolloff.mean()
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_centroid_mean = spectral_centroid.mean()
    features.append({
                    'genre': 'disco',
                     **{f'mfcc_{i+1}': mfcc_mean[i] for i in range(len(mfcc_mean))},
                     **{f'chroma_{i+1}': chroma_mean[i] for i in range(len(chroma_mean))},
                     **{f'chromaCQT_{i+1}': chroma_cqt_mean[i] for i in range(len(chroma_cqt_mean))},
                     **{f'chromaCENS_{i+1}': chroma_cens_mean[i] for i in range(len(chroma_cens_mean))},
                     **{f'spec_contrast_{i+1}': spec_contrast_mean[i] for i in range(len(spec_contrast_mean))},
                     'zcr': zcr_mean,
                     'rms': rms_mean,
                     'tempo': float(np.squeeze(tempo)),
                     'spectral_rolloff': spectral_rolloff_mean,
                     'spectral_centroid': spectral_centroid_mean
                     })

# Load 'HipHop' into Librosa and Extract Features
for file in hiphop_audio:
    y, sr = librosa.load(file, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc = num_mfcc)
    mfcc_mean = mfcc.mean(axis=1)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = chroma.mean(axis=1)
    chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_cqt_mean = chroma_cqt.mean(axis=1)
    chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr)
    chroma_cens_mean = chroma_cens.mean(axis=1)
    spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    spec_contrast_mean = spec_contrast.mean(axis=1)
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean = zcr.mean()
    rms = librosa.feature.rms(y=y)
    rms_mean = rms.mean()
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    spectral_rolloff_mean = spectral_rolloff.mean()
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_centroid_mean = spectral_centroid.mean()
    features.append({
                    'genre': 'hiphop',
                     **{f'mfcc_{i+1}': mfcc_mean[i] for i in range(len(mfcc_mean))},
                     **{f'chroma_{i+1}': chroma_mean[i] for i in range(len(chroma_mean))},
                     **{f'chromaCQT_{i+1}': chroma_cqt_mean[i] for i in range(len(chroma_cqt_mean))},
                     **{f'chromaCENS_{i+1}': chroma_cens_mean[i] for i in range(len(chroma_cens_mean))},
                     **{f'spec_contrast_{i+1}': spec_contrast_mean[i] for i in range(len(spec_contrast_mean))},
                     'zcr': zcr_mean,
                     'rms': rms_mean,
                     'tempo': float(np.squeeze(tempo)),
                     'spectral_rolloff': spectral_rolloff_mean,
                     'spectral_centroid': spectral_centroid_mean
                     })

# Load 'Jazz' into Librosa and Extract Features
for file in jazz_audio:
    y, sr = librosa.load(file, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc = num_mfcc)
    mfcc_mean = mfcc.mean(axis=1)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = chroma.mean(axis=1)
    chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_cqt_mean = chroma_cqt.mean(axis=1)
    chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr)
    chroma_cens_mean = chroma_cens.mean(axis=1)
    spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    spec_contrast_mean = spec_contrast.mean(axis=1)
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean = zcr.mean()
    rms = librosa.feature.rms(y=y)
    rms_mean = rms.mean()
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    spectral_rolloff_mean = spectral_rolloff.mean()
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_centroid_mean = spectral_centroid.mean()
    features.append({
                    'genre': 'jazz',
                     **{f'mfcc_{i+1}': mfcc_mean[i] for i in range(len(mfcc_mean))},
                     **{f'chroma_{i+1}': chroma_mean[i] for i in range(len(chroma_mean))},
                     **{f'chromaCQT_{i+1}': chroma_cqt_mean[i] for i in range(len(chroma_cqt_mean))},
                     **{f'chromaCENS_{i+1}': chroma_cens_mean[i] for i in range(len(chroma_cens_mean))},
                     **{f'spec_contrast_{i+1}': spec_contrast_mean[i] for i in range(len(spec_contrast_mean))},
                     'zcr': zcr_mean,
                     'rms': rms_mean,
                     'tempo': float(np.squeeze(tempo)),
                     'spectral_rolloff': spectral_rolloff_mean,
                     'spectral_centroid': spectral_centroid_mean
                     })

# Load 'Metal' into Librosa and Extract Features
for file in metal_audio:
    y, sr = librosa.load(file, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc = num_mfcc)
    mfcc_mean = mfcc.mean(axis=1)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = chroma.mean(axis=1)
    chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_cqt_mean = chroma_cqt.mean(axis=1)
    chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr)
    chroma_cens_mean = chroma_cens.mean(axis=1)
    spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    spec_contrast_mean = spec_contrast.mean(axis=1)
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean = zcr.mean()
    rms = librosa.feature.rms(y=y)
    rms_mean = rms.mean()
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    spectral_rolloff_mean = spectral_rolloff.mean()
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_centroid_mean = spectral_centroid.mean()
    features.append({
                    'genre': 'metal',
                     **{f'mfcc_{i+1}': mfcc_mean[i] for i in range(len(mfcc_mean))},
                     **{f'chroma_{i+1}': chroma_mean[i] for i in range(len(chroma_mean))},
                     **{f'chromaCQT_{i+1}': chroma_cqt_mean[i] for i in range(len(chroma_cqt_mean))},
                     **{f'chromaCENS_{i+1}': chroma_cens_mean[i] for i in range(len(chroma_cens_mean))},
                     **{f'spec_contrast_{i+1}': spec_contrast_mean[i] for i in range(len(spec_contrast_mean))},
                     'zcr': zcr_mean,
                     'rms': rms_mean,
                     'tempo': float(np.squeeze(tempo)),
                     'spectral_rolloff': spectral_rolloff_mean,
                     'spectral_centroid': spectral_centroid_mean
                     })

# Load 'Pop' into Librosa and Extract Features
for file in pop_audio:
    y, sr = librosa.load(file, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc = num_mfcc)
    mfcc_mean = mfcc.mean(axis=1)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = chroma.mean(axis=1)
    chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_cqt_mean = chroma_cqt.mean(axis=1)
    chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr)
    chroma_cens_mean = chroma_cens.mean(axis=1)
    spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    spec_contrast_mean = spec_contrast.mean(axis=1)
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean = zcr.mean()
    rms = librosa.feature.rms(y=y)
    rms_mean = rms.mean()
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    spectral_rolloff_mean = spectral_rolloff.mean()
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_centroid_mean = spectral_centroid.mean()
    features.append({
                    'genre': 'pop',
                     **{f'mfcc_{i+1}': mfcc_mean[i] for i in range(len(mfcc_mean))},
                     **{f'chroma_{i+1}': chroma_mean[i] for i in range(len(chroma_mean))},
                     **{f'chromaCQT_{i+1}': chroma_cqt_mean[i] for i in range(len(chroma_cqt_mean))},
                     **{f'chromaCENS_{i+1}': chroma_cens_mean[i] for i in range(len(chroma_cens_mean))},
                     **{f'spec_contrast_{i+1}': spec_contrast_mean[i] for i in range(len(spec_contrast_mean))},
                     'zcr': zcr_mean,
                     'rms': rms_mean,
                     'tempo': float(np.squeeze(tempo)),
                     'spectral_rolloff': spectral_rolloff_mean,
                     'spectral_centroid': spectral_centroid_mean
                     })

# Load 'Reggae' into Librosa and Extract Features
for file in reggae_audio:
    y, sr = librosa.load(file, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc = num_mfcc)
    mfcc_mean = mfcc.mean(axis=1)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = chroma.mean(axis=1)
    chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_cqt_mean = chroma_cqt.mean(axis=1)
    chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr)
    chroma_cens_mean = chroma_cens.mean(axis=1)
    spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    spec_contrast_mean = spec_contrast.mean(axis=1)
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean = zcr.mean()
    rms = librosa.feature.rms(y=y)
    rms_mean = rms.mean()
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    spectral_rolloff_mean = spectral_rolloff.mean()
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_centroid_mean = spectral_centroid.mean()
    features.append({
                    'genre': 'reggae',
                     **{f'mfcc_{i+1}': mfcc_mean[i] for i in range(len(mfcc_mean))},
                     **{f'chroma_{i+1}': chroma_mean[i] for i in range(len(chroma_mean))},
                     **{f'chromaCQT_{i+1}': chroma_cqt_mean[i] for i in range(len(chroma_cqt_mean))},
                     **{f'chromaCENS_{i+1}': chroma_cens_mean[i] for i in range(len(chroma_cens_mean))},
                     **{f'spec_contrast_{i+1}': spec_contrast_mean[i] for i in range(len(spec_contrast_mean))},
                     'zcr': zcr_mean,
                     'rms': rms_mean,
                     'tempo': float(np.squeeze(tempo)),
                     'spectral_rolloff': spectral_rolloff_mean,
                     'spectral_centroid': spectral_centroid_mean
                     })

# Load 'Rock' into Librosa and Extract Features
for file in rock_audio:
    y, sr = librosa.load(file, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc = num_mfcc)
    mfcc_mean = mfcc.mean(axis=1)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = chroma.mean(axis=1)
    chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_cqt_mean = chroma_cqt.mean(axis=1)
    chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr)
    chroma_cens_mean = chroma_cens.mean(axis=1)
    spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    spec_contrast_mean = spec_contrast.mean(axis=1)
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean = zcr.mean()
    rms = librosa.feature.rms(y=y)
    rms_mean = rms.mean()
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    spectral_rolloff_mean = spectral_rolloff.mean()
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_centroid_mean = spectral_centroid.mean()
    features.append({
                    'genre': 'rock',
                     **{f'mfcc_{i+1}': mfcc_mean[i] for i in range(len(mfcc_mean))},
                     **{f'chroma_{i+1}': chroma_mean[i] for i in range(len(chroma_mean))},
                     **{f'chromaCQT_{i+1}': chroma_cqt_mean[i] for i in range(len(chroma_cqt_mean))},
                     **{f'chromaCENS_{i+1}': chroma_cens_mean[i] for i in range(len(chroma_cens_mean))},
                     **{f'spec_contrast_{i+1}': spec_contrast_mean[i] for i in range(len(spec_contrast_mean))},
                     'zcr': zcr_mean,
                     'rms': rms_mean,
                     'tempo': float(np.squeeze(tempo)),
                     'spectral_rolloff': spectral_rolloff_mean,
                     'spectral_centroid': spectral_centroid_mean
                     })


# Creates a dataframe from the extracted features
df = pd.DataFrame(features)
print("\n\nInitial Dataframe:")
print(df.head())
print(df.shape)


# Turns 'genre' into a numerical value
encoder = LabelEncoder()
df['label'] = encoder.fit_transform(df['genre'])
print("\n\nEncoded Dataframe:")
print(df.head())

# Creates X (features) dataset
X_init = df.drop(columns=['genre', 'label'])
print("\n\nX Dataframe:")
print(X_init.head())
print(X_init.shape)

# Normalizes X (features) dataset
normalizer = Normalize()
X_norm = normalizer.fit_transform(X_init)
print("\n\nNormalized X Dataframe:")
print(X_norm.head())
print(X_norm.shape)
print(X_norm.mean(axis=0))
print(X_norm.std(axis=0))

# Scales X (features) dataset
scaler = Scaler()
X_norm_scale = scaler.fit_transform(X_norm)
print("\n\nNormalized and Scaled X Dataframe:")
print(X_norm_scale.head())
print(X_norm_scale.shape)
print(X_norm_scale.mean(axis=0))
print(X_norm_scale.std(axis=0))

# Performs PCA on X (features) dataset
pca = PCA(n_components=0.99)
X_norm_scale_pca = pca.fit_transform(X_norm_scale)
print("\n\nNormalized and Scaled and PCA X Dataframe:")
print(X_norm_scale_pca)
X = pd.DataFrame(X_norm_scale_pca)
print(X.head())
print(X.shape)

# Creates y (genre) dataset
y = df['label']
print("\n\ny Dataframe:")
print(y)
print(y.shape)

# Creates Training and Validation datasets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

print("\n\nX Training Dataset:")
print(X_train.head())
print("\n\ny Training Dataset:")
print(y_train.head())
print("\n\nX Validation Dataset:")
print(X_val.head())
print("\n\ny Validation Dataset:")
print(y_val.head())

# Before Normalization
mfcc_cols = [c for c in df.columns if c.startswith('mfcc_')]
plt.figure(figsize=(12, 5))
sns.boxplot(data=df[mfcc_cols])
plt.xticks(rotation=45)
plt.title("MFCC feature spread")
plt.show()

# After Normalization
mfcc_cols = [c for c in df.columns if c.startswith('mfcc_')]
plt.figure(figsize=(12, 5))
sns.boxplot(data=X_norm[mfcc_cols])
plt.xticks(rotation=45)
plt.title("MFCC feature spread after normalization")
plt.show()

# After Normalization + Scaling
mfcc_cols = [c for c in df.columns if c.startswith('mfcc_')]
plt.figure(figsize=(12, 5))
sns.boxplot(data=X_norm_scale[mfcc_cols])
plt.xticks(rotation=45)
plt.title("MFCC feature spread after normalization and scaling")
plt.show()

# After Normalization + Scaling + PCA
plt.figure(figsize=(12, 5))
sns.boxplot(data=X)
plt.xticks(rotation=45)
plt.title("Feature spread after normalization and scaling and PCA")
plt.show()

# Before Normalization
plt.figure(figsize=(10, 5))
sns.boxplot(x='genre', y='mfcc_1', data=df)
plt.xticks(rotation=45)
plt.title("mfcc_1 by genre")
plt.show()

# After Normalization
xlabel = X_norm
xlabel['label'] = y
plt.figure(figsize=(10, 5))
sns.boxplot(x='label', y='mfcc_1', data=xlabel)
plt.xticks(rotation=45)
plt.title("mfcc_1 by genre after normalization")
plt.show()

# After Normalization + Scaling
xlabel = X_norm_scale
xlabel['label'] = y
plt.figure(figsize=(10, 5))
sns.boxplot(x='label', y='mfcc_1', data=xlabel)
plt.xticks(rotation=45)
plt.title("mfcc_1 by genre after normalization and scaling")
plt.show()

# After Normalization + Scaling + PCA
xlabel = X
xlabel['label'] = y
plt.figure(figsize=(10, 5))
sns.boxplot(x='label', y=0, data=xlabel)
plt.xticks(rotation=45)
plt.title("PCA feature 0 by genre after normalization and scaling and PCA")
plt.show()

# Before Normalization
df_num = df.drop(columns=['genre'])
corr_matrix = df_num.corr()
sns.heatmap(corr_matrix, cmap='coolwarm', center=0)
plt.title("heatmap")
plt.show()

# After Normalization
corr_matrix = X_norm.corr()
sns.heatmap(corr_matrix, cmap='coolwarm', center=0)
plt.title("heatmap after normalization")
plt.show()

# After Normalization + Scaling
corr_matrix = X_norm_scale.corr()
sns.heatmap(corr_matrix, cmap='coolwarm', center=0)
plt.title("heatmap after normalization and scaling")
plt.show()

# After Normalization + Scaling + PCA
corr_matrix = X.corr()
sns.heatmap(corr_matrix, cmap='coolwarm', center=0)
plt.title("heatmap after normalization and scaling and PCA")
plt.show()



# TRAINING AND VALIDATING (LOGISTIC REGRESSION)
# BEST ONE
# Train Logistic Regression Model
logres = LogisticRegression(alpha=0.1, iterations=5000, lambdaparam=0.1)
print("\n\nLogistic Regression Model Training...")
logres.fit(X_train, y_train)
print("Logistic Regression Model Training Complete")
print("\n\nPredicting from Trained Logisitc Regression Model")
y_pred = logres.predict(X_val)
print("Logistic Regression Model Predictions Complete")


print("\n\nEvaluating Logistic Regression Model\n")
print("Logistic Regression Model Evaluation:")
lraccuracy = accuracy_score(y_val, y_pred)
lrbalancedaccuracy = balanced_accuracy_score(y_val, y_pred)
lrconfusionmatrix = confusion_matrix(y_val, y_pred)
lrreport = classification_report(y_val, y_pred)

print(f"Accuracy: {lraccuracy:.4f}")
print(f"Balanced Accuracy: {lrbalancedaccuracy:.4f}")
print(f"Confusion Matrix:\n{lrconfusionmatrix}")
print(f"Classification Report:\n{lrreport}")


# TESTING
# Train Altered-Hyperparameter Logistic Regression Model
logres_alt = LogisticRegression(alpha=0.10, iterations=5000)
print("\n\nAltered Logistic Regression Model Training...")
logres_alt.fit(X_train, y_train)
print("Altered Logistic Regression Model Training Complete")
print("\n\nPredicting from Trained Altered Logistic Regression Model")
y_pred_alt = logres_alt.predict(X_val)
print("Altered Logistic Regression Model Predictions Complete")


print("\n\nEvaluating Altered Logistic Regression Model\n")
print("Altered Logistic Regression Model Evaluation:")
lraccuracy = accuracy_score(y_val, y_pred_alt)
lrbalancedaccuracy = balanced_accuracy_score(y_val, y_pred_alt)
lrconfusionmatrix = confusion_matrix(y_val, y_pred_alt)
lrreport = classification_report(y_val, y_pred_alt)

print(f"Accuracy: {lraccuracy:.4f}")
print(f"Balanced Accuracy: {lrbalancedaccuracy:.4f}")
print(f"Confusion Matrix:\n{lrconfusionmatrix}")
print(f"Classification Report:\n{lrreport}")


# TESTING
# Train Altered-Hyperparameter Simple Logistic Regression Model
logres_alt_simple = LogisticRegressionSimplified(alpha=0.025, iterations=7500)
print("\n\nAltered Simplified Logistic Regression Model Training...")
logres_alt_simple.fit(X_train, y_train)
print("Altered Simplified Logistic Regression Model Training Complete")
print("\n\nPredicting from Trained Altered Simplified Logistic Regression Model")
y_pred_alt_simple = logres_alt_simple.predict(X_val)
print("Altered Simplified Logistic Regression Model Predictions Complete")


print("\n\nEvaluating Altered Simplified Logistic Regression Model\n")
print("Altered Simplified Logistic Regression Model Evaluation:")
lraccuracy = accuracy_score(y_val, y_pred_alt_simple)
lrbalancedaccuracy = balanced_accuracy_score(y_val, y_pred_alt_simple)
lrconfusionmatrix = confusion_matrix(y_val, y_pred_alt_simple)
lrreport = classification_report(y_val, y_pred_alt_simple)

print(f"Accuracy: {lraccuracy:.4f}")
print(f"Balanced Accuracy: {lrbalancedaccuracy:.4f}")
print(f"Confusion Matrix:\n{lrconfusionmatrix}")
print(f"Classification Report:\n{lrreport}")


# SVM ANALYSIS
svm = SVC(kernel='rbf', C=200, gamma=0.01)
print("\n\nSVM Model Training...")
svm.fit(X_train, y_train)
print("SVM Training Complete")
print("\n\nPredicting from SVM Model")
svm_pred = svm.predict(X_val)
print("SVM Predictions Complete")

print("\n\nEvaluating SVM Model\n")
print("SVM Model Evaluation:")
svm_accuracy = accuracy_score(y_val, svm_pred)
svm_balancedaccuracy = balanced_accuracy_score(y_val, svm_pred)
svm_confusionmatrix = confusion_matrix(y_val, svm_pred)
svm_report = classification_report(y_val, svm_pred)

print(f"Accuracy: {svm_accuracy:.4f}")
print(f"Balanced Accuracy: {svm_balancedaccuracy:.4f}")
print(f"Confusion Matrix:\n{svm_confusionmatrix}")
print(f"Classification Report:\n{svm_report}")


# PREDICTING ON KAGGLE TEST DATA (LOGISTIC REGRESSION)
# PATH TO KAGGLE TEST DATA IS DEFINED HERE
# Loads files for Kaggle test data
kaggle_test_audio = glob(f'{BASE_PATH}/data/test/*.au')

# Creates Kaggle Test Features
kaggle_test_features = []

# Load Kaggle test data into Librosa and Extract Features
print("\n\nProcessing Kaggle Test Data...")
for file in kaggle_test_audio:
    y, sr = librosa.load(file, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc = num_mfcc)
    mfcc_mean = mfcc.mean(axis=1)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = chroma.mean(axis=1)
    chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_cqt_mean = chroma_cqt.mean(axis=1)
    chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr)
    chroma_cens_mean = chroma_cens.mean(axis=1)
    spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    spec_contrast_mean = spec_contrast.mean(axis=1)
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean = zcr.mean()
    rms = librosa.feature.rms(y=y)
    rms_mean = rms.mean()
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    spectral_rolloff_mean = spectral_rolloff.mean()
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_centroid_mean = spectral_centroid.mean()
    kaggle_test_features.append({
                     **{f'mfcc_{i+1}': mfcc_mean[i] for i in range(len(mfcc_mean))},
                     **{f'chroma_{i+1}': chroma_mean[i] for i in range(len(chroma_mean))},
                     **{f'chromaCQT_{i+1}': chroma_cqt_mean[i] for i in range(len(chroma_cqt_mean))},
                     **{f'chromaCENS_{i+1}': chroma_cens_mean[i] for i in range(len(chroma_cens_mean))},
                     **{f'spec_contrast_{i+1}': spec_contrast_mean[i] for i in range(len(spec_contrast_mean))},
                     'zcr': zcr_mean,
                     'rms': rms_mean,
                     'tempo': float(np.squeeze(tempo)),
                     'spectral_rolloff': spectral_rolloff_mean,
                     'spectral_centroid': spectral_centroid_mean
                     })


# Creates a dataframe from the extracted features
kaggle_test_df = pd.DataFrame(kaggle_test_features)

# Normalizes Kaggle Test (features) dataset
kaggle_test_norm = normalizer.transform(kaggle_test_df)

# Scales Kaggle Test (features) dataset
kaggle_test_norm_scale = scaler.transform(kaggle_test_norm)

# Performs PCA on Kaggle Test (features) dataset
kaggle_test_norm_scale_pca = pca.transform(kaggle_test_norm_scale)
X_test = pd.DataFrame(kaggle_test_norm_scale_pca)

# Predict Kaggle Test Data
print("Predicting from Kaggle Test Data...")
y_final_pred = svm.predict(X_test)

# Creates CSV for Kaggle Submission
print("Creating Kaggle CSV")
kaggle_test_audio = [item.split('/')[-1] for item in kaggle_test_audio]
kaggle_test_predictions = pd.DataFrame({'id': kaggle_test_audio, 'class': encoder.inverse_transform(y_final_pred)})
kaggle_test_predictions.to_csv('TheOutlierDetectivesPredictions.csv', index=False)
print("Kaggle Test Complete")


# EVALUATION OF OTHER ML METHODS

# GBC ANALYSIS
gbc = GradientBoostingClassifier()
print("\n\nGBC Model Training...")
gbc.fit(X_train, y_train)
print("GBC Training Complete")
print("\n\nPrediciting from GBC Model")
gbc_pred = gbc.predict(X_val)
print("GBC Predictions Complete")

print("\n\nEvaluating GBC Model\n")
print("GBC Model Evaluation:")
gbc_accuracy = accuracy_score(y_val, gbc_pred)
gbc_balancedaccuracy = balanced_accuracy_score(y_val, gbc_pred)
gbc_confusionmatrix = confusion_matrix(y_val, gbc_pred)
gbc_report = classification_report(y_val, gbc_pred)

print(f"Accuracy: {gbc_accuracy:.4f}")
print(f"Balanced Accuracy: {gbc_balancedaccuracy:.4f}")
print(f"Confusion Matrix:\n{gbc_confusionmatrix}")
print(f"Classification Report:\n{gbc_report}")


# GNB ANALYSIS
gnb = GaussianNB()
print("\n\nGNB Model Training...")
gnb.fit(X_train, y_train)
print("GNB Training Complete")
print("\n\nPredicting from GNB Model")
gnb_pred = gnb.predict(X_val)
print("GNB Predictions Complete")

print("\n\nEvaluating GNB Model\n")
print("GNB Model Evaluation:")
gnb_accuracy = accuracy_score(y_val, gnb_pred)
gnb_balancedaccuracy = balanced_accuracy_score(y_val, gnb_pred)
gnb_confusionmatrix = confusion_matrix(y_val, gnb_pred)
gnb_report = classification_report(y_val, gnb_pred)

print(f"Accuracy: {gnb_accuracy:.4f}")
print(f"Balanced Accuracy: {gnb_balancedaccuracy:.4f}")
print(f"Confusion Matrix:\n{gnb_confusionmatrix}")
print(f"Classification Report:\n{gnb_report}")


# RANDOM FOREST ANALYSIS
rf = RandomForestClassifier(n_estimators=100, random_state=42)
print("\n\nRandom Forest Model Training...")
rf.fit(X_train, y_train)
print("Random Forest Training Complete")
print("\n\nPredicting from Random Forest Model")
rf_pred = rf.predict(X_val)
print("Random Forest Predictions Complete")

print("\n\nEvaluating Random Forest Model\n")
print("Random Forest Model Evaluation:")
rf_accuracy = accuracy_score(y_val, rf_pred)
rf_balancedaccuracy = balanced_accuracy_score(y_val, rf_pred)
rf_confusionmatrix = confusion_matrix(y_val, rf_pred)
rf_report = classification_report(y_val, rf_pred)

print(f"Accuracy: {rf_accuracy:.4f}")
print(f"Balanced Accuracy: {rf_balancedaccuracy:.4f}")
print(f"Confusion Matrix:\n{rf_confusionmatrix}")
print(f"Classification Report:\n{rf_report}")


"""

#########################################
### Hyperparameter Tuning Experiments ###
#########################################

# Using GridSearchCV to find best hyperparameters for 
# all models using max 4 variations per hyperparameter

# Logistic Regression
class SklearnLogisticRegressionWrapper:
    def __init__(self, alpha=0.2, iterations=1000, lambdaparam=0.01):
        self.alpha = alpha
        self.iterations = iterations
        self.lambdaparam = lambdaparam
        self.model = None

    def fit(self, X, y):
        self.model = LogisticRegression(alpha=self.alpha, iterations=self.iterations, lambdaparam=self.lambdaparam)
        X = X.values if hasattr(X, 'values') else X
        self.model.fit(X, y)
        return self

    def predict(self, X):
        if self.model is None:
            raise Exception("Model has not been fitted yet.")
        X = X.values if hasattr(X, 'values') else X
        return self.model.predict(X)
    
    def get_params(self, deep=True):
        return {'alpha': self.alpha, 'iterations': self.iterations, 'lambdaparam': self.lambdaparam}
    
    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self
    
param_grid_lr = {'alpha': [0.0001, 0.001, 0.01, 0.1],
    'iterations': [500, 1000, 2500, 5000],
    'lambdaparam': [0.0001, 0.01, 0.1, 1.0]}

grid_search_lr = GridSearchCV(SklearnLogisticRegressionWrapper(), param_grid_lr, cv=3, scoring='balanced_accuracy', n_jobs=-1, verbose=1, return_train_score=True)
grid_search_lr.fit(X_train, y_train)
print("Best Hyperparameters for Logistic Regression:")
print(grid_search_lr.best_params_)
print("Best cross-validation score: {:.4f}".format(grid_search_lr.best_score_))

print("\n\nPredicting from Logistic Regression Model with Best Hyperparameters")
lr_pred = grid_search_lr.predict(X_val)
print("Logistic Regression Model Predictions Complete")
# Evaluate Logistic Regression with Best Hyperparameters
lr_balancedaccuracy = balanced_accuracy_score(y_val, lr_pred)
print(f"Balanced Accuracy with Best Hyperparameters: {lr_balancedaccuracy:.4f}")

#########################################

# Support Vector Machine
param_grid_svm = {'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'C': [0.1, 25, 50, 100],
    'gamma': [0.001, 0.01, 0.1, 1]}

grid_search_svm = GridSearchCV(SVC(), param_grid_svm, cv=3, scoring='balanced_accuracy', n_jobs=-1, verbose=1, return_train_score=True)
grid_search_svm.fit(X_train, y_train)
print("\n\nBest Hyperparameters for SVM:")
print(grid_search_svm.best_params_)
print("Best cross-validation score: {:.4f}".format(grid_search_svm.best_score_))

print("\n\nPredicting from SVM Model with Best Hyperparameters")
svm_pred = grid_search_svm.predict(X_val)
print("SVM Model Predictions Complete")
# Evaluate SVM with Best Hyperparameters
svm_balancedaccuracy = balanced_accuracy_score(y_val, svm_pred)
print(f"Balanced Accuracy with Best Hyperparameters: {svm_balancedaccuracy:.4f}")

#########################################

# Gradient Boosting Classifier
param_grid_gbc = {'n_estimators': [100, 250, 500, 1000],
    'learning_rate': [0.01, 0.1, 0.2, 0.3],
    'max_depth': [3, 5, 8, 10]}

grid_search_gbc = GridSearchCV(GradientBoostingClassifier(), param_grid_gbc, cv=3, scoring='balanced_accuracy', n_jobs=-1, verbose=1, return_train_score=True)
grid_search_gbc.fit(X_train, y_train)
print("\n\nBest Hyperparameters for GBC:")
print(grid_search_gbc.best_params_)
print("Best cross-validation score: {:.4f}".format(grid_search_gbc.best_score_))

print("\n\nPredicting from GBC Model with Best Hyperparameters")
gbc_pred = grid_search_gbc.predict(X_val)
print("GBC Model Predictions Complete")
# Evaluate GBC with Best Hyperparameters
gbc_balancedaccuracy = balanced_accuracy_score(y_val, gbc_pred)
print(f"Balanced Accuracy with Best Hyperparameters: {gbc_balancedaccuracy:.4f}")

#########################################

# Random Forest Classifier
param_grid_rf = {'n_estimators': [100, 250, 500, 1000],
    'max_depth': [3, 8, 14, 20]}

grid_search_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf, cv=3, scoring='balanced_accuracy', n_jobs=-1, verbose=1, return_train_score=True)
grid_search_rf.fit(X_train, y_train)
print("\n\nBest Hyperparameters for Random Forest:")
print(grid_search_rf.best_params_)
print("Best cross-validation score: {:.4f}".format(grid_search_rf.best_score_)) 

print("\n\nPredicting from Random Forest Model with Best Hyperparameters")
rf_pred = grid_search_rf.predict(X_val)
print("Random Forest Model Predictions Complete")
# Evaluate Random Forest with Best Hyperparameters
rf_balancedaccuracy = balanced_accuracy_score(y_val, rf_pred)
print(f"Balanced Accuracy with Best Hyperparameters: {rf_balancedaccuracy:.4f}")

"""

# PLOT 1 — Model Accuracy Comparison (all 5 models)
# ─────────────────────────────────────────────────────────────────────────────

model_names = ['Logistic\nRegression', 'SVM', 'Gradient\nBoosting', 'Naive\nBayes', 'Random\nForest']
bal_accuracies = [
    balanced_accuracy_score(y_val, y_pred),
    balanced_accuracy_score(y_val, svm_pred),
    balanced_accuracy_score(y_val, gbc_pred),
    balanced_accuracy_score(y_val, gnb_pred),
    balanced_accuracy_score(y_val, rf_pred)
]

fig, ax = plt.subplots(figsize=(9, 5))
fig.patch.set_facecolor('#0b0f1a')
ax.set_facecolor('#111520')

bar_colors = ['#5ee7b0', '#3b9eff', '#f7b731', '#ff6b6b', '#a29bfe']
bars = ax.bar(model_names, bal_accuracies, color=bar_colors, width=0.5, edgecolor='none', zorder=3)

for bar, val in zip(bars, bal_accuracies):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.008,
            f'{val:.2%}', ha='center', va='bottom',
            color='#c0c8e0', fontsize=11, fontweight='bold', fontfamily='monospace')

ax.set_title('Balanced Accuracy — Model Comparison', color='#e8eaf0',
             fontsize=13, fontweight='bold', pad=14)
ax.set_ylabel('Balanced Accuracy', color='#5a6080', fontsize=10)
ax.set_ylim(0, max(bal_accuracies) + 0.12)
ax.tick_params(colors='#7a8099', labelsize=10)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.0%}'))
ax.spines[:].set_visible(False)
ax.grid(axis='y', color='#1e2535', linewidth=0.8, zorder=0)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig('plot1_model_comparison.png', dpi=180, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.show()
print("Saved: plot1_model_comparison.png")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 2 — Confusion Matrix (Best Model: Logistic Regression)
# ─────────────────────────────────────────────────────────────────────────────

cm = confusion_matrix(y_val, svm_pred)
genre_labels = encoder.classes_

fig, ax = plt.subplots(figsize=(10, 8))
fig.patch.set_facecolor('#0b0f1a')
ax.set_facecolor('#111520')

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=genre_labels)
disp.plot(ax=ax, colorbar=False, cmap='Blues')

acc = accuracy_score(y_val, svm_pred)
bal_acc = balanced_accuracy_score(y_val, svm_pred)

ax.set_title(
    f'Confusion Matrix — SVM\nAccuracy: {acc:.2%}  |  Balanced Accuracy: {bal_acc:.2%}',
    color='#e8eaf0', fontsize=12, fontweight='bold', pad=12
)
ax.tick_params(colors='#7a8099', labelsize=9)
ax.xaxis.label.set_color('#7a8099')
ax.yaxis.label.set_color('#7a8099')
ax.spines[:].set_color('#1e2535')
plt.xticks(rotation=45, ha='right')

for text in disp.text_.ravel():
    text.set_color('white')
    text.set_fontsize(9)
    text.set_fontweight('bold')

plt.tight_layout()
plt.savefig('plot2_confusion_matrix.png', dpi=180, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.show()
print("Saved: plot2_confusion_matrix.png")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 3 — Genre Distribution (samples per class)
# ─────────────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor('#0b0f1a')
ax.set_facecolor('#111520')

genre_counts = df['genre'].value_counts().sort_index()
genre_colors = ['#5ee7b0', '#3b9eff', '#f7b731', '#ff6b6b', '#a29bfe',
                '#fd79a8', '#00cec9', '#e17055', '#74b9ff', '#55efc4']

bars = ax.bar(genre_counts.index, genre_counts.values,
              color=genre_colors, width=0.6, edgecolor='none', zorder=3)

for bar, val in zip(bars, genre_counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            str(val), ha='center', va='bottom',
            color='#c0c8e0', fontsize=10, fontweight='bold', fontfamily='monospace')

ax.set_title('Training Samples per Genre', color='#e8eaf0',
             fontsize=13, fontweight='bold', pad=14)
ax.set_ylabel('Number of Audio Files', color='#5a6080', fontsize=10)
ax.tick_params(colors='#7a8099', labelsize=10)
plt.xticks(rotation=30, ha='right')
ax.spines[:].set_visible(False)
ax.grid(axis='y', color='#1e2535', linewidth=0.8, zorder=0)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig('plot3_genre_distribution.png', dpi=180, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.show()
print("Saved: plot3_genre_distribution.png")

print("\nAll 3 plots saved in your project folder.")