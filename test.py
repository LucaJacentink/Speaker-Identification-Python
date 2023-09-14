import pickle as cPickle
import numpy as np
from scipy.io.wavfile import read
from sklearn.mixture import GaussianMixture as Gmm
from featureextraction import extract_features
import os
import warnings
import time
from report import write_report


source = "SampleData/"

# path where training speakers will be saved
modelpath = "Speakers_models/"

gmm_files = [os.path.join(modelpath, fname) for fname in os.listdir(modelpath) if fname.endswith('.gmm')]

# Load the Gaussian gender Models
models = [cPickle.load(open(fname, 'rb')) for fname in gmm_files]
speakers = [fname.split("/")[-1].split(".gmm")[0] for fname in gmm_files]

correct = 0
total_sample = 0.0

print("Do you want to Test a Single Audio: Press '1' or The complete Test Audio Sample: Press '0' ?")
take = int(input().strip())

if take == 1:
    print("Enter the File name from Test Audio Sample Collection :")
    path = input().strip()
    print("Testing Audio : ", path)
    sr, audio = read(os.path.join(source, path))
    vector = extract_features(audio, sr)

    log_likelihood = np.zeros(len(models))

    for i in range(len(models)):
        gmm = models[i]  # checking with each model one by one
        scores = np.array(gmm.score(vector))
        log_likelihood[i] = scores.sum()

    winner = np.argmax(log_likelihood)
    detected_speaker = speakers[winner]
    print("\tdetected as - ", detected_speaker)

    true_speaker = path.split('-')[1]  # Extrair o locutor real do nome do arquivo
    if detected_speaker == true_speaker:
        correct += 1

    print("Score da amostra predita:", log_likelihood[winner])  # Adicionando a saída do score
    time.sleep(1.0)

elif take == 0:
    test_file = "testSamplePath.txt"
    file_paths = open(test_file, 'r')

    # Read the test directory and get the list of test audio files
    for path in file_paths:
        total_sample += 1.0
        path = path.strip()
        print("Testing Audio : ", path)
        sr, audio = read(os.path.join(source, path))
        vector = extract_features(audio, sr)

        log_likelihood = np.zeros(len(models))

        for i in range(len(models)):
            gmm = models[i]  # checking with each model one by one
            scores = np.array(gmm.score(vector))
            log_likelihood[i] = scores.sum()

        winner = np.argmax(log_likelihood)
        detected_speaker = speakers[winner]
        print("\tdetected as - ", detected_speaker)

        true_speaker = path.split('-')[1]
        if detected_speaker == true_speaker:
            correct += 1

        print("Score da amostra predita:", log_likelihood[winner])  # Adicionando a saída do score
        time.sleep(1.0)
        
       
        relatorio_info = {
            "Amostra testada": path,
            "Chute do programa": detected_speaker,
            "Distancia individual": log_likelihood[winner],
            "Distancias para cada modelo": log_likelihood
        }


        relatorio_nome = f"relatorio_{path.split('.')[0]}.txt"
        write_report(relatorio_nome, report_info=relatorio_info)
accuracy = (correct / total_sample) * 100 if total_sample > 0 else 0

print("The Accuracy Percentage for the current testing Performance with MFCC + GMM is : ", accuracy, "%")