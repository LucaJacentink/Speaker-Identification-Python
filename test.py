import pickle as cPickle
import numpy as np
from scipy.io.wavfile import read
from featureextraction import extract_features
import os
import time
from report import write_report
from captura_de_voz import captura_voz
def testa_voz(take):
    source = "SampleData/"
    modelpath = "Speakers_models/"
    #Inicializa os caminhos de destino e entrada de data

    gmm_files = [os.path.join(modelpath, fname) for fname in os.listdir(modelpath) if fname.endswith('.gmm')]
    #Cria uma lista com os modelos gaussianos

    models = [cPickle.load(open(fname, 'rb')) for fname in gmm_files]
    speakers = [fname.split("/")[-1].split(".gmm")[0] for fname in gmm_files]

    #Cria duas listas, separando entre modelos e quem é o falante do modelo

    correct = 0
    total_sample = 0.0
    count=0
    


    #Se take==1 testa um audio apenas, se take==0 testa o SampleData
    if take == 1:
        total_sample+=1
        nome = input("Nome\n")
        captura_voz(nome, 1)
        path=f"voice-{nome}-0.wav"
        print("Testing Audio : ", path)
        sr, audio = read(os.path.join(path))
        vector = extract_features(audio, sr)
        #Adiciona o vetor das caracteristicas a variavel vector
        log_likelihood = np.zeros(len(models))
        lista_somas=np.zeros(6)
        indice=0
        #Inicializa a lista de semelhança do tamanho da lista de modelos com zeros
        for i in range(len(models)):
            gmm = models[i]  
            scores = np.array(gmm.score(vector))
            log_likelihood[i] = scores.sum()
            lista_somas[indice]=lista_somas[indice]+log_likelihood[i]
            count+=1
            if count ==3:
                indice+=1
                count=0
            
            #adiciona a lista de semelhança a distancia do vetor a cada modelo

        winner = np.argmax(log_likelihood)
        detected_speaker = speakers[winner].split("-")[0]
        print(f"\tdetected as - {detected_speaker}")
        #Detecta o chute do programa
        
        true_speaker = path.split('-')[1]  # Extrair o locutor real do nome do arquivo
        if detected_speaker == true_speaker:
            correct += 1
        lista_nomes=["antonio", "bandeira", "betim", "luca", "patrick", "viktor"]
        indice_soma=np.argmax(lista_somas)
        distancias_para_cada_modelo = [(speakers[i], log_likelihood[i]) for i in range(len(models))]
        lista_tuplas = [(lista_somas[i], lista_nomes[i]) for i in range (len(lista_nomes))]
            
         # Criar informações para o relatório
        relatorio_info = {
                "Amostra testada": path,
                "Chute do programa": detected_speaker,
                "Distancia individual": f"({log_likelihood[winner]}, {speakers[winner]})",
                "Distancia somas": f"({lista_somas[indice_soma]}, {lista_nomes[indice_soma]})",
                "Distancias para cada modelo": distancias_para_cada_modelo,  # Use a lista de tuplas
                "Distancia total": lista_tuplas
            }
            
            
            # Nome do relatório
        relatorio_nome = f"relatorio_{path.split('.')[0]}.txt"
            
            # Chame a função para escrever o relatório
        write_report(relatorio_nome, report_info=relatorio_info)
            
        #verifica se o chte do programa foi correto



    elif take == 0:
        test_file = "testSamplePath.txt"
        file_paths = open(test_file, 'r')

        # Lê o diretorio de teste
        for path in file_paths:
            total_sample += 1.0
            path = path.strip()
            print("Testing Audio : ", path)
            sr, audio = read(os.path.join(source, path))
            vector = extract_features(audio, sr)
            #Adiciona o vetor das caracteristicas a variavel vector

            log_likelihood = np.zeros(len(models))
            #Inicializa a lista de semelhança do tamanho da lista de modelos com zeros
            lista_somas=np.zeros(7)
            indice=0
            for i in range(len(models)):
                gmm = models[i]  
                scores = np.array(gmm.score(vector))
                log_likelihood[i] = scores.sum()
                #adiciona a lista de semelhança a distancia do vetor a cada modelo
                lista_somas[indice]=lista_somas[indice]+log_likelihood[i]
                count+=1
                if count ==3:
                    indice+=1
                    count=0
            winner = np.argmax(log_likelihood)
            detected_speaker = speakers[winner].split("-")[0]
            print("\tdetected as - ", detected_speaker)
            #Detecta o chute do programa

            true_speaker = path.split('-')[1]   # Extrair o locutor real do nome do arquivo
            if detected_speaker == true_speaker:
                correct += 1
            #verifica se o chte do programa foi correto

            print("Score da amostra predita:", log_likelihood[winner])  # Adicionando a saída do score
            lista_tuplas=[]
            lista_nomes=["antonio", "bandeira", "betim", "luca", "patrick", "viktor"]
            for i in range(len(lista_somas)-1):
                lista_tuplas.append((lista_somas[i], lista_nomes[i]))
                
            distancias_para_cada_modelo = [(speakers[i], log_likelihood[i]) for i in range(len(models))]
            indice_soma=np.argmax(lista_somas[:-1])
            
            
            # Criar informações para o relatório
            relatorio_info = {
                "Amostra testada": path,
                "Chute do programa": detected_speaker,
                "Distancia individual": f"({log_likelihood[winner]}, {speakers[winner]})",
                "Distancia somas": f"({lista_somas[indice_soma]}, {lista_nomes[indice_soma]})",
                "Distancias para cada modelo": distancias_para_cada_modelo,  # Use a lista de tuplas
                "Distancia total": lista_tuplas
            }
            
            # Nome do relatório
            relatorio_nome = f"relatorios/relatorio_{path.split('.')[0]}.txt"
            
            # Chame a função para escrever o relatório
            write_report(relatorio_nome, report_info=relatorio_info)
            
            time.sleep(1.0)
    accuracy = (correct / total_sample) * 100 if total_sample > 0 else 0

    print(f"Acurácia:{accuracy}")