import pickle as cPickle
import numpy as np
from scipy.io.wavfile import read
from featureextraction import extract_features
import os
from report import write_report
from captura_de_voz import captura_voz

class tester():
    def __init__(self) -> None:
        self.source = "SampleData/"
        self.modelpath = "speakers_models/"
        self.test_file = "testSamplePath.txt"
        
        self.file_paths = open(self.test_file, 'r')
        
        
        self.gmm_files = [os.path.join(self.modelpath, fname) for fname in os.listdir(self.modelpath) if fname.endswith('.gmm')]
        self.models = [cPickle.load(open(fname, 'rb')) for fname in self.gmm_files]
        self.speakers = [fname.split("/")[-1].split(".gmm")[0] for fname in self.gmm_files]
        
        
        self.correct=0
        self.total_sample=0
        self.count=0
        self.indice=0
        
        self.lista_somas=[]
        self.log_likelihood=[]
        self.vector=[]
                
        self.path=""
        self.nome=""
        
        
        
        

    def testa_voz(self, take):
        if take==1:
            self.testa_unico()
           
            
        elif take==0:
            self.testa_sample()
            
                
                
  
                
                
                
    def testa_unico(self):
        self.total_sample+=1
        self.nome = input("Nome\n")
        captura_voz(self.nome, 1)
        self.path=f"voice-{self.nome}-0.wav"
        print("Testing Audio : ", self.path)
        sr, audio = read(os.path.join(self.path))
        self.vector = extract_features(audio, sr)
        self.log_likelihood = np.zeros(len(self.models))
        self.lista_somas=np.zeros(6)
        self.compara_sample()
        self.prepara_relatorio()



    def testa_sample(self):
        for self.path in self.file_paths:
                self.total_sample += 1.0
                self.path = self.path.strip()
                print("Testing Audio : ", self.path)
                sr, audio = read(os.path.join(self.source, self.path))
                self.vector = extract_features(audio, sr)
                #Adiciona o vetor das caracteristicas a variavel vector
                self.log_likelihood = np.zeros(len(self.models))
                #Inicializa a lista de semelhança do tamanho da lista de modelos com zeros
                self.lista_somas=np.zeros(6)
                self.compara_sample()
                self.prepara_relatorio()
        
        
        
        
    def compara_sample(self):
      
        #Inicializa a lista de semelhança do tamanho da lista de modelos com zeros
        for i in range(len(self.models)):
                gmm = self.models[i]  
                scores = np.array(gmm.score(self.vector))
                self.log_likelihood[i] = scores.sum()
                self.lista_somas[self.indice]=self.lista_somas[self.indice]+self.log_likelihood[i]
                self.count+=1
                if self.count ==3:
                    self.indice+=1
                    self.count=0
        self.indice=0
                    
                    
                    
    def prepara_relatorio(self):

            winner = np.argmax(self.log_likelihood)
            lista_nomes=["antonio", "bandeira", "betim", "luca", "patrick", "viktor"]
            indice_soma=np.argmax(self.lista_somas[:-1])
            distancias_para_cada_modelo = [(self.speakers[i], self.log_likelihood[i]) for i in range(len(self.models))]
            lista_tuplas = [(self.lista_somas[i], lista_nomes[i]) for i in range (len(lista_nomes))]
            if self.speakers[winner].split("-")[0]== lista_nomes[indice_soma]:
                detected_speaker = self.speakers[winner].split("-")[0]
            else:
                detected_speaker = "Inconclusivo"
                
                
            print(f"\tdetected as - {detected_speaker}")
            #Detecta o chute do programa
            
            true_speaker = self.path.split('-')[1]  # Extrair o locutor real do nome do arquivo
            if detected_speaker == true_speaker:
                self.correct += 1
                
            distancias_para_cada_modelo = [(self.speakers[i], self.log_likelihood[i]) for i in range(len(self.models))]
            indice_soma=np.argmax(self.lista_somas)
            
                
            # Criar informações para o relatório
            relatorio_info = {
                    "Amostra testada": self.path,
                    "Chute do programa": detected_speaker,
                    "Distancia individual": f"({self.log_likelihood[winner]}, {self.speakers[winner]})",
                    "Distancia somas": f"({self.lista_somas[indice_soma]}, {lista_nomes[indice_soma]})",
                    "Distancias para cada modelo": distancias_para_cada_modelo,  # Use a lista de tuplas
                    "Distancia total": lista_tuplas
                }
                
                
                # Nome do relatório
            relatorio_nome = f"relatorios/relatorio_{self.path.split('.')[0]}.txt"
                
                # Chame a função para escrever o relatório
            write_report(relatorio_nome, report_info=relatorio_info)
                
            #verifica se o chte do programa foi correto

        