import pickle as cPickle
import numpy as np
from scipy.io.wavfile import read
from featureextraction import extract_features
import os
from report import write_report
from captura_de_voz import captura_voz
import re
import shutil
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
        self.lista_nomes=[]
                
        self.path=""
        self.nome=""
        
        
        
        

    def testa_voz(self, take):
        if take==1:
            self.testa_unico() 
        elif take==0:
            self.testa_sample()
        print(f"Acuracia: {100*self.correct/int(self.total_sample)}%")    
                
                
                
    def testa_unico(self):
        self.total_sample+=1
        self.nome = input("Nome\n")
        captura_voz(self.nome, 1)
        self.path=f"voice-{self.nome}-0.wav"
        self.teste_geral()
       
      


    def testa_sample(self):
        for self.path in self.file_paths:
                self.total_sample += 1.0
                self.path = self.path.strip()
                self.teste_geral()
             
        
        
        
    def teste_geral(self):
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
            self.lista_nomes=["antonio", "bandeira", "betim", "luca", "patrick", "viktor"]
            indice_soma=np.argmax(self.lista_somas)
            distancias_para_cada_modelo = [(self.speakers[i], self.log_likelihood[i]) for i in range(len(self.models))]
            lista_tuplas = [(self.lista_somas[i], self.lista_nomes[i]) for i in range (len(self.lista_nomes))]
            
            if self.speakers[winner].split("-")[0]== self.lista_nomes[indice_soma]:
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
                    "Distancia somas": f"({self.lista_somas[indice_soma]}, {self.lista_nomes[indice_soma]})",
                    "Distancias para cada modelo": distancias_para_cada_modelo,  # Use a lista de tuplas
                    "Distancia total": lista_tuplas
                }
                
                
                # Nome do relatório
            relatorio_nome = f"relatorios/relatorio_{self.path.split('.')[0]}.txt"
                
                # Chame a função para escrever o relatório
            write_report(relatorio_nome, report_info=relatorio_info)
            
            if self.total_sample==1:
                if self.nome in self.lista_nomes:
                    self.adicionar_ao_sample()
                else:
                    os.remove(os.path.join(self.source, self.path))
               
                    
                
                
    def adicionar_ao_sample(self):
        caminho=os.path.join(self.source, self.path)
        caminho_destino_completo=f"voice/voice-{self.nome}-0.wav"
        shutil.move(caminho, caminho_destino_completo)
        
        x=os.listdir(f"voice/{self.nome}-003")
        x=len(x)
        caminho=self.update_string(caminho_destino_completo, x)
        os.rename(caminho_destino_completo, caminho)
        caminho_final=f"voice/{self.nome}-003/voice-{self.nome}-{x}.wav"
        shutil.move(caminho, caminho_final)
        
        self.reescreve_data_path()
        
    def reescreve_data_path(self):
        arquivos = []

        for i in self.lista_nomes:
            pasta = f"voice/{i}-003"
            arquivos_na_pasta = [os.path.join(pasta, f) for f in os.listdir(pasta) if os.path.isfile(os.path.join(pasta, f))]
            arquivos.extend(arquivos_na_pasta)

        # Função para extrair o número do nome do arquivo
        def extrair_numero(nome_arquivo):
            match = re.search(r'-(\d+)\.wav$', nome_arquivo)
            if match:
                return int(match.group(1))
            return 0

        # Ordenar os arquivos numericamente
        arquivos_ordenados = sorted(arquivos, key=extrair_numero)

        # Agora arquivos_ordenados contém todos os caminhos dos arquivos, ordenados numericamente
        with open("trainingDataPath2.txt", 'w') as arquivo:
            for linha in arquivos_ordenados:
                arquivo.write(linha + '\n')

                            
       
    def update_string(self, s, x):
        result = re.sub(r'(\d+)\.wav', lambda m: str(int(m.group(1)) + x) + '.wav', s)
        return result

        