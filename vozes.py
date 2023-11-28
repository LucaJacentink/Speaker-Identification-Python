import numpy as np
import pickle as cPickle
from scipy.io.wavfile import read
from sklearn.mixture import GaussianMixture as Gmm
from featureextraction import extract_features
import os



class voz:
        def __init__(self, destino, train_file):
                self.destino=destino
                self.train_file=train_file
                self.file_paths = open(self.train_file, 'r')
                self.features=[]
                self.count=1
                self.vector=" "
                self.lista_nomes=["antonio", "bandeira", "betim", "luca", "patrick", "viktor"]
        
        def obter_caminhos_arquivos(self):
                arquivos = []

                for nome in self.lista_nomes:
                        pasta = f"voice/{nome}-003"
                        arquivos_na_pasta = [os.path.join(pasta, f) for f in os.listdir(pasta) if os.path.isfile(os.path.join(pasta, f))]
                        arquivos.extend(arquivos_na_pasta)

                return arquivos

        def treinar(self):
                self.file_paths = self.obter_caminhos_arquivos()
                self.features = np.asarray(())
                self.count = 0
                limite_por_nome = len(self.file_paths) // len(self.lista_nomes)

                for path in self.file_paths:
                        path = path.strip()
                        print(path)

                        sr, audio = read(path)
                        vector = extract_features(audio, sr)

                        if self.features.size == 0:
                                self.features = vector
                        else:
                                self.features = np.vstack((self.features, vector))

                        if self.count == limite_por_nome:
                                nome_speaker = path.split("/")[1]
                                gmm = Gmm(n_components=16, max_iter=200, covariance_type='diag', n_init=3)
                                gmm.fit(self.features)
                                picklefile = f"{nome_speaker}.gmm"
                                cPickle.dump(gmm, open(os.path.join(self.destino, picklefile), 'wb'))
                                print('+ modeling completed for speaker:', picklefile, " with data point = ", self.features.shape)
                                self.features = np.asarray(())
                                self.count = 0

                self.count += 1
                                
        
        
            
        

        
        
