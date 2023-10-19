import numpy as np
import pickle as cPickle
from scipy.io.wavfile import read
from sklearn.mixture import GaussianMixture as Gmm
from featureextraction import extract_features



class voz:
    def __init__(self, destino, train_file, limite):
        self.destino=destino
        self.train_file=train_file
        self.file_paths = open(self.train_file, 'r')
        self.features=[]
        self.count=1
        self.vector=" "
        self.limite=limite
        self.lista_nomes=[]
    def treinar(self):
        self.features=np.asarray(())
        for path in self.file_paths:
                path=path.strip()
                print(path)
                
                sr, audio = read(path)
                
                self.vector = extract_features(audio, sr)
                if self.features.size == 0:
                        self.features = self.vector
                else: 
                        self.features = np.vstack((self.features, self.vector))
            
                if self.count==self.limite:
                        gmm = Gmm(n_components=16, max_iter=200, covariance_type='diag', n_init=3)
                        gmm.fit(self.features)
                        picklefile = path.split("/")[1]+".gmm"
                        cPickle.dump(gmm, open(self.destino + picklefile, 'wb'))
                        print('+ modeling completed for speaker:', picklefile, " with data point = ", self.features.shape)
                        self.features = np.asarray(())
                        self.count = 0
                self.count = self.count + 1
            
        
            
        

        
        
