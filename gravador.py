import speaker_verification_toolkit.tools as svt
import librosa
import numpy
from captura_de_voz import captura_voz


Nome=input("Digite seu nome: ")
gravacoes=int(input("Escolha o número de gravações: "))
captura_voz(Nome, gravacoes)



