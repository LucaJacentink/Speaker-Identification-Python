
from vozes import voz
from test import tester

def main():
    pergunta=int(input("Voce deseja testar ou treinar o algoritimo? 1 para testar e 0 para treinar\n").strip())
    if pergunta==0:
        tipo_treino=int(input("Você deseja o treino padrão? 1 para sim e 0 para não\n").strip())
        if tipo_treino==1:
            treino=voz("Speakers_models/", "trainingDataPath.txt", 6)
            treino.treinar()
            treino=voz("Speakers_models/", "trainingDataPath2.txt", 12)
            treino.treinar()
        elif tipo_treino==0:
            destino=input("Diga o arquivo de destino")
            training_path=input("Diga o arquivo de treino")
            limite=int(input("Diga o limite de count"))
            treino=voz(destino, training_path, limite)
            treino.treinar()
        else:
            print("Valor invalido")
            
            
    elif pergunta==1:
        take = int(input("Teste, 1 para audio gravado, 0 para lista de audios?\n").strip())
        teste=tester()
        teste.testa_voz(take)
    else:
        print("Valor invalido")
        
        
        
        
        
        
if __name__=="__main__":
    main()  