
from test import testa_voz

def main():
    take = int(input("Teste, 1 para audio gravado, 0 para lista de audios?\n").strip())
    testa_voz(take)
if __name__=="__main__":
    main()  