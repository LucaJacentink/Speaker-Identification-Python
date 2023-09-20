from typing import Dict



def write_report(report_name, report_info):
    try:
        with open(report_name, 'w') as report_file:
            for key, value in report_info.items():
                if key == "Distancias para cada modelo":
                    report_file.write(f"{key}:\n")
                    for nome, distancia in value:
                        report_file.write(f"{nome}: {distancia}\n")
                else:
                    report_file.write(f"{key}: {value}\n")
        print(f"Relatório '{report_name}' criado com sucesso.")
    except Exception as e:
        print(f"Erro ao criar o relatório: {str(e)}")
