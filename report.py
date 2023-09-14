from typing import Dict



def write_report(path: str, report_info: dict) -> None:
    """Escreve o arquivo de relatório do experimento"""
    with open(f"relatorios/{path}", 'w') as relatorio:
        for key, value in report_info.items():
            relatorio.write(f'{key}:\n')
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    relatorio.write(f'\t{sub_key}: {sub_value:.2f}\n')
            else:
                relatorio.write(f'\t{value}\n')