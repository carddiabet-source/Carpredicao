import os
import subprocess
import sys
from flask import Flask, render_template, redirect, url_for, send_from_directory

app = Flask(__name__)

# Define o diretório onde os gráficos e o relatório são salvos.
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')

@app.route('/')
def index():
    """
    Renderiza a página principal do relatório.
    Procura pelo arquivo de relatório HTML gerado pelo script.
    """
    report_filename = 'relatorio_analise_completa.html'
    report_path = os.path.join(OUTPUT_DIR, report_filename)

    # Verifica se o relatório já existe para exibi-lo.
    if os.path.exists(report_path):
        with open(report_path, 'r', encoding='utf-8') as f:
            report_content = f.read()
    else:
        # Mensagem para exibir se o relatório ainda não foi gerado.
        report_content = """
        <div class='card' style='text-align: center; padding: 40px;'>
            <h2>Relatório ainda não gerado.</h2>
            <p>Clique no botão 'Atualizar Análise' para gerar o primeiro relatório.</p>
        </div>
        """

    return render_template('index.html', report_content=report_content)

@app.route('/atualizar')
def atualizar_dados():
    """
    Executa o script de análise 'cardio.py' para gerar um novo relatório.
    """
    print("Iniciando a execução do script de análise 'cardio.py'...")
    script_path = os.path.join(os.path.dirname(__file__), 'cardio.py')
    
    # Executa o script em um processo separado.
    subprocess.run([sys.executable, script_path], check=True)
    
    print("Script concluído. Redirecionando para a página principal.")
    return redirect(url_for('index'))

@app.route('/output/<path:filename>')
def serve_output_file(filename):
    """Serve um arquivo diretamente do diretório de output."""
    return send_from_directory(OUTPUT_DIR, filename)