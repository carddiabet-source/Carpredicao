import os
import sys

# Adiciona o diretório da aplicação ao path do Python
sys.path.insert(0, os.path.dirname(__file__))

# Importa a instância 'app' do seu arquivo principal 'app.py'
from app import app as application
