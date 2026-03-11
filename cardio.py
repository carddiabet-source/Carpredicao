# ==============================
# 1. IMPORTAR BIBLIOTECAS
# ==============================


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from scipy.stats import shapiro, mannwhitneyu, chi2_contingency
import scipy.stats as stats
from warnings import filterwarnings


# ==============================
# 2. CARREGAR BASE DE DADOS
# ==============================

csv_path = os.path.join(os.path.dirname(__file__), 'base_completa_final.csv')
df = pd.read_csv(csv_path, sep=';')

# Garantir que a coluna alvo 'chd_confirmada' é numérica (0 ou 1) desde o início
df['chd_confirmada'] = df['chd_confirmada'].astype(int)

# Ignorar avisos para um output mais limpo
filterwarnings('ignore')

# ==============================
# 2.A GERAR DADOS SINTÉTICOS (OPCIONAL)
# ==============================
print("Iniciando a geração de dados sintéticos...")

def gerar_dados_sinteticos(df_original, valor_chd, n_amostras):
    """
    Gera dados sintéticos baseados em um subconjunto do dataframe.
    
    Args:
        df_original (pd.DataFrame): O DataFrame completo para referência de colunas.
        valor_chd (int): O valor de 'chd_confirmada' (0 ou 1) para o grupo base.
        n_amostras (int): O número de amostras a serem geradas.
        
    Returns:
        pd.DataFrame: Um DataFrame com os novos dados sintéticos.
    """
    grupo_str = "'com CHD'" if valor_chd == 1 else "'sem CHD'"
    print(f"Gerando {n_amostras} novas amostras para o grupo {grupo_str}...")
    
    # Usar o grupo oposto como base se o grupo alvo não existir
    df_base = df_original[df_original['chd_confirmada'] == valor_chd]
    if df_base.empty:
        print(f"Aviso: Não há dados para o grupo {grupo_str}. Usando o grupo oposto como base para a geração.")
        df_base = df_original[df_original['chd_confirmada'] != valor_chd]
    
    if df_base.empty:
        print("Erro: Não há dados em nenhum dos grupos para gerar amostras sintéticas.")
        return pd.DataFrame()

    # Copiando para evitar SettingWithCopyWarning
    df_base = df_base.copy()

    novos_dados = {}

    # Identificar colunas numéricas e categóricas
    colunas_numericas = df_original.select_dtypes(include=np.number).columns.drop(['gestante_id', 'consulta_numero', 'chd_confirmada'])
    colunas_categoricas = df_original.select_dtypes(include=['object', 'category']).columns

    # Gerar dados para colunas numéricas
    for col in colunas_numericas:
        if not df_base[col].dropna().empty:
            amostra = df_base[col].dropna().sample(n_amostras, replace=True).values
            novos_dados[col] = amostra

    # Gerar dados para colunas categóricas
    for col in colunas_categoricas:
        if not df_base[col].dropna().empty:
            freq = df_base[col].value_counts(normalize=True)
            if not freq.empty:
                amostra = np.random.choice(freq.index, size=n_amostras, p=freq.values)
                novos_dados[col] = amostra

    # Definir valor fixo para a coluna alvo
    novos_dados['chd_confirmada'] = valor_chd
    
    return pd.DataFrame(novos_dados)

# Gerar 135 amostras para o grupo "sem CHD" para balancear o dataset
df_novos_sem_chd = gerar_dados_sinteticos(df, valor_chd=0, n_amostras=135)

# Concatenar todos os DataFrames
df = pd.concat([df, df_novos_sem_chd], ignore_index=True)

# Gerar IDs únicos para as novas gestantes para evitar duplicatas
id_max_existente = df['gestante_id'].dropna().max()
num_novas_linhas = len(df_novos_sem_chd)
df.loc[df['gestante_id'].isnull(), 'gestante_id'] = range(int(id_max_existente) + 1, int(id_max_existente) + 1 + num_novas_linhas)
df['gestante_id'] = df['gestante_id'].astype(int)

print("Geração de dados sintéticos concluída. DataFrame atualizado.")

# Criar diretório para salvar os resultados
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)

# ==============================
# 2.B SALVAR O NOVO DATAFRAME
# ==============================

output_csv_path = os.path.join(output_dir, 'base_completa_com_sinteticos.csv')
df.to_csv(output_csv_path, sep=';', index=False, encoding='utf-8')
print(f"Novo DataFrame com dados sintéticos salvo em: {os.path.abspath(output_csv_path)}")


# ==============================
# SETUP DO RELATÓRIO HTML
# ==============================

# O conteúdo do relatório será construído e inserido em um container principal.
html_content = """
    <div class="container">
        <h1>Relatório de Análise Descritiva Completa</h1>
        <div class="tab">
            <button class="tablinks" onclick="openTab(event, 'Tables')" id="defaultOpen">Tabelas</button>
            <button class="tablinks" onclick="openTab(event, 'Graphs')">Gráficos</button>
        </div>

        <div id="Tables" class="tabcontent">
            <!-- Conteúdo das tabelas será inserido aqui -->
        </div>

        <div id="Graphs" class="tabcontent">
            <!-- Conteúdo dos gráficos será inserido aqui -->
        </div>
    </div>"""

# ==============================
# 3. INFORMAÇÕES GERAIS
# ==============================

html_tables = "<h2>Informações Gerais</h2>"

# --- Valores Faltantes ---
missing_values_df = pd.DataFrame({
    'Total Faltante': df.isnull().sum(),
    '% Faltante': (df.isnull().sum() / len(df)) * 100
})
html_tables += f"""
<div class="card">
    <h3>Valores Faltantes</h3>
    {missing_values_df.to_html(classes='table')}
</div>
"""

# ==============================
# 2.1 PREPARAÇÃO DOS DADOS
# ==============================

# ==============================
# 6. ESTATÍSTICA DESCRITIVA
# ==============================

html_tables += '<div class="card grid-container">'

numericas = df.select_dtypes(include=['float64','int64'])

desc = numericas.describe().T

# Selecionando apenas as colunas de interesse e renomeando
estatisticas_principais = desc[['mean', 'std', 'min', '25%', '50%', '75%', 'max']]
estatisticas_principais.rename(columns={'mean': 'Média', '50%': 'Mediana', 'std': 'Desvio Padrão'}, inplace=True)
estatisticas_principais.rename(columns={
'mean':'Média',
'std':'Desvio Padrão',
'min':'Mínimo',
'25%':'Q1',
'50%':'Mediana',
'75%':'Q3',
'max':'Máximo'
}, inplace=True)
html_tables += f"""
<div>
    <h3>Estatísticas Descritivas (Variáveis Numéricas)</h3>
    {estatisticas_principais.to_html(classes='table')}
</div>
"""

# --- Estatísticas Descritivas para Categóricas ---
desc_categoricas = df.describe(include=['object']).T
desc_categoricas.rename(columns={
    'count': 'Contagem',
    'unique': 'Valores Únicos',
    'top': 'Mais Frequente',
    'freq': 'Frequência do Top'
}, inplace=True)

html_tables += f"""
<div>
    <h3>Estatísticas Descritivas (Variáveis Categóricas)</h3>
    {desc_categoricas.to_html(classes='table')}
</div>
"""


# ==============================
# 7. TESTE DE NORMALIDADE
# ==============================

def highlight_significant(s, threshold=0.05):
    """Highlight p-values less than a threshold."""
    is_significant = s < threshold
    return ['background-color: #ffebee' if v else '' for v in is_significant]


print("\nTeste de normalidade (Shapiro-Wilk)")
normalidade = {}

for col in numericas.columns:
    
    dados = df[col].dropna()
    
    if len(dados) > 3:
        stat, p = shapiro(dados)
        normalidade[col] = p

normalidade_df = pd.DataFrame.from_dict(
    normalidade,
    orient='index',
    columns=['p_valor']
)

# Adiciona interpretação e formata
normalidade_df['Significante (p < 0.05)'] = normalidade_df['p_valor'].apply(lambda p: 'Sim' if p < 0.05 else 'Não')


# 8. SEPARAR GRUPOS CHD
# ==============================

chd = df[df['chd_confirmada'] == 1]
sem_chd = df[df['chd_confirmada'] == 0]

html_tables += f"""
<div>
    <h3>Contagem de Casos</h3>
    <p>Casos com CHD: {len(chd)}</p>
    <p>Casos sem CHD: {len(sem_chd)}</p>
</div>
"""


# ==============================
# 9. TESTE MANN-WHITNEY
# ==============================

variaveis = [
'idade',
'imc',
'pressao_sistolica',
'frequencia_cardiaca_fetal',
'idade_gestacional'
]

resultados = []

for var in variaveis:
    
    if var in df.columns:
        
        grupo1 = chd[var].dropna()
        grupo2 = sem_chd[var].dropna()
        
        if len(grupo1) > 0 and len(grupo2) > 0:
            
            stat, p = mannwhitneyu(grupo1, grupo2)
            
            resultados.append([var, p])

teste_mw = pd.DataFrame(
    resultados,
    columns=['variavel','p_valor']
)

# Adiciona interpretação e formata
teste_mw['Significante (p < 0.05)'] = teste_mw['p_valor'].apply(lambda p: 'Sim' if p < 0.05 else 'Não')

html_tables += f"""
<div>
    <h3>Teste Mann-Whitney (Numéricas vs. CHD)</h3>
    {teste_mw.to_html(classes='table', index=False)}
</div>
"""


# 10. TESTE QUI-QUADRADO
# ==============================

categoricas = [
'diabetes_gestacional',
'hipertensao',
'hipertensao_pre_eclampsia',
'obesidade_pre_gestacional',
'tabagismo',
'alcoolismo'
]

resultados_chi = []

for var in categoricas:
    
    if var in df.columns:
        
        tabela = pd.crosstab(df[var], df['chd_confirmada'])
        
        if tabela.shape[0] > 1:
            
            chi2, p, dof, exp = chi2_contingency(tabela)
            
            resultados_chi.append([var, p])

chi_df = pd.DataFrame(
    resultados_chi,
    columns=['variavel','p_valor']
)

# Adiciona interpretação e formata
chi_df['Significante (p < 0.05)'] = chi_df['p_valor'].apply(lambda p: 'Sim' if p < 0.05 else 'Não')

html_tables += f"""
<div>
    <h3>Teste Qui-Quadrado (Categóricas vs. CHD)</h3>
    {chi_df.to_html(classes='table', index=False)}
</div>
"""


# ==============================
# 11. CORRELAÇÃO SPEARMAN
# ==============================
# Calcula a matriz de correlação de Spearman
corr = df.select_dtypes(include=np.number).corr(method='spearman')
corr = numericas.corr(method='spearman')

# ==============================
# 12. CORRELAÇÃO COM CHD
# ==============================

corr_chd = corr['chd_confirmada'].sort_values(ascending=False)

html_tables += f"""
<div>
    <h3>Correlação de Spearman com 'chd_confirmada'</h3>
    {corr_chd.to_frame().to_html(classes='table')}
</div>
"""

html_tables += "</div>" # Fecha o grid-container da análise estatística

# ==============================
# ANÁLISES DE TABELA ADICIONAIS
# ==============================

html_tables += "<h2>Análises de Tabela Adicionais</h2>"

# --- Tabelas de Contingência para Alterações Estruturais ---
html_tables += '<div class="card">'
html_tables += "<h3>Tabelas de Contingência: Alterações Estruturais vs. CHD</h3>"
alteracoes_estruturais = ['doppler_ducto_venoso', 'eixo_cardiaco', 'quatro_camaras']
for col in alteracoes_estruturais:
    contingency_table = pd.crosstab(df[col], df['chd_confirmada'])
    html_tables += f"<h4>{col.replace('_', ' ').title()}</h4>"
    html_tables += contingency_table.to_html(classes='table')
html_tables += "</div>"

# --- Estatísticas Descritivas por Grupo CHD ---
grouped_stats_vars = [
    'idade',
    'imc',
    'pressao_sistolica',
    'translucencia_nucal_mm',
    'frequencia_cardiaca_fetal'
]

grouped_stats = df.groupby('chd_confirmada')[grouped_stats_vars].agg(['mean', 'median', 'std']).T
grouped_stats.rename(columns={0: 'Sem CHD', 1: 'Com CHD'}, inplace=True)

html_tables += f"""
<div class="card">
    <h3>Estatísticas Descritivas por Grupo de Diagnóstico</h3>
    {grouped_stats.to_html(classes='table')}
</div>
"""


# ==============================
# 13. FREQUÊNCIA FATORES DE RISCO
# ==============================

html_tables += "<h2>Análise de Fatores de Risco e Prevalência</h2>"
fatores_risco = [
'diabetes_gestacional',
'hipertensao',
'hipertensao_pre_eclampsia',
'obesidade_pre_gestacional',
'tabagismo',
'alcoolismo'
]

fatores_freq = {}
for fator in fatores_risco:
    if fator in df.columns:
        fatores_freq[fator] = df[fator].sum()

fatores_df = pd.DataFrame.from_dict(fatores_freq, orient='index', columns=['Total'])

html_tables += f"""
<div class="card">
    <h3>Frequência Absoluta de Fatores de Risco</h3>
    {fatores_df.to_html(classes='table')}
</div>
"""

# ==============================
# NOVA SEÇÃO: ANÁLISE FETAL (TABELAS)
# ==============================

html_tables += "<h2>Análise Fetal (Tabelas)</h2>"

# --- Estatísticas Descritivas para Variáveis Fetais Categóricas ---
fetal_categorical_vars = ['doppler_ducto_venoso', 'eixo_cardiaco', 'quatro_camaras']

# Garante que as colunas existem antes de tentar descrevê-las
fetal_categorical_vars_exist = [col for col in fetal_categorical_vars if col in df.columns]
fetal_desc_categoricas = df[fetal_categorical_vars_exist].describe(include=['object']).T
fetal_desc_categoricas.rename(columns={
    'count': 'Contagem',
    'unique': 'Valores Únicos',
    'top': 'Mais Frequente (Top)',
    'freq': 'Frequência do Top'
}, inplace=True)

html_tables += f"""
<div class="card"><h3>Estatísticas Descritivas (Variáveis Fetais Categóricas)</h3>{fetal_desc_categoricas.to_html(classes='table')}</div>
"""

# ==============================
# ANÁLISE DE OUTLIERS
# ==============================

html_tables += "<h2>Análise de Outliers</h2>"

outlier_summary = {}
numeric_cols_for_outliers = df.select_dtypes(include=np.number).columns.drop(['gestante_id', 'consulta_numero', 'chd_confirmada'], errors='ignore')

for col in numeric_cols_for_outliers:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    num_outliers = len(outliers)
    if num_outliers > 0:
        outlier_summary[col] = {
            'Nº de Outliers': num_outliers,
            '% de Outliers': f"{(num_outliers / len(df[col].dropna())) * 100:.2f}%"
        }

outlier_df = pd.DataFrame.from_dict(outlier_summary, orient='index')
html_tables += f"""
<div class="card"><h3>Resumo de Outliers (Método IQR)</h3>{outlier_df.to_html(classes='table')}</div>
"""

# ==============================
# 14. FREQUÊNCIA DE CHD
# ==============================

# Esta informação já foi adicionada na seção 8

# ==============================
# 15. VISUALIZAÇÃO DE DADOS
# ==============================

html_graphs = "<h2>Visualizações Gerais</h2><div class='plots-container'>"

import json

# --- Gráfico de Barras com Chart.js: Frequência de Fatores de Risco ---
print("\nGerando dados para o gráfico de barras de Fatores de Risco (Chart.js)...")
fatores_df_chartjs = df[fatores_risco].sum().sort_values(ascending=False)

# Preparar dados para JSON
labels_fatores = fatores_df_chartjs.index.tolist()
data_fatores = fatores_df_chartjs.values.tolist()

# Gerar o HTML e o script para o gráfico
html_graphs += """
<div class="card" style="flex: 1 1 48%;">
    <h3>Frequência Absoluta de Fatores de Risco (Chart.js)</h3>
    <canvas id="fatoresRiscoChart"></canvas>
    <script>
        const ctx = document.getElementById('fatoresRiscoChart').getContext('2d');
        new Chart(ctx, {
            type: 'bar',
            data: {
                labels: """ + json.dumps(labels_fatores) + """,
                datasets: [{
                    label: 'Contagem Total',
                    data: """ + json.dumps(data_fatores) + """,
                    backgroundColor: 'rgba(52, 152, 219, 0.6)',
                    borderColor: 'rgba(52, 152, 219, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                scales: {
                    y: {
                        beginAtZero: true
                    }
                },
                responsive: true,
                maintainAspectRatio: true
            }
        });
    </script>
</div>
"""

# --- Novo Painel: Distribuição de Sinais Vitais Maternos ---
print("\nGerando painel de distribuição de sinais vitais maternos...")
sinais_vitais_maternos = ['pressao_sistolica', 'bpm_materno', 'saturacao', 'temperatura_corporal']
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle('Distribuição Geral de Sinais Vitais Maternos', fontsize=18)
axes = axes.flatten()

for i, sinal in enumerate(sinais_vitais_maternos):
    if sinal in df.columns:
        sns.histplot(ax=axes[i], data=df, x=sinal, kde=True, bins=20)
        axes[i].set_title(f'Distribuição de {sinal.replace("_", " ").title()}', fontsize=14)
        axes[i].set_xlabel('Valor')
        axes[i].set_ylabel('Contagem')

# Ocultar eixos não utilizados, se houver
for j in range(len(sinais_vitais_maternos), len(axes)):
    axes[j].set_visible(False)

img_path = os.path.join(output_dir, 'plot_geral_sinais_vitais.png')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(img_path, bbox_inches='tight')
plt.close()
html_graphs += f"<div class='card' style='flex: 1 1 100%;'><h3>Painel de Distribuição de Sinais Vitais Maternos</h3><img src='/output/{os.path.basename(img_path)}'></div>"

# --- Novo Painel: Perfil Materno Geral (IMC e Risco Acumulado) ---
print("\nGerando painel de Perfil Materno Geral...")
fig, axes = plt.subplots(1, 2, figsize=(20, 7))
fig.suptitle('Perfil Materno Geral', fontsize=18)

# Subplot 1: Distribuição por Categoria de IMC
imc_bins = [0, 18.5, 25, 30, np.inf]
imc_labels = ['Abaixo do Peso', 'Normal', 'Sobrepeso', 'Obesidade']
df['imc_categoria'] = pd.cut(df['imc'], bins=imc_bins, labels=imc_labels, right=False)
sns.countplot(ax=axes[0], data=df, x='imc_categoria', palette='plasma', order=imc_labels)
axes[0].set_title('Distribuição por Categoria de IMC', fontsize=14)
axes[0].set_xlabel('Categoria de IMC')
axes[0].set_ylabel('Número de Gestantes')

# Subplot 2: Contagem de Fatores de Risco Acumulados
sns.countplot(ax=axes[1], data=df, x='total_fatores_risco', palette='magma')
axes[1].set_title('Contagem de Fatores de Risco Acumulados', fontsize=14)
axes[1].set_xlabel('Número de Fatores de Risco')
axes[1].set_ylabel('Número de Gestantes')

img_path = os.path.join(output_dir, 'plot_geral_perfil_materno.png')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(img_path, bbox_inches='tight')
plt.close()
html_graphs += f'<div class="card" style="flex: 1 1 100%;"><h3>Painel: Perfil Materno Geral</h3><img src="/output/{os.path.basename(img_path)}"></div>'

# --- Novo Painel: Perfil Demográfico e Gestacional ---
print("\nGerando painel de Perfil Demográfico e Gestacional...")
fig, axes = plt.subplots(1, 2, figsize=(20, 7))
fig.suptitle('Perfil Demográfico e Gestacional', fontsize=18)

# Subplot 1: Distribuição por Faixa Etária
age_bins = [0, 20, 35, np.inf]
age_labels = ['Adolescente (<20)', 'Adulto (20-34)', 'Idade Avançada (35+)']
df['faixa_etaria'] = pd.cut(df['idade'], bins=age_bins, labels=age_labels, right=False)
sns.countplot(ax=axes[0], data=df, x='faixa_etaria', palette='Set2', order=age_labels)
axes[0].set_title('Distribuição por Faixa Etária', fontsize=14)
axes[0].set_xlabel('Faixa Etária')
axes[0].set_ylabel('Número de Gestantes')

# Subplot 2: Distribuição da Idade Gestacional
sns.histplot(ax=axes[1], data=df, x='idade_gestacional', kde=True, bins=25, color='skyblue')
axes[1].set_title('Distribuição da Idade Gestacional (Semanas)', fontsize=14)
axes[1].set_xlabel('Idade Gestacional')
axes[1].set_ylabel('Número de Consultas')

img_path = os.path.join(output_dir, 'plot_geral_perfil_demografico.png')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(img_path, bbox_inches='tight')
plt.close()
html_graphs += f'<div class="card" style="flex: 1 1 100%;"><h3>Painel: Perfil Demográfico e Gestacional</h3><img src="/output/{os.path.basename(img_path)}"></div>'

# --- Novo Painel: Perfil de Estilo de Vida e Condições Pré-existentes ---
print("\nGerando painel de Estilo de Vida e Condições Pré-existentes...")
fig, axes = plt.subplots(2, 2, figsize=(16, 14))
fig.suptitle('Perfil de Estilo de Vida e Condições Pré-existentes', fontsize=18)

lifestyle_vars = {
    'uso_medicamentos': 'Uso de Medicamentos',
    'tabagismo': 'Tabagismo',
    'alcoolismo': 'Alcoolismo',
    'obesidade_pre_gestacional': 'Obesidade Pré-Gestacional'
}

axes = axes.flatten()

for i, (col, title) in enumerate(lifestyle_vars.items()):
    if col in df.columns:
        counts = df[col].value_counts()
        axes[i].pie(counts, labels=counts.index.map({1: 'Sim', 0: 'Não', True: 'Sim', False: 'Não'}), autopct='%1.1f%%',
                    startangle=90, colors=['#ff9999','#66b3ff'])
        axes[i].set_title(title, fontsize=14)
        axes[i].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

img_path = os.path.join(output_dir, 'plot_geral_perfil_lifestyle.png')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(img_path, bbox_inches='tight')
plt.close()
html_graphs += f'<div class="card" style="flex: 1 1 100%;"><h3>Painel: Estilo de Vida e Condições Pré-existentes</h3><img src="/output/{os.path.basename(img_path)}"></div>'


# --- Gráfico 2: Boxplots para Variáveis Numéricas vs CHD ---
img_path = os.path.join(output_dir, 'plot_02_boxplots_numericas.png')
print("\nGerando boxplots para variáveis numéricas...")
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Comparação de Variáveis Numéricas entre Grupos', fontsize=18)

sns.boxplot(ax=axes[0, 0], data=df, x='chd_confirmada', y='imc')
axes[0, 0].set_title('IMC vs CHD')

sns.boxplot(ax=axes[0, 1], data=df, x='chd_confirmada', y='pressao_sistolica')
axes[0, 1].set_title('Pressão Sistólica vs CHD')

sns.boxplot(ax=axes[1, 0], data=df, x='chd_confirmada', y='frequencia_cardiaca_fetal')
axes[1, 0].set_title('Frequência Cardíaca Fetal vs CHD')

sns.boxplot(ax=axes[1, 1], data=df, x='chd_confirmada', y='idade_gestacional')
axes[1, 1].set_title('Idade Gestacional vs CHD')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(img_path, bbox_inches='tight')
plt.close()
html_graphs += f"<div class='card' style='flex: 1 1 48%;'><h3>Boxplots: Numéricas vs CHD</h3><img src='/output/{os.path.basename(img_path)}'></div>"

# # --- Gráfico 4: Matriz de Correlação ---
# print("\nGerando heatmap de correlação...")
# # Calcula a matriz de correlação de Spearman
# corr = df.select_dtypes(include=np.number).corr(method='spearman')
# img_path = os.path.join(output_dir, 'plot_04_corr_matrix.png')
# plt.figure(figsize=(12, 10))

# # Criando uma máscara para o triângulo superior
# mask = np.triu(np.ones_like(corr, dtype=bool))

# sns.heatmap(corr, 
#             mask=mask, 
#             annot=True, 
#             fmt=".2f", 
#             cmap='coolwarm', 
#             vmin=-1, vmax=1, 
#             center=0,
#             linewidths=.5,
#             cbar_kws={"shrink": .8})

# plt.title('Matriz de Correlação de Spearman', fontsize=16, pad=20)
# plt.savefig(img_path, bbox_inches='tight')
# plt.close()
# html_graphs += f'<div class="card" style="flex: 1 1 48%;"><h3>Matriz de Correlação</h3><img src="{os.path.basename(img_path)}"></div>'
html_graphs += '</div>' # Fecha o plots-container

# ==============================
# 16. ANÁLISE DESCRITIVA ADICIONAL
# ==============================

# html_graphs += "<h2>Análises Descritivas Adicionais</h2><div class='plots-container'>"

# # --- Gráfico 5: Gráfico de Violino para Translucência Nucal ---
# img_path = os.path.join(output_dir, 'plot_05_violino_tn.png')
# print("\nGerando gráfico de violino para translucência nucal...")
# plt.figure(figsize=(10, 7))
# sns.violinplot(data=df, x='chd_confirmada', y='translucencia_nucal_mm', palette='pastel', inner='quartile')
# plt.title('Distribuição da Translucência Nucal (mm) vs. CHD', fontsize=16)
# plt.xlabel('CHD Confirmada')
# plt.ylabel('Translucência Nucal (mm)')
# plt.xticks([0, 1], ['Não', 'Sim'])
# plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.savefig(img_path, bbox_inches='tight')
# plt.close()
# html_graphs += f'<div class="card" style="flex: 1 1 48%;"><h3>Violino: Translucência Nucal</h3><img src="{os.path.basename(img_path)}"></div>'

# html_graphs += '</div>' # Fecha o plots-container

# ==============================
# 17. ANÁLISE CRUZADA DE VARIÁVEIS
# ==============================

html_graphs += "<h2>Análises Cruzadas</h2><div class='plots-container'>"

# --- Gráfico 7: Proporção de CHD por Fator de Risco Categórico ---
print("\nGerando gráficos de proporção para fatores de risco...")

# Criar uma figura com subplots para o painel de proporções
fig, axes = plt.subplots(2, 3, figsize=(22, 12))
fig.suptitle('Painel: Proporção de CHD por Fator de Risco', fontsize=20)
axes = axes.flatten() # Transforma a matriz de eixos em um array 1D

for i, fator in enumerate(fatores_risco):
    if fator in df.columns and i < len(axes):
        # Calcula a proporção e cria um gráfico de barras empilhadas
        props = df.groupby(fator)['chd_confirmada'].value_counts(normalize=True).unstack()
        if props is not None and not props.empty and props.shape[1] == 2:
            props.plot(kind='bar', stacked=True, ax=axes[i], colormap='coolwarm', legend=False)
            axes[i].set_title(f'Proporção por {fator.replace("_", " ").title()}', fontsize=14)
            axes[i].set_ylabel('Proporção')
            axes[i].set_xlabel('')
            axes[i].tick_params(axis='x', rotation=0)

img_path = os.path.join(output_dir, 'plot_07_painel_proporcoes.png')
plt.tight_layout(rect=[0, 0.03, 1, 0.96])
plt.savefig(img_path, bbox_inches='tight')
plt.close()
html_graphs += f'<div class="card" style="flex: 1 1 100%;"><h3>Painel de Proporções por Fator de Risco</h3><img src="/output/{os.path.basename(img_path)}"></div>'

# # --- Gráfico 8: Idade Materna vs. Translucência Nucal ---
# img_path = os.path.join(output_dir, 'plot_08_scatter_idade_tn.png')
# print("\nGerando gráfico de dispersão: Idade vs. Translucência Nucal...")
# plt.figure(figsize=(12, 8))
# sns.scatterplot(data=df, x='idade', y='translucencia_nucal_mm', hue='chd_confirmada', palette='viridis', alpha=0.7, s=80)
# plt.title('Idade Materna vs. Translucência Nucal', fontsize=16)
# plt.xlabel('Idade da Mãe')
# plt.ylabel('Translucência Nucal (mm)')
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.legend(title='CHD Confirmada', labels=['Não', 'Sim'])
# plt.savefig(img_path, bbox_inches='tight')
# plt.close()
# html_graphs += f'<div class="card" style="flex: 1 1 48%;"><h3>Idade vs. Translucência Nucal</h3><img src="{os.path.basename(img_path)}"></div>'
html_graphs += "</div>" # Fecha o plots-container

# ==============================
# NOVA SEÇÃO: ANÁLISE FETAL
# ==============================

html_graphs += "<h2>Análise Fetal</h2><div class='plots-container'>"

# --- Painel de Variáveis Numéricas Fetais ---
print("\nGerando painel de análise de variáveis fetais numéricas...")
fetal_numeric_vars = [
    'peso_fetal', 'frequencia_cardiaca_fetal', 'circunferencia_cefalica_fetal_mm',
    'circunferencia_abdominal_mm', 'comprimento_femur_mm', 'translucencia_nucal_mm'
]
fig, axes = plt.subplots(2, 3, figsize=(22, 12))
fig.suptitle('Painel: Análise de Variáveis Fetais Numéricas vs. CHD', fontsize=20)
axes = axes.flatten()

for i, col in enumerate(fetal_numeric_vars):
    if col in df.columns and not df[col].isnull().all():
        sns.boxplot(ax=axes[i], data=df, x='chd_confirmada', y=col, palette='pastel')
        axes[i].set_title(f'Distribuição de {col.replace("_", " ").title()}', fontsize=14)
        axes[i].set_xlabel('CHD Confirmada')
        axes[i].set_ylabel('')
        axes[i].set_xticklabels(['Não', 'Sim'])

img_path = os.path.join(output_dir, 'plot_fetal_painel_numerico.png')
plt.tight_layout(rect=[0, 0.03, 1, 0.96])
plt.savefig(img_path, bbox_inches='tight')
plt.close()
html_graphs += f'<div class="card" style="flex: 1 1 100%;"><h3>Painel de Variáveis Fetais Numéricas</h3><img src="/output/{os.path.basename(img_path)}"></div>'

# --- Painel de Alterações Estruturais (Movido para cá) ---
alteracoes_estruturais = ['doppler_ducto_venoso', 'eixo_cardiaco', 'quatro_camaras']
print("\nGerando painel de contagem para alterações estruturais...")
fig, axes = plt.subplots(1, 3, figsize=(22, 7))
fig.suptitle('Painel: Contagem por Alterações Estruturais', fontsize=20)
for i, col in enumerate(alteracoes_estruturais):
    sns.countplot(ax=axes[i], data=df, x=col, hue='chd_confirmada', palette='viridis')
    axes[i].set_title(f'Contagem por {col.replace("_", " ").title()}', fontsize=14)
    axes[i].set_ylabel('Número de Casos')
    axes[i].set_xlabel('')
    axes[i].legend(title='CHD', labels=['Não', 'Sim'])
img_path = os.path.join(output_dir, 'plot_06_painel_alteracoes_estruturais.png')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(img_path, bbox_inches='tight')
plt.close()
html_graphs += f'<div class="card" style="flex: 1 1 100%;"><h3>Painel de Contagem: Alterações Estruturais</h3><img src="/output/{os.path.basename(img_path)}"></div>'

html_graphs += "</div>" # Fecha o plots-container

# ==============================
# 18. ANÁLISE FOCADA EM CADA COMORBIDADE
# ==============================

html_graphs += "<h2>Análise Focada por Comorbidade</h2><div class='plots-container'>"

# Variáveis numéricas chave para cruzar com os fatores de risco
variaveis_numericas_chave = ['idade', 'imc', 'translucencia_nucal_mm']

for fator in fatores_risco:
    if fator in df.columns:
        # Criar uma figura com subplots para cada fator de risco
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        fig.suptitle(f'Análise de Comorbidade: {fator.replace("_", " ").title()}', fontsize=18)
        
        for i, var_num in enumerate(variaveis_numericas_chave):
            if var_num in df.columns and not df[var_num].isnull().all():
                sns.boxplot(ax=axes[i], data=df, x=fator, y=var_num, hue='chd_confirmada', palette='coolwarm')
                axes[i].set_title(f'{var_num.replace("_", " ").title()}')
                axes[i].set_xlabel(f'Possui {fator.replace("_", " ")}')
                axes[i].set_ylabel('') # Limpa o y-label para não poluir
                axes[i].legend(title='CHD', labels=['Não', 'Sim'])

        img_path = os.path.join(output_dir, f'plot_18_painel_{fator}.png')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(img_path, bbox_inches='tight')
        plt.close()
        html_graphs += f'<div class="card" style="flex: 1 1 100%;"><h3>Painel de Análise: {fator.replace("_", " ").title()}</h3><img src="/output/{os.path.basename(img_path)}"></div>'

html_graphs += "</div>" # Fecha o plots-container

# ==============================
# 19. ANÁLISE DE SINAIS VITAIS POR COMORBIDADE
# ==============================

html_graphs += "<h2>Análise de Sinais Vitais por Comorbidade</h2><div class='plots-container' style='flex-direction: column;'>"

# Sinais vitais para cruzar com os fatores de risco
sinais_vitais = [
    'pressao_sistolica',
    'bpm_materno',
    'saturacao',
    'frequencia_cardiaca_fetal'
]

for fator in fatores_risco:
    if fator in df.columns:
        # Criar uma figura com subplots para cada fator de risco
        fig, axes = plt.subplots(1, len(sinais_vitais), figsize=(24, 5))
        fig.suptitle(f'Análise de Sinais Vitais por {fator.replace("_", " ").title()}', fontsize=18)

        for i, sinal_vital in enumerate(sinais_vitais):
            if sinal_vital in df.columns and not df[sinal_vital].isnull().all():
                sns.boxplot(ax=axes[i], data=df, x=fator, y=sinal_vital, hue='chd_confirmada', palette='viridis')
                axes[i].set_title(f'{sinal_vital.replace("_", " ").title()}')
                axes[i].set_xlabel(f'Possui {fator.replace("_", " ")}')
                axes[i].set_ylabel('')
                axes[i].legend(title='CHD', labels=['Não', 'Sim'])

        img_path = os.path.join(output_dir, f'plot_19_painel_sinais_{fator}.png')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(img_path, bbox_inches='tight')
        plt.close()
        html_graphs += f'<div class="card" style="flex: 1 1 100%;"><h3>Painel de Sinais Vitais: {fator.replace("_", " ").title()}</h3><img src="/output/{os.path.basename(img_path)}"></div>'
html_graphs += "</div>" # Fecha o plots-container

# ==============================
# 20. ANÁLISE SISTEMÁTICA DE TODAS AS VARIÁVEIS POR CHD
# ==============================

# html_graphs += "<h2>Análise Sistemática (Todas as Variáveis vs. CHD)</h2><div class='plots-container'>"

# # --- Análise de Variáveis Numéricas ---
# print("\nAnalisando variáveis numéricas vs. CHD...")
# # Lista de variáveis fetais já plotadas na seção específica
# fetal_vars_already_plotted = [
#     'peso_fetal', 'frequencia_cardiaca_fetal', 'circunferencia_cefalica_fetal_mm',
#     'circunferencia_abdominal_mm', 'comprimento_femur_mm', 'translucencia_nucal_mm'
# ]
# # Exclui colunas de ID e as que já são fatores de risco
# numeric_cols_to_plot = df.select_dtypes(include=np.number).columns.drop(
#     ['gestante_id', 'consulta_numero', 'chd_confirmada'] + fatores_risco + fetal_vars_already_plotted, 
#     errors='ignore'
# )

# cols_per_panel = 9 # Aumentado para 9
# for i in range(0, len(numeric_cols_to_plot), cols_per_panel):
#     cols_subset = numeric_cols_to_plot[i:i+cols_per_panel]
#     fig, axes = plt.subplots(3, 3, figsize=(22, 18)) # Grade 3x3
#     fig.suptitle(f'Painel de Análise Numérica Sistemática (Parte {i//cols_per_panel + 1})', fontsize=20)
#     axes = axes.flatten()

#     for j, col in enumerate(cols_subset):
#         if col in df.columns and not df[col].isnull().all():
#             sns.boxplot(ax=axes[j], data=df, x='chd_confirmada', y=col, palette='pastel')
#             axes[j].set_title(f'Distribuição de {col.replace("_", " ").title()}', fontsize=14)
#             axes[j].set_xlabel('CHD Confirmada')
#             axes[j].set_ylabel('')
#             axes[j].set_xticklabels(['Não', 'Sim'])    

#     # Ocultar eixos não utilizados
#     for j in range(len(cols_subset), len(axes)):
#         axes[j].set_visible(False)
    
#     img_path = os.path.join(output_dir, f'plot_20_painel_num_{i//cols_per_panel + 1}.png')
#     plt.tight_layout(rect=[0, 0.03, 1, 0.96])
#     plt.savefig(img_path, bbox_inches='tight')
#     plt.close()
#     html_graphs += f'<div class="card" style="flex: 1 1 100%;"><h3>Painel de Análise Numérica Sistemática (Parte {i//cols_per_panel + 1})</h3><img src="{os.path.basename(img_path)}"></div>'

# # --- Análise de Variáveis Categóricas ---
# print("\nAnalisando variáveis categóricas vs. CHD...")
# categorical_cols_to_plot = [
#     'uso_medicamentos', 'doppler_ducto_venoso', 'eixo_cardiaco', 
#     'quatro_camaras', 'class_pressao', 'trimestre', 'alteracao_estrutural'
# ]
# # Garante que as colunas existem no dataframe e remove duplicatas
# categorical_cols_to_plot = sorted(list(set([col for col in categorical_cols_to_plot if col in df.columns])))


# for i in range(0, len(categorical_cols_to_plot), cols_per_panel):
#     cols_subset = categorical_cols_to_plot[i:i+cols_per_panel]
#     fig, axes = plt.subplots(3, 3, figsize=(22, 20)) # Grade 3x3
#     fig.suptitle(f'Painel de Análise Categórica Sistemática (Parte {i//cols_per_panel + 1})', fontsize=20)
#     axes = axes.flatten()

#     for j, col in enumerate(cols_subset):
#         if col in df.columns and not df[col].isnull().all():
#             sns.countplot(ax=axes[j], data=df, x=col, hue='chd_confirmada', palette='coolwarm')
#             axes[j].set_title(f'Contagem por {col.replace("_", " ").title()}', fontsize=14)
#             axes[j].set_ylabel('Número de Casos')
#             axes[j].set_xlabel('')
#             axes[j].set_xticklabels(axes[j].get_xticklabels(), rotation=30, ha='right')
#             axes[j].legend(title='CHD', labels=['Não', 'Sim'])
    
#     # Ocultar eixos não utilizados para um painel mais limpo
#     for j in range(len(cols_subset), len(axes)):
#         axes[j].set_visible(False)


#     img_path = os.path.join(output_dir, f'plot_20_painel_cat_{i//cols_per_panel + 1}.png')
#     plt.tight_layout(rect=[0, 0.03, 1, 0.96])
#     plt.savefig(img_path, bbox_inches='tight')
#     plt.close()
#     html_graphs += f'<div class="card" style="flex: 1 1 100%;"><h3>Painel de Análise Categórica Sistemática (Parte {i//cols_per_panel + 1})</h3><img src="{os.path.basename(img_path)}"></div>'

# html_graphs += "</div>" # Fecha o plots-container

# ==============================
# FINALIZAR E SALVAR HTML
# ==============================

# Adicionar o CSS e o JS diretamente no conteúdo que será salvo.
full_html_content = f"""
<style>
    .container {{ max-width: 1200px; margin: auto; background: #fff; padding: 20px 40px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }}
    h1 {{ color: #2c3e50; text-align: center; border-bottom: 2px solid #3498db; padding-bottom: 10px;}}
    h2 {{ color: #34495e; border-bottom: 1px solid #ecf0f1; padding-bottom: 5px; margin-top: 40px;}}
    h3 {{ color: #34495e; margin-top: 30px;}}
    .card {{ background: #ffffff; border: 1px solid #e0e0e0; border-radius: 8px; padding: 20px; margin-top: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); overflow: auto; }}
    img {{ max-width: 100%; height: auto; border-radius: 5px; display: block; margin: 1em auto; }}
    .table {{ width: 100%; border-collapse: collapse; margin-top: 1em; font-size: 0.9em; }}
    .table th, .table td {{ padding: 12px; border: 1px solid #ddd; text-align: left; }}
    .table th {{ background-color: #3498db; color: white; }}
    .table tr:nth-child(even) {{ background-color: #f9f9f9; }}
    .plots-container {{ display: flex; flex-wrap: wrap; gap: 20px; }}
    .grid-container {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(500px, 1fr)); gap: 20px; }}
    .tab {{ overflow: hidden; border: 1px solid #ccc; background-color: #f1f1f1; border-radius: 8px 8px 0 0; }}
    .tab button {{ background-color: inherit; float: left; border: none; outline: none; cursor: pointer; padding: 14px 16px; transition: 0.3s; font-size: 17px; }}
    .tab button:hover {{ background-color: #ddd; }}
    .tab button.active {{ background-color: #3498db; color: white; }}
    .tabcontent {{ display: none; padding: 6px 12px; border: 1px solid #ccc; border-top: none; border-radius: 0 0 8px 8px; background-color: white; animation: fadeEffect 1s; }}
    @keyframes fadeEffect {{ from {{opacity: 0;}} to {{opacity: 1;}} }}
</style>
{html_content}
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script>
    function openTab(evt, tabName) {{
        var i, tabcontent, tablinks;
        tabcontent = document.getElementsByClassName("tabcontent");
        for (i = 0; i < tabcontent.length; i++) {{
            tabcontent[i].style.display = "none";
        }}
        tablinks = document.getElementsByClassName("tablinks");
        for (i = 0; i < tablinks.length; i++) {{
            tablinks[i].className = tablinks[i].className.replace(" active", "");
        }}
        document.getElementById(tabName).style.display = "block";
        if (evt) {{ evt.currentTarget.className += " active"; }}
    }}
    // Garante que a primeira aba seja aberta por padrão.
    if (document.getElementById("defaultOpen")) {{
        document.getElementById("defaultOpen").click();
    }}
</script>
"""

# Inserir o conteúdo das tabelas e gráficos no corpo do HTML
full_html_content = full_html_content.replace('<!-- Conteúdo das tabelas será inserido aqui -->', html_tables)
full_html_content = full_html_content.replace('<!-- Conteúdo dos gráficos será inserido aqui -->', html_graphs)

report_path = os.path.join(output_dir, 'relatorio_analise_completa.html')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(full_html_content)

print(f"\nAnálise concluída! Relatório salvo em: {os.path.abspath(report_path)}")

# ==============================
# 21. UPLOAD AUTOMÁTICO PARA SERVIDOR WEB (SFTP)
# ==============================

# A funcionalidade de upload SFTP não é mais necessária neste fluxo,
# pois a aplicação Flask servirá o conteúdo diretamente.
# Você pode remover esta seção se desejar.
