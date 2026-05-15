"""
Módulo de Predição de Desempenho Acadêmico via Regressão Linear.
Este script implementa uma aplicação web interativa utilizando Streamlit e 
Scikit-Learn para modelar a relação entre horas de estudo e notas obtidas.
"""

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression


def gerar_dados_historicos() -> pd.DataFrame:
    """
    Instancia o conjunto de dados empíricos para o treinamento do estimador.
    
    Returns:
        pd.DataFrame: Dados contendo as variáveis 'notas' e 'horas'.
    """
    return pd.DataFrame({
        'notas': [1.0, 2.0, 4.0, 6.0, 8.0, 10.0],
        'horas': [2.0, 4.0, 5.0, 7.0, 9.0, 10.0]
    })


def treinar_modelo_regressao(df: pd.DataFrame) -> LinearRegression:
    """
    Ajusta um modelo de Regressão Linear Simples a partir das features fornecidas.
    
    Args:
        df (pd.DataFrame): DataFrame com as colunas constitutivas do modelo.
        
    Returns:
        LinearRegression: O estimador ajustado (fitted).
    """
    X = df[['horas']]  # Matriz de features (bidimensional)
    y = df['notas']    # Vetor de targets (unidimensional)
    
    modelo = LinearRegression()
    modelo.fit(X, y)
    return modelo


def executar_inferencia(modelo: LinearRegression, horas: float) -> float:
    """
    Realiza a predição da nota a partir de um valor escalar de horas de estudo,
    aplicando a restrição de domínio do problema (limite superior da nota = 10.0).
    
    Args:
        modelo (LinearRegression): O modelo treinado.
        horas (float): A variável independente de entrada.
        
    Returns:
        float: A nota predita truncada no intervalo regulamentar.
    """
    # Conversão do input escalar para estrutura bidimensional aceita pelo sklearn
    input_dados = np.array([[horas]])
    predicao_bruta = modelo.predict(input_dados)[0]
    
    # Restrição física/acadêmica: a nota máxima permitida é 10.0
    return min(predicao_bruta, 10.0)


def renderizar_interface(df: pd.DataFrame, modelo: LinearRegression):
    """
    Renderiza os componentes visuais da aplicação via Streamlit.
    """
    st.header('ANÁLISE DE NOTAS - PREVENDO DESEMPENHO')
    
    # Visualização de Dados: Gráfico de Dispersão (Scatter Plot)
    st.subheader('Distribuição dos Dados de Treino')
    st.scatter_chart(data=df, x='horas', y='notas')
    
    # Componente de Entrada de Dados (Slider)
    st.subheader('Simulação de Cenários')
    horas_usuario = st.slider(
        label='Selecione a quantidade de horas de estudos dedicadas:',
        min_value=0,
        max_value=12,
        value=5,
        step=1
    )
    
    # Processamento da Inferência
    nota_estimada = executar_inferencia(modelo, horas_usuario)
    
    # Exibição do Resultado Mnemônico
    st.metric(
        label='Nota Final Estimada',
        value=f'{nota_estimada:.1f}'
    )


def main():
    # Inicialização do pipeline
    dados_estudos = gerar_dados_historicos()
    modelo_escola = treinar_modelo_regressao(dados_estudos)
    
    # Construção da UI
    renderizar_interface(dados_estudos, modelo_escola)


if __name__ == '__main__':
    main()