"""
Módulo: Detector de Sono Gamer
Objetivo: Predizer o nível de cansaço de um jogador com base nas horas
          contínuas de jogo utilizando Regressão Linear.
Público-Alvo: Desenvolvedores
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def inicializar_dataset() -> pd.DataFrame:
    """
    Instancia o conjunto de dados empíricos de telemetria gamer.
    
    Returns:
        pd.DataFrame: Dados estruturados de horas jogadas vs nível de cansaço.
    """
    return pd.DataFrame({
        'horas_jogo': [1, 2, 4, 6, 8, 10],
        'cansaco': [1, 2, 3, 5, 8, 10]
    })


def treinar_detector_sono(df: pd.DataFrame) -> LinearRegression:
    """
    Isola as variáveis e ajusta o estimador linear de telemetria.
    
    Nota de Implementação: A variável independente X precisa ser um objeto 
    bidimensional. Usar df[['horas_jogo']] retorna um DataFrame (2D), atendendo
    à assinatura do método fit() sem a necessidade de .reshape().
    
    Args:
        df (pd.DataFrame): Dataset contendo o histórico do usuário.
        
    Returns:
        LinearRegression: O modelo treinado com os coeficientes calculados.
    """
    X = df[['horas_jogo']]  # Features: Matriz Bidimensional [n_samples, n_features]
    y = df['cansaco']       # Target: Vetor Unidimensional [n_samples]
    
    modelo = LinearRegression()
    modelo.fit(X, y)
    return modelo


def calcular_fadiga(modelo: LinearRegression, horas: float) -> float:
    """
    Realiza a inferência do nível de cansaço limitando o output aos limiares
    da escala de medição adotada (Escala de 0 a 10).
    
    Args:
        modelo (LinearRegression): O estimador linear ajustado.
        horas (float): Quantidade de horas jogadas a ser avaliada.
        
    Returns:
        float: Nível de cansaço normalizado entre 0.0 e 10.0.
    """
    # Conversão do escalar de entrada para o formato matricial exigido (1, 1)
    input_dados = np.array([[horas]])
    predicao_bruta = modelo.predict(input_dados)[0]
    
    # Restrição de domínio: O nível de cansaço satura em 10.0 e não pode ser menor que 0.0
    cansaco_normalizado = max(0.0, min(predicao_bruta, 10.0))
    return cansaco_normalizado


def main():
    # 1. Pipeline de Dados e Modelagem
    dados_gamer = inicializar_dataset()
    modelo_detector = treinar_detector_sono(dados_gamer)
    
    # 2. Log de Auditoria dos Parâmetros do Modelo
    # Equação gerada: Cansaço = Intercepto + (Coeficiente * Horas)
    print("=" * 65)
    print("MÉTRICAS DO DETECTOR DE SONO (REGRESSÃO LINEAR)")
    print("=" * 65)
    print(f"Intercepto (Cansaço Inicial/Vies - Beta 0): {modelo_detector.intercept_:.4f}")
    print(f"Coeficiente Angular (Taxa de Fadiga - Beta 1): {modelo_detector.coef_[0]:.4f}")
    print("-" * 65)
    
    # 3. Simulação de Monitoramento em Tempo Real (Fase de Inferência)
    cenarios_horas = [0.5, 3.0, 5.0, 7.5, 12.0]
    print("MONITORAMENTO DE SESSÃO GAMER:")
    print("-" * 65)
    for horas in cenarios_horas:
        nivel_fadiga = calcular_fadiga(modelo_detector, horas)
        
        # Alerta lógico baseado no nível de exaustão predito
        status = "ESTÁVEL" if nivel_fadiga < 7.0 else "ALERTA DE SONO / CRÍTICO"
        
        print(f"Tempo de Tela: {horas:>4.1f}h | Cansaço Estimado: {nivel_fadiga:>4.1f}/10.0 | Status: {status}")
    print("=" * 65)


if __name__ == '__main__':
    main()