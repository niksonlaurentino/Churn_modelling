# Previsão de Churn: Cancelamento de Apólices de Seguro

Este repositório contém a pipeline completa para análise exploratória, pré-processamento, treinamento e avaliação de modelos de Machine Learning aplicados à predição de Churn (Cancelamento de Apólices de Seguro).

O objetivo principal é identificar precocemente clientes com alta probabilidade de cancelamento, permitindo que a equipe de retenção/CRM atue de forma preventiva antes do término do contrato.

---

## 1. Visão Geral do Problema de Negócio

No setor de seguros, a aquisição de novos clientes é significativamente mais cara do que a retenção dos ativos. A perda de segurados (Churn) afeta diretamente o volume de prêmios emitidos e a rentabilidade da carteira.

* Objetivo: Construir um modelo preditivo capaz de identificar clientes em risco de cancelamento.
* Foco da Métrica: Maximizar o Recall (Sensibilidade) para a classe de Churn, garantindo que o menor número possível de cancelamentos passe despercebido.
* Estratégia Operacional: Ajuste do limiar de decisão (Threshold) de probabilidade para 0.20 no algoritmo campeão.

---

## 2. Estrutura dos Dados e Atributos

O dataset passa por uma etapa inicial de limpeza na qual colunas sem valor preditivo (RowNumber, CustomerId, Surname) são removidas. As variáveis restantes são organizadas da seguinte forma:

* CreditScore (Numérica Contínua): Score de crédito do titular | Tratamento: StandardScaler
* Age (Numérica Contínua): Idade do segurado | Tratamento: StandardScaler
* Tenure (Numérica Discreta): Tempo de permanência na seguradora em anos | Tratamento: StandardScaler
* Balance (Numérica Contínua): Saldo em conta / Reserva acumulada | Tratamento: StandardScaler
* NumOfProducts (Numérica Discreta): Quantidade de seguros/produtos contratados | Tratamento: StandardScaler
* EstimatedSalary (Numérica Contínua): Renda anual estimada do cliente | Tratamento: StandardScaler
* Geography (Categórica): Localização geográfica (País/Região) | Tratamento: OneHotEncoder(drop='first')
* Gender (Categórica): Gênero do segurado | Tratamento: OneHotEncoder(drop='first')
* HasCrCard (Binária): Possui cartão de crédito ativo (0 ou 1) | Tratamento: Passthrough (Mantida 0/1)
* IsActiveMember (Binária): Engajamento/Membro ativo (0 ou 1) | Tratamento: Passthrough (Mantida 0/1)
* Exited (Target): Variável Alvo (1 = Cancelou a Apólice, 0 = Permaneceu) | Tratamento: N/A

---

## 3. Pipeline de Pré-processamento e Modelagem

A arquitetura do código foi estruturada utilizando a biblioteca scikit-learn para garantir reprodutibilidade e evitar o vazamento de dados (data leakage):

1. Análise de Correlação: Aplicação das matrizes de Pearson (relações lineares) e Spearman (relações monotônicas) para entendimento preliminar do comportamento das variáveis frente à variável alvo Exited.
2. Divisão Treino/Teste: Divisão de 80% para treino e 20% para teste, com estratificação (stratify=df_y) para preservar a proporção da classe minoritária (Churn).
3. Transformação de Atributos (ColumnTransformer):
   - Padronização de variáveis numéricas contínuas via StandardScaler.
   - Codificação de variáveis categóricas via OneHotEncoder com remoção da primeira categoria para evitar multicolinearidade.
   - Manutenção de variáveis binárias (0/1) sem alteração de escala.
4. Algoritmos Avaliados:
   - Random Forest Classifier: Modelo baseado em ensamble de árvores de decisão.
   - Regressão Logística: Modelo linear baseline com ajuste de pesos de classe (class_weight='balanced').
5. Ajuste de Limiar (Thresholding): Redução do ponto de corte padrão (0.50) para 0.20, priorizando a captura do Churn.
6. Validação Cruzada: Validação do pipeline completo com StratifiedKFold (k=5) para verificação de estabilidade em múltiplos subconjuntos de dados.

---

## 4. Requisitos de Execução

Para rodar o script localmente, certifique-se de ter as seguintes bibliotecas instaladas em seu ambiente Python:

pip install numpy pandas matplotlib seaborn scikit-learn

### Como Executar

1. Clone este repositório:
   git clone https://github.com/seu-usuario/nome-do-repositorio.git
2. Atualize o caminho da variável DATASET_PATH no arquivo principal para o local onde seu arquivo Churn_Modelling.csv está armazenado.
3. Execute o script:
   python churn_insurance_analysis.py

---

## 5. Resultados e Conclusões de Negócio

Ao adotar o limiar ajustado de 0.20, os modelos apresentaram o seguinte comportamento:

* Random Forest (Modelo Recomendado): Demostrou alta capacidade de separação das probabilidades entre as duas classes. Alcançou um Recall superior a 76% na detecção de cancelamentos de apólices, mantendo o volume de falsos alarmes em um nível operacionalmente sustentável para as equipes de retenção.
* Regressão Logística: Apresentou sobreposição expressiva nas probabilidades calculadas para clientes fiéis e clientes em churn. No limiar de 0.20, o modelo gerou um volume excessivo de Falsos Positivos, inviabilizando sua aplicação prática no ambiente de negócios.

---

## 6. Próximos Passos

* Otimização de hiperparâmetros no Random Forest utilizando GridSearchCV ou RandomizedSearchCV.
* Teste de algoritmos avançados de Gradient Boosting (XGBoost, LightGBM e CatBoost).
* Análise de Feature Importance para identificar os principais gatilhos operacionais que levam ao cancelamento das apólices.
