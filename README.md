# Previsão de Churn Bancário (Churn Modelling)

Script em Python para análise exploratória e previsão de **churn (cancelamento de clientes)** de um banco, comparando os desempenhos de `RandomForestClassifier` e `LogisticRegression`, com ajuste de limiar de decisão (threshold) para priorizar recall, validação cruzada estratificada e visualizações completas (correlações, matrizes de confusão e distribuição de probabilidades).

## 📋 Sumário

- [Requisitos](#-requisitos)
- [Instalação](#-instalação)
- [Estrutura do dataset](#-estrutura-do-dataset)
- [Como usar](#-como-usar)
- [Exemplos de uso](#-exemplos-de-uso)
- [Saídas geradas](#-saídas-geradas)
- [Observações](#-observações)
- [Licença](#-licença)

## ✅ Requisitos

- Python 3.9+
- Bibliotecas:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `seaborn`
  - `scikit-learn`

## 🚀 Instalação

1. Clone o repositório:

   ```bash
   git clone https://github.com/seu-usuario/churn-bancario-predictor.git
   cd churn-bancario-predictor
   ```

2. (Opcional, mas recomendado) Crie um ambiente virtual:

   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

3. Instale as dependências:

   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn
   ```

   Ou crie um arquivo `requirements.txt` com o conteúdo abaixo e instale com `pip install -r requirements.txt`:

   ```
   numpy>=1.23.0
   pandas>=1.5.0
   matplotlib>=3.6.0
   seaborn>=0.12.0
   scikit-learn>=1.2.0
   ```

## 📊 Estrutura do dataset

O script espera um arquivo `Churn_Modelling.csv` no mesmo diretório, com (entre outras) as seguintes colunas:

| Coluna              | Tipo             | Uso no script                              |
|---------------------|------------------|---------------------------------------------|
| `RowNumber`         | -                | Descartada                                   |
| `CustomerId`        | -                | Descartada                                   |
| `Surname`           | -                | Descartada                                   |
| `CreditScore`       | Numérica         | Padronizada (`StandardScaler`)               |
| `Geography`         | Categórica       | One-Hot Encoding                             |
| `Gender`            | Categórica       | One-Hot Encoding                             |
| `Age`               | Numérica         | Padronizada (`StandardScaler`)                |
| `Tenure`            | Numérica         | Padronizada (`StandardScaler`)                |
| `Balance`           | Numérica         | Padronizada (`StandardScaler`)                |
| `NumOfProducts`     | Numérica         | Padronizada (`StandardScaler`)                |
| `HasCrCard`         | Binária (0/1)    | Passthrough (mantida como está)               |
| `IsActiveMember`    | Binária (0/1)    | Passthrough (mantida como está)               |
| `EstimatedSalary`   | Numérica         | Padronizada (`StandardScaler`)                |
| `Exited`            | Binária (0/1)    | **Target** (variável alvo, indica o churn)   |

> 💡 O dataset "Churn Modelling" é amplamente utilizado em tutoriais de ML e pode ser encontrado publicamente, por exemplo, no [Kaggle](https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling).

Coloque o arquivo `Churn_Modelling.csv` na raiz do projeto, no mesmo diretório do script, antes de executá-lo.

## 🖥️ Como usar

Execute o script diretamente, sem parâmetros:

```bash
python churn_predictor.py
```

O script executa, em sequência:
1. Carregamento do `Churn_Modelling.csv` e remoção de colunas irrelevantes (`RowNumber`, `CustomerId`, `Surname`).
2. Cálculo e exibição (em heatmaps) das correlações de **Pearson** e **Spearman** entre os atributos e a variável alvo `Exited`.
3. Separação em features (`df_x`) e target (`df_y`).
4. Divisão treino/teste (80/20, com `stratify` no target).
5. Pré-processamento via `ColumnTransformer`: `StandardScaler` para colunas numéricas, `OneHotEncoder(drop='first')` para colunas categóricas e `passthrough` para colunas binárias.
6. Treinamento de dois modelos: `RandomForestClassifier` (com `class_weight='balanced'`) e `LogisticRegression` (com `class_weight='balanced'`).
7. Ajuste do **limiar de decisão (threshold)** para `0.20`, priorizando recall na detecção de clientes com maior propensão ao churn.
8. Impressão de acurácia e relatório de classificação para os dois modelos.
9. Validação cruzada estratificada (`StratifiedKFold`, 5 folds) comparando a consistência dos dois modelos.
10. Exibição de um painel com matrizes de confusão e histogramas de distribuição das probabilidades previstas por classe.

## 💡 Exemplos de uso

### Executando o script como está

```bash
python churn_predictor.py
```

O script exibirá, na sequência, dois painéis gráficos (`plt.show()`):
1. Matrizes de correlação de Pearson e Spearman.
2. Matrizes de confusão e histogramas de separação de probabilidades para os dois modelos.

E imprimirá no console, entre outras informações:

```
=================================================================
 CORRELAÇÃO COM A VARIÁVEL ALVO (EXITED / CHURN) 
=================================================================

--- PEARSON (Linear) ---
Exited            1.000000
Age               0.285...
...

--- DESEMPENHO: RANDOM FOREST ---
Acuracia Geral: 0.79xx
              precision    recall  f1-score   support
  Permaneceu       ...
       Churn       ...
```

### Ajustando o limiar de decisão (threshold)

Para priorizar precisão em vez de recall (ou vice-versa), altere a variável `novo_threshold` no script:

```python
novo_threshold = 0.35  # valor original: 0.20
```

E execute novamente:

```bash
python churn_predictor.py
```

### Reutilizando os modelos treinados em uma sessão interativa

Como o script não encapsula a lógica em funções, a forma mais simples de reaproveitar os modelos (`rf_model`, `lr_model`) e o pré-processador (`preprocessor`) após a execução é rodá-lo com `python -i`, mantendo as variáveis disponíveis no console:

```bash
python -i churn_predictor.py
```

```python
>>> nova_amostra_processada = preprocessor.transform(x_test.head(1))
>>> rf_model.predict_proba(nova_amostra_processada)
```

## 📊 Saídas geradas

- **Console**: correlações de Pearson e Spearman com a variável alvo, acurácia e relatório de classificação (`RandomForest` e `LogisticRegression`) com o threshold ajustado, e resultados da validação cruzada (média e desvio padrão da acurácia em 5 folds).
- **Gráficos (via `plt.show()`)**:
  - Painel 1: heatmaps de correlação de Pearson e Spearman entre os atributos.
  - Painel 2: matrizes de confusão dos dois modelos e histogramas comparando a distribuição das probabilidades previstas para as classes "Permaneceu" e "Churn", com o limiar de decisão destacado.

## ⚠️ Observações

- O script está estruturado de forma **sequencial/procedural** (sem funções ou classes) e deve ser executado de uma só vez, do início ao fim.
- O caminho do dataset está fixado na variável `DATASET_PATH = 'Churn_Modelling.csv'`; para usar outro arquivo, edite essa variável diretamente no script.
- O limiar de decisão (`novo_threshold = 0.20`) foi definido para priorizar recall (reduzir falsos negativos de churn); ajuste esse valor conforme o objetivo do negócio (maior precisão vs. maior recall).
- Os hiperparâmetros dos modelos (`n_estimators=100`, `max_depth=20` para o Random Forest; `max_iter=1000` para a Regressão Logística) estão fixos no código; ajuste-os conforme necessário para o seu dataset.
- A validação cruzada (`cross_val_score`) é executada sobre os dados **não escalonados/originais** (`df_x`, `df_y`), pois o próprio pipeline (`pip_rf`/`pip_lr`) já inclui o pré-processamento em cada fold.

## 📄 Licença

Este projeto está licenciado sob os termos da licença MIT. Sinta-se livre para usar, modificar e distribuir.
