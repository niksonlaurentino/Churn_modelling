# IMPORTAÇÃO DE BIBLIOTECAS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# ESTILIZAÇÃO DE GRÁFICOS
sns.set_theme(style="whitegrid")
plt.rcParams.update({"font.size": 10, "figure.autolayout": True})

# CARREGAMENTO DO DATASET
DATASET_PATH = 'Churn_Modelling.csv'
df = pd.read_csv(DATASET_PATH)

# DROP DE COLUNAS IRRELEVANTES
df = df.drop(columns=['RowNumber', 'CustomerId', 'Surname'], errors='ignore')

# ANÁLISE DE CORRELAÇÃO DE PEARSON E SPEARMAN
cols_analise = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Exited'] 
corr_pearson = df[cols_analise].corr(method='pearson')
corr_spearman = df[cols_analise].corr(method='spearman')

# GRÁFICOS DAS CORRELAÇÕES
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sns.heatmap(
    corr_pearson, 
    annot=True, 
    fmt=".2f", 
    cmap="coolwarm", 
    vmin=-1, 
    vmax=1, 
    ax=axes[0], 
    cbar=False
)
axes[0].set_title("Correlação de Pearson (Relação Linear)", fontsize=12, fontweight='bold')

sns.heatmap(
    corr_spearman, 
    annot=True, 
    fmt=".2f", 
    cmap="coolwarm", 
    vmin=-1, 
    vmax=1, 
    ax=axes[1], 
    cbar=True
)
axes[1].set_title("Correlação de Spearman (Relação Monotônica)", fontsize=12, fontweight='bold')

plt.suptitle("Análise Exploratória: Matrizes de Correlação dos Atributos", fontsize=14, fontweight='bold')
plt.show()

print("=" * 65)
print(" CORRELAÇÃO COM A VARIÁVEL ALVO (EXITED / CHURN) ")
print("=" * 65)
print("\n--- PEARSON (Linear) ---")
print(corr_pearson['Exited'].sort_values(ascending=False))

print("\n--- SPEARMAN (Postos/Ranks) ---")
print(corr_spearman['Exited'].sort_values(ascending=False))

# SEPARAÇÃO DE FEATURES E TARGET
df_x = df.drop(columns=['Exited'])
df_y = df['Exited']

# SEPARAÇÃO EM TREINO E TESTE
x_train, x_test, y_train, y_test = train_test_split(
    df_x, df_y, test_size=0.2, random_state=42, stratify=df_y
)

# LISTAS DE PREPROCESSAMENTO
numerical_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary'] # Padronização
categorical_cols = ['Geography', 'Gender'] # One Hot Encoder
bin_cols = ['HasCrCard', 'IsActiveMember'] # Mantidas como passthrough (0 e 1)

# PREPROCESSAMENTO DE ESCALONAMENTO / ENCODING
preprocessor = ColumnTransformer(
    transformers=[
        ("num_scaler", StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(drop='first'), categorical_cols),
        ('bin', 'passthrough', bin_cols)
    ]
)

# ESCALONAMENTO DE DADOS
x_train_scaled = preprocessor.fit_transform(x_train)
x_test_scaled = preprocessor.transform(x_test)

# APLICAÇÃO DOS DOIS ALGORITMOS - RF E LR
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    random_state=42,
    class_weight='balanced'
)

lr_model = LogisticRegression(
    random_state=42,
    class_weight='balanced',
    max_iter=1000
)

# TREINAMENTO (AJUSTE) DOS MODELOS COM OS DADOS PROCESSADOS
rf_model.fit(x_train_scaled, y_train)
lr_model.fit(x_train_scaled, y_train)

# LIMIAR DE CORTE AJUSTADO PARA 0.20 (PRIORIZANDO RECALL)
novo_threshold = 0.20

# APLICAÇÃO DO NOVO THRESHOLD
probs_rf = rf_model.predict_proba(x_test_scaled)[:, 1]
pred_rf_custom = (probs_rf >= novo_threshold).astype(int)

probs_lr = lr_model.predict_proba(x_test_scaled)[:, 1]
pred_lr_custom = (probs_lr >= novo_threshold).astype(int)

# AVALIAÇÃO DE DESEMPENHO
print("=" * 65)
print(f" AVALIAÇÃO DE DESEMPENHO (LIMIAR DE CORTE AJUSTADO PARA {novo_threshold}) ")
print("=" * 65)

print("\n--- DESEMPENHO: RANDOM FOREST ---")
print(f'Acuracia Geral: {accuracy_score(y_test, pred_rf_custom):.4f}')
print(classification_report(y_test, pred_rf_custom, target_names=['Permaneceu', 'Churn']))

print("\n--- DESEMPENHO: REGRESSÃO LOGÍSTICA ---")
print(f'Acuracia Geral: {accuracy_score(y_test, pred_lr_custom):.4f}')
print(classification_report(y_test, pred_lr_custom, target_names=['Permaneceu', 'Churn']))

# VALIDAÇÃO CRUZADA ESTRATIFICADA (CROSS-VALIDATION)
pip_rf = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", rf_model)])
pip_lr = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", lr_model)])

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores_rf = cross_val_score(pip_rf, df_x, df_y, cv=skf, n_jobs=-1)
scores_lr = cross_val_score(pip_lr, df_x, df_y, cv=skf, n_jobs=-1)

print("\n" + "=" * 65)
print(" VALIDAÇÃO CRUZADA COMPARATIVA (CONSISTÊNCIA EM 5 FOLDS) ")
print("=" * 65)
print(f'Random Forest   - Acuracia Media: {scores_rf.mean():.4f} (Desvio Padrao: {scores_rf.std():.4f})')
print(f'Reg. Logistica  - Acuracia Media: {scores_lr.mean():.4f} (Desvio Padrao: {scores_lr.std():.4f})')

# MATRIZES DE CONFUSÃO E DISTRIBUIÇÃO DE PROBABILIDADES
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

cm_rf = confusion_matrix(y_test, pred_rf_custom)
cm_lr = confusion_matrix(y_test, pred_lr_custom)

ConfusionMatrixDisplay(cm_rf, display_labels=['Permaneceu', 'Churn']).plot(
    ax=axes[0, 0], cmap='Blues', colorbar=False
)
axes[0, 0].set_title(f'Random Forest (Matriz de Confusão - Corte {novo_threshold})')
axes[0, 0].grid(False)

ConfusionMatrixDisplay(cm_lr, display_labels=['Permaneceu', 'Churn']).plot(
    ax=axes[0, 1], cmap='Greens', colorbar=False
)
axes[0, 1].set_title(f'Regressão Logística (Matriz de Confusão - Corte {novo_threshold})')
axes[0, 1].grid(False)

# HISTOGRAMAS DE SEPARAÇÃO DE PROBABILIDADES
sns.histplot(probs_rf[y_test == 0], color='blue', label='Classe 0 (Permaneceu)', ax=axes[1, 0], kde=True, stat="density", alpha=0.4)
sns.histplot(probs_rf[y_test == 1], color='orange', label='Classe 1 (Churn)', ax=axes[1, 0], kde=True, stat="density", alpha=0.4)
axes[1, 0].axvline(novo_threshold, color='red', linestyle='--', label=f'Limiar ({novo_threshold})')
axes[1, 0].set_title('Random Forest: Separação de Probabilidades')
axes[1, 0].set_xlabel('Probabilidade Calculada de Churn - (Cancelamento de Seguro)')
axes[1, 0].legend()

sns.histplot(probs_lr[y_test == 0], color='blue', label='Classe 0 (Permaneceu)', ax=axes[1, 1], kde=True, stat="density", alpha=0.4)
sns.histplot(probs_lr[y_test == 1], color='orange', label='Classe 1 (Churn)', ax=axes[1, 1], kde=True, stat="density", alpha=0.4)
axes[1, 1].axvline(novo_threshold, color='red', linestyle='--', label=f'Limiar ({novo_threshold})')
axes[1, 1].set_title('Regressão Logística: Separação de Probabilidades')
axes[1, 1].set_xlabel('Probabilidade Calculada de Churn - (Cancelamento de Seguro)')
axes[1, 1].legend()

plt.suptitle(f"Painel Comparativo de Desempenho e Diagnóstico dos Modelos (Threshold {novo_threshold})", fontsize=14, fontweight='bold')
plt.show()