# MVP-GABRIEL-BESSA-
MVP SSD E CEP 
README — Previsão de Chargeback com Machine Learning
Universidade de Brasília | Machine Learning & Deep Learning | Prof. Dr. André Luiz Marques Serrano
1. Descrição do Problema
O chargeback é o estorno de uma transação financeira realizado pela operadora do cartão de crédito a pedido do portador, geralmente quando este não reconhece a compra. Para empresas, cada chargeback representa prejuízo direto: perda da mercadoria ou serviço, multas operacionais e risco de descredenciamento.
O objetivo deste MVP é construir modelos de Machine Learning capazes de identificar antecipadamente transações com risco elevado de chargeback (fraude), permitindo que sistemas de pagamento bloqueiem ou sinalizem operações suspeitas em tempo real.

2. Sobre o Dataset
Dataset Credit Card Fraud Detection — Machine Learning Group da Université Libre de Bruxelles (ULB). Contém 284.807 transações de titulares de cartões europeus em setembro de 2013.

Característica	Detalhe
Total de registros	284.807 transações
Fraudes (Classe 1)	492 transações (~0,172%)
Legítimas (Classe 0)	284.315 transações
Atributos	31 (V1–V28 via PCA, Time, Amount, Class)
Valores nulos	0

As features originais foram transformadas via PCA por confidencialidade, resultando nos componentes V1 a V28. As únicas variáveis não transformadas são Time (segundos desde a primeira transação) e Amount (valor em euros).

3. Hipóteses e Premissas
•	Hipótese principal: transações fraudulentas possuem padrões latentes nos componentes PCA que as diferenciam das legítimas.
•	Hipótese secundária: Amount e Time contribuem marginalmente para a separação entre classes.
•	Premissa de dados: dataset anonimizado e limpo pelo ULB, sem valores nulos.
•	Restrição crítica: o forte desbalanceamento (~0,17% de fraudes) é o principal desafio e deve ser tratado com oversampling.

Seção 1 — Configuração do Ambiente
Prepara o ambiente importando bibliotecas e definindo configurações globais de reprodutibilidade.

Instalação de dependências
# !pip install imbalanced-learn xgboost lightgbm -q
Comentado por padrão; descomente se as bibliotecas não estiverem instaladas no Colab. A flag -q suprime o output de instalação.

Imports principais
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings, time

Biblioteca	Finalidade
numpy	Operações vetorizadas e numéricas de alto desempenho
pandas	Estruturas de dados tabulares e manipulação de dados
matplotlib / seaborn	Visualizações e gráficos estatísticos
warnings	Supressão de avisos não críticos no output
time	Medição de tempo de execução (benchmarking)

Imports scikit-learn
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, StratifiedKFold,
    cross_val_score, RandomizedSearchCV
from sklearn.feature_selection import mutual_info_classif

Classe / Função	Finalidade
RobustScaler	Escalonador baseado em mediana e IQR — resistente a outliers
train_test_split	Divisão do dataset em treino e teste
StratifiedKFold	Validação cruzada que preserva a proporção das classes em cada fold
cross_val_score	Executa CV e retorna métricas por fold
RandomizedSearchCV	Busca aleatória de hiperparâmetros com validação cruzada
mutual_info_classif	Calcula informação mútua entre cada feature e o alvo

Imports de modelos
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,
    GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

Imports de métricas
from sklearn.metrics import confusion_matrix, classification_report,
    roc_auc_score, average_precision_score, precision_recall_curve,
    roc_curve, f1_score, precision_score, recall_score, ConfusionMatrixDisplay

Métrica	Descrição
average_precision_score (AUPRC)	Métrica principal — área sob a curva Precisão-Recall
roc_auc_score	Complementar — área sob a curva ROC
f1_score	Harmônica entre precisão e recall (threshold 0.5)
precision_score / recall_score	Precisão e recall individuais
confusion_matrix	Tabela de TP, FP, FN, TN
classification_report	Relatório completo por classe

SMOTE
from imblearn.over_sampling import SMOTE
Synthetic Minority Oversampling Technique: gera amostras sintéticas da classe minoritária por interpolação entre vizinhos mais próximos, ao invés de simplesmente duplicar amostras existentes.

Configurações globais
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
plt.rcParams.update({'figure.dpi': 120, 'font.size': 11})
sns.set_theme(style='whitegrid', palette='muted')

•	RANDOM_STATE = 42 — semente aleatória global; garante reprodutibilidade total: qualquer execução produz os mesmos resultados.
•	np.random.seed — semente para operações numéricas aleatórias do numpy.
•	figure.dpi: 120 — resolução das figuras adequada para tela e exportação.
•	style='whitegrid' / palette='muted' — estilo visual limpo e profissional para todos os gráficos.

Seção 2 — Carregamento dos Dados
O dataset é carregado diretamente via URL pública (espelho Google/TensorFlow), sem necessidade de download manual ou configuração de API key do Kaggle.

Carregamento via URL
URL = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
t0 = time.time()
df = pd.read_csv(URL)
elapsed = time.time() - t0

•	pd.read_csv aceita URLs diretamente, sem download prévio.
•	O tempo é medido com time.time() antes e depois, permitindo reportar a duração do download+parsing.

Inspeção inicial
print(df.shape)                          # dimensoes
print(df.memory_usage(deep=True).sum())  # uso de memoria
print(df.isnull().sum().sum())            # valores nulos

•	df.shape — confirma dimensões esperadas (284.807 × 31).
•	memory_usage(deep=True) — calcula uso de memória real incluindo objetos Python internos; deep=True é necessário para colunas object.
•	isnull().sum().sum() — contagem total de nulos em todas as colunas; resultado esperado: 0.

Distribuição da variável alvo
class_counts = df['Class'].value_counts()
class_pct    = df['Class'].value_counts(normalize=True) * 100

•	value_counts() — contagem absoluta por classe.
•	value_counts(normalize=True) — proporção percentual por classe.
•	A razão de desbalanceamento (~578:1) quantifica o desequilíbrio e justifica o uso do SMOTE.
•	ATENÇÃO: com ~0,17% de fraudes, acurácia é métrica enganosa. Um modelo que classifica tudo como legítimo atinge 99,83% de acurácia sem detectar nenhuma fraude. Métricas principais do projeto: AUPRC e F1-Score.

Seção 3 — Análise Exploratória de Dados (EDA)
Investiga a estrutura dos dados, identifica padrões visuais e levanta evidências para guiar as decisões de modelagem.

3.1 — Distribuição Temporal (Time)
axes[0].hist(df[df['Class'] == 0]['Time'] / 3600, bins=96)
axes[1].hist(df[df['Class'] == 1]['Time'] / 3600, bins=48)

•	df[df['Class'] == 0] — filtra apenas transações legítimas.
•	Divisão por 3600 converte segundos em horas, tornando o eixo x interpretável.
•	bins=96 para legítimas (maior volume) vs bins=48 para fraudes (menor volume).
•	Resultado: fraudes apresentam menor concentração nos horários de pico e maior distribuição nas madrugadas (horas 0–8 e 24–32), confirmando valor discriminativo de Time.

3.2 — Distribuição do Valor (Amount)
axes[0].hist(np.log1p(subset), bins=60)
axes[1].set_yscale('log')

•	np.log1p — transformação log(1+x) necessária porque a distribuição de Amount é extremamente assimétrica (right-skewed); o log1p trata corretamente Amount = 0, evitando log(0) = -infinito.
•	set_yscale('log') no boxplot — eixo y em escala log para visualizar outliers sem comprimir os quartis da maioria.

3.3 — Top Features Discriminativas
corr_with_class = df.corr()['Class'].drop('Class').abs().sort_values(ascending=False)

•	df.corr() — matriz de correlação de Pearson entre todas as variáveis numéricas.
•	['Class'] — extrai apenas correlações com a variável alvo.
•	.drop('Class') — remove a autocorrelação da própria variável (Class × Class = 1.0).
•	.abs() — valor absoluto; correlações negativas fortes são tão informativas quanto positivas.

3.4 — Boxplots das Top 8 Features
top_features = corr_with_class.head(8).index.tolist()
bp = axes[i].boxplot([data_leg, data_fra], ..., notch=True)

•	notch=True — boxplot entalhado: se os entalhes de dois boxplots não se sobrepõem, há evidência estatística de que as medianas são diferentes (equivalente informal a um teste de hipótese visual).
•	Features como V14, V4, V11 e V12 mostram separação clara entre classes, confirmando alto valor discriminativo.

Seção 4 — Preparação dos Dados

4.1 — RobustScaler em Time e Amount
df_processed = df.copy()
scaler_robust = RobustScaler()
df_processed[['Time', 'Amount']] = scaler_robust.fit_transform(
    df_processed[['Time', 'Amount']]
)

•	df.copy() — cria cópia profunda para não modificar os dados brutos originais.
•	RobustScaler — usa mediana e IQR no lugar de média e desvio padrão; muito mais resistente a outliers extremos comuns em transações financeiras. O StandardScaler seria distorcido por esses outliers.
•	Apenas Time e Amount são escalonados; V1–V28 já foram padronizados internamente pelo PCA.
•	fit_transform — aplica fit (aprende mediana e IQR) e transform (aplica a escala) em uma única chamada.

4.2 — Divisão Treino/Teste Estratificada (80/20)
X = df_processed.drop('Class', axis=1)
y = df_processed['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

•	drop('Class', axis=1) — separa features (X) da variável alvo (y); axis=1 indica coluna.
•	test_size=0.2 — 20% para teste (~56.961 amostras), 80% para treino (~227.846 amostras).
•	stratify=y — PARÂMETRO CRÍTICO: garante que a proporção de fraudes (~0,17%) seja preservada em ambos os conjuntos. Sem estratificação, poderia resultar em zero fraudes no teste, tornando a avaliação inviável.

4.3 — SMOTE apenas no conjunto de treino
smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

•	k_neighbors=5 — número de vizinhos usados na interpolação; valor padrão e bem estabelecido na literatura.
•	fit_resample — gera amostras sintéticas balanceando as classes (resultado: 50/50 no treino).
•	REGRA FUNDAMENTAL: SMOTE é aplicado SOMENTE no treino, APÓS o split. Aplicar antes causaria data leakage — o modelo seria avaliado em dados que vazaram indiretamente para o treino, inflando artificialmente as métricas. O teste permanece com distribuição real para avaliação honesta.

Seção 5 — Seleção de Features (Feature Selection)
A Informação Mútua (MI) é calculada entre cada feature e o alvo para selecionar as 20 mais informativas.

Cálculo da Mutual Information
mi_scores = mutual_info_classif(
    X_train_bal, y_train_bal,
    random_state=RANDOM_STATE,
    n_neighbors=3
)

•	mutual_info_classif — estima a informação mútua I(X;Y) usando estimador k-NN. Captura QUALQUER tipo de dependência estatística, incluindo relações não-lineares; ao contrário da correlação de Pearson, que é restrita a relações lineares.
•	n_neighbors=3 — número de vizinhos no estimador k-NN; valor menor aumenta sensibilidade a estruturas locais.

Seleção das Top 20 features
TOP_K = 20
selected_features = mi_series.head(TOP_K).index.tolist()
X_train_sel = X_train_bal[selected_features]
X_test_sel  = X_test[selected_features]

•	Como os componentes PCA são ortogonais (sem correlação linear entre si), não há redundância — o critério é puramente baseado na informatividade individual de cada feature.
•	Os subsets de treino e teste são criados com as mesmas features, garantindo consistência na avaliação.

Seção 6 — Modelagem e Treinamento
Sete algoritmos são definidos, treinados e avaliados via Cross-Validation Estratificada com 5 folds.

Modelo	Tipo	Justificativa
Regressão Logística	Linear	Baseline linear; interpretável; limite mínimo de comparação
Decision Tree	Não-linear	Baseline interpretável; referência de overfitting
Random Forest	Bagging	Robusto a outliers; baixa variância; paralelizável
Gradient Boosting	Boosting	Boosting sequencial clássico sklearn
XGBoost	Boosting	Regularização L1/L2 integrada; eficiente
LightGBM	Boosting	Leaf-wise growth; state-of-the-art em datasets grandes
AdaBoost	Boosting	Boosting adaptativo; referência para comparar com XGB/LGBM

Parâmetros dos modelos — destaques
•	LogisticRegression — max_iter=1000 (o padrão de 100 não converge com 30 features); C=0.1 (regularização L2 mais forte); solver='lbfgs' (eficiente para multiclasse).
•	DecisionTree — max_depth=8 e min_samples_leaf=10 para controlar overfitting; sem limite a árvore memoriza os dados de treino completamente.
•	RandomForest — n_jobs=-1 utiliza todos os núcleos de CPU disponíveis para paralelizar o treinamento das 100 árvores.
•	GradientBoosting — subsample=0.8 implementa stochastic gradient boosting: usa 80% das amostras por iteração, reduzindo variância.
•	XGBoost — eval_metric='aucpr' define AUPRC como métrica interna, alinhado com a métrica principal do projeto.
•	LightGBM — learning_rate=0.05 menor que o XGBoost pois o leaf-wise converge mais rápido; verbose=-1 suprime output verboso.

Cross-Validation Estratificada (5-fold)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

roc_scores = cross_val_score(model, X_train_sel, y_train_bal,
                              cv=cv, scoring='roc_auc', n_jobs=-1)
ap_scores  = cross_val_score(model, X_train_sel, y_train_bal,
                              cv=cv, scoring='average_precision', n_jobs=-1)
f1_scores  = cross_val_score(model, X_train_sel, y_train_bal,
                              cv=cv, scoring='f1', n_jobs=-1)

•	StratifiedKFold(shuffle=True) — divide o treino em 5 partições estratificadas com embaralhamento para eliminar viés de ordenação temporal.
•	Três métricas calculadas em paralelo: roc_auc (complementar), average_precision / AUPRC (principal) e f1 (harmônica P/R no threshold 0.5).
•	O resultado de cada cross_val_score é um array de 5 valores; média e desvio padrão são reportados para estimar a variância do modelo.

Seção 7 — Otimização de Hiperparâmetros
Os 3 melhores modelos do CV (LightGBM, XGBoost e Random Forest) são otimizados com RandomizedSearchCV buscando maximizar o AUPRC.

Espaço de busca e RandomizedSearchCV
lgbm_search = RandomizedSearchCV(
    lgbm_base, param_grid_lgbm,
    n_iter=30, cv=cv,
    scoring='average_precision',
    random_state=RANDOM_STATE, n_jobs=-1
)
lgbm_search.fit(X_train_sel, y_train_bal)
best_lgbm = lgbm_search.best_estimator_

•	RandomizedSearchCV vs GridSearchCV — o grid completo do LightGBM testaria mais de 155.000 treinamentos. O Random Search com n_iter=30 testa apenas 150, sendo ~1.000x mais eficiente com performance comparável (Bergstra & Bengio, 2012).
•	scoring='average_precision' — otimiza pelo AUPRC, alinhado com a métrica principal.
•	best_estimator_ — retorna o modelo com os melhores hiperparâmetros já treinado no conjunto completo de treino.
•	Parâmetros buscados no LightGBM: n_estimators, learning_rate, max_depth, num_leaves, min_child_samples, reg_alpha (L1), reg_lambda (L2), subsample e colsample_bytree.
•	Parâmetros buscados no XGBoost: n_estimators, learning_rate, max_depth, subsample, colsample_bytree, reg_alpha, reg_lambda, min_child_weight e gamma.

Seção 8 — Ensemble (Soft Voting Classifier)
Um Soft Voting Classifier combina as probabilidades dos três melhores modelos com pesos calibrados.

ensemble = VotingClassifier(
    estimators=[
        ('lgbm', best_lgbm),  # peso 2
        ('xgb',  best_xgb),   # peso 2
        ('rf',   best_rf),    # peso 1
    ],
    voting='soft',
    weights=[2, 2, 1]
)

•	voting='soft' — cada modelo fornece a probabilidade P(fraude). O ensemble calcula a média ponderada. Diferente do hard voting (contagem de votos), o soft voting considera a confiança de cada predição.
•	weights=[2, 2, 1] — LightGBM e XGBoost recebem peso 2 (melhores AUPRC no CV); Random Forest recebe peso 1 como modelo complementar por diversidade.
•	A combinação de modelos reduz variância: modelos diferentes cometem erros em amostras diferentes, e a combinação ponderada produz predições mais estáveis.

Seção 9 — Avaliação no Conjunto de Teste
Todos os modelos são avaliados no conjunto de teste com distribuição real (desbalanceada, ~0,17% de fraudes).

Predição e métricas
y_prob = model.predict_proba(X_test_sel)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)

auprc = average_precision_score(y_test, y_prob)  # principal
roc_a = roc_auc_score(y_test, y_prob)             # complementar
f1    = f1_score(y_test, y_pred)
rec   = recall_score(y_test, y_pred)
prec  = precision_score(y_test, y_pred)

•	predict_proba — retorna probabilidades para cada classe; [:, 1] seleciona P(fraude).
•	(y_prob >= 0.5).astype(int) — converte probabilidades em labels binários pelo threshold 0.5; em produção esse threshold seria ajustado via análise custo-benefício.

Métrica	Fórmula	Justificativa
AUPRC	Área sob curva P-R	Principal: avalia P vs R na classe de fraude
ROC-AUC	Área sob curva ROC	Complementar: separabilidade geral das classes
F1-Score	2×(P×R)/(P+R)	Harmônica entre precisão e recall
Recall	TP/(TP+FN)	Taxa de detecção de fraudes — impacto financeiro direto
Precisão	TP/(TP+FP)	Taxa de alarmes corretos — evitar sobrecarga operacional

Curvas ROC e Precision-Recall
fpr, tpr, _ = roc_curve(y_test, y_prob)
prec_arr, rec_arr, _ = precision_recall_curve(y_test, y_prob)

•	roc_curve — calcula FPR e TPR para todos os thresholds possíveis.
•	precision_recall_curve — mais informativa que ROC-AUC com classes desbalanceadas, pois não considera os Verdadeiros Negativos (extremamente abundantes) que distorceriam a curva ROC.

Matriz de Confusão Normalizada
cm_norm = confusion_matrix(y_test, y_pred_best, normalize='true')

•	normalize='true' — normaliza por linha (por classe real), mostrando taxas em vez de contagens absolutas; essencial dado o extremo desequilíbrio.

Feature Importance
lgbm_imp = pd.Series(best_lgbm.feature_importances_, index=selected_features)

•	feature_importances_ — importância baseada em GAIN: redução média de impureza ponderada pelo número de amostras. Features com maior gain reduziram mais a incerteza nas árvores do LightGBM.

Seção 10 — Resultados e Conclusão

Comparativo por Tier
Tier	Modelos	Característica
Tier 1 (melhor)	Voting Ensemble, LightGBM Otimizado	AUPRC superior, alta estabilidade, recall elevado
Tier 2 (bom)	XGBoost Otimizado, Random Forest	Boa generalização, menor risco de overfitting
Tier 3 (referência)	Gradient Boosting, AdaBoost	Boa performance, abaixo dos otimizados
Tier 4 (baseline)	Regressão Logística, Decision Tree	Limite inferior de comparação

Boas práticas implementadas
Prática	Implementação
Reprodutibilidade	RANDOM_STATE = 42 em todas as operações
Sem data leakage	SMOTE apenas no treino, após o split
Validação robusta	StratifiedKFold k=5 com estratificação
Feature selection	Mutual Information (captura não-linearidade)
Otimização eficiente	RandomizedSearchCV — 30 iterações vs 155.000+ do GridSearch
Ensemble	Soft Voting com pesos calibrados por CV
Métricas adequadas	AUPRC como principal (não acurácia)

Pontos de atenção para produção
•	Threshold — o padrão 0.5 pode não ser ótimo; ajuste para ~0.3 aumenta recall às custas de mais falsos positivos.
•	Interpretabilidade — V1–V28 não são diretamente interpretáveis; SHAP values permitiriam explicar cada decisão ao time de compliance.
•	Concept Drift — padrões de fraude mudam rapidamente; retraining periódico e monitoramento de deriva são essenciais.
•	Custo assimétrico — scale_pos_weight no XGBoost/LightGBM pode penalizar falsos negativos explicitamente.
•	Dados temporais — modelos sequenciais (LSTM) poderiam capturar padrões temporais por usuário, além do escopo deste MVP.

Referências
•	Bergstra, J. & Bengio, Y. (2012). Random Search for Hyper-Parameter Optimization. JMLR, 13, 281-305.
•	Chawla, N.V. et al. (2002). SMOTE: Synthetic Minority Over-sampling Technique. JAIR, 16, 321-357.
•	Chen, T. & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. KDD.
•	Dal Pozzolo, A. et al. (2015). Calibrating Probability with Undersampling for Unbalanced Classification. IEEE SSCI.
•	Ke, G. et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. NeurIPS.

Notebook desenvolvido para a disciplina de Machine Learning — Universidade de Brasília, 2026.

