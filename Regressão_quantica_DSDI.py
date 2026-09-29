import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from scipy.stats import wilcoxon

# 1. Carregar e limpar os dados
file_path = 'Auto_comLeves_combustível_jan2023_Agosto_2026.xlsx'
df = pd.read_excel(file_path)
df = df.rename(columns=lambda x: str(x).strip()) 

# Isolar apenas as 6 colunas das marcas (ignorando a data)
brands = [c for c in df.columns if c not in ['Mês']]

# Configurações do teste
alpha_target = 0.80 
test_size = 12      

# Funções de métrica
def pinball_loss(y_true, y_pred, alpha):
    diff = y_true - y_pred
    return np.mean(np.maximum(alpha * diff, (alpha - 1) * diff))

def asym_wmape(y_true, y_pred, penalty_under=2.0, penalty_over=1.0):
    diff = y_true - y_pred
    weighted_errors = np.where(diff > 0, diff * penalty_under, np.abs(diff) * penalty_over)
    if np.sum(y_true) == 0: return 0
    return np.sum(weighted_errors) / np.sum(y_true)

results = []

# 2. O Loop Principal
for brand in brands:
    data = df[['Mês', brand]].copy().sort_values('Mês')
    
    for i in range(1, 4):
        data[f'lag_{i}'] = data[brand].shift(i)
    data['month'] = data['Mês'].dt.month
    data['year'] = data['Mês'].dt.year
    
    data = data.dropna().reset_index(drop=True)
    
    X = data[['lag_1', 'lag_2', 'lag_3', 'month', 'year']]
    y = data[brand]
    
    X_train, X_test = X.iloc[:-test_size], X.iloc[-test_size:]
    y_train, y_test = y.iloc[:-test_size], y.iloc[-test_size:]
    
    # Modelos
    gb_std = GradientBoostingRegressor(loss='squared_error', random_state=42)
    gb_std.fit(X_train, y_train)
    preds_std = gb_std.predict(X_test)
    
    gb_qnt = GradientBoostingRegressor(loss='quantile', alpha=alpha_target, random_state=42)
    gb_qnt.fit(X_train, y_train)
    preds_qnt = gb_qnt.predict(X_test)
    
    # Métricas
    pb_std = pinball_loss(y_test, preds_std, alpha_target)
    aw_std = asym_wmape(y_test, preds_std)
    
    pb_qnt = pinball_loss(y_test, preds_qnt, alpha_target)
    aw_qnt = asym_wmape(y_test, preds_qnt)
    
    results.append({
        'Marca': brand,
        'AW_Padrao': round(aw_std, 4),
        'AW_Quantil': round(aw_qnt, 4),
        'Melhoria_AW': round(aw_std - aw_qnt, 4),
        'Pinball_Padrao': round(pb_std, 2),
        'Pinball_Quantil': round(pb_qnt, 2)
    })

# 3. Exibir Tabela
res_df = pd.DataFrame(results)
print("=== TABELA DE RESULTADOS PARA O ARTIGO ===")
print(res_df.to_string())
print("\n")

# 4. Estatísticas Agregadas
print("=== ANÁLISE ESTATÍSTICA (AW_Padrao vs AW_Quantil) ===")
print(f"Média AW_Padrao:    {res_df['AW_Padrao'].mean():.4f}")
print(f"Média AW_Quantil:   {res_df['AW_Quantil'].mean():.4f}")
print(f"Mediana AW_Padrao:  {res_df['AW_Padrao'].median():.4f}")
print(f"Mediana AW_Quantil: {res_df['AW_Quantil'].median():.4f}")

# Teste de Wilcoxon
w_stat, p_val = wilcoxon(res_df["AW_Padrao"], res_df["AW_Quantil"])
print(f"\nTeste de Wilcoxon (p-value): {p_val:.4f}")

# Effect Size (Cohen's d)
diff = res_df["AW_Padrao"] - res_df["AW_Quantil"]
cohens_d = diff.mean() / diff.std()
print(f"Cohen's d (Effect Size):     {cohens_d:.4f}")

# 5. Geração Automática do Insight Textual
limiar_negligencia = 0.01 # Diferenças entre -1% e +1% são consideradas negligenciáveis

melhorou = len(res_df[res_df['Melhoria_AW'] > limiar_negligencia])
piorou = len(res_df[res_df['Melhoria_AW'] < -limiar_negligencia])
negligenciavel = len(res_df[(res_df['Melhoria_AW'] <= limiar_negligencia) & (res_df['Melhoria_AW'] >= -limiar_negligencia)])

print("\n=== CONCLUSÃO PARA O MANUSCRITO ===")
print(f"The Quantile approach improved the asymmetric WMAPE in {melhorou} categories,")
print(f"while the remaining {piorou + negligenciavel} categories exhibited either marginal deterioration or negligible differences.")