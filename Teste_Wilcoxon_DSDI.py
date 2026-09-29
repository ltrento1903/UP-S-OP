import numpy as np
from scipy.stats import wilcoxon

# -------------------------------------------------------------------
# 1. Dados de Viés Absoluto (Absolute Bias) para as 6 marcas
# -------------------------------------------------------------------

# Coluna 'Viés Abs' do cenário WMAPE (Symmetric Optimisation)
vies_abs_wmape = [
    366, 4141, 20030, 3203, 8649, 
    197
]

# Coluna 'Viés Abs' do cenário Decision Score (DS)
vies_abs_ds = [
    366, 3077, 9473, 2460, 8550, 
    190
]

# -------------------------------------------------------------------
# 2. Execução do Teste de Postos Sinalizados de Wilcoxon
# -------------------------------------------------------------------

# Converter para arrays numpy para garantir eficiência computacional
array_wmape = np.array(vies_abs_wmape)
array_ds = np.array(vies_abs_ds)

# O parâmetro alternative='greater' testa a hipótese de que os 
# erros do WMAPE são estatística e sistematicamente MAIORES que os do DS.
# O parâmetro zero_method='zsplit' ou 'wilcox' (padrão) lida com os empates 
# (as 8 marcas onde o viés foi idêntico).
stat, p_value = wilcoxon(array_wmape, array_ds, alternative='greater')

# -------------------------------------------------------------------
# 3. Impressão dos Resultados
# -------------------------------------------------------------------

print("--- Teste de Wilcoxon (Postos Sinalizados) ---")
print(f"Estatística de Teste (W): {stat}")
print(f"Valor de p: {p_value:.5f}")

print("\n--- Conclusão ---")
if p_value < 0.01:
    print("A redução do viés absoluto pelo Decision Score é ESTATISTICAMENTE SIGNIFICATIVA a 1% (p < 0.01).")
    print("A hipótese nula é rejeitada. A melhoria não ocorreu ao acaso.")
elif p_value < 0.05:
    print("A redução do viés absoluto pelo Decision Score é ESTATISTICAMENTE SIGNIFICATIVA a 5% (p < 0.05).")
    print("A hipótese nula é rejeitada. A melhoria não ocorreu ao acaso.")
else:
    print("O resultado NÃO é estatisticamente significativo.")