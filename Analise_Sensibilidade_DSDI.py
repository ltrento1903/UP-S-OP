import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data_cu = {
    'Cu': [1.5, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Co': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    'WMAPE %': [8.21, 8.21, 8.21, 8.21, 8.21, 8.21, 8.21, 8.21, 8.21, 8.21],
    'Bias ABS': [24116, 24116, 24116, 24116, 24116, 24116, 24116, 24116, 24116, 24116],
    'Bias %': [-0.43, -0.43, -0.43, -0.43, -0.43, -0.43, -0.43, -0.43, -0.43, -0.43]
}
df_cu = pd.DataFrame(data_cu)

data_co = {
    'Cu': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    'Co': [1.5, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'WMAPE %': [8.19, 8.19, 8.19, 8.19, 8.19, 8.19, 8.19, 8.19, 8.19, 8.19],
    'Bias ABS': [24859, 24859, 24859, 24859, 24859, 24859, 24859, 24859, 24859, 24859],
    'Bias %': [-0.63, -0.63, -0.63, -0.63, -0.63, -0.63, -0.63, -0.63, -0.63, -0.63]
}
df_co = pd.DataFrame(data_co)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(df_cu['Cu'], df_cu['Bias %'], marker='o', linestyle='-', color='blue')
axes[0].set_title('Impact of Increasing Shortage Cost (Cu)\non Systemic Bias (Co=1)')
axes[0].set_xlabel('Cu (Under-forecasting Penalty)')
axes[0].set_ylabel('Net Bias %')
axes[0].grid(True)
axes[0].axhline(0, color='black', linestyle='--')

axes[1].plot(df_co['Co'], df_co['Bias %'], marker='s', linestyle='-', color='red')
axes[1].set_title('Impact of Increasing Excess Cost (Co)\non Systemic Bias (Cu=1)')
axes[1].set_xlabel('Co (Over-forecasting Penalty)')
axes[1].set_ylabel('Net Bias %')
axes[1].grid(True)
axes[1].axhline(0, color='black', linestyle='--')

plt.tight_layout()
plt.savefig('sensitivity_analysis.png')
plt.show()