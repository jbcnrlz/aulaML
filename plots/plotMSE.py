import numpy as np
import matplotlib.pyplot as plt

# 1. Dados para o MSE (Erro Quadrático Médio)
# Imaginando que o valor real (y) seja 0
y_true_mse = 0
y_pred_mse = np.linspace(-2, 2, 100)
mse_loss = (y_pred_mse - y_true_mse)**2

# 2. Dados para a Cross-Entropy (Entropia Cruzada)
# Imaginando que a classe correta seja 1. 
# y_pred_ce representa a probabilidade prevista (entre 0 e 1)
y_pred_ce = np.linspace(0.01, 0.99, 100) # Evita log(0)
ce_loss = -np.log(y_pred_ce)

# Criando a figura com dois subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot do MSE
ax1.plot(y_pred_mse, mse_loss, color='blue', lw=2)
ax1.set_title('Mean Squared Error (MSE)\n$L = (y - \hat{y})^2$', fontsize=14)
ax1.set_xlabel('Valor Previsto ($\hat{y}$)', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.annotate('Mínimo Global', xy=(0, 0), xytext=(-1, 2),
             arrowprops=dict(facecolor='black', shrink=0.05))

# Plot da Cross-Entropy
ax2.plot(y_pred_ce, ce_loss, color='red', lw=2)
ax2.set_title('Categorical Cross-Entropy\n$L = -\log(\hat{y}_{correto})$', fontsize=14)
ax2.set_xlabel('Probabilidade Prevista para a Classe Correta', fontsize=12)
ax2.set_ylabel('Loss', fontsize=12)
ax2.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()