import numpy as np
import tensorflow as tf
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Определяем дифференциальное уравнение и краевые условия для каждого уравнения
def fun1(x, y):
    return np.vstack((y[1], 1 * y[0]+5))

def bc1(ya, yb):
    return np.array([ya[0], yb[0] - 1.01])

def fun2(x, y):
    return np.vstack((y[1], 2 * y[0]+6))

def bc2(ya, yb):
    return np.array([ya[0], yb[0] - 1.02])

def fun3(x, y):
    return np.vstack((y[1], 3 * y[0]+7))

def bc3(ya, yb):
    return np.array([ya[0], yb[0] - 1.03])

def fun4(x, y):
    return np.vstack((y[1], 4 * y[0]+8))

def bc4(ya, yb):
    return np.array([ya[0], yb[0] - 1.04])

def fun5(x, y):
    return np.vstack((y[1], 5 * y[0]+9))

def bc5(ya, yb):
    return np.array([ya[0], yb[0] - 1.05])

def fun6(x, y):
    return np.vstack((y[1], 6 * y[0]+10))

def bc6(ya, yb):
    return np.array([ya[0], yb[0] - 1.06])

def fun7(x, y):
    return np.vstack((y[1], 7 * y[0]+11))

def bc7(ya, yb):
    return np.array([ya[0], yb[0] - 1.07])

def fun8(x, y):
    return np.vstack((y[1], 8 * y[0]+12))

def bc8(ya, yb):
    return np.array([ya[0], yb[0] - 1.08])

def fun9(x, y):
    return np.vstack((y[1], 9 * y[0]+13))

def bc9(ya, yb):
    return np.array([ya[0], yb[0] - 1.09])

def fun10(x, y):
    return np.vstack((y[1], 10 * y[0]+14))

def bc10(ya, yb):
    return np.array([ya[0], yb[0] - 1.1])

# Добавляем новые уравнения и их краевые условия
equations = [
    {'equation': fun1, 'boundary_conditions': bc1},
    {'equation': fun2, 'boundary_conditions': bc2},
    {'equation': fun3, 'boundary_conditions': bc3},
    {'equation': fun4, 'boundary_conditions': bc4},
    {'equation': fun5, 'boundary_conditions': bc5},
    {'equation': fun6, 'boundary_conditions': bc6},
    {'equation': fun7, 'boundary_conditions': bc7},
    {'equation': fun8, 'boundary_conditions': bc8},
    {'equation': fun9, 'boundary_conditions': bc9},
    {'equation': fun10, 'boundary_conditions': bc10}
]

# Генерируем случайные точки для x
x = np.linspace(0, 1, 100)

# Генерируем случайные начальные условия для каждого уравнения
y_initial1 = np.random.rand(2, len(x))
y_initial2 = np.random.rand(2, len(x))
y_initial3 = np.random.rand(2, len(x))
y_initial4 = np.random.rand(2, len(x))
y_initial5 = np.random.rand(2, len(x))
y_initial6 = np.random.rand(2, len(x))
y_initial7 = np.random.rand(2, len(x))
y_initial8 = np.random.rand(2, len(x))
y_initial9 = np.random.rand(2, len(x))
y_initial10 = np.random.rand(2, len(x))

# Решаем каждое дифференциальное уравнение и получаем обучающий датасет
solutions = []
for i in range(10):
    sol = solve_bvp(equations[i]['equation'], equations[i]['boundary_conditions'], x, locals()[f"y_initial{i+1}"])
    y_plot = sol.sol(x)
    solutions.append(y_plot[0])

# Создаем модель нейронной сети
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(1,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Компилируем модель
model.compile(optimizer='adam', loss='mse')

# Обучаем модель на каждом датасете
for i in range(10):
    model.fit(x, solutions[i], epochs=100, verbose=1)

# Проверяем работу модели на новом уравнении
# Например, y'' - 11y = 0 с краевыми условиями y(0) = 0, y(1) = 1,001

def fun_new(x, y):
    return np.vstack((y[1], 11 * y[0]+15))

def bc_new(ya, yb):
    return np.array([ya[0], yb[0] - 1.001])

x_new = np.linspace(0, 1, 100)
y_guess = np.zeros((2, len(x_new)))

sol_new = solve_bvp(fun_new, bc_new, x_new, y_guess)

# Предсказываем значения с помощью модели на новом тестовом наборе данных x_new
y_pred = model.predict(x_new)

# Вычисляем абсолютное значение ошибки между фактическими и предсказанными значениями
mse = mean_squared_error(sol_new.sol(x_new)[0], y_pred.flatten())

# Визуализируем результаты
plt.plot(x_new, y_pred, '--', label='Predicted')  # Пунктирная линия для предсказания
plt.plot(x_new, sol_new.sol(x_new)[0], label='Actual')  # Обычная линия для фактических данных
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

print(f"Mean Squared Error (MSE): {mse}")