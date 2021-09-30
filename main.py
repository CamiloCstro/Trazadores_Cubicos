import numpy as np
import pandas as pd
import sympy as sym
from matplotlib import pyplot as plt


def traza3natural(xi, yi):
    n = len(xi)

    # Valores h
    h = np.zeros(n - 1, dtype=float)
    for j in range(0, n - 1, 1):
        h[j] = xi[j + 1] - xi[j]

    # Sistema de ecuaciones
    A = np.zeros(shape=(n - 2, n - 2), dtype=float)
    B = np.zeros(n - 2, dtype=float)
    S = np.zeros(n, dtype=float)

    A[0, 0] = 2 * (h[0] + h[1])
    A[0, 1] = h[1]
    B[0] = 6 * ((yi[2] - yi[1]) / h[1] - (yi[1] - yi[0]) / h[0])

    for i in range(1, n - 3, 1):
        A[i, i - 1] = h[i]
        A[i, i] = 2 * (h[i] + h[i + 1])
        A[i, i + 1] = h[i + 1]
        factor21 = (yi[i + 2] - yi[i + 1]) / h[i + 1]
        factor10 = (yi[i + 1] - yi[i]) / h[i]
        B[i] = 6 * (factor21 - factor10)

    A[n - 3, n - 4] = h[n - 3]
    A[n - 3, n - 3] = 2 * (h[n - 3] + h[n - 2])
    factor12 = (yi[n - 1] - yi[n - 2]) / h[n - 2]
    factor23 = (yi[n - 2] - yi[n - 3]) / h[n - 3]
    B[n - 3] = 6 * (factor12 - factor23)

    # Resolver sistema de ecuaciones S
    r = np.linalg.solve(A, B)
    for j in range(1, n - 1, 1):
        S[j] = r[j - 1]
    S[0] = 0
    S[n - 1] = 0

    # Coeficientes
    a = np.zeros(n - 1, dtype=float)
    b = np.zeros(n - 1, dtype=float)
    c = np.zeros(n - 1, dtype=float)
    d = np.zeros(n - 1, dtype=float)
    for j in range(0, n - 1, 1):
        a[j] = (S[j + 1] - S[j]) / (6 * h[j])
        b[j] = S[j] / 2
        factor10 = (yi[j + 1] - yi[j]) / h[j]
        c[j] = factor10 - (2 * h[j] * S[j] + h[j] * S[j + 1]) / 6
        d[j] = yi[j]

    # Polinomio trazador
    x = sym.Symbol('x')
    px_tabla = []
    for j in range(0, n - 1, 1):
        pxtramo = a[j] * (x - xi[j]) ** 3 + b[j] * (x - xi[j]) ** 2
        pxtramo = pxtramo + c[j] * (x - xi[j]) + d[j]

        pxtramo = pxtramo.expand()
        px_tabla.append(pxtramo)

    return (px_tabla)


# PROGRAMA -----------------------
# INGRESO , Datos de prueba
data = pd.read_csv('DatosEMA-2020-03-31-Radiación.csv', sep=';')
x = np.array(data['MIN'])
y = np.array(data['RADI'])
muestras = 10  # entre cada par de puntos

# PROCEDIMIENTO
# Tabla de polinomios por tramos
n = len(x)
px_tabla = traza3natural(x, y)

# SALIDA
print('Polinomios por tramos: ')
for tramo in range(1, n, 1):
    print(' x = [' + str(x[tramo - 1])
          + ',' + str(x[tramo]) + ']')
    print(str(px_tabla[tramo - 1]))

# GRAFICA
# Puntos para graficar cada tramo
xtraza = np.array([])
ytraza = np.array([])
tramo = 1
while not (tramo >= n):
    a = x[tramo - 1]
    b = x[tramo]
    xtramo = np.linspace(a, b, muestras)

    # evalua polinomio del tramo
    pxtramo = px_tabla[tramo - 1]
    pxt = sym.lambdify('x', pxtramo)
    ytramo = pxt(xtramo)

    # vectores de trazador en x,y
    xtraza = np.concatenate((xtraza, xtramo))
    ytraza = np.concatenate((ytraza, ytramo))
    tramo = tramo + 1


# Gráfica 1
plt.scatter(x, y)
plt.xlabel("Minuto")
plt.ylabel("Nivel de Radiación")
plt.title("Radiación 31-03-2020")
plt.show()

# Gráfica 2
plt.plot(x, y, 'ro', label='puntos')
plt.plot(xtraza, ytraza, label='trazador'
         , color='blue')
plt.title('Trazadores Cúbicos Naturales')
plt.xlabel('Minuto')
plt.ylabel('Radiación')
plt.legend()
plt.show()

