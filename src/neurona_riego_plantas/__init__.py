import numpy as np

print("Neurona  artificial sencilla que determine si una planta necesita ser regada. La neurona  recibirá dos variables ")

#Entradas

X = np.array([[80, 18], [70, 22], [65, 28], [55, 25], [50, 32],[40, 30], [35, 25], [30, 32], [20, 35], [10, 38]], dtype=float)
y = np.array([[0], [0], [0], [0], [0],[1], [1], [1], [1], [1]], dtype=float)

# Normalización de entradas

escala = np.array([100, 50])
X_normalizado = X / escala
print("\nDatos normalizados:")
print(X_normalizado)


#Funcion de activacion
def sigmoide(z):
        return 1 / (1+np.exp(-z))

rng = np.random.default_rng(7)
pesos = rng.normal(size=(2,1))
sesgo = 0.0
tasa_aprendizaje = 2.0
epocas = 10000

for epoca in range(epocas):
        z = X_normalizado @ pesos + sesgo
        predicciones = sigmoide(z)
# Error cuadratico
        error = predicciones -y
        gradiente_z = 2 * error *predicciones * (1 - predicciones) / len(X_normalizado)
        gradiente_pesos = X_normalizado.T @ gradiente_z
        gradiente_sesgo = np.sum(gradiente_z)

        pesos -= tasa_aprendizaje * gradiente_pesos
        sesgo -= tasa_aprendizaje * gradiente_sesgo

        if epoca % 2000 == 0:
            perdida = np.mean(error**2)
            print(f"Epoca {epoca:5d} | error: {perdida:.4f}")
#Usamos el ravel o flatten para aplanar una matri
print("Pesos aprendido:" , pesos.ravel())
print("Sesgo aprendido:" , sesgo)

print("Probar lo que el perceptron aprendio ")
print("Si la salida es >= 0.5 == 1")
print("En caso contrario la salida sera  ==0")

probabilidades = sigmoide(X_normalizado @ pesos + sesgo)
respuestas = (probabilidades >=0.5).astype(int)

for entrada, probabilidad, respuesta in zip(X, probabilidades.ravel(), respuestas.ravel()):
        print(f"{entrada.astype(int)} -> probabilidad: {probabilidad:.3f} -> respuesta: {respuesta}")
assert np.array_equal(respuestas, y.astype(int))
 
