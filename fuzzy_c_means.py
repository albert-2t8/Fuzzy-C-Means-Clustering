import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


plt.close('all')





image_ruta = 'perico.jpg'

try:
    imagen_original = Image.open(image_ruta).convert('L')  
except FileNotFoundError:
    print(f"La imagen no se encontró en la ruta: {image_ruta}")
    exit(1)

##################################################################################
############# Histograma
ancho, alto = imagen_original.size
histograma = [0] * 256
pixels = np.array(imagen_original).ravel()

for pixel in pixels:
    histograma[pixel] += 1
    

##################################################################################
############# Inicializar matriz de pertenencia
def generar_matriz_u(c, rango=256):
    return np.random.dirichlet(alpha=np.ones(c), size=rango).T


##################################################################################
############# Cálculo de la entropía de partición
def entropy_partition(u, c):
    suma_2 = 0  
    for k in range(256):  
        suma_1 = 0  
        for i in range(c):  
            if u[i, k] > 0:
                suma_1 += u[i, k] * np.log(u[i, k])
        suma_2 += suma_1  
    PE = -(1 / (256 * np.log(1 / c))) * suma_2  
    return PE

##################################################################################
############# Calcular centroides
def calcular_centroides(c, u, m, v, histograma):
    for k in range(c):  
        suma_1 = 0
        suma_2 = 1e-9  

        for i in range(256):  
            peso = (u[k, i] ** m) * histograma[i]  
            suma_1 += peso * i
            suma_2 += peso  

        v[k] = suma_1 / suma_2 

    return v


##################################################################################
############# Matriz de pertenencia
def actualizar_matriz_u(c, u, m, v):
    distancias = np.abs(np.arange(256)[:, np.newaxis] - v[np.newaxis, :]) + 1e-9  
    potencias = (distancias ** (-2 / (m - 1)))  
    u = potencias/ np.sum(potencias, axis=1, keepdims=True)  
    return u.T

##################################################################################
############# Función gaussiana
def funcion_gaussiana(x, v_i, m=0.0005):
    return np.exp(-m * ((x - v_i) ** 2))


##################################################################################
############# Fuzzy C-Means con selección automática de clusters
 
m = 3
max_iter = 100  
rango_clusters = range(2, 11)  
mejor_entropia = np.inf
mejor_c = None


for c in rango_clusters:
    u = generar_matriz_u(c)  
    v = np.linspace(np.min(histograma), np.max(histograma), c)   
    
    for _ in range(max_iter):
        v = calcular_centroides(c, u, m, v, histograma)
        u = actualizar_matriz_u(c, u, m, v)
    
    entropia_actual = np.abs(entropy_partition(u, c))
    print(f"Clusters: {c}, Entropía: {entropia_actual}")
    
    if entropia_actual < mejor_entropia:
        mejor_entropia = entropia_actual
        mejor_c = c

print(f"Número óptimo de clusters: {mejor_c}")

###################################################################################
################# Ejecutar Fuzzy C-Means con el número óptimo de clusters
c = mejor_c
u = generar_matriz_u(c)
v = np.linspace(np.min(histograma), np.max(histograma), c) 

for _ in range(max_iter):
    v = calcular_centroides(c, u, m, v, histograma)
    u = actualizar_matriz_u(c, u, m, v)


cluster_labels = np.argmax(u, axis=0)
imagen_clasificada = np.array([cluster_labels[p] for p in pixels]).reshape((alto, ancho))
imagen_clasificada = (imagen_clasificada * (255 // c)).astype(np.uint8)


##############################################################################
########## Mostrar imagen
plt.figure(figsize=(8, 5))
plt.imshow(imagen_original, cmap='gray') 
plt.title('Imagen en escala de grises')
plt.axis("off")

###############################################################################
######### Mostrar la imagen clasificada
plt.figure(figsize=(8, 5))
plt.imshow(imagen_clasificada, cmap='gray') #jet
plt.title(f"Imagen Segmentada (FCM, {c} Clusters)")
plt.axis("off")


###############################################################################
############## Mostrar el histograma 

x_vals = np.arange(0, 256)
f= np.array([funcion_gaussiana(x_vals, v_i) for v_i in v])


max_histograma = np.max(histograma)
max_f = np.max(f)

f_ajust = max_histograma / max_f
funciones_membresia_ajustadas = f * f_ajust


plt.figure(figsize=(8,5))
plt.bar(range(256), histograma, alpha=0.5, label="Histograma")
plt.title('Histograma')


for i in range(c):
    plt.plot(x_vals, funciones_membresia_ajustadas[i], label=f"Función de Membresia {i+1}")

for i in range(c):
    plt.scatter(v[i], 0, color='red', marker='x', label=f"Centroide {i+1} ({v[i]:.2f})")

plt.legend()
plt.show()


