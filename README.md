# TALLER NUMERO 2 - CLUSTERING
## Numeral 1.
1. Se utiliza mucho en la deteccion de comunidades de una red  (redes sociales por ejemplo) de tal manera que minimice el numero de aristas entre los conjuntos.  Spectral clustering es un método de agrupamiento de datos que utiliza información de la estructura de los datos en su espacio de características para encontrar agrupamientos naturales.
2. Utiliza la matriz de Laplaciando que se define como la matriz de Grados (diagnonal con el numero de aristas) menos la matriz de Adyacencia (matriz binaria con las conexiones de cada elemento) .  Tienen las siguientes propiedades
   1. Es simetrica 
   2. Los eagen valores son reales positivos
   3. El vector de unos es un eigenvector de L (la matriz laplaciano)
   4.   La suma de lor renglones de esta matriz siempre es cero
   5. Lambda 0 = tambien es un eigenvalor
3. El algoritmo en resumen es:
   1.  Encontrar el Laplaciano de la grafica (del grafo que representa por ejemplo las comunidades)
   2.  Encontrar los eigenvalores de L(G) Laplaciando
   3.  Consideramos los segundos menores valores de los eigenvalores y se calcula el eigenvector correspondiente a este eigenvalor.
   4.  Asignamos a cada vertice de la grafica el valor correspondiente a este eigenvector y agrupamos los vertices de G de acuerdo a su signo.
   5.  separamos los vertices negativos {A} y los vertices positivos {B}.  --> estos conjuntos ya entregan la particion de la gráfica.
4. Se relaciona con los conceptos de clustering vistos en clase, aunque utilizan una tecnica que en lugar de usar directamente las caracteristicas para su seperacion, Clustering espectral primero transforma los datros en una representacion de un espacio de caracteristicas (a travez de los eigenvalores y eigenvectores).

## Numeral 2.

El algoritmo DBSCAN (Density-Based Spatial Clustering of Applications with Noise) es un algoritmo de agrupamiento de datos que se basa en la densidad de los puntos. Es decir, agrupa los puntos que están cerca unos de otros y que están rodeados por otros puntos cercanos, y los puntos que están aislados se consideran como ruido.

El algoritmo DBSCAN es útil para identificar grupos en datos que tienen formas arbitrarias y tamaños diferentes, y puede manejar datos ruidosos y con diferentes densidades. El algoritmo es especialmente útil en conjuntos de datos de alta dimensionalidad.

El algoritmo DBSCAN se usa comúnmente en la minería de datos, la segmentación de imágenes, la identificación de anomalías y la clasificación de objetos.

El fundamento matemático del algoritmo DBSCAN se basa en la definición de la densidad de un conjunto de puntos en un espacio. El algoritmo utiliza dos parámetros clave: el radio epsilon, que define la distancia máxima entre dos puntos para que se consideren vecinos cercanos, y el número mínimo de puntos que deben estar dentro del radio epsilon para que un punto sea considerado denso. Un punto se considera denso si tiene al menos el número mínimo de puntos dentro del radio epsilon, y un grupo se forma a partir de puntos densos que están conectados.

El algoritmo DBSCAN no está directamente relacionado con Spectral Clustering, que es otro algoritmo de agrupamiento de datos. Sin embargo, ambos algoritmos pueden utilizarse para agrupar conjuntos de datos no lineales y de alta dimensionalidad. Spectral Clustering se basa en la teoría de grafos y utiliza técnicas de álgebra lineal para agrupar los datos. Ambos algoritmos pueden ser complementarios en la identificación de grupos en diferentes tipos de conjuntos de datos.

## Numeral 3.
El método del codo es una técnica comúnmente utilizada en Clustering para determinar el número óptimo de clusters en un conjunto de datos. El nombre del método se debe a que se busca el punto en el gráfico que se asemeja a un "codo", es decir, el punto donde la tasa de disminución de la varianza intra-cluster se aplanó significativamente.

El método del codo implica trazar un gráfico de la varianza intra-cluster para diferentes valores de k (el número de clusters). La varianza intra-cluster es la suma de las distancias al cuadrado de cada punto en un cluster a su centroide. A medida que aumenta el número de clusters, la varianza intra-cluster tiende a disminuir, ya que los grupos se vuelven más homogéneos. Sin embargo, en algún punto, añadir más clusters no proporciona una mejora significativa en la varianza intra-cluster, y ese es el punto donde se encuentra el codo.

Una de las principales desventajas del método del codo es que puede ser subjetivo, ya que el punto donde se encuentra el codo puede ser difícil de determinar en algunos casos. Además, el método del codo no siempre es efectivo en la identificación del número óptimo de clusters en conjuntos de datos con formas complejas o densidades variables. En tales casos, pueden ser necesarias técnicas más avanzadas para determinar el número óptimo de clusters, como la validación interna o externa de clusters, o la utilización de índices de evaluación de la calidad de la agrupación, como el índice de silueta

## Numeral 5.
1. Implementandos
2. Despues de plotear el dataset y sin ejecutar ninguna rutina de agrupamiento se pueden observar 4 clusters
3. Calculados
4. 4