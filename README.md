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

## Numeral 5.
1. Implementandos
2. Despues de plotear el dataset y sin ejecutar ninguna rutina de agrupamiento se pueden observar 4 clusters
3. Calculados
4. 4