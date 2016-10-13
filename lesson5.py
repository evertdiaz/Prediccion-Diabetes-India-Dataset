from keras.models import Sequential
from keras.layers import Dense
import numpy
# seed of reproductibilty, este es random. se debería sacar mediante fórmula.
seed = 7
numpy.random.seed(seed)
# cargando el dataset de pima-indians
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
print("DATASET ORIGINAL")
print(dataset)

# Dividir el dataset en variables input(x) y output(y)
X = dataset[:,0:8]
#Corte del array: [row inicial:rowfinal, datoinicial:datofinal]
Y = dataset[:,8]
print("VALORES DE X")
print(X)
print("VALORES DE Y")
print(Y)

# Crear el modelo
model = Sequential()
model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))
model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))
# Configurar Aprendizaje del modelo
model.compile(loss='binary_crossentropy' , optimizer='adam', metrics=['accuracy'])
# Fit model
model.fit(X, Y, nb_epoch=15, batch_size=5) 
#epoch: Entrenamientos
#batch_size: Cantidad de Muestras

# Evaluar el modelo
scores = model.evaluate(X, Y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
print(X)
Z=numpy.array([[0,89,66,23,94,28.1,0.167,21]])
#Predicciones del Dataset, para comparar con el original Y
#predicciones = model.predict(X, batch_size=5, verbose=0)
#print(predicciones)
print("PREDICCION DE PRUEBA")
prediccion_prueba=model.predict(Z, batch_size=5, verbose=0)
print(prediccion_prueba)