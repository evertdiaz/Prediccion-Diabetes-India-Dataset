#Uso de Keras
from keras.models import Sequential
model = Sequential()
print(model)

#Agregar Capas
from keras.layers import Dense, Activation
model.add(Dense(output_dim=64, input_dim=100))
model.add(Activation("relu"))
model.add(Dense(output_dim=10))
model.add(Activation("softmax"))
print(model)

#Configurar proceso de aprendizaje
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
print(model)

#Configurar optimizador
from keras.optimizers import SGD
model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True))

#Iterar en data de entrenamiento (X y Y)
model.fit(X_train, Y_train, nb_epoch=5, batch_size=32)

#Ingresar batches manualmente
model.train_on_batch(X_batch, Y_batch)

#Evaluar rendimiento
loss_and_metrics = model.evaluate(X_test, Y_test, batch_size=32)

#Generar predicciones de nueva data
classes = model.predict_classes(X_test, batch_size=32)
proba = model.predict_proba(X_test, batch_size=32)

#Por default Keras utiliza el backend de Theano, para utilizar el de tensorflow
#Se busca el archivo de configuracion que está en ~/.keras/keras.json
#Y se muestra {"epsilon": 1e-07, "floatx": "float32", "backend": "theano"} Modificar theano -> tensorflow