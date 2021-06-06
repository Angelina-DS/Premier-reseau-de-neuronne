from keras.datasets import mnist
from keras import layers
from keras import models
import matplotlib.pyplot as plt #Pour afficher un élément de mnist
from numpy.core.fromnumeric import argmax
import tensorflow as tf
from keras import layers
from keras import models

#Import des données, training set et test set
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

#Remodelage des données dans la forme attendue par le réseau et pour que toutes les valeurs soit dans l'intervalle [0,1]
train_images = train_images.reshape((60000, 28*28))
train_images = train_images.astype('float32')/255

test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype('float32')/255

#On encode catégoriquement les labels
train_labels = tf.keras.utils.to_categorical(train_labels)
test_labels = tf.keras.utils.to_categorical(test_labels)

#L'architecture du réseau : 2 couches denses ou "fully connected"
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28*28,)))
network.add(layers.Dense(10, activation='softmax'))
"""La deuxième couche renvoie une liste de 10 scores, chaque score correspondant à 
la probabilité que le chiffre actuel coïncide avec un des 10 chiffres."""

#L'étape de compilation
network.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

#L'entraînement du réseau sur les données train
network.fit(train_images, train_labels, epochs=5, batch_size=128)

#Les résultats du réseau sur les vraies données (données test)
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test acc : ', test_acc)

#On fait une prédiction sur les 500 1ères images 
prediction = network.predict(test_images[:500])
print(prediction)
"""prediction est une liste de liste : c'est une liste qui va contenir une liste
de probabilités d'appartenance aux 10 chiffres pour chaque image prédite."""

predictions = [argmax(p) for p in prediction]
"""argmax() prend l'indice de la liste tel que l'élément associé à cet indice 
soi le plus grand de la liste. Ici on construit donc une liste contenant les 
résultats, ou les vraies prédictions, du réseau pour chaque image."""

print(predictions[:10]) #On affiche les 10 1ères prédictions du modèle

#On redéfinit les données car elles ont été modifiées plus haut
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

#On affiche les 10 1ères images du set pour vérifier qu'elles correspondent bien aux prédictions précédentes
#Pour passer d'une image à l'autre, cliquer sur la croix rouge
for i in range(10):
    plt.imshow(test_images[i])
    plt.show()

#Ce qui suit ne s'affiche qu'une fois avoir fermé les 10 images des 10 1ère prédictions précédentes
for i in range(500):
    if test_labels[i] != predictions[i]:
        print('Label : ', test_labels[i])
        print('Predicted : ', predictions[i])
        plt.imshow(test_images[i])
        plt.show()
"""Pour chacun des 1000 1ers éléments on regarde si le label est différent de la 
prédiction, si c'est le cas on affiche le label et la prédiction du modèle ainsi
que l'image du chiffre."""