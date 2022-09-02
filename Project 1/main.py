from statistics import mode
import tensorflow as tf
import matplotlib.pyplot as plt
import keras.api._v2.keras as keras
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense,Conv2D,Dropout,Flatten,MaxPooling2D
    
(x_train,y_train),(x_test,y_test)=  tf.keras.datasets.mnist.load_data()
print(x_train.shape)
fig, axis = plt.subplots(3,3, figsize = (10,10))
plt.gray()
x_train = x_train.reshape(x_train.shape[0], 28,28,1)
x_test = x_test.reshape(x_test.shape[0], 28,28,1)
input_shape = (28,28,1)

x_train =x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /=255
print('x_train shape :',x_train.shape)
print('number of images in x train :',x_train.shape[0])
print('number of images in xtest :',x_test.shape[0])

for i, axs in enumerate(axis.flat):
    axs.matshow(x_train[i])
    axs.axis('off')
    axs.set_title('Number {}'.format(y_train[i])) 
    fig.show()
model = keras.models.Sequential()
model.add(keras.layers.Conv2D(28, kernel_size=(3,3),input_shape=input_shape))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128,activation=tf.nn.relu))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(10,activation=tf.nn.softmax))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(x=x_train,y=y_train,epochs=10)
model.evaluate(x_train,y_train)




