import tensorflow as tf
from tensorflow.keras import layers,models
import matplotlib.pyplot as plt
import cv2
import pickle

(X_train,y_train),(X_test,y_test) = tf.keras.datasets.mnist.load_data()
X_train.shape

X_train = X_train/255.0
X_test = X_test/255.0

model = tf.keras.models.Sequential([
    layers.Flatten(input_shape=[28,28]),
    layers.Dense(300,activation='relu'),
    layers.Dense(100,activation='relu'),
    layers.Dense(10,activation='softmax')
])

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(X_train,y_train,epochs=20,validation_split=0.1)

#model usage 
# def resize_image(img_gray):
#     resized_gray = cv2.resize(img_gray, (28, 28), interpolation=cv2.INTER_AREA)
#     normalized = resized_gray.astype('float32') / 255.0
#     input_img = normalized.reshape(1, 28, 28)
#     return input_img

# img_gray = cv2.imread('./Test_image_2.png', cv2.IMREAD_GRAYSCALE) 
# function_data = resize_image(img_gray)
# predictions = model.predict(function_data)
# predicted_class = predictions[0].argmax() 
# print("Predicted class:", predicted_class)

model.save("digit_classifier.h5") 


