
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

model = tf.keras.models.load_model('models/cnn_model.h5')

img = image.load_img('sample.jpg', target_size=(128,128))
img_array = image.img_to_array(img)/255.0
img_array = np.expand_dims(img_array, axis=0)

pred = model.predict(img_array)

print("Dog" if pred[0][0] > 0.5 else "Cat")
