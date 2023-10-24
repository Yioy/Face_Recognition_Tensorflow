import matplotlib.pyplot as plt
import numpy as np
import logging
import tensorflow as tf
from tensorflow import keras
import math
import numpy as np
import matplotlib.pyplot as plt
import logging
from keras.preprocessing.image import ImageDataGenerator
import pathlib

BATCH_SIZE = 8
# logger = tf.get_logger()
# logger.setLevel(logging.ERROR)

#display 8 images
def plot_images(images, labels):
  plt.figure(figsize=(12, 6))
  for i in range(len(images)):
      plt.subplot(3, 3, i + 1)
      plt.title(f"Class: {labels[i]}")
      plt.imshow(images[i], cmap='binary_r')
      plt.axis('off')
  plt.show()

#plot images avec predictions
def plot_image(i, predictions_array, true_labels, images):

  predictions_array, true_label, img = predictions_array[i], true_labels[i], images[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  
  plt.imshow(img[...,0], cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  print("True labels : ", true_label)
  print("Predictions labels : ", predicted_label)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'
  
  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

#plot bar graph
def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(13), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)
  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')


#dataset URL
train_datasetUrl = "c:/Users/sebdo/OneDrive/Bureau/UTBM/Etudes/Semestre5/VA53/Face/Train"
validation_datasetUrl =  "c:/Users/sebdo/OneDrive/Bureau/UTBM/Etudes/Semestre5/VA53/Face/Test"

trainingSet = tf.keras.utils.image_dataset_from_directory(
    train_datasetUrl,
    labels='inferred',
    label_mode='int',
    class_names=None,
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    image_size=(125, 100),
    shuffle=True
)

#display all classes
class_names = trainingSet.class_names
print(class_names)

validationSet = tf.keras.utils.image_dataset_from_directory(
    validation_datasetUrl,
    labels='inferred',
    label_mode='int',
    class_names=None,
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    image_size=(125, 100),
    shuffle=True
)
images, label = next(iter(validationSet))
plot_images(images, label)


AUTOTUNE = tf.data.AUTOTUNE
train_dataset = trainingSet.cache().prefetch(buffer_size=AUTOTUNE)
validation_dataset = validationSet.cache().prefetch(buffer_size=AUTOTUNE)

#create architecture of the model
model = tf.keras.Sequential([
  tf.keras.layers.Rescaling(1./255),
  tf.keras.layers.Conv2D(16, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D((2,2)),
  tf.keras.layers.Dropout(0.5),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Dense(len(class_names), activation='softmax')
])

#compile the model
model.compile(
  optimizer='adam',
  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

#-----------------------------------Data augmentation to avoid overfitting--------------------------------------------------------------------
# image_gen_train = ImageDataGenerator(
#       rescale=1./255,
#       horizontal_flip=True,
#       fill_mode='nearest')

# train_data_gen = image_gen_train.flow_from_directory(batch_size=2,
#                                                      directory=validation_datasetUrl,
#                                                      shuffle=False,
#                                                      target_size=(125,100),
#                                                      color_mode="grayscale",
#                                                      class_mode='binary')
#----------------------------------------------------------------------------------------------------------------------------

#------Train the model---------
history = model.fit_generator(
    trainingSet,
    epochs=18,
    validation_data=validationSet,
)

#-------------------display accuracy and loss evolution during training----------------------------
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(18)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

#-------------------display predictions----------------------------

images, labels = next(iter(validationSet))
labels = np.array(labels)
images = images[:8]
predictions = []
for i, test_images in enumerate(images):
  test_images = np.expand_dims(test_images, axis=0)
  predictions.append(model.predict(test_images))

predictions = np.array(predictions)
flattened_table = [inner[0] for inner in predictions]
predictions = np.array(flattened_table)

num_rows = 5
num_cols = 2
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(8):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, labels, images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, labels)
plt.show()