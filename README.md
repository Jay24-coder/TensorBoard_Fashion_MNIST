# Fashion MNIST Classification using CNN

This repository contains a Convolutional Neural Network (CNN) implementation using Keras to classify images from the Fashion MNIST dataset. The project includes model training, evaluation, and visualization of results, along with TensorBoard integration for monitoring performance metrics.

---

## 🧠 Project Overview

Fashion MNIST is a dataset of 28x28 grayscale images of 10 fashion categories. The model is trained to classify these categories using a CNN architecture built with Keras.

---

## 📦 Requirements

Install the required Python packages:

```bash
pip install numpy pandas matplotlib seaborn keras tensorflow
```

To use TensorBoard:
```
pip install tensorboard
```

---

## 🧾 Dataset
The dataset is automatically loaded via Keras:
```python
from keras.datasets import fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
```

---

## 🏗️ Model Architecture
- Conv2D (20 filters, 3x3 kernel) – ReLU activation
- Dropout (0.5)
- MaxPooling2D (2x2)
- Conv2D (20 filters, 3x3 kernel) – ReLU activation
- Dropout (0.3)
- Flatten
- Dense (10 units, ReLU)
- Dense (10 units, Softmax)

---

## 🚀 Training
```python
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=20, validation_data=(x_test, y_test), batch_size=40)
```
Model is trained on normalized and reshaped images, and labels are one-hot encoded.

---

## 📊 Visualization
Training loss and validation loss over epochs:
```python
plt.plot(history['loss'], label='Loss')
plt.plot(history['val_loss'], label='Val_loss')
plt.legend(loc='best')
plt.show()

```

---

## 📈 TensorBoard Integration
You can monitor the training using TensorBoard:
```bash
%load_ext tensorboard
%tensorboard --logdir /content/my_log
```

Logs are saved using:
```python
log_ = '/content/my log' + datetime.datetime.now().strftime("%d-%m-%Y-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_, histogram_freq=1)
```

---

## 💾 Saving & Loading the Model
Save the trained model:
```python
model.save('My_model')
```

Load the model later:
```python
from keras.models import load_model
loaded_model = load_model('My_model')
```

---

## 🎯 Results
After training, the model achieves reasonable accuracy on the Fashion MNIST dataset with the given architecture. The use of Dropout and validation monitoring helps reduce overfitting.