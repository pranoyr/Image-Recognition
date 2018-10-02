# Image Classification in Keras
How to develop an Image Classifiier in keras using tensorflow backend.

## Getting Started
### Prerequisites
1. TensorFlow
2. Keras

### Dataset
1. Download https://www.kaggle.com/c/dogs-vs-cats
![alt text](http://adilmoujahid.com/images/cats-dogs.jpg)
2. Create a folder named **"dataset_image"** in the root directory.
3. Create two folders -  **"cat"** and **"dog"** inside the folder **"dataset_image"**.
4. Put the downloaded images into the respective folders.

### Training
Run train.py

### Testing
1. Put an image of a dog/cat in the folder named **"images"**.
2. Run predict.py

### Model
```
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), padding='same', activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
```

## License
This project is licensed under the MIT License. 
