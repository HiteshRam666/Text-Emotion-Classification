# Text Emotion Classification 📚😊😂

Welcome to the Text Emotion Classification project! This project aims to classify emotions from text data using a deep learning model.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)

## Overview 🚀

This project leverages deep learning to classify emotions from text. The model is built using TensorFlow/Keras and processes input text to predict the associated emotion.

## Features 🌟

- Text preprocessing and tokenization ✨
- Deep learning model with Embedding and Dense layers 🧠
- Training and validation accuracy and loss plots 📈
- Predicting emotions from new text inputs 🎯

## Installation 🔧

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/text-emotion-classification.git
    ```
2. Navigate to the project directory:
    ```bash
    cd text-emotion-classification
    ```
3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage 🛠️

1. **Train the Model**:
    ```python
    import matplotlib.pyplot as plt
    from keras.models import Sequential
    from keras.layers import Embedding, Flatten, Dense
    from keras.preprocessing.sequence import pad_sequences

    # Define the model
    model = Sequential()
    model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=128, input_length=max_length))
    model.add(Flatten())
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=len(one_hot_labels[0]), activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Fit the model
    history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    # Plot loss and accuracy curves
    plt.figure(figsize=(14, 5))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()
    ```

2. **Predict Emotions from New Text**:
    ```python
    import numpy as np
    from keras.preprocessing.sequence import pad_sequences

    input_text = "I am very happy, I can't stop laughing"

    # Preprocess the input text
    input_sequences = tokenizer.texts_to_sequences([input_text])
    padded_input_sequence = pad_sequences(input_sequences, maxlen=max_length)

    # Predict the sentiment
    prediction = model.predict(padded_input_sequence)
    predicted_label = label_encoder.inverse_transform([np.argmax(prediction[0])])

    # Print the result based on the prediction probabilities
    if np.max(prediction[0]) > 0.5:
        print(f"The given text is positive: {predicted_label[0]}")
    else:
        print(f"The given text is negative: {predicted_label[0]}")
    ```

## Results 📊

The model shows promising results with training and validation accuracy and loss plots. It can accurately predict emotions from new text inputs.
