import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import LambdaCallback
import numpy as np
import matplotlib.pyplot as plt

print("Hello")

class AutoencoderTrainer:
    def __init__(self, input_dim, learning_rate=0.001):
        self.model = self.create_autoencoder(input_dim)
        self.compile_autoencoder(learning_rate)
        self.input_dim = input_dim
        self.loss_history = []  # Initialize the list to store loss values

    def create_autoencoder(self, input_dim):
        model = Sequential()
        # Encoder
        model.add(Dense(units=4, activation='relu', input_shape=(input_dim,)))
        # Decoder
        model.add(Dense(units=input_dim, activation='sigmoid'))
        return model

    def compile_autoencoder(self, learning_rate):
        # Configure the optimizer with the specified learning rate
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    def print_epoch_output(self, epoch, logs):
        print(f'\nEpoch {epoch + 1}:')
        for i, sample in enumerate(self.X_train):
            predictions = self.model.predict(np.array([sample]))
            print(f'Model predictions for input {sample}:')
            print(predictions)

    def train_autoencoder(self, X_train, epochs=10, batch_size=1):
        self.X_train = X_train  # Save the training data for use in the callback

        # Callback to capture loss values
        class LossHistory(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                if logs is not None:
                    trainer.loss_history.append(logs.get('loss'))
        
        loss_history = LossHistory()
        print_callback = LambdaCallback(on_epoch_end=self.print_epoch_output)
        history = self.model.fit(X_train, X_train, epochs=epochs, batch_size=batch_size, verbose=0, callbacks=[print_callback, loss_history])

        # Plot the loss values
        plt.plot(self.loss_history)
        plt.title('Model Loss vs Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.show()

# Example usage
input_dim = 3
learning_rate = 1.01
trainer = AutoencoderTrainer(input_dim, learning_rate)

# Training on multiple data points
X_train = np.array([
    [1, 0, 1],
    [1, 1, 1],
    [0, 1, 0],
    [0, 0, 1]
])

trainer.train_autoencoder(X_train)