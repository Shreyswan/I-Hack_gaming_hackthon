###############################################################
# Owners: Aditi Pawar, Himanish Goel, Shreyas Sawant
# File name: data_preprocessor.py
# Purpose: Code for building, training and testing the model.
###############################################################

from config import *
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, BatchNormalization, MaxPooling2D
from tensorflow.keras.layers import Input, Conv2D, ReLU, Reshape, GlobalAveragePooling2D, Subtract, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model
from tensorflow.keras.saving import register_keras_serializable

@register_keras_serializable(package="Custom", name="contrastive_loss")
def contrastive_loss(y_true, y_pred, margin = 0.1):
    positive_loss = y_true * tf.square(y_pred)
    negative_loss = (1-y_true)*tf.square(tf.maximum(margin - y_pred , 0))
    return tf.reduce_mean(positive_loss+negative_loss)

class Model_rnn:
    def build_model(self, label_encoder):
        # Build the RNN model
        self.model = Sequential()
        self.model.add(Embedding(input_dim=MAX_WORDS, output_dim=64, input_length=MAX_LEN))
        self.model.add(Bidirectional(LSTM(64, return_sequences = True)))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.2))

        self.model.add(Bidirectional(LSTM(64)))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.2))

        # self.model.add(Bidirectional(LSTM(32)))
        # self.model.add(BatchNormalization())
        # self.model.add(Dropout(0.3))

        self.model.add(Dense(32, activation = "relu"))
        self.model.add(Dense(len(label_encoder.classes_), activation = "softmax"))
        self.model.build((None, MAX_LEN))

        # Compile the model
        self.model.compile(optimizer = Adam(), loss = 'categorical_crossentropy', metrics = ['accuracy'])

    def train_model(self, X_train_pad, y_train, X_val_pad, y_val):
        early_stopping = EarlyStopping(monitor = 'val_loss', patience = 5, restore_best_weights = True)

        # Train the model
        self.history = self.model.fit(X_train_pad, y_train, 
                                      epochs = EPOCHS, 
                                      batch_size = BATCH_SIZE, 
                                      validation_data = (X_val_pad, y_val),
                                      callbacks=[early_stopping])

    def evaluate_model(self, X_test_pad, y_test):
        loss, accuracy = self.model.evaluate(X_test_pad, y_test)
        print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

    def save_model(self):
        # Save the model
        self.model.save(MODEL_FILE_NAME)

class Siamese_Net:
    def build_model(self):
        img_left = Input((256, 256, 3), name = "img_left")
        img_right = Input((256, 256, 3), name = "img_right")

        cnn = Sequential()
        cnn.add(Reshape((256, 256, 3)))
        cnn.add(Conv2D(64, (5, 5), padding = "same"))
        cnn.add(BatchNormalization())
        cnn.add(MaxPooling2D())
        cnn.add(ReLU())
        cnn.add(Conv2D(64, (5, 5), padding = "same"))
        cnn.add(BatchNormalization())
        cnn.add(MaxPooling2D())
        cnn.add(ReLU())
        cnn.add(GlobalAveragePooling2D())
        self.cnn = cnn
        # cnn.add(Dense(64, activation = "relu"))

        feature_left  = cnn(img_left)
        feature_right = cnn(img_right)

        # distance = Lambda(lambda tensors:abs(tensors[0] - tensors [1]))([feature_left, feature_right])
        difference_layer = Subtract()([feature_left, feature_right])
        output = Dense(1, activation = "sigmoid")(difference_layer)

        self.model = Model(inputs=[img_left, img_right], outputs = output)
        self.model.compile(optimizer = Adam(learning_rate = 0.001), loss = contrastive_loss, metrics = ['accuracy'])

    def train_model(self, x_pair, y_pair, x_test_pair, y_test_pair):
        early_stopping = EarlyStopping(monitor = 'val_accuracy', patience = 3, restore_best_weights = True)

        # Train the model
        self.model.fit(x = [x_pair[:, 0, :, :], x_pair[:, 1, :, :]],
                       y = y_pair,
                       validation_data = ([x_test_pair[:, 0, :, :], x_test_pair[:, 1, :, :]],y_test_pair),
                       epochs = 3, 
                       batch_size = 16,
                       callbacks=[early_stopping])
        
    def evaluate_model(self, X_test_pad, y_test):
        loss, accuracy = self.model.evaluate(
        [X_test_pad[:, 0], X_test_pad[:, 1]], 
        y_test
        )

        print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

    def save_model(self):
        # Save the model
        self.cnn.save(SIAMESE_FILE_NAME)

    def visualize_predictions(self, test_pairs, test_pair_labels, num_pairs=10):
        preds = self.model.predict([test_pairs[:, 0], test_pairs[:, 1]])
        fig, axes = plt.subplots(num_pairs, 2, figsize=(10, 20))
        
        for i in range(num_pairs):
            axes[i, 0].imshow(test_pairs[i, 0], cmap="gray")
            axes[i, 0].set_title(f"Image 1 Label: {test_pair_labels[i]}")
            axes[i, 0].axis("off")
            
            axes[i, 1].imshow(test_pairs[i, 1], cmap="gray")
            axes[i, 1].set_title(f"Image 2 - Predicted: {preds[i][0]:.4f}")
            axes[i, 1].axis("off")
        
        plt.tight_layout()
        plt.show()