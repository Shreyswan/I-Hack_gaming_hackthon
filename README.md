# Gaming protection system 💪
## Files in the repo: 
* **Config.py**: This is the file which contains all the constants in the code
* **data_handler.py**: This is the file that cleans the text data for RNN.
* **data_handler_siamese.py**: This is the file that preprocesses and makes pairs of the image dataset for the Siamese model.
* **data_preprocessor.py**: This is the file that preprocesses textual data by tokenizing and padding it for the RNN.
* **model.py**: This file contains classes for both the models, it builds, trains, evaluates and saves the models.
* **main.py**: This file is the main file which simply calls all the functions and trains the models.
* **game.py**: This file is the front-end code that prepares the front-end using the streamlit library from Python. It leverages the power of our models for classification.
* **cyberbullying_rnn_model.h5**: This file is the H5 file for RNN model.
* **game_classifier.keras**: This is the CNN used by the Siamese network.

## Siamese Network: 
This is the state-of-the-art model which is trained from images that have been scraped from the web. It does Binary classification of a game into either Gambling or Skill-based games. It uses a multilayer CNN for extracting features of the images and provides it's embeddings as the output. There is one CNN which is given 2 images for generating embeddings. The difference between these images is calculated and then the similarity score is calculated with the help of the Sigmoid Activation function. Thus the Siamese Net understands the differences/similarities between 2 images
Training Accuracy: 68.48% Test Accuracy: 69.44%

## RNN Model: 
This is a Recurrent Neural Network that is trained on Twitter's abusive comments. It also has multiple Bidirectional LSTM layers which capture the variation in the dataset nicely. It is able to classify between 6 classes of cyber bullying namely: Gender-based, Age-based, Race-based, etc. 
Training Accuracy: 91.00% Test Accuracy: 66.67%
