#############################################################
# Owners: Aditi Pawar, Himanish Goel, Shreyas Sawant
# File name: data_preprocessor.py
# Purpose: Making the data suitable for feeding to a model.
#############################################################

from config import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

def preprocess_data(data):
    # Encode the labels
    label_encoder = LabelEncoder()
    data['label'] = label_encoder.fit_transform(data['cyberbullying_type'])

    # Split the data into training and testing sets
    X = data['cleaned_text']
    y = data['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.05, random_state=42, stratify=y_train)

    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)

    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_val_seq = tokenizer.texts_to_sequences(X_val)
    X_test_seq = tokenizer.texts_to_sequences(X_test)

    X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_LEN, padding='post', truncating='post')
    X_val_pad = pad_sequences(X_val_seq, maxlen=MAX_LEN, padding='post', truncating='post')
    X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_LEN, padding='post', truncating='post')

    y_train = to_categorical(y_train, num_classes = len(label_encoder.classes_))
    y_val = to_categorical(y_val, num_classes = len(label_encoder.classes_))
    y_test = to_categorical(y_test, num_classes = len(label_encoder.classes_))

    return X_train_pad, X_test_pad, X_val_pad, y_train, y_test, y_val, label_encoder