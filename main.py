#############################################################
# Owners: Aditi Pawar, Himanish Goel, Shreyas Sawant
# File name: main.py
# Purpose: Main file for calling all the other codes. 
#############################################################

import pandas as pd
from config import *
from data_handler import *
from data_preprocessor import *
from model import *
from data_handler_siamese import *

def RNN_ops():
    data = pd.read_csv(DATA_PATH)
    data1 = data[5000:9000]
    # data2 = data[12000:14000]
    data3 = data[18000:20000]
    # data4 = data[24000:26000]
    # data5 = data[28000:30000]
    # data6 = data[36000:38000]
    
    final_data = pd.concat([data1, data3])
    # final_data2 = pd.concat([final_data1, data3])
    # final_data = pd.concat([final_data1, data4])
    # final_data4 = pd.concat([final_data3, data5])
    # final_data = pd.concat([final_data4, data6])
    final_data['cleaned_text'] = final_data['tweet_text'].apply(data_cleaner)

    # Remove rows with None values (non-English sentences)
    final_data = final_data.dropna(subset=['cleaned_text'])

    X_train_pad, X_test_pad, X_val_pad, y_train, y_test, y_val, label_encoder = preprocess_data(final_data)

    model_obj = Model_rnn()
    model_obj.build_model(label_encoder)
    print(model_obj.model.summary())
    model_obj.train_model(X_train_pad, y_train, X_val_pad, y_val)
    model_obj.evaluate_model(X_test_pad, y_test)
    model_obj.save_model()

def siamese_ops():
    X_train, X_test, y_train, y_test = image_data_reader()

    X_pairs, y_pairs = make_paired_dataset(X_train, y_train)
    X_test_pair, y_test_pair = make_paired_dataset(X_test, y_test)

    model_obj = Siamese_Net()
    model_obj.build_model()
    print(model_obj.model.summary())
    model_obj.train_model(X_pairs, y_pairs, X_test_pair, y_test_pair)
    model_obj.evaluate_model(X_test_pair, y_test_pair)
    model_obj.save_model()
    model_obj.visualize_predictions(X_test_pair, y_test_pair)

def main():
    RNN_ops()
    # siamese_ops()

    
if __name__ == '__main__':
    main()