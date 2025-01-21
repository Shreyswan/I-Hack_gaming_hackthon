#######################################################
# Owners: Aditi Pawar, Himanish Goel, Shreyas Sawant
# File name: config.py
# Purpose: Contains config data.
#######################################################

DATA_PATH = "clean_dataset.csv"
FALLBACK_STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
    'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
    'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
    'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
    'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
    'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
    'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
    'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
    'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
    'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
    'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'
}

TOKENIZER_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"
MAX_WORDS = 10000 # Vocabluary size
MAX_LEN = 100 # Vocabulary size

BATCH_SIZE = 32
EPOCHS = 20

MODEL_FILE_NAME = "cyberbullying_rnn_model.h5"

############################################################################################
# Config for the Siamese network:
############################################################################################

DATA_PATH_SIAMESE = "/Users/shreyassawant/mydrive/shreyus_workspace/IIT_gaming_hackathon/siamese_dataset"
SIAMESE_FILE_NAME = "game_classifier.keras"

GAMBLING_PATH = "/Users/shreyassawant/mydrive/shreyus_workspace/IIT_gaming_hackathon/siamese_dataset/gambling/"
SKILL_PATH = "/Users/shreyassawant/mydrive/shreyus_workspace/IIT_gaming_hackathon/siamese_dataset/skill/"