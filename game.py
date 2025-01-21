#############################################################
# Owners: Aditi Pawar, Himanish Goel, Shreyas Sawant
# File name: game.py
# Purpose: code for front-end.
#############################################################

import streamlit as st
import random
import numpy as np
import tensorflow as tf
from config import *
from data_handler_siamese import *
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import register_keras_serializable
import os
import cv2

# Constants
MAX_WORDS = 10000
MAX_LEN = 50

# GAMBLING_PATH = r"siamese_dataset\gambling"
# SKILL_PATH = r"siamese_dataset\skill"

# Register the custom contrastive loss function
@register_keras_serializable()
def contrastive_loss(y_true, y_pred, margin=0.1):
    """
    Custom contrastive loss function.
    """
    positive_loss = y_true * tf.square(y_pred)
    negative_loss = (1 - y_true) * tf.square(tf.maximum(margin - y_pred, 0))
    return tf.reduce_mean(positive_loss + negative_loss)

@st.cache_resource
def load_game_classifier():
    return load_model('game_classifier.keras',custom_objects={"contrastive_loss": contrastive_loss})

game_model = load_game_classifier()

# Load the pre-trained model
@st.cache_resource
def load_cached_model():
    return load_model('cyberbullying_rnn_model.h5')

try:
    model = load_cached_model()
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

# Predefined labeled conversations
chat_data = [
    {
        "label": "Not Cyberbullying",
        "messages": [
            ("Person 1", "Hey, saw your post about that game you’re playing. Any good?"),
            ("Person 2", "Yeah, it’s awesome. The graphics and story are next-level."),
            ("Person 1", "Nice. I’ve been looking for something new to try."),
            ("Person 2", "You should definitely check it out. It’s worth every penny."),
            ("Person 1", "Cool, I’ll grab it this weekend."),
            ("Person 2", "Let me know what you think. We can compare notes."),
            ("Person 1", "For sure. Do they have a multiplayer mode?"),
            ("Person 2", "Yeah, and it’s pretty solid. Super fun with friends."),
            ("Person 1", "Perfect. We should team up sometime."),
            ("Person 2", "Absolutely. Just send me your gamer tag."),
            ("Person 1", "Will do. By the way, any tips for beginners?"),
            ("Person 2", "Focus on the main quests first. The side stuff can wait."),
            ("Person 1", "Got it. Thanks, man!"),
            ("Person 2", "No problem. Happy gaming!")
        ]
    },
    {
        "label": "Religion-based Cyberbullying",
        "messages": [
            ("Person 1", "Why are they still allowing those people to practice their religion here?"),
            ("Person 2", "It’s a joke. They’re taking over everything."),
            ("Person 1", "Exactly. They don’t even try to respect our culture."),
            ("Person 2", "They never will. Their beliefs are too primitive."),
            ("Person 1", "And they expect us to just accept it? No way."),
            ("Person 2", "Right? It’s their fault things are getting worse."),
            ("Person 1", "Everywhere they go, they bring trouble. It’s a pattern."),
            ("Person 2", "But if we say anything, we’re the bad guys. Ridiculous."),
            ("Person 1", "The double standards are insane. It’s always about them."),
            ("Person 2", "They don’t contribute anything positive. Just chaos."),
            ("Person 1", "Exactly. They’re ruining the country one step at a time."),
            ("Person 2", "And we’re supposed to sit back and watch? Not happening."),
            ("Person 1", "Someone needs to put an end to this."),
            ("Person 2", "The sooner, the better.")
        ]
    },
    {
        "label": "Sexual Orientation-based Cyberbullying",
        "messages": [
            ("Person 1", "Look at that guy’s profile picture. What a joke."),
            ("Person 2", "Haha, yeah. He looks like he’s trying way too hard."),
            ("Person 1", "Probably thinks he’s some kind of hero for 'being himself.'"),
            ("Person 2", "More like a clown. No one cares about his 'struggles.'"),
            ("Person 1", "Exactly. He’s just looking for pity."),
            ("Person 2", "It’s pathetic. Why can’t he just act normal?"),
            ("Person 1", "Right? He’s so desperate for attention it’s embarrassing."),
            ("Person 2", "Someone should tell him to log off and stop humiliating himself."),
            ("Person 1", "I would, but I don’t think he’d get it. Too delusional."),
            ("Person 2", "True. People like him are always playing the victim."),
            ("Person 1", "It’s the only thing they’re good at."),
            ("Person 2", "No wonder no one takes them seriously."),
            ("Person 1", "Exactly. He’s a total waste of space."),
            ("Person 2", "Couldn’t have said it better.")
        ]
    },
    {
        "label": "Race-based Cyberbullying",
        "messages": [
            ("Person 1", "Why are they always so loud in public? It’s like they have no manners."),
            ("Person 2", "Exactly. It’s like they don’t know how to behave in civilized society."),
            ("Person 1", "And the way they dress? So tacky and cheap-looking."),
            ("Person 2", "Right? It’s embarrassing to even be around them."),
            ("Person 1", "I heard one of them yelling at a cashier the other day. Typical."),
            ("Person 2", "They always think they’re entitled to everything."),
            ("Person 1", "And then they wonder why no one respects them."),
            ("Person 2", "Respect? They don’t even respect themselves."),
            ("Person 1", "True. They just keep proving the stereotypes right."),
            ("Person 2", "Exactly. They bring all the hate on themselves."),
            ("Person 1", "I don’t even feel bad for them anymore."),
            ("Person 2", "Why should you? They’ll never change."),
            ("Person 1", "Exactly. They’re a lost cause."),
            ("Person 2", "Couldn’t agree more.")
        ]
    },
    {
        "label": "Gender-based Cyberbullying",
        "messages": [
            ("Person 1", "Why does she always have to post pictures of herself? So annoying."),
            ("Person 2", "She’s just desperate for likes. It’s pathetic."),
            ("Person 1", "Exactly. She’s not even that pretty, but she acts like she’s a model."),
            ("Person 2", "Right? Someone needs to tell her she’s not special."),
            ("Person 1", "I bet she spends hours editing those photos to look decent."),
            ("Person 2", "Of course she does. That’s all she’s good for."),
            ("Person 1", "She probably thinks she’s better than everyone else, too."),
            ("Person 2", "She’s nothing but a wannabe. So fake."),
            ("Person 1", "And the captions she writes? Cringe-worthy."),
            ("Person 2", "Totally. It’s like she’s trying to sound deep, but it’s just dumb."),
            ("Person 1", "I can’t stand girls like her. They’re all the same."),
            ("Person 2", "Exactly. She’s just another attention-seeking nobody."),
            ("Person 1", "Someone should call her out for how fake she is."),
            ("Person 2", "She wouldn’t handle it. Too fragile.")
        ]
    }
]

def get_random_image():
    """Get a random image from either gambling or skill folder"""
    folders = [GAMBLING_PATH, SKILL_PATH]
    chosen_folder = random.choice(folders)
    
    try:
        images = [f for f in os.listdir(chosen_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
        if not images:
            st.error(f"No images found in {chosen_folder}")
            return None, None
        
        chosen_image = random.choice(images)
        image_path = os.path.join(chosen_folder, chosen_image)

        return image_path
    except Exception as e:
        st.error(f"Error accessing image folders: {str(e)}")
        return None, None

def get_support_set():
    folders = [GAMBLING_PATH, SKILL_PATH]
    image_path = []
    for folder in folders:
        try:
            images = [f for f in os.listdir(folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
            if not images:
                st.error(f"No images found in {folder}")
                return None, None
            
            chosen_image = random.choice(images)
            image_path.append(os.path.join(folder, chosen_image))

        except Exception as e:
            st.error(f"Error accessing image folders: {str(e)}")
            return None, None
        
    return image_path

def preprocess_image(image_path):
    """Preprocess the image for prediction"""
    image = cv2.imread(image_path)
    image = cv2.resize(image, (256, 256))  # Resize to match model's expected input size
    image = img_to_array(image) / 255.0  # Normalize pixel values
    # image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def get_embeddings(data):
    embeddings = game_model.predict(data, verbose = 0)
    return embeddings

def predict_image(image):
    """Make prediction using the game classifier model"""
    predictions = game_model.predict(image, verbose=0)
    predicted_label_index = np.argmax(predictions[0])
    
    # Assume the model has been trained on two classes: 'Gambling' and 'Skill'
    # classes = ['Gambling', 'Skill']
    
    # predicted_label = classes[predicted_label_index % 2]
    confidence = float(predictions[0][predicted_label_index])
    
    return predictions, confidence

def make_decision(support_embeddings, predictions, labels):
    distances = np.linalg.norm(support_embeddings - predictions, axis=1)
    best_match_index = np.argmin(distances)

    return labels[best_match_index]

def preprocess_chat(chat_messages):
    """Preprocess chat messages for model input"""
    try:
        # Join all messages into a single string
        chat_text = " ".join([msg for _, msg in chat_messages])

        # Initialize tokenizer
        tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
        tokenizer.fit_on_texts([chat_text])

        # Convert text to sequences and pad
        chat_sequence = tokenizer.texts_to_sequences([chat_text])
        chat_padded = pad_sequences(chat_sequence, maxlen=MAX_LEN, padding='post', truncating='post')

        return chat_padded
    except Exception as e:
        st.error(f"Error in preprocessing: {str(e)}")
        return None

def make_prediction(chat_padded):
    """Make prediction using the model"""
    try:
        predictions = model.predict(chat_padded, verbose=0)
        predicted_label_index = np.argmax(predictions[0])
        
        # Initialize label encoder with known classes
        enc = LabelEncoder()
        enc.classes_ = np.array([
            'Gender-based Cyberbullying',
            'Not Cyberbullying',
            'Race-based Cyberbullying',
            'Religion-based Cyberbullying',
            'Sexual Orientation-based Cyberbullying'
        ])
        
        predicted_label = enc.classes_[predicted_label_index]
        confidence = float(predictions[0][predicted_label_index])
        
        return predicted_label, confidence
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        return None, None

# Sidebar with two tabs
st.sidebar.title("Navigation")
tabs = ["Predict the Type of the Game", "Chat"]
selection = st.sidebar.radio("Choose a tab:", tabs)

# Tab 1: Predict the Type of the Game
if selection == "Predict the Type of the Game":
    st.title("Predict the Type of the Game")
    support_set = []
    support_set_paths = get_support_set()
    image_path = get_random_image()
    
    if image_path:
        for path in support_set_paths:
            support_set.append(preprocess_image(path))
        
        support_set = np.array(support_set)
        support_labels = np.array(["Gambling", "Skills"])
        image = preprocess_image(image_path)
        image = np.expand_dims(image, axis=0)
        st.image(image[0])  # Display the image
        
        # Make prediction
        support_embeddings = get_embeddings(support_set)
        predictions, confidence = predict_image(image)

        predicted_label = make_decision(support_embeddings, predictions, support_labels)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Predicted Type:")
            st.write(predicted_label)
        with col2:
            st.subheader("Confidence:")
            st.write(f"{confidence:.2%}")
# Tab 2: Chat
elif selection == "Chat":
    st.title("Chat")

    # Initialize session state for current chat
    if "current_chat" not in st.session_state:
        st.session_state["current_chat"] = random.choice(chat_data)

    # Display the messages with person labels
    chat_display = ""
    for speaker, message in st.session_state["current_chat"]["messages"]:
        chat_display += f"**{speaker}:** {message}\n\n"

    st.text_area("Chat Box", value=chat_display, height=300, disabled=True)

    # Process chat and make prediction
    chat_padded = preprocess_chat(st.session_state["current_chat"]["messages"])
    
    if chat_padded is not None:
        predicted_label, confidence = make_prediction(chat_padded)
        
        if predicted_label is not None:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Predicted Type:")
                st.write(predicted_label)
            with col2:
                st.subheader("Confidence:")
                st.write(f"{confidence:.2%}")

    # Refresh button to randomize the chat
    if st.button("Refresh"):
        st.session_state["current_chat"] = random.choice(chat_data)
        st.rerun()

# Add some spacing at the bottom
st.markdown("<br><br>", unsafe_allow_html=True)