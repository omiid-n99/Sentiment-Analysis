import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding, SpatialDropout1D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from google.colab import drive

# Load dataset
drive.mount('/content/drive')
df = pd.read_csv('/content/drive/My Drive/Prozhe Karshenasi/traintrain.csv')
# df = pd.read_csv('traintrain.csv')

df.head()

df.columns

# Select columns of interest
tweet_df = df[['text', 'sentiment']]
print(tweet_df.shape)
tweet_df.head(5)

# Remove neutral sentiment rows
# tweet_df = tweet_df[tweet_df['sentiment'] != 'neutral']
# print(tweet_df.shape)
# tweet_df.head(5)

# Convert sentiment labels to numerical values
sentiment_dict = {'negative': 0, 'neutral': 1, 'positive': 2}
tweet_df['sentiment'] = tweet_df['sentiment'].map(sentiment_dict)

sentiment_dict

tweet_df["sentiment"].value_counts()

# Tokenize text data
tweet = tweet_df.text.values
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(tweet.astype('str'))
vocab_size = len(tokenizer.word_index) + 1
encoded_docs = tokenizer.texts_to_sequences(tweet.astype('str'))
padded_sequence = pad_sequences(encoded_docs, maxlen=200)

print(tokenizer.word_index)

print(tweet[0])
print(encoded_docs[0])

print(padded_sequence[0])

# Split data into training and testing sets
train_x, test_x, train_y, test_y = train_test_split(padded_sequence, tweet_df['sentiment'], test_size=0.2, random_state=42)

# Define the model architecture
embedding_vector_length = 32
model = Sequential()
model.add(Embedding(vocab_size, embedding_vector_length, input_length=200))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(128, dropout=0.4, recurrent_dropout=0.4))
model.add(Dense(3, activation='softmax'))

# Define the optimizer with a lower learning rate
optimizer = Adam(learning_rate=0.0001)

# Compile the model with the new optimizer
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Define early stopping callback
earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto')

# Fit the model
history = model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=20, batch_size=64, callbacks=[earlystop])

# Evaluate the model
loss, accuracy = model.evaluate(test_x, test_y, verbose=0)
print('Accuracy: {:.2f}%'.format(accuracy*100))

# Plot the accuracyand loss curves
plt.plot(history.history['accuracy'], label='Training accuracy')
plt.plot(history.history['val_accuracy'], label='Validation accuracy')
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.title('Model accuracy and loss')
plt.xlabel('Epoch')
plt.ylabel('Accuracy/Loss')
plt.legend()
plt.show()

# Define function to predict sentiment of input text
def predict_sentiment(model, tokenizer, input_text):
    # Tokenize and pad the input text
    encoded_text = tokenizer.texts_to_sequences([input_text])
    padded_text = pad_sequences(encoded_text, maxlen=200)
    
    # Predict the sentiment using the trained model
    sentiment_prediction = model.predict(padded_text)
    
    # Return the predicted sentiment label
    sentiment_dict = {0: 'negative', 1: 'neutral', 2: 'positive'}
    predicted_sentiment = sentiment_dict[np.argmax(sentiment_prediction)]
    return predicted_sentiment

# Take input text from user
while True:
    input_text = input("Enter the text to predict the sentiment: ")

    if input_text == "":
        break
    # Predict the sentiment label of the input text
    predicted_sentiment = predict_sentiment(model, tokenizer, input_text)

    # Print the predicted sentiment label
    print("Predicted sentiment: ", predicted_sentiment)