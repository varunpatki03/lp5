# %%
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
#df = pd.read_csv("//content/IMDB Dataset.csv")


# Map sentiment labels to numerical values
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# Split data into training and testing sets
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['review'], df['sentiment'], test_size=0.2, random_state=42
)

# Tokenization parameters
vocab_size = 20000  # Limit vocabulary size
max_length = 500  # Max length of sequences
embedding_dim = 128  # Embedding dimension

# Tokenize and convert text to sequences
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(train_texts)

train_sequences = tokenizer.texts_to_sequences(train_texts)
test_sequences = tokenizer.texts_to_sequences(test_texts)

# Pad sequences
train_padded = pad_sequences(train_sequences, maxlen=max_length, padding='post', truncating='post')
test_padded = pad_sequences(test_sequences, maxlen=max_length, padding='post', truncating='post')

# Convert labels to numpy arrays
train_labels = np.array(train_labels)
test_labels = np.array(test_labels)

# Build DNN model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Conv1D(64, 5, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(train_padded, train_labels, validation_data=(test_padded, test_labels), epochs=5, batch_size=64)

# Evaluate model
predictions = (model.predict(test_padded) > 0.5).astype("int32")
accuracy = accuracy_score(test_labels, predictions)
print(f"Test Accuracy: {accuracy:.4f}")


# %%
import kagglehub
import pandas as pd
import os

# Download the dataset
path = kagglehub.dataset_download("lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")

# Find the actual CSV file inside the downloaded path
csv_file = os.path.join(path, "IMDB Dataset.csv")

# Load the dataset into pandas
df = pd.read_csv(csv_file, encoding="utf-8")

# Display dataset info
print("Dataset Loaded Successfully!")
print(df.head())


# %%
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

# Ensure TensorFlow uses GPU
print("Num GPUs Available:", len(tf.config.experimental.list_physical_devices('GPU')))
tf.config.experimental.list_physical_devices('GPU')


# Convert sentiment labels to numerical values
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# Split data into training and testing sets
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['review'], df['sentiment'], test_size=0.2, random_state=42
)

# Tokenization parameters
vocab_size = 20000  # Limit vocabulary size
max_length = 500  # Max length of sequences
embedding_dim = 128  # Embedding dimension

# Tokenize and convert text to sequences
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(train_texts)

train_sequences = tokenizer.texts_to_sequences(train_texts)
test_sequences = tokenizer.texts_to_sequences(test_texts)

# Pad sequences
train_padded = pad_sequences(train_sequences, maxlen=max_length, padding='post', truncating='post')
test_padded = pad_sequences(test_sequences, maxlen=max_length, padding='post', truncating='post')

# Convert labels to numpy arrays
train_labels = np.array(train_labels)
test_labels = np.array(test_labels)

# Activation functions to test
activation_functions = ['relu', 'tanh', 'sigmoid', 'leaky_relu']

# Store results
results = []

for activation in activation_functions:
    print(f"\nTraining model with activation: {activation}")

    # Build model
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.Conv1D(64, 5, activation='relu'),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(32, activation=activation),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')  # Final layer for binary classification
    ])

    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train model
    history = model.fit(train_padded, train_labels, validation_data=(test_padded, test_labels), epochs=5, batch_size=64, verbose=1)

    # Predictions
    predictions = (model.predict(test_padded) > 0.5).astype("int32")

    # Metrics
    accuracy = accuracy_score(test_labels, predictions)
    mse = mean_squared_error(test_labels, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(test_labels, predictions)
    loss = history.history['val_loss'][-1]  # Last validation loss

    # Store results
    results.append([activation, accuracy, loss, mse, rmse, r2])

# Convert results to DataFrame and print
results_df = pd.DataFrame(results, columns=['Activation Function', 'Accuracy', 'Loss', 'MSE', 'RMSE', 'R² Score'])
print("\n📌 Model Performance Summary:")
print(results_df)

# Print best result
best_model = results_df.loc[results_df['Accuracy'].idxmax()]
print("\n🏆 Best Model Configuration:")
print(best_model)
