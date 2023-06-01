import tensorflow as tf
import numpy as np
from keras.preprocessing import sequence


# Define the training data
training_data = [
    [60, 62, 64, 65, 67, 69, 71, 72],  # C major scale
    [60, 62, 63, 65, 67, 68, 70, 72],  # C minor scale
    [60, 62, 64, 67, 69, 71, 74, 76],  # C major arpeggio
    [60, 63, 67, 70, 73],  # C5 chord
    [67, 71, 74, 76, 79],  # G5 chord
]

# Convert the training data into input-output pairs
sequences = []
next_note = []
for melody in training_data:
    for i in range(len(melody) - 1):
        sequences.append(melody[:i+1])
        next_note.append(melody[i+1])

# Pad sequences to ensure equal length
max_sequence_length = max(map(len, sequences))
sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_sequence_length)

# Convert the sequences and next_note to numpy arrays
sequences = np.array(sequences)
next_note = np.array(next_note)

# Define the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(88, 10, input_length=max_sequence_length),
    tf.keras.layers.GRU(128, return_sequences=True),
    tf.keras.layers.GRU(128),
    tf.keras.layers.Dense(88, activation='softmax')
])

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

# Train the model
model.fit(sequences, next_note, epochs=100, verbose=2)

# Generate a new melody
# seed_sequence = [60] 
# modify the code to ensure that seed_sequence is a 2D array before passing it to the pad_sequences function.
seed_sequence = np.array([[60]])  # Seed sequence for the generator
generated_melody = seed_sequence.copy()

for _ in range(10):  # Generate 10 notes
    input_sequence = sequence.pad_sequences([generated_melody[-1]], maxlen=max_sequence_length, padding='pre')
    predicted_note = model.predict(input_sequence)[0]
    generated_melody = np.column_stack((generated_melody, [predicted_note]))
    seed_sequence = np.column_stack((seed_sequence, [predicted_note]))

# Print the generated melody
print("Generated Melody:", np.array(generated_melody))
