import numpy as np
from tensorflow.keras.models import load_model
from music21 import converter, instrument, note, chord, stream, duration

# Load the trained model
model = load_model('trained_model.h5')

# Set the sequence length for generating new sequences
sequence_length = 128

# Generate a random seed sequence
seed = np.random.randint(0, 128, size=(1, sequence_length))

# Generate notes using the model
prediction_output = model.predict(seed)

# Create a MIDI stream
output_stream = stream.Stream()
output_stream.append(instrument.Piano())

# Add notes and chords to the stream
for pattern in prediction_output:
    try:
        # If the pattern is a chord
        notes = [note.Note(int(n), duration=duration.Duration(0.5)) for n in pattern]
        chord_obj = chord.Chord(notes)
        output_stream.append(chord_obj)
    except:
        # If not a chord, consider it as a note
        note_obj = note.Note(int(pattern), duration=duration.Duration(0.5))
        output_stream.append(note_obj)

# Write the stream to a MIDI file
output_stream.write('midi', fp='path_to_output_midi_file.mid')
