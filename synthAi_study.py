from IPython import display
import collections
import datetime
import fluidsynth
import glob
import numpy as np
import pathlib
import pandas as pd
import pretty_midi
import seaborn as sns
import tensorflow as tf

from matplotlib import pyplot as plt
from typing import Dict, List, Optional, Sequence, Tuple

seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

# Sampling rate for audio playback
_SAMPLING_RATE = 16000

# Download the maestro dataset of 1282 files
data_dir = pathlib.Path('data/maestro-v2.0.0')
if not data_dir.exists():
  tf.keras.utils.get_file(
      'maestro-v2.0.0-midi.zip',
      origin='https://storage.googleapis.com/magentadata/datasets/maestro/v2.0.0/maestro-v2.0.0-midi.zip',
      extract=True,
      cache_dir='.', cache_subdir='data',
  )

# The data above contains 1200+ Midi files
filenames = glob.glob(str(data_dir/'**/*.mid*'))
print('Number of files:', len(filenames))

# Process a MIDI file
sample_file = filenames[1]
print(sample_file)

# generate a prettyMIDI object for the sample MIDI file
pm = pretty_midi.PrettyMIDI(sample_file)

# Play the sample file
def display_audio(pm: pretty_midi.PrettyMIDI, seconds=60):
    waveform = pm.fluidsynth(fs=float(_SAMPLING_RATE))
    # Take a sample of the generated waveform to mitigate kernel resets
    waveform_short = waveform[:seconds*_SAMPLING_RATE]
    return display.Audio(waveform_short, rate=_SAMPLING_RATE)

display_audio(pm)