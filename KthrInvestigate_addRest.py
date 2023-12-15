"""Investigat add_rest nodes function of the KthrPianoroll AI"""

from IPython import display
import collections
import datetime
import fluidsynth
import glob
import numpy as np
import pathlib
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import pretty_midi
import seaborn as sns
import tensorflow as tf

from matplotlib import pyplot as plt
from typing import Dict, List, Optional, Sequence, Tuple

# Tomb added
import random
import pdb
import sys

data_dir = pathlib.Path('/Volumes/MAGIC1/CS50/myMusicGen/data/chorales')
if not data_dir.exists():
  tf.keras.utils.get_file(
      'midi',
      origin='https://github.com/jamesrobertlloyd/infinite-bach/tree/master/data/chorales/midi',
      extract=True,
      cache_dir='.', cache_subdir='data',
  )
filenames = glob.glob(str(data_dir/'**/*.mid*'))
# print(filenames)
print('Number of files:', len(filenames))

class UnsupportedMidiFileException(Exception):
  "Unsupported MIDI File"


def check_ones_zeros(array):
        count_ones = np.count_nonzero(array == 1)
        count_zeros = np.count_nonzero(array == 0)

        if count_ones == 1 and count_zeros == array.size - 1:
            print("There is only one '1' and the rest are '0's in the array.")
        elif count_zeros == array.size:
            print("All elements in the array are '0's.")
        else:
            print("There are either multiple '1's or different values in the array.")


def read_midi(filename, sop_alto, seqlen):
  
  def get_pianoroll(midi, nn_from, nn_thru, seqlen, tempo):
    pianoroll = midi.get_piano_roll(fs=2*tempo/60) # shape(128, 1262) This is the core line which makes this matrix based on 8th note
    if pianoroll.shape[1] < seqlen:
        raise UnsupportedMidiFileException

    pianoroll = pianoroll[nn_from:nn_thru, 0:seqlen] # (48, 64) Pinoroll's value still NOT binary since it has velocity
    
    binary_pianoroll = np.heaviside(pianoroll, 0) # converting as a binary matrix
    transposed_pianoroll = np.transpose(binary_pianoroll) #(64, 48)
    # transposed_pianoroll = np.transpose(pianoroll)
    # return binary_pianoroll
    return transposed_pianoroll # type numpy.ndarray


  def add_rest_nodes(pianoroll):  # If all the elemets are zero, the rest node says 1, else 0
    
    count_ones = np.count_nonzero(pianoroll == 1)
    count_zeros = np.count_nonzero(pianoroll == 0)
    
    #"There is only one '1' and the rest are '0's in the array."
    if count_ones == 1 and count_zeros == pianoroll.size - 1:
        rests = 1 - np.sum(pianoroll, axis=1)
        rests = np.expand_dims(rests, 1)
        return np.concatenate([pianoroll, rests], axis=1)
    
    #"All elements in the array are '0's."
    elif count_zeros == pianoroll.size:
        rests = 1 - np.sum(pianoroll, axis=1)
        rests = np.expand_dims(rests, 1)
        return np.concatenate([pianoroll, rests], axis=1)
    
    #"There are either multiple '1's or different values in the array."
    else:
        total_sum = np.sum(pianoroll, axis=1)
        rests = total_sum-total_sum 
        print(rests)
        rests = np.expand_dims(rests, 1)
        return np.concatenate([pianoroll, rests], axis=1)
    
    

  # read midi file
  midi = pretty_midi.PrettyMIDI(filename)

  # An Exception error is thrown if there is a modulation(key change)
  if len(midi.key_signature_changes) !=1:
    raise UnsupportedMidiFileException

  # Modulate the given key to C major or C minor
  key_number = midi.key_signature_changes[0].key_number
  # transpose_to_c(midi, key_number)

  # Get Major key(keynode=0) or Minor key(keynode=1)
  keymode = np.array([int(key_number / 12)])

  # The Exception error thrown when tempo changes
  tempo_time, tempo = midi.get_tempo_changes()
  if len(tempo) != 1:
    raise UnsupportedMidiFileException
 
    # The exception thrown if there are less than 2 parts
  if len(midi.instruments) < 2:
      raise UnsupportedMidiFileException
    
  else:
      #Get a pianoroll which gathered all the parts
      pr = get_pianoroll(midi, nn_from=36, nn_thru=84, seqlen=seqlen, tempo=tempo[0])
      pr_rest = add_rest_nodes(pr)
      return pr_rest, keymode
    

np.set_printoptions(threshold=np.inf) # Show the entire print, esp Matrix

x_all = [] 
y_all = [] 
files = [] 
# keymodes = [] 

raw_seq_length =64

for file in glob.glob(str(data_dir/"**/*.mid*")):
  files.append(file)
  try:
    # make a window to get sequence pairs (Xn, Xn+1) -> Xn+2
    wholePianoroll, keymode = read_midi(file, sop_alto=True, seqlen=raw_seq_length)
    desiredData = wholePianoroll
    for i in range(len(desiredData)-2): 
      
      input_sequence = desiredData[i:i+2]
      output_target = desiredData[i+2]

      x_all.append(input_sequence)
      y_all.append(output_target)

  except UnsupportedMidiFileException:
    print("nah")

# x_all = np.array(x_all)
# print(x_all[:,:, :]) # print the entire array
# print(x_all[:,:, -1:]) # print the last rest nodes
