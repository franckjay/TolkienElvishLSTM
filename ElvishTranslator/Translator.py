# -*- coding: utf-8 -*-
"""
Spyder Editor
written_by: J Franck

All Data scraped from: http://eldamo.org/
Inspired by: https://github.com/keras-team/keras/blob/master/examples/lstm_seq2seq.py
"""
import numpy as np
import pandas as pd
import keras

def raw_data_reader(raw_file_path,out_path):
    dat=open(raw_file_path,"r",encoding="utf-8")
    out=open(out_path,"w",encoding="utf-8")
    for i in dat:
        try:
            tmp=i.split("“")
        except:
            print ("****BAD***LINE***********")
        tmp[0]=tmp[0].strip("”*[]†²\n\t")
        tmp[1]=tmp[1].strip("”*[]†²\n\t")
        out.write(tmp[1]+",\t"+tmp[0]+"\n")
    out.close()

def encodeData(csvFilePath):
    inSeq=[]#Input phrases/sequences
    outSeq=[]
    inChars=set()#Characters used in input
    outChars=set()
    
    csvFile=open(csvFilePath,"r",encoding="utf-8")
    for i in csvFile:
        tmp=(i.split(","))
        inSeq.append(tmp[0])#Append sentences to list
        outSeq.append(tmp[1])#Elvish phrases list
        #Add to the list of English Characters
        for j in tmp[0]:
            for tmpChar in j:
                if tmpChar not in inChars:
                    inChars.add(tmpChar)
        #...significantly more elvish Characters            
        for j in tmp[1]:
            for tmpChar in j:
                if tmpChar not in outChars:
                    outChars.add(tmpChar)
                    
    return inSeq,outSeq,sorted(list(inChars)),sorted(list(outChars))
    
    

process_raw=True
if process_raw:
    inPath,outPath="raw_data/","processed_data/"
    raw_data_reader(inPath+"Sindarin_Phrases.dat",outPath+"Sindarin_Phrases.csv")
    raw_data_reader(inPath+"Quenya_Phrases.dat",outPath+"Quenya_Phrases.csv")

quenya=True
if quenya:
    print ("Loading the Quenya Phrase dataset")
    input_texts,target_texts,input_characters,target_characters=encodeData(outPath+"Quenya_Phrases.csv")
    modelPath="models/Quenya_LSTM.h5"
else:
    input_texts,target_texts,input_characters,target_characters=encodeData(outPath+"Sindarin_Phrases.csv") 
    modelPath="models/Sindarin_LSTM.h5"


num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])

print('Number of samples:', len(input_texts))
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)

input_token_index = dict(
    [(char, i) for i, char in enumerate(input_characters)])
target_token_index = dict(
    [(char, i) for i, char in enumerate(target_characters)])

encoder_input_data = np.zeros(
    (len(input_texts), max_encoder_seq_length, num_encoder_tokens),
    dtype='float32')
decoder_input_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')
decoder_target_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')

for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, char in enumerate(input_text):
        encoder_input_data[i, t, input_token_index[char]] = 1.
    for t, char in enumerate(target_text):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data[i, t, target_token_index[char]] = 1.
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            decoder_target_data[i, t - 1, target_token_index[char]] = 1.



from keras.models import Model
from keras.layers import Input, LSTM, Dense          

batch_size = 64  # Batch size for training.
epochs = 100 # Number of epochs to train for.
latent_dim = 512  # Latent dimensionality of the encoding space.
num_samples = 10000  # Number of samples to train on.  

# Define an input sequence and process it.
encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None, num_decoder_tokens))
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                     initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
elvenLSTM = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Run training
elvenLSTM.compile(optimizer='rmsprop', loss='categorical_crossentropy')

elvenLSTM.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2)
# Save model
elvenLSTM.save(modelPath)

# Next: inference mode (sampling).
# Here's the drill:
# 1) encode input and retrieve initial decoder state
# 2) run one step of decoder with this initial state
# and a "start of sequence" token as target.
# Output will be the next target token
# 3) Repeat with the current target token and current states

# Define sampling models
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

# Reverse-lookup token index to decode sequences back to
# something readable.
reverse_input_char_index = dict(
    (i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict(
    (i, char) for char, i in target_token_index.items())


def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, target_token_index['\t']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '\n' or
           len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence


for seq_index in range(100):
    # Take one sequence (part of the training set)
    # for trying out decoding.
    input_seq = encoder_input_data[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print('-')
    print ('Input sentence:', input_texts[seq_index])
    print ('Decoded sentence:', decoded_sentence)
    print ("Actual sentence: ",target_texts[seq_index])
