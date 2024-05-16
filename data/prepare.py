"""
Prepare the EPL Match Commentary dataset for character-level language modeling.
Instead of encoding with 'GPT-2 BPE tokens', we just map characters to integers.
`train.bin` & `val.bin` will contain the dataset binaries; `meta.pkl` will contain
the encoder and decoder checkpoints, and some other related info.
"""

import os
import pickle
import requests
import numpy as np

# download the epl match-commentary dataset
__file__ = "C:/Users/salva/Desktop/projects/eplGPT/data/match_commentary.txt"
input_file_path = os.path.join(os.path.dirname(__file__), "match_commentary.txt")
if not os.path.exists(input_file_path):
    data_url = ""
    with open(input_file_path, "w", encoding="utf-8") as file:
        file.write(requests.get(data_url).text)

# read data as single text
with open(input_file_path, "r", encoding="utf-8") as file:
    data = file.read()
print(f"Length of dataset in characters: {len(data):,}")

# get all unique characters present in text
chars = sorted(list(set(data)))
vocab_size = len(chars)
print(f"Unique characters present in dataset are: {''.join(chars)} ==>> {vocab_size:,} characters")

# mapping from characters to integers and vice-versa
char_to_int = {c: i for i, c in enumerate(chars)}
int_to_char = {i: c for i, c in enumerate(chars)}
# encoder --> takes string `s` and outputs list of integers
encode = lambda s: [char_to_int[c] for c in s]
# decoder --> takes list of integers `l` and outputs string
decode = lambda l: [int_to_char[i] for i in l]

# split train & val sets
n = len(data)
train_data = data[:int(n * 0.9)]
val_data = data[int(n * 0.9):]

# encode both sets to integers
train_ids = encode(train_data)
val_ids = encode(val_data)
print(f"train set has {len(train_ids):,} tokens")
print(f"val set has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), "train.bin"))
val_ids.tofile(os.path.join(os.path.dirname(__file__), "val.bin"))

# save meta info to help encode/decode later
meta = {
    "vocab_size": vocab_size,
    "char_to_int": char_to_int,
    "int_to_char": int_to_char
}
with open(os.path.join(os.path.dirname(__file__), "meta.pkl"), "wb") as file:
    pickle.dump(meta, file)

# Length of dataset in characters: 5,245,353
# Unique characters present in dataset are: `
#  !"%&'()+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz£­ÁÜßáãéíïñóøúćšž–‘’“”…` ==>> 102 characters
# train set has 4,720,817 tokens
# val set has 524,536 tokens
