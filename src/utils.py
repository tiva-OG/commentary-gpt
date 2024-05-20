import torch
import pickle
import numpy as np

batch_size = 4
block_size = 8
pkl_path = "C:/Users/salva/Desktop/projects/eplGPT/data/meta.pkl"


def read_data(data_path):
    data = np.fromfile(data_path, dtype=np.uint16)

    return torch.from_numpy(data).type(torch.long)


def read_pkl(pkl_path):
    with open(pkl_path, "rb") as file:
        data = pickle.load(file)

    return data


def get_batch(split):
    assert split in ['train', 'val']
    data_path = f"C:/Users/salva/Desktop/projects/eplGPT/data/{split}.bin"
    data = read_data(data_path)

    # generate a batch of data of inputs X and targets y
    ix = torch.randint(len(data) - block_size, (batch_size,))
    X = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])

    return X, y


meta = read_pkl(pkl_path)
chars = meta["char_to_int"].keys()
char_to_int = {c: i for i, c in enumerate(chars)}
int_to_char = {i: c for i, c in enumerate(chars)}

# encoder --> takes string `s` and outputs list of integers
encode = lambda s: [char_to_int[c] for c in s]
# decoder --> takes list of integers `l` and outputs string
decode = lambda l: "".join([int_to_char[i] for i in l])


if __name__ == "__main__":
    # train_path = "train.bin"
    # val_path = "val.bin"

    Xb, yb = get_batch("val")
    print("INPUTS:", Xb.shape)
    print(Xb)
    print("TARGETS:", yb.shape)
    print(yb)
