import torch


def read_vocab(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read()
    return text


def create_mappings(chars):
    stoi = {char: idx for idx, char in enumerate(chars)}
    itos = {idx: char for idx, char in enumerate(chars)}
    return stoi, itos


def encode_string(mapping, s):
    return [mapping[char] for char in s]


def decode_list(mapping, l):
    return ''.join([mapping[idx] for idx in l])


# text = read_vocab("combined_lyrics.txt")
# vocab = sorted(list(set(text)))
# vocab_size = len(vocab)
# stoi, itos = create_mappings(vocab)
# encoded = encode_string(stoi, "hello")
# decoded = decode_list(itos, encoded)
#
# # Make the train and test split
#
# data = torch.tensor(encode_string(stoi, text), dtype=torch.long)
# n = int(0.9*len(data)) # first 90% will be train, rest val
# train_data = data[:n]
# val_data = data[n:]

# print(stoi)
# print(itos)
# print(encoded)
# print(decoded)
