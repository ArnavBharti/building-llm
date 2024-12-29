import urllib.request
import re
import tiktoken
import torch
from torch.utils.data import DataLoader, Dataset

# class SimpleTokensizerV1:
#     """Throws KeyError when word is not in vocabulary."""
#     def __init__(self, vocab):
#         self.str_to_int = vocab
#         self.int_to_str = {i:s for s,i in vocab.items()}

#     def encode(self, text):
#         preprocessed = re.split(r"([,.:;?_!\"()']|--|\s)", text)
#         preprocessed = [item.strip() for item in preprocessed if item.strip()]
#         ids = [self.str_to_int[s] for s in preprocessed]
#         return ids
    
#     def decode(self, ids):
#         text = " ".join([self.int_to_str[i] for i in ids])
#         # Remove whitespace before specified punctuation marks.
#         text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)

# class SimpleTokensizerV2:
#     """Throws KeyError when word is not in vocabulary."""
#     def __init__(self, vocab):
#         self.str_to_int = vocab
#         self.int_to_str = {i:s for s,i in vocab.items()}

#     def encode(self, text):
#         preprocessed = re.split(r"([,.:;?_!\"()']|--|\s)", text)
#         preprocessed = [item.strip() for item in preprocessed if item.strip()]
#         preprocessed = [item if item in self.str_to_int else "<|unk|>" for item in preprocessed]
#         ids = [self.str_to_int[s] for s in preprocessed]
#         return ids
    
#     def decode(self, ids):
#         text = " ".join([self.int_to_str[i] for i in ids])
#         # Remove whitespace before specified punctuation marks.
#         text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
#         return text
    
class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size=4, max_length=256, 
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):

    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader

def main():
    # Download and store 'The Verdict'
    # the_verdict_url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/refs/heads/main/ch02/01_main-chapter-code/the-verdict.txt"
    # the_verdict_file_path = "the-verdict.txt"
    # urllib.request.urlretrieve(the_verdict_url, the_verdict_file_path)

    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
    # print("Length of raw text:", len(raw_text))
    # print("First 100 characters of raw text:", raw_text[:99])

    # Tokenise raw_text.
    # Split on whitespace and punctuation.
    # preprocessed = re.split(r"([,.:;?_!\"()']|--|\s)", raw_text)
    # Remove whitespace.
    # preprocessed = [item.strip() for item in preprocessed if item.strip()]
    # print("Length of preprocessed text after removing whitespace:", len(preprocessed))
    # print("First 30 preprocessed words:", preprocessed[:30])

    # Convert tokens into token IDs
    # all_words = sorted(set(preprocessed))
    # vocab_size = len(all_words)
    # print("Size of vocabulary:", vocab_size)
    # print("First 30 words from vocabulary:", all_words[:30])

    # vocab = {token:integer for integer,token in enumerate(all_words)}
    # for i, item in enumerate(vocab.items()):
    #     print(item)
    #     if i > 10:
    #         break

    # This throws KeyError.
    # tokenizer = SimpleTokensizerV1(vocab)
    # ids = tokenizer.encode(raw_text[:30])
    # print(ids)

    # all_tokens = list(all_words)
    # all_tokens.extend(["<|unk|>","<|endoftext|>"])
    # print(all_tokens[-5:])
    # vocab = {token:integer for integer, token in enumerate(all_tokens)}

    # tokenizer = SimpleTokensizerV2(vocab)
    # ids = tokenizer.encode(raw_text[:30])
    # print(raw_text[:30])
    # print(ids)
    # print(tokenizer.decode(ids))

    # tokenizer = tiktoken.get_encoding("gpt2")
    # test_sentence = "akwirw ier"
    # test_sentence = "Arnav Bharti is my name. Luziputi."
    # ids = tokenizer.encode(test_sentence)
    # print(ids)
    # print(tokenizer.decode(ids))

    # enc_text = tokenizer.encode(raw_text)
    # print(len(enc_text))
    # enc_sample = enc_text[50:]
    # context_size = 4
    # x = enc_sample[:context_size]
    # y = enc_sample[1:context_size+2]
    # for i in range(1, context_size+1):
    #     context = enc_sample[:i]
    #     desired = enc_sample[i]
    #     print(tokenizer.decode(context), "--->", tokenizer.decode([desired]))

    dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=4, stride=4, shuffle=False)
    # data_iter = iter(dataloader)
    # first_batch = next(data_iter)
    # print(first_batch)

    # input_ids = torch.tensor([2,5,6,1])
    vocab_size = 50257 
    output_dim = 256
    # torch.manual_seed(123)
    token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
    # print(embedding_layer.weight)
    # print(embedding_layer(torch.tensor([6])))
    # print(token_embedding_layer(input_ids))

    max_length = 4
    dataloader = create_dataloader_v1(
        raw_text, batch_size=8, max_length=max_length,
        stride=max_length, shuffle=False
    )
    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)
    # print("Token IDs:\n", inputs)
    # print("\nInputs shape:\n", inputs.shape)
    # print("Targets:", targets)
    token_embeddings = token_embedding_layer(inputs)
    # print(token_embeddings.shape)


if __name__ == "__main__":
    main()
