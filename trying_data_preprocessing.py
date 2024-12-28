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
    def __init__(self, text, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(text)

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i+max_length]
            output_chunk = token_ids[i+1:i+max_length+1]
            self.input_ids.append(input_chunk)
            self.target_ids.append(output_chunk)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return self.input_ids[index], self.target_ids[index]

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

    tokenizer = tiktoken.get_encoding("gpt2")
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

if __name__ == "__main__":
    main()
