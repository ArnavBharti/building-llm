import urllib.request
import re

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
    # Split on space and punctuation.
    preprocessed = re.split(r"([,.:;?_!\"()']|--|\s)", raw_text)
    # Remove spaces.
    preprocessed = [item.strip() for item in preprocessed if item.strip()]
    # print("Length of preprocessed text after removing whitespace:", len(preprocessed))
    # print("First 30 preprocessed words:", preprocessed[:30])

    # Convert tokens into token IDs
    all_words = sorted(set(preprocessed))
    vocab_size = len(all_words)
    # print("Size of vocabulary:", vocab_size)
    # print("First 30 words from vocabulary:", all_words[:30])

    vocab = {token:integer for integer,token in enumerate(all_words)}
    # for i, item in enumerate(vocab.items()):
    #     print(item)
    #     if i > 10:
    #         break


if __name__ == "__main__":
    main()
