import urllib.request

def main():
    # Download and store 'The Verdict'
    the_verdict_url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/refs/heads/main/ch02/01_main-chapter-code/the-verdict.txt"
    the_verdict_file_path = "the-verdict.txt"
    urllib.request.urlretrieve(the_verdict_url, the_verdict_file_path)


if __name__ == "__main__":
    main()
