from datasets import load_dataset

data = load_dataset("allenai/pixmo-points", split="train")

def main():
    print(data)

if __name__ == "__main__":
    main()