import numpy as np
def main():
  wordmap = {}
  with open('./vectors.txt') as infile:
    line = infile.readline()
    while line:
      split = line.strip().split(" ")
      word = split[0]
      vecs = list(map(float, split[1:]))
      wordmap[word] = vecs
      line = infile.readline()

    print(wordmap['the'])


if __name__ == "__main__":
  main()
