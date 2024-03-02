import numpy as np
import random


def main():
  wordmap = {}
  with open('./vectors.txt') as infile:
    line = infile.readline()
    while line:
      split = line.strip().split(" ")
      word = split[0]
      vecs = np.array(list(map(float, split[1:])))
      wordmap[word] = vecs
      line = infile.readline()
      
  secret = random.choice(list(wordmap.keys()))
  
  while True:
    guess = input("Enter word: ")
    # dist to answer
    try:
      guessVec = wordmap[guess]
      dist = np.linalg.norm(guessVec - wordmap[secret])
      getRecs(guessVec, wordmap)
      print("dist to answer:", dist) 

    except Exception as e:
      print(e)
      guessVec = wordmap['<unk>']
    

def getRecs(wordVec, wordMap):
  dists = []
  for word in list(wordMap.keys()):
    dists.append((word, np.linalg.norm(wordVec - wordMap[word])))

  dt = np.dtype([('string', object), ('float', np.float32)])
  print(np.array(dists, dtype=dt))
  dists = np.array(dists, dtype=dt)
  print(dists.sort(order='float'))



if __name__ == "__main__":
  main()
