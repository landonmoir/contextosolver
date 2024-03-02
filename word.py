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
  secretDists = getRecs(wordmap[secret], wordmap)
  
  while True:
    guess = input("Enter word: ")
    # dist to answer
    try:
      guessVec = wordmap[guess]
      dist = np.linalg.norm(guessVec - wordmap[secret])
      guessDists = getRecs(guessVec, wordmap)
      rank = secretDists
      index = np.where(secretDists[:, 0] == guess)[0]

      if len(index) > 0:
    # Extract the corresponding number value from the second column
        number = secretDists[index[0]][1]
        print(f"Number associated with '{guess}': {number}")
      else:
        print(f"Word '{guess}' not found in the data.")

      print("dist to answer:", dist) 

    except Exception as e:
      print(e)
      guessVec = wordmap['<unk>']
    

def getRecs(wordVec, wordMap):
  dists = []
  for word in list(wordMap.keys()):
    dists.append((word, np.linalg.norm(wordVec - wordMap[word])))

  dt = [('x', object), ('y', np.float32)]
  dists = np.array(dists, dtype=dt)
  dists.sort(order='y')
  return dists



if __name__ == "__main__":
  main()
