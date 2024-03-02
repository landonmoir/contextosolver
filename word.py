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
      
  secret = random.choice(list(wordmap.keys())[100:1200])
  secretDists = getRecs(wordmap[secret], wordmap)
  print('secret word is:',  secret)
  print('top 10 words')
  print(secretDists[0:9])
  
  while True:
    guess = input("Enter word: ")
    # dist to answer
    try:
      guessVec = wordmap[guess]
      guessDists = getRecs(guessVec, wordmap)
      value = [item for item in secretDists if item[0] == guess]
      index = np.where(secretDists == value)

      print("dist to answer:", value[0][1])
      print('guess rank:', index[0][0]+1)
      print('')

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
