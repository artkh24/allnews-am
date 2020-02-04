import NER.sentence
import random
from transformers import AutoTokenizer

def partitionRankings(data, percent):
    howManyNumbers = int(round(percent*len(data)))
    shuffled = list(data[:])
    random.shuffle(shuffled)
    return shuffled[howManyNumbers:], shuffled[:howManyNumbers]

def writeInfile(data, filename):
    with open(filename, 'wt',encoding='utf-8') as f:
        for sentence in data:
            f.write((sentence+'\n'))

if __name__ == '__main__':
    testgetter=NER.sentence.Sentence('pioner-silver/test.conll03')
    writeInfile(list(testgetter.sentences), 'data/test.txt')