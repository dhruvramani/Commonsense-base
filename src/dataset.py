import re
import os
import json
import torch 
import bcolz
import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader


_DIR = "/home/nevronas/Projects/Personal-Projects/Dhruv/NeuralDialog-CVAE/"
_GLOVE_PATH = '/home/nevronas/word_embeddings/glove_twitter'
_EMB_DIM = 50

def init_glove(glove_path=_GLOVE_PATH): # Run only first time
    words = []
    idx = 0
    word2idx = {}
    vectors = bcolz.carray(np.zeros(1), rootdir='{}/27B.50.dat'.format(glove_path), mode='w')
    with open('{}/glove.twitter.27B.50d.txt'.format(glove_path), 'rb') as f:
        for l in f:
            line = l.decode().split()
            word = line[0]
            words.append(word)
            word2idx[word] = idx
            idx += 1
            vect = np.array(line[1:]).astype(np.float)
            vectors.append(vect)

    vectors = bcolz.carray(vectors.reshape((1193514, _EMB_DIM)), rootdir='{}/27B.50.dat'.format(glove_path), mode='w')
    vectors.flush()
    pickle.dump(words, open('{}/27B.50_words.pkl'.format(glove_path), 'wb'))
    pickle.dump(word2idx, open('{}/27B.50_idx.pkl'.format(glove_path), 'wb'))
    return idx

class CommonSenseDataset(Dataset):
    def __init__(self, step_size, tr="train", annotation_path="data/commonsense/json_version/annotations.json", partition_path="data/commonsense/storyid_partition.txt", pickle_path="data/commonsense/data.pkl", glove_path=_GLOVE_PATH):
        self.tr = tr
        self.step_size = step_size
        self.pickle_path = pickle_path
        self.glove_path = glove_path
        self.partition_path = partition_path
        self.annotation_path = annotation_path

        self.count, self.zerc = 0, 0
        self.glove = self.load_glove()

        file = open(_DIR + self.annotation_path, "r")
        self.data = json.load(file)

        self.to_write = {"train" : list(), "valid" : list(), "test" : list()}
        self.classes = ["joy", "trust", "fear", "surprise", "sadness", "disgust", "anger", "anticipation"]
        self.main_run()

    def load_glove(self):
        vectors = bcolz.open('{}/27B.50.dat'.format(self.glove_path))[:]
        words = pickle.load(open('{}/27B.50_words.pkl'.format(self.glove_path), 'rb'))
        word2idx = pickle.load(open('{}/27B.50_idx.pkl'.format(self.glove_path), 'rb'))

        return {w: vectors[word2idx[w]] for w in words}

    def get_labels(self, charay):
        ann = []
        try :
            ann = [charay["emotion"]["ann0"]["plutchik"]]
        except:
            print("ann0 ignored")
        try :
            ann.append(charay["emotion"]["ann1"]["plutchik"])
        except:
            print("ann1 ignored")
        try :
            ann.append(charay["emotion"]["ann2"]["plutchik"])
        except:
            print("ann2 ignored")
        if(len(ann) == 0):
            return None

        final_dict = dict()
        for classi in self.classes:
            final_dict[classi] = [1, 1, 1]

        for idx in range(len(ann)):
            for i in ann[idx]:
                if(i[:-2] in final_dict.keys()):
                    final_dict[i[:-2]][idx] = int(i[-1])

        majority = []
        for key in final_dict.keys():
            if(int(sum(final_dict[key]) / 3) >= 2):
                majority.append(key) #[key if(floor(sum(final_dict[key]) / 3) >= 2) for key in final_dict.keys()]
        onehot = [1 if i in majority else 0 for i in self.classes]
        return onehot

    def main_run(self):
        if(os.path.isfile(_DIR + self.pickle_path)):
            file = open(_DIR + self.pickle_path, 'rb+')
            self.to_write = pickle.load(file)
            file.close()
            return True

        with open(_DIR + self.partition_path, "r") as f:
            for line in f:
                idkey =  line.split("\t")[0]
                story = self.data[idkey]
                tdt = self.story["partition"].replace("dev", "valid")
                dialog = dict()
                chars = list(self.story["lines"]["1"]["characters"].keys())
                if(len(chars) < 2):
                    continue
                dialog["A"] = chars[0]
                dialog["B"] = chars[1] # TODO : Change
                utterances = list()
                for stline in range(5):
                    linei = self.story["lines"]["{}".format(stline + 1)]
                    characters = linei["characters"]

                    charA, charB = characters[dialog["A"]], characters[dialog["B"]]
                    tr = False
                    if(charA["app"] == True):
                        onehotm = self.get_labels(charA)
                        if(onehotm != None):
                            uttr = ("A", linei["text"], [onehotm])
                            utterances.append(uttr)
                            count += 1
                            tr = True
                            try : 
                                _ = onehotm.index(1)
                            except ValueError:
                                zerc += 1
                    if(charB["app"] == True):
                        onehotm = self.get_labels(charB)
                        if(onehotm != None):
                            uttr = ("B", linei["text"], [onehotm])
                            utterances.append(uttr)
                            if(not tr):
                                count += 1
                            try : 
                                _ = onehotm.index(1)
                            except ValueError:
                                zerc += 1
                print(utterances)
                dialog["utts"] = utterances
                self.to_write[tdt].append(dialog)
            dev_len = len(to_write["valid"])
            self.to_write["train"], self.to_write["valid"] = self.to_write["valid"][:int(0.8 * dev_len)], self.to_write["valid"][int(0.8 * dev_len) + 1:]
            print(count, zerc)

            with open(_DIR + self.pickle_path, "wb+") as handle:
                pickle.dump(self.to_write, handle)


    def __len__(self):
       return len(self.to_write[self.tr])

    def __getitem__(self, idx):
        data = self.to_write[self.tr]
        if(idx + self.step_size > len(data)):
            idx = (idx + self.step_size) - len(data) 
        start_elem = data[idx]
        charA, charB = start_elem["A"], start_elem["B"]
        embA, embB = np.zeros((1, self.step_size, _EMB_DIM)), np.zeros((1, self.step_size, _EMB_DIM))
        predA, predB = None, None
        countA, countB = 0, 0

        while countA <= min(self.step_size, len(data)):
            element = data[idx + countA]
            if(element["A"] == charA):
                uttr_text = element["utts"][1]
                print(uttr_text)
                embed_string = re.sub(r"[^a-zA-Z]+", ' ', uttr_text)
                embedding = np.asarray([self.glove.get(word, self.glove['unk']) for word in embed_string.split(" ")])
                embA[0, countA, :] = embedding
                countA += 1
        predA = data[idx + countA]["utts"][-1][0]
                
        while countB <= min(self.step_size, len(data)):
            element = data[idx + countB]
            if(element["A"] == charA):
                uttr_text = element["utts"][1]
                embed_string = re.sub(r"[^a-zA-Z]+", ' ', uttr_text)
                embedding = np.asarray([self.glove.get(word, self.glove['unk']) for word in embed_string.split(" ")])
                embB[0, countB, :] = embedding
                countB += 1
        predB = data[idx + countB]["utts"][-1][0]

        return embA, predA, embB, predB

if __name__ == '__main__':
    dset = CommonSenseDataset(10)
    dataloader = DataLoader(dset, batch_size=128, shuffle=True, num_workers=1)
    dataloader = iter(dataloader)
    for i in range(0, len(dataloader)):
        embA, predA, embB, predB = next(dataloader)
        print(embA.shape, predA.shape, embB.shape, predB.shape)
        #break


