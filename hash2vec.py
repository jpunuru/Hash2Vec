
import sys
import numpy as np
import re
import json

_WORD_SPLIT = re.compile(b"([.,!?\"':;)(])\+")

NB_DIMS    = 200
CTX_WIDTH  = 5




class Hash2Vec:

    def __init__(self):
        self.dic       = {}
        self.rev_dic   = {}
        self.vec       = []
        self.nbWords   = 0
        self.sigma     =  0.0
        distVec        = [ -i+CTX_WIDTH for i in range(CTX_WIDTH) ] 
        distVec.extend([i for i in range(CTX_WIDTH)])
        self.std       = np.std(distVec)
        
        
    def __call__(self, fname):
        self.genVec(fname)
        return self

    def wordToId(self, w):
        if not self.dic.has_key(w):
            self.dic[w] = self.nbWords
            self.rev_dic[self.nbWords] = w
            self.vec.append([0.0 for _ in xrange(NB_DIMS)])
            self.nbWords += 1
        return self.dic[w]
    

    def getContextWords(self, words, pos):
        spos = 0
        epos = pos+1+CTX_WIDTH
        if pos > CTX_WIDTH:
            spos = pos-CTX_WIDTH
        if epos > len(words):
            epos = len(words)
        cw = words[spos:pos]+words[pos+1:epos]
        d =  range(1, pos-spos+1)+range(epos-pos-1,0,-1)
        assert len(d) == len(cw)
        return cw, d

    def getHash(self, word):
        val = hash(word)
        hid = val%NB_DIMS
        hsn = val%2
        sign = -1
        if hsn:
            sign = 1
        return hid, sign 

   
    def normalize(self):
        for i in xrange(len(self.vec)):
            v    = [abs(val) for val in self.vec[i]]
            vsum = sum(v)
            self.vec[i] = [val*1./vsum  for val in v]

        
    def genVec(self, dataFile):
        with open(dataFile, mode="r") as dataF:
            lines = dataF.readlines()
            for ln in lines:
                ln = ln.strip("\r\n")
                if len(ln) == 0: continue
                words = ln.lower().split()
                if len(words) <= 1: continue
                words = [re.sub(_WORD_SPLIT, '', w) for w in words]
                wordids = [ self.wordToId(w) for w in words]
                pos = 0;
                for w in wordids:
                    cwl, d = self.getContextWords(wordids, pos)
                    i = 0;
                    for cw in cwl:
                        dist =  np.exp(-1*pow(d[i]/self.std, 2))
                        hid,sign = self.getHash(self.rev_dic[cw])
                        self.vec[w][hid] = self.vec[w][hid] +  sign*dist
                        i += 1
                    pos += 1
        #normalize the vectors
        self.normalize()        

    def toJson(self):
        data = {'NB_DIMS':NB_DIMS, 'CTX_WIDTH':CTX_WIDTH, 
                 'VOCAB':self.dic,'VECTORS':self.vec}
        return data
         
                        
    def write(self, fileName):
         data = self.toJson()
         with open(fileName, mode="w") as outf:
             outf.write(json.dumps(self.toJson()))
             outf.flush()
         
    def read(self, filename):
        global NB_DIMS
        global CTX_WIDTH
        with open(filename, mode="r") as inpF:
            data = json.load(inpF)
            NB_DIMS = data['NB_DIMS']
            CTX_WIDTH = data['CTX_WIDTH']
            self.dic = data['VOCAB']
            for k in self.dic.keys():
                self.rev_dic[self.dic[k]] = k
            self.vec = data['VECTORS']


    def __insert(self, index, val, top10):
        for i in xrange(len(top10)):
            if top10[i][1] < val:
                top10[i] = [index, val]
                break
        return


    def getSimilar(self, sw):
        sw = sw.lower()
        if not self.dic.has_key(sw): 
            return None
        pos = self.dic[sw]
        swVec = self.vec[pos]
        top10 = [] 
        nbTop     = 0;
        #den1 = np.sqrt(sum([pow(v, 2) for v in swVec]))
        den1 =1
        for index, v in enumerate(self.vec):
            if index == pos: continue
            den2 = np.sqrt(sum([pow(ve, 2) for ve in v]))
            simVal = np.dot(swVec, v)/(den1*den2)
            if nbTop < 10:
                top10.append([index,simVal])
                nbTop += 1
            else:
                self.__insert(index, simVal, top10)
        top10.sort(key=lambda tup:tup[1], reverse=True)
        similarTerms = []
        for elm in top10:
            similarTerms.append([self.rev_dic[elm[0]],elm[1]])
        return similarTerms 
         

def similarityEval(h2v):
    w = raw_input("enter a word to get similar words(press exit to exit).")
    w = w.strip()
    while w <> "exit":
        simW = h2v.getSimilar(w)
        if simW is None:
            print("word '%s' is not found"%w)
        else:
            print("Words similar to %s" % w)
            print("-----------------------------\n")
            for item in simW:
                print("\t%-40s\t\t\t%0.5f" % (item[0],item[1]))

            print("-----------------------------\n")
        w = raw_input("enter a word to get similar words(press exit to exit).")
        w = w.strip()


if __name__ == '__main__':
    if sys.argv[1] == '--data':
        h2v = Hash2Vec()
        h2v = h2v(sys.argv[2])
        h2v.write('hash2vec.out')
        h2v.read('hash2vec.out')
        similarityEval(h2v) 
    
