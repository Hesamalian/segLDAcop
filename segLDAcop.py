"(C) Copyright 2017, Hesam Amoualian"
"hesam.amoualian@univ-grenoble-alpes.fr"
# References paper: Topical Coherence in LDA-based Models through Induced Segmentation
""""@InProceedings{P17-1165,
    author =     "Amoualian, Hesam
    and Lu, Wei
    and Gaussier, Eric
    and Balikas, Georgios
    and Amini, Massih R
    and Clausel, Marianne",
    title =     "Topical Coherence in LDA-based Models through Induced Segmentation",
    booktitle =     "Proceedings of the 55th Annual Meeting of the Association for      Computational Linguistics (Volume 1: Long Papers)    ",
    year =     "2017",
    publisher =     "Association for Computational Linguistics",
    pages =     "1799--1809",
    location =     "Vancouver, Canada",
    doi =     "10.18653/v1/P17-1165",
    url =     "http://www.aclweb.org/anthology/P17-1165"
    }
"""








import numpy as np, codecs, json,  cPickle as pickle, sys, random, itertools
from datetime import datetime
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.metrics import precision_recall_fscore_support
from collections import Counter
from rpy2.robjects.packages import importr
import vocabulary
from sklearn.preprocessing import MultiLabelBinarizer
import rpy2.robjects as robjects
from functions import *
utils = importr("copula")
import time
import math

def fibo(n): # number of segmentation possibilities    
    i=4
    l=[1,1,2,4]
    if n>3:
        while i<=n:
            l[i%4]= l[(i-1) % 4] + l[(i-2) % 4]+l[(i-3)%4]
            i+=1
    return l[n%4]

def fprob(n,m): # probability of segment with different length
    p=np.zeros(n)
    for i in range(0,n):
        if i+1>m:
            p[i]=0
        else :
            p[i]=math.exp(math.log(fibo(m-(i+1)))-math.log(fibo(m)))
    return p


class lda_gibbs_sampling_copula:
    def __init__(self, K=None, alpha=None, beta=None, copulaFamily="Frank", docs= None, V= None, copula_parameter=None,maxl=None,segt=None, it=None):
        self.K = K
        self.lastit=it-1
        self.copPar = copula_parameter
        self.family = copulaFamily
        self.alpha = alpha # parameter of topics prior
        self.beta = beta   # parameter of words prior
        self.docs = docs # a list of lists, each inner list contains the indexes of the words in a doc, e.g.: [[1,2,3],[2,3,5,8,7],[1, 5, 9, 10 ,2, 5]]
        self.V = V # how many different words in the vocabulary i.e., the number of the features of the corpus
        self.C = segt # diffrent type of segment
        self.L =maxl # maximum length of segment
        self.z_m_n = [] # topic assignements for each of the N words in the corpus. N: total number of words in the corpus (not the vocabulary size).
        self.l_m_ch =[] # segment length assign
        self.n_m_z = np.zeros((len(self.docs), K), dtype=np.float64) + alpha     # |docs|xK topics: number of sentences assigned to topic z in document m  
        self.n_l_z = np.zeros((len(self.docs),self.L, K), dtype=np.float64)+alpha*2
        self.n_m_l = np.zeros((len(self.docs), self.L), dtype=np.float64)   # different segment with different length within document
        self.n_z_t = np.zeros((K, V), dtype=np.float64) + beta # (K topics) x |V| : number of times a word v is assigned to a topic z
        self.n_z = np.zeros(K) + V * beta    # (K,) : overal number of words assigned to a topic z
        self.n_l = np.zeros(self.L) # number of different segments with different length within corpus
        self.n_m = np.zeros(len(self.docs),dtype=np.int)    # length of each document
        self.N = 0
        newleveldocs=[]
        for m, doc in enumerate(docs):         # Initialization of the data structures I need and initial segmentation
            z_doc = []
            l_doc =[]
            accn=0
            while True:
                l=np.random.randint(0, self.L)
                accn+=l+1
                while accn>len(doc):
                    l=l-1
                    accn-=1
    
                self.n_l[l]+=1
                self.n_m_l[m,l]+=1
                l_doc.append(l)
                if accn==len(doc):
                    break
            self.l_m_ch.append(np.array(l_doc))
            newleveldoc=[]
            for i in self.l_m_ch[m]:
                newleveldoc.append(doc[sum(len(x) for x in newleveldoc):sum(len(x) for x in newleveldoc)+i+1])    
            newleveldocs.append(newleveldoc)
            
            # initial topic assignment
            for chunk in newleveldoc: 
                self.N += len(chunk)
                z_n = []
                for t in chunk:
                    z = np.random.randint(0, K) # Randomly assign a topic to a segement. Recall, topics have ids 0 ... K-1. randint: returns integers to [0,K[
                    z_n.append(z)                  # Keep track of the topic assigned 
                    self.n_m_z[m, z] += 1          # increase the number of words assigned to topic z in the m doc.
                    self.n_z_t[z, t] += 1   #  .... number of times a word is assigned to this particular topic
                    self.n_z[z] += 1   # increase the counter of words assigned to z topic
                    self.n_l_z[m,len(chunk)-1,z]+=1 #number of times a topic assign to this segment and document
                    self.n_m[m] += 1    #
                z_doc.append(z_n)
            self.z_m_n.append(np.array(z_doc)) # update the array that keeps track of the topic assignements in the sentences of the corpus.
            
        self.docs=newleveldocs

    def inference(self,iteration):
        """    The learning process. Here only one iteration over the data. 
               A loop will be calling this function for as many iterations as needed.     """
        newleveldocs=[]
        newlevelzs=[]
        for m, doc in enumerate(self.docs):
            z_n, n_m_z = self.z_m_n[m], self.n_m_z[m] #Take the topics of the words and the number of words assigned to each topic
            
            for sid, chunk in enumerate(doc): #Sentence stands for a segment, that is contiguous words that are generated by topics that are bound.
                # Get Sample from copula
                
                if len(chunk) > 1: # If the size of segment is bigger than one, sample the copula, else back-off to standard LDA Gibbs sampling
                    command = "U = rnacopula(1, onacopula('%s', C(%s, 1:%d)))"%(self.family, self.copPar, len(chunk))
                    U = robjects.r(command)

                for n, t in enumerate(chunk):
                    z = z_n[sid][n]
                    n_m_z[z] -= 1
                    self.n_z_t[z, t] -= 1
                    self.n_z[z] -= 1

                    f=np.random.randint(0, 2)     #random decision between segment and document topic distribution
                    if f==0:
                        p = (self.n_z_t[:, t]) * (n_m_z) / (self.n_z) #Update probability distributions
                        p = p / p.sum() # normalize the updated distributions

                    if f==1:
                        p = (self.n_z_t[:, t]) *(n_m_z)*(self.n_l_z[m,len(chunk)-1]+0.02) / (self.n_z) #Update probability distributions
                        p = p / p.sum() # normalize the updated distributions
                        
                    if len(chunk)>1: # Copula mechanism over the words of a segment (noun-phrase or sentence)
                        new_z = self.getTopicIndexOfCopulaSample(p, U[n])
                    else:
                        new_z = np.random.multinomial(1, p).argmax() # Back-off to Gibbs sampling if len(sentence) == 1 for speed.

                    z_n[sid][n] = new_z
                    n_m_z[new_z] += 1
                    self.n_z_t[new_z, t] += 1
                    self.n_z[new_z] += 1
                    self.n_l_z[m,len(chunk)-1,new_z]+=1

            #re-segmentation for new possible structure to apply copula
            newzdoc=[]
            newdoc=[]
            for i in z_n:
                for j in i:
                    newzdoc.append(j)
            for i in doc:
                for j in i:
                    newdoc.append(j)
            l_doc =[]
            accn=self.n_m[m]
            pl=0.01
            while accn>0:

                pfib=fprob(self.L,accn)   
                if accn==1:
                    pnew=pfib
                else :
                    pnew=pl*pfib*100
                l=np.random.multinomial(1, pnew/pnew.sum()).argmax()
                accn-=l+1
                self.n_l[l]+=1
                self.n_m_l[m,l]+=1
                l_doc.append(l)
                if accn==len(newdoc):
                    break
            self.l_m_ch[m]=np.array(l_doc)
            newleveldoc=[]
            newlevelz=[]
            for i in self.l_m_ch[m]:
                newleveldoc.append(newdoc[sum(len(x) for x in newleveldoc):sum(len(x) for x in newleveldoc)+i+1])
                newlevelz.append(newzdoc[sum(len(x) for x in newlevelz):sum(len(x) for x in newlevelz)+i+1])
            newleveldocs.append(newleveldoc)
            newlevelzs.append(newlevelz)
        self.docs=newleveldocs
        self.z_m_n=newlevelzs
        
        #save the topic assignments with segmentation boundary
        if iteration==self.lastit:
            g = open('boundtopicslast', "w+b")
            g.write("\n".join(str(elem) for elem in self.z_m_n))
            g.close()
                   
        

    def getTopicIndexOfCopulaSample(self, probs, sample): #Probability integral transform: given a uniform sample from the copula, use the quantile $F^{-1}$ to tranform it to a sample from f
        cdf = 0
        for key, val in enumerate(probs):
            cdf += val
            if sample <= cdf:
                return key


    def topicdist(self):
        topcDist = self.n_m_z / (self.n_m[:, np.newaxis]+ self.K * self.alpha)
        return topcDist     


    def worddist(self):
        """get topic-word distribution, \phi in Blei's paper. Returns the distribution of topics and words. (Z topics) x (V words)  """
        phi=self.n_z_t / self.n_z[:, np.newaxis]
        return  phi #Normalize each line (lines are topics), with the number of words assigned to this topics to obtain probs.  *neaxis: Create an array of len = 1
        

if __name__ == "__main__":
    path2input = sys.argv[1]
    Numberoftopic = sys.argv[2]
    Numberofiteration = sys.argv[3]
    having_stopword = json.loads(sys.argv[4])
    corpus = codecs.open(path2input, 'r', encoding='utf8').read().splitlines()
    iterations = int(Numberofiteration)
    topics = int(Numberoftopic)
    al = 1.0/topics
    be = 1.0/topics
    voca = vocabulary.Vocabulary(excluds_stopwords=having_stopword)
    classificationData = corpus
    docs = [voca.doc_to_ids(doc) for doc in classificationData]
    print 'whole number of documents:',len(docs)
    print 'whole number of unique words:',voca.size()
    lda = lda_gibbs_sampling_copula(K=topics, alpha=al, beta=be,copulaFamily="Frank", docs= docs, V= voca.size(), copula_parameter=5,maxl=3,segt=5,it=iterations)
    outper=open('perp.txt','w')
    for i in range(iterations):
        lda.inference(i)
        if  i%1==0:
                starting = datetime.now()
                print "iteration:", i
                d = lda.worddist()
                th = lda.topicdist()
                per=0
                b=0
                c=0
                for m, doc in enumerate(docs):
                        b+=len(doc)
                        for n, w in enumerate(doc):
                            l=0
                            for i in range(topics):
                                l+=th[m,i]*d[i,w]
                            c+=np.log(l)
 
                per=np.exp(-c/b)
                print "perpelixity", per
                outper.write(str(per)+'\n')
               
    outper.close()
    for i in range(topics):
        ind = np.argpartition(d[i], -10)[-10:] # an array with the indexes of the 10 words with the highest probabilitity in the topic
        for j in ind:
            print voca[j],
        print


