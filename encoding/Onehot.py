import numpy as np
from Bio import SeqIO


def read_fasta(file_path):
    one=list(SeqIO.parse(file_path,'fasta'))
    return one


def onehot(seq):
    bases = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']
    X = np.zeros((len(seq),len(seq[0]),len(bases)))
    for i,m in enumerate(seq):
        for l,s in enumerate(m):
            # print(s)
            if s in bases:
                X[i,l,bases.index(s)] = 1
    return X


def col_delete(data):  # delete the columns woth same elements
    col_del = [200,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,219]
    data_new = np.delete(data, col_del, axis=1)
    return data_new






