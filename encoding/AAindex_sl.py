import numpy as np
import re
from Bio import SeqIO

def AAindex_HM_encoding(file_name):
    with open(r'D:\python code\N_Link\Mouse\encoding\AAindex_normalized.txt') as f:
        records = f.readlines()[1:]
    AA_aaindex = 'ARNDCQEGHILKMFPSTWYV'
    AAindex = []
    AAindexName = []
    for i in records:
        AAindex.append(i.rstrip().split()[1:] if i.rstrip() != '' else None)
        AAindexName.append(i.rstrip().split()[0] if i.rstrip() != '' else None)
    props = 'OOBM850104:WEBA780101:QIAN880124:FAUJ880113:ZIMJ680104:NAKH920101:PRAM820101:HOPT810101:QIAN880136:TSAJ990101:NADH010107:ZHOH040103:BEGF750103:QIAN880128:QIAN880108:GEOR030106:FUKS010101:QIAN880121:FAUJ880111:AURR980117'.split(
        ':')  ## human挑选出的20个理化特征
    if props:
        tmpIndexNames = []
        tmpIndex = []
        for p in props:
            if AAindexName.index(p) != -1:
                tmpIndexNames.append(p)
                tmpIndex.append(AAindex[AAindexName.index(p)])
        if len(tmpIndexNames) != 0:
            AAindexName = tmpIndexNames
            AAindex = tmpIndex

    index = {}
    for i in range(len(AA_aaindex)):
        index[AA_aaindex[i]] = i

    encoding_aaindex = []
    for seq_record in SeqIO.parse(file_name, "fasta"):
        sequence = seq_record.seq
        code = []
        for aa in sequence:
            if aa == 'X' or aa == 'U':
                for j in AAindex:
                    code.append(0)
                continue
            for j in AAindex:
                code.append(j[index[aa]])
        encoding_aaindex.append(code)
    return encoding_aaindex


def AAindex_MS_encoding(file_name):
    with open(r'D:\python code\N_Link\Mouse\encoding\AAindex_normalized.txt') as f:
        records = f.readlines()[1:]
    AA_aaindex = 'ARNDCQEGHILKMFPSTWYV'
    AAindex = []
    AAindexName = []
    for i in records:
        AAindex.append(i.rstrip().split()[1:] if i.rstrip() != '' else None)
        AAindexName.append(i.rstrip().split()[0] if i.rstrip() != '' else None)
    props = 'OOBM850103:WEBA780101:QIAN880114:FAUJ880112:PRAM820101:JUKT750101:HUTJ700102:QIAN880112:OOBM850104:TANS770110:KUMS000104:AURR980112:VASM830101:TANS770105:QIAN880102:PRAM900103:QIAN880136:SNEP660103:HOPT810101:PRAM820103'.split(
        ':')  ## human挑选出的20个理化特征
    if props:
        tmpIndexNames = []
        tmpIndex = []
        for p in props:
            if AAindexName.index(p) != -1:
                tmpIndexNames.append(p)
                tmpIndex.append(AAindex[AAindexName.index(p)])
        if len(tmpIndexNames) != 0:
            AAindexName = tmpIndexNames
            AAindex = tmpIndex

    index = {}
    for i in range(len(AA_aaindex)):
        index[AA_aaindex[i]] = i

    encoding_aaindex = []
    for seq_record in SeqIO.parse(file_name, "fasta"):
        sequence = seq_record.seq
        code = []
        for aa in sequence:
            if aa == 'X' or aa == 'U':
                for j in AAindex:
                    code.append(0)
                continue
            for j in AAindex:
                code.append(j[index[aa]])
        encoding_aaindex.append(code)
    return encoding_aaindex


def col_delete(data):  # delete the columns woth same elements
    col_del = [200,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,219]
    data_new = np.delete(data, col_del, axis=1)
    return data_new
