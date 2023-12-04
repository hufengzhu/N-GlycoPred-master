import numpy as np
from Bio import SeqIO
import os
import math
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from keras.layers import *
from keras.models import *
from sklearn import preprocessing
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Callback
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, LearningRateScheduler
# from N_Link.Human.encoding import Onehot
from N_Link.Mouse.encoding import AAindex_sl
from keras_pos_embd import PositionEmbedding, TrigPosEmbedding
import tensorflow as tf
from keras import backend as K


# Pairwise attention module
def P_Attention(input, use_embedding='PositionEmbedding.MODE_ADD'):
    lstm_len = int(input.shape[1])
    lstm_dim = int(input.shape[2])
    if use_embedding == 'PositionEmbedding.MODE_CONCAT ':
        pe_layer = PositionEmbedding(input_dim=lstm_len, output_dim=lstm_dim,
                                     mode=PositionEmbedding.MODE_CONCAT)
    elif use_embedding == 'PositionEmbedding.MODE_ADD':
        pe_layer = PositionEmbedding(input_dim=lstm_len, output_dim=lstm_dim,
                                     mode=PositionEmbedding.MODE_ADD)
    elif use_embedding == 'TrigPosEmbedding.MODE_CONCAT ':
        pe_layer = PositionEmbedding(input_dim=lstm_len, output_dim=lstm_dim,
                                     mode=TrigPosEmbedding.MODE_CONCAT)
    elif use_embedding == 'TrigPosEmbedding.MODE_ADD':
        pe_layer = PositionEmbedding(input_dim=lstm_len, output_dim=lstm_dim,
                                     mode=TrigPosEmbedding.MODE_ADD)
    fw_lstm_pe = pe_layer(input)

    dk = int(input.shape[2]) // 2
    dv = int(input.shape[2])

    Q = Dense(dk, use_bias=False)  # (bn,L,dk)
    K = Dense(dk, use_bias=False)  # (bn,L,dk)
    V = Dense(dv, use_bias=False)  # (bn,L,dv)
    QKt = Lambda(lambda x: tf.matmul(x[0], x[1]) / np.sqrt(dk))
    attention_score = Softmax(2, name='attention_score')
    attention_output = Lambda(lambda x: tf.matmul(x[0], x[1]))
    add_layer = Add()

    q_fw = Q(fw_lstm_pe)
    k_fw = K(fw_lstm_pe)
    v_fw = V(fw_lstm_pe)

    Kt = Permute((2, 1))  # (bn,dk,L)
    kt_fw = Kt(k_fw)
    QKt_fw = QKt([q_fw, kt_fw])  # (bn,L,L)

    attention_score_fw = attention_score(QKt_fw)  # (bn,L,L)
    attention_output_fw = attention_output([attention_score_fw, v_fw])  # (bn,L,dv)

    output = add_layer([input, attention_output_fw])
    return output


def Pair_Attention2():
    word_input = Input(shape=(20, 20), name='word_input')

    overallResult = Convolution1D(filters=32, kernel_size=1, padding='same', activation="relu", name='Cov1')(word_input)

    overallResult = Convolution1D(filters=16, kernel_size=1, padding='same', activation="relu", name='Cov2')(overallResult)

    overallResult = P_Attention(input=overallResult, use_embedding='PositionEmbedding.MODE_ADD')

    overallResult = Bidirectional(LSTM(50, dropout=0.5, activation='tanh', return_sequences=True), name='Bil')(
        overallResult)

    overallResult = Flatten()(overallResult)

    overallResult = Dense(32, activation='sigmoid', name='den')(overallResult)

    ss_output = Dense(1, activation='sigmoid', name='ss_output')(overallResult)

    return Model(inputs=[word_input], outputs=[ss_output])


scale = preprocessing.StandardScaler()

seed = 7

def Twoclassfy_evalu(y_test, y_predict):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    FP_index = []
    FN_index = []
    for i in range(len(y_test)):
        if y_predict[i] > 0.5 and y_test[i] == 1:
            TP += 1
        if y_predict[i] > 0.5 and y_test[i] == 0:
            FP += 1
            FP_index.append(i)
        if y_predict[i] < 0.5 and y_test[i] == 1:
            FN += 1
            FN_index.append(i)
        if y_predict[i] < 0.5 and y_test[i] == 0:
            TN += 1
    Sn = TP / (TP + FN)
    Sp = TN / (FP + TN)
    MCC = (TP * TN - FP * FN) / math.sqrt((TN + FN) * (FP + TN) * (TP + FN) * (TP + FP))
    Acc = (TP + TN) / (TP + FP + TN + FN)
    precision = TP / (TP + FP)
    f1_score = (2 * precision * Sn) / (precision + Sn)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_predict, pos_label=1)  # poslabel正样本的标签
    auc = metrics.auc(fpr, tpr)

    return Sn, Sp, Acc, MCC, auc, precision, f1_score


## BOVIN
data = "./BOVIN/BOVIN_73_142"

## CAEEL
# data = "./CAEEL/CAEEL_202_829"

## DROME
# data = "./DROME/DROME_58_374"

## RAT
# data = "./RAT/RAT_88_188"


## HM
# BOVIN_data = Onehot.read_fasta(data)
# BOVIN_data = Onehot.onehot(BOVIN_data)
# BOVIN_data = BOVIN_data.reshape(BOVIN_data.shape[0], 420)
# test_word = Onehot.col_delete(BOVIN_data).reshape(BOVIN_data.shape[0], 20, 20)


## MS
encodings_P = AAindex_sl.AAindex_MS_encoding(data)
XP = np.array(encodings_P).reshape(-1, 420)
XP = AAindex_sl.col_delete(XP)
test_word = scale.fit_transform(XP).reshape(XP.shape[0], 20, 20)


test_label = np.array([1] * 73 + [0] * 142, dtype='float32')

model = Pair_Attention2()
# model.load_weights(r"D:\python code\N_Link\Human\model\hm_onehot_PAtt2.h5")    # human model
model.load_weights(r"D:\python code\N_Link\Mouse\ms_model\ms_AAindex_PAtt2.h5")   # mouse model

y_predict1 = model.predict({'word_input': test_word})

# -----------------------------------------------
prediction_file = 'MS_BOVIN_Result.txt'

Y_pred = []
f = open(prediction_file, 'w')

for i in range(len(y_predict1)):
        pre = y_predict1[i][0]
        f.write(str(pre))
        f.write('\n')

f.close()

# -------------------------------------------------

(Sn1, Sp1, Acc1, MCC1, AUC1, precision1, f1_score1) = Twoclassfy_evalu(test_label, y_predict1)

print('SN', Sn1 * 100)
print('SP', Sp1 * 100)
print('ACC', Acc1 * 100)
print('MCC', MCC1 * 100)
print('AUC', AUC1 * 100)
# print('Pre', precision1 * 100)
# print('f1', f1_score1 * 100)








