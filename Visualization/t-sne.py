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
from N_Link.Human.encoding import Onehot
from keras_pos_embd import PositionEmbedding, TrigPosEmbedding
import tensorflow as tf



ftrain = r'D:\python code\N_Link\Human\data\HM_Train2436.txt'
ftest = r'D:\python code\N_Link\Human\data\HM_test513.txt'

# ---------------------------------------Onehot
tr = Onehot.read_fasta(ftrain)
f_train = Onehot.onehot(tr)
f_train = f_train.reshape(f_train.shape[0], 420)
f_train = Onehot.col_delete(f_train).reshape(f_train.shape[0], 20, 20)

te = Onehot.read_fasta(ftest)
f_test = Onehot.onehot(te)
f_test = f_test.reshape(f_test.shape[0], 420)
f_test = Onehot.col_delete(f_test).reshape(f_test.shape[0], 20, 20)


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

    overallResult = Convolution1D(filters=16, kernel_size=1, padding='same', activation="relu", name='Cov2')(
        overallResult)

    overallResult = P_Attention(input=overallResult, use_embedding='PositionEmbedding.MODE_ADD')

    overallResult = Bidirectional(LSTM(50, dropout=0.5, activation='tanh', return_sequences=True), name='Bil')(
        overallResult)

    overallResult = Flatten()(overallResult)

    overallResult = Dense(32, activation='sigmoid', name='den')(overallResult)

    ss_output = Dense(1, activation='sigmoid', name='ss_output')(overallResult)

    model = Model(inputs=[word_input], outputs=[ss_output])

    return model


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


def step_decay(epoch):
    initial_lrate = 0.0005
    drop = 0.8
    epochs_drop = 5.0
    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return lrate


word = f_train
train_label = np.array([1] * 1218 + [0] * 1218, dtype='float32')

test_word = f_test
test_label = np.array([1] * 135 + [0] * 378, dtype='float32')


model = Pair_Attention2()
model.load_weights(r'D:\python code\N_Link\Human\model\hm_onehot_PAtt2.h5')


feature_model = Model(inputs=model.inputs, outputs=model.get_layer('add').output)
features = feature_model.predict(word).reshape(len(word), 320)
print(features.shape)

from sklearn.manifold import TSNE
embedded = TSNE(n_components=2).fit_transform(features)

import matplotlib.pyplot as plt
capsul_feature = plt.figure(figsize=(8,6))
for i, labels in enumerate(train_label):
    if int(labels)==0:
        plt.scatter(embedded[i,0],embedded[i,1],c='r',s=0.9,facecolors='none',label='Non-N-Linked')
    else:
        plt.scatter(embedded[i,0],embedded[i,1],c='b',s=0.9,facecolors='none',label='N-Linked')
plt.ylabel('Dimension2',fontweight='bold')
plt.xlabel('Dimension1',fontweight='bold')
l1=plt.Line2D(range(0),range(0),marker='o',color='r',linestyle='')
l2=plt.Line2D(range(0),range(0),marker='o',color='b',linestyle='')
plt.legend((l1,l2),('Non-N-Linked','N-Linked'),loc='upper right',numpoints=1)
capsul_feature.savefig('train_add.jpg')    # for saving figure
plt.show()
