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
from N_Link.Human.encoding import Onehot, AAindex_sl
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
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_predict, pos_label=1)  # poslabel正样本的标签
    auc = metrics.auc(fpr, tpr)

    return Sn, Sp, Acc, MCC, auc


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


# ---------------------------------------AAindex
# encodings_tr = AAindex_sl.AAindex_HM_encoding(ftrain)
# Xtr = np.array(encodings_tr).reshape(-1, 420)
# Xtr = AAindex_sl.col_delete(Xtr)
# f_train = scale.fit_transform(Xtr).reshape(Xtr.shape[0], 20, 20)
# 
# 
# encodings_te = AAindex_sl.AAindex_HM_encoding(ftest)
# Xte = np.array(encodings_te).reshape(-1, 420)
# Xte = AAindex_sl.col_delete(Xte)
# f_test = scale.fit_transform(Xte).reshape(Xte.shape[0], 20, 20)



def Pair_Attention1():
    word_input = Input(shape=(20, 20), name='word_input')

    overallResult = Convolution1D(filters=32, kernel_size=1, padding='same', activation="relu", name='Cov1')(word_input)

    overallResult = P_Attention(input=overallResult, use_embedding='PositionEmbedding.MODE_ADD')

    overallResult = Bidirectional(LSTM(50, dropout=0.5, activation='tanh', return_sequences=True), name='Bil')(
        overallResult)

    overallResult = Flatten()(overallResult)

    overallResult = Dense(32, activation='sigmoid', name='den')(overallResult)

    ss_output = Dense(1, activation='sigmoid', name='ss_output')(overallResult)

    return Model(inputs=[word_input], outputs=[ss_output])


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

    return Model(inputs=[word_input], outputs=[ss_output])


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


MODEL_PATH = './'
filepath = os.path.join(MODEL_PATH, 'hm_onehot_PAtt2.h5')
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

word = f_train
train_label = np.array([1] * 1218 + [0] * 1218, dtype='float32')

test_word = f_test
test_label = np.array([1] * 135 + [0] * 378, dtype='float32')

np.random.seed(seed)
KF = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
SN = []
SP = []
ACC = []
MCC = []
Precision = []
F1_score = []
AUC = []

tprs = []
aucs1 = []
mean_fpr = np.linspace(0, 1, 100)
batchSize = 32
maxEpochs = 200

# data为数据集,利用KF.split划分训练集和测试集
for train_index, val_index in KF.split(word, train_label):
    # 建立模型，并对训练集进行测试，求出预测得分
    # 划分训练集和测试集

    x_train_word, x_val = word[train_index], word[val_index]
    y_train_word, y_val = train_label[train_index], train_label[val_index]

    model = Pair_Attention2()
    model.count_params()
    model.summary()
    model.compile(optimizer='adam', loss={'ss_output': 'binary_crossentropy'}, metrics=['accuracy'])

    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=0, save_best_only=True, mode='auto')

    history = model.fit(
        {'word_input': x_train_word},
        {'ss_output': y_train_word},
        epochs=maxEpochs,
        batch_size=batchSize,
        callbacks=[EarlyStopping(monitor='val_accuracy', patience=20, verbose=0, mode='auto'),
                   checkpoint, LearningRateScheduler(step_decay)],
        verbose=2,
        validation_data=({'word_input': x_val},
                         {'ss_output': y_val}),
        shuffle=True)

    import matplotlib.pyplot as plt

    # 绘制损失函数图
    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.plot(title='Loss function')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()

    # 绘制准确率图
    plt.plot(history.history['accuracy'], label='train acc')
    plt.plot(history.history['val_accuracy'], label='val acc')
    plt.plot(title='accuracy function')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()

    model_save_path = "hm_onehot_PAtt2.h5"
    # 保存模型
    model.save(model_save_path)

    score = model.evaluate({'word_input': x_val}, y_val)

    y_pred1 = model.predict({'word_input': x_val})

    (Sn1, Sp1, Acc1, MCC1, auc) = Twoclassfy_evalu(y_val, y_pred1)

    SN.append(Sn1 * 100)
    SP.append(Sp1 * 100)
    MCC.append(MCC1 * 100)
    ACC.append(Acc1 * 100)
    AUC.append(auc * 100)

print('SN', SN)
print('SP', SP)
print('ACC', ACC)
print('MCC', MCC)
print('AUC', AUC)

meanSN = np.mean(SN)
meanSP = np.mean(SP)
meanACC = np.mean(ACC)
meanMCC = np.mean(MCC)
meanAUC = np.mean(AUC)

print("meanSN", meanSN)
print("meanSP", meanSP)
print("meanACC", meanACC)
print("meanMCC", meanMCC)
print("meanAUC", meanAUC)

model = Pair_Attention2()
model.load_weights("hm_onehot_PAtt2.h5")

y_predict1 = model.predict({'word_input': test_word})

(Sn1, Sp1, Acc1, MCC1, AUC) = Twoclassfy_evalu(test_label, y_predict1)

print('SN', Sn1 * 100)
print('SP', Sp1 * 100)
print('ACC', Acc1 * 100)
print('MCC', MCC1 * 100)
print('AUC', AUC * 100)



