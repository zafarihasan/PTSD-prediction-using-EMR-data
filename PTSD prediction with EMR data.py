
**********************************************************************************************************************
*                            #    Loading the data, sampling and spliting                                            *
**********************************************************************************************************************
df_st_nt_PTSD = pd.read_csv('/PTSD.csv')

X_tobe_RUS,X_test_skewed,y_tobe_RUS,y_test_skewed = train_test_split(df_st_nt_PTSD,y, test_size=0.15, random_state=40)

pid_sbsmple=X_tobe_RUS.idx # this is development dataset
list_of_idx = X_tobe_RUS['idx'].to_list()
df_tot_tobe_resampled=df_st_nt_PTSD[df_st_nt_PTSD.idx.isin(list_of_idx)]
df=df_tot_tobe_resampled
#1. Find Number of samples which are positive
PTSD_pos_cnt = len(df[df['PTSD'] == 1])

#2. Get indices of negative and positive samples
PTSD_neg_indices = df[df.PTSD == 0].index
PTSD_pos_indices = df[df.PTSD == 1].index

#3. Random sample negative indices
random_indices = np.random.choice(PTSD_neg_indices,PTSD_pos_cnt, replace=False)

#4. Concat positive indices with the sampled negative ones
under_sample_indices = np.concatenate([PTSD_pos_indices,random_indices])

#5. Get Balance Dataframe
under_sample = df.loc[under_sample_indices]

X_RUS_train = under_sample.loc[:,under_sample.columns != 'PTSD']
y_RUS_train = under_sample.loc[:,under_sample.columns == 'PTSD']
print(y_RUS_train[y_RUS_train.PTSD==1].count())

#--------------------- choose negative samples to create a skewed dataset --------------
# select those records that are not in the balanced sampled dataset
only_negative_sample=df_tot_tobe_resampled[~df_tot_tobe_resampled.index.isin(under_sample.index)]
PTSD_neg_indices_skewed = only_negative_sample.index
skewed_cnt=3400 # add 100 times mote PTSD negative to validate model in the training dataset
random_indices_skewed = np.random.choice(PTSD_neg_indices_skewed,skewed_cnt, replace=False)
extra_negative_samples_for_test_crossvalidation=only_negative_sample.loc[random_indices_skewed]

**********************************************************************************************************************
*                                                   functions                                                        *
**********************************************************************************************************************

from numpy.random import seed
seed(1)
from tensorflow import random
random.set_seed(1)

import spacy
import re
import string
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
spacy.load('en')
lemmatizer = spacy.lang.en.English()

from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Conv1D, Dense, MaxPool1D, Flatten, Input,GlobalMaxPooling1D,GlobalMaxPool1D
from keras.models import Model
from keras.layers.embeddings import Embedding
from keras.layers.core import Activation, Dropout
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from keras.layers.merge import concatenate
from numpy import array
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from keras import layers

def my_tokenizer(doc):
    tokens = lemmatizer(doc)
    return([token.lemma_ for token in tokens if len(token)>1])

def clean_all_but_alphabet(s):
  out = re.sub(r'[^a-zA-Z\s]', ' ', s)
  out=re.sub("\s\s+", " ", out)
  out=out.lower()
  return(out)

def wm2df(wm, feat_names):
    # create an index for each row
    doc_names = ['Doc{:d}'.format(idx) for idx, _ in enumerate(wm)]
    df = pd.DataFrame(data=wm.toarray(), index=doc_names,columns=feat_names)
    return(df)


def show_metrics(y_test, y_pred, message):
    #print('confusion matrix: \n', confusion_matrix(y_test, y_pred))
    auc_val= roc_auc_score(y_test, y_pred, average='macro')
    TN, FP, FN, TP = confusion_matrix(y_test, y_pred).ravel()

    AUC= roc_auc_score(y_test, y_pred, average='macro')
    TN, FP, FN, TP = confusion_matrix(y_test, y_pred).ravel()
    # Sensitivity, hit rate, recall, or true positive rate
    SN = TP/(TP+FN)
    # Specificity or true negative rate
    SP = TN/(TN+FP) 
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    # Negative predictive value
    NPV = TN/(TN+FN)
    # Overall accuracy
    ACC = (TP+TN)/(TP+FP+FN+TN)

    F_score=2*(PPV*SN)/(PPV+SN)

    metrics=[round(PPV,2),round(NPV,2),round(SN,2),round(SP,2),round(ACC,2),round(F_score,2),round(AUC,2)]
    return metrics
	
	

**********************************************************************************************************************
*                                                     ST_Data MLNN model                                             *
**********************************************************************************************************************

threshoulds=[0.15,0.20,0.25,0.30,0.35,0.4,0.45,0.50,0.55,0.60,0.65,0.7,0.75,0.8,0.85,0.90,0.95]
cols=7 # number of mertrics
rows=len(threshoulds)

sum_vector= [[0]*cols]*rows

fold=0


cv = StratifiedKFold(n_splits=2)
for train,test in cv.split(X_RUS_train,y_RUS_train):

  print("----------",fold,"-----------")
  print(test)
  pid=X_RUS_train.iloc[train].idx
  X_st_train=X_RUS_train.iloc[train].drop(['idx','note','recNo','rand'],axis=1)
  X_st_test=X_RUS_train.iloc[test].drop(['idx','note','recNo','rand'],axis=1)
  X_st_test_extra_neg = extra_negative_samples_for_test_crossvalidation.drop(['idx','note','recNo','rand','PTSD'],axis=1)
  y_st_test_extra_neg=extra_negative_samples_for_test_crossvalidation.PTSD


  X_test_st_skewed=pd.concat([X_st_test, X_st_test_extra_neg])
  

  y_train=y_RUS_train.iloc[train].PTSD
  y_test=y_RUS_train.iloc[test].PTSD
  y_nt_test_extra_neg_i = y_nt_test_extra_neg
  y_test_skewed=pd.concat([y_test, y_nt_test_extra_neg_i],axis=0)



  #--------------------------------------- st data -------------------------------
  seed(1)
  random.set_seed(1)
  
  
  input_st = Input(shape=(399,))
  dense_layer_1 = Dense(50, activation='relu')(input_st)
  dense_layer_2 = Dense(50, activation='relu')(dense_layer_1)
  dense_layer_3 = Dense(50, activation='relu')(dense_layer_2)
  dense20_st = Dense(20, activation='relu')(dense_layer_3)
  dense1_st = Dense(1, activation='sigmoid')(dense20_st)
  
  X_st_train = np.asarray(X_train_just_st).astype(np.float32)

  model_mlp = Model(inputs=[input_st], outputs=dense1_st)
  # compile
  model_mlp.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

  model_mlp.fit(X_train_just_st, array(y_train), epochs=7, batch_size=16 ,verbose=1)

  
  y_pred_mlp = model_mlp.predict(X_test_just_st, verbose=False)

  for loop in range(len(threshoulds)):
    threshould=threshoulds[loop]
    print("------------------------------ threshould: ",threshould,"---------------------------------",fold,"-----------")
    y_pred_mlp_rounded = (y_pred_mlp > threshould)
    cm=confusion_matrix(y_test_skewed, y_pred_mlp_rounded)  
    metrics=show_metrics(y_test_skewed, y_pred_mlp_rounded, "ST MLNN")
    sum_vector[loop]=sum_vector[loop]+np.array(metrics)
    print(cm)
    print(metrics)

  fold+=1
#----------------------- calculate the average metrics for each threshould -----------
print("\n-------------------- average for each threshold -----------------")
print("\tPPV \t NPV \t SN \t SP \t ACC \t F1 \t AUC")
for loop in range(len(threshoulds)):
  v=sum_vector[loop]
  print(threshoulds[loop],end=':\t')
  for n in v:
    print(round(n/5,2),end='\t')
  print()

########################################## test the model with the hold-out dataset ############################################

threshoulds=0.65


X_st_holdout=X_test_skewed.drop(['idx','note','recNo','rand'],axis=1)


X_st_holdout = np.asarray(X_st_holdout).astype(np.float32)


y_pred = model_mlp.predict(X_st_holdout, verbose=False)

y_pred_rounded = (y_pred > threshould)
cm=confusion_matrix(y_test_skewed, y_pred_rounded)  
metrics=show_metrics(y_test_skewed, y_pred_rounded, "MLNN_ST")
print(metrics)
print(cm)



**********************************************************************************************************************
*                                                    ST_Data RF model                                                *
**********************************************************************************************************************

threshoulds=[0.15,0.20,0.25,0.30,0.35,0.4,0.45,0.50,0.55,0.60,0.65,0.7,0.75,0.8,0.85,0.90,0.95]
cols=7 # number of mertrics
rows=len(threshoulds)

sum_vector= [[0]*cols]*rows

fold=0


cv = StratifiedKFold(n_splits=2)
for train,test in cv.split(X_RUS_train,y_RUS_train):

  print("----------",fold,"-----------")
  print(test)
  pid=X_RUS_train.iloc[train].idx
  X_st_train=X_RUS_train.iloc[train].drop(['idx','note','recNo','rand'],axis=1)
  X_st_test=X_RUS_train.iloc[test].drop(['idx','note','recNo','rand'],axis=1)
  X_st_test_extra_neg = extra_negative_samples_for_test_crossvalidation.drop(['idx','note','recNo','rand','PTSD'],axis=1)
  y_st_test_extra_neg=extra_negative_samples_for_test_crossvalidation.PTSD


  X_test_st_skewed=pd.concat([X_st_test, X_st_test_extra_neg])
  

  y_train=y_RUS_train.iloc[train].PTSD
  y_test=y_RUS_train.iloc[test].PTSD
  y_nt_test_extra_neg_i = y_nt_test_extra_neg
  y_test_skewed=pd.concat([y_test, y_nt_test_extra_neg_i],axis=0)

  #--------------------------------------- st data -------------------------------
  from sklearn.ensemble import RandomForestClassifier
  RF= RandomForestClassifier(n_estimators=100, random_state=96)

  RF_model_nt = RF.fit(X_st_train, y_train)
  y_pred = RF_model_nt.predict_proba(X_test_st_skewed)[::,1]

  for loop in range(len(threshoulds)):
    threshould=threshoulds[loop]
    print("------------------------------ threshould: ",threshould,"---------------------------------",fold,"-----------")
    y_pred_rounded = (y_pred_mlp > threshould)
    cm=confusion_matrix(y_test_skewed, y_pred_rounded)  
    metrics=show_metrics(y_test_skewed, y_pred_rounded, "ST MLNN")
    sum_vector[loop]=sum_vector[loop]+np.array(metrics)
    print(cm)
    print(metrics)

  fold+=1
#----------------------- calculate the average metrics for each threshould -----------
print("\n-------------------- average for each threshold -----------------")
print("\tPPV \t NPV \t SN \t SP \t ACC \t F1 \t AUC")
for loop in range(len(threshoulds)):
  v=sum_vector[loop]
  print(threshoulds[loop],end=':\t')
  for n in v:
    print(round(n/5,2),end='\t')
  print()

########################################## test the model with the hold-out dataset ############################################

threshoulds=0.5


X_st_holdout=X_test_skewed.drop(['idx','note','recNo','rand'],axis=1)


X_st_holdout = np.asarray(X_st_holdout).astype(np.float32)


y_pred = model_mlp.predict(X_st_holdout, verbose=False)

y_pred_rounded = (y_pred > threshould)
cm=confusion_matrix(y_test_skewed, y_pred_rounded)  
metrics=show_metrics(y_test_skewed, y_pred_rounded, "MLNN_ST")
print(metrics)
print(cm)


**********************************************************************************************************************
*                                                 Note_Data BOW RF                                                   *
**********************************************************************************************************************

######################### the baseline model, apply random-forest directly on note data ########################
import spacy
from html import unescape
import re
import string
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
# create a spaCy tokenizer
spacy.load('en')
lemmatizer = spacy.lang.en.English()

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

def show_metrics(y_test, y_pred, message):
    #print('confusion matrix: \n', confusion_matrix(y_test, y_pred))
    auc_val= roc_auc_score(y_test, y_pred, average='macro')
    TN, FP, FN, TP = confusion_matrix(y_test, y_pred).ravel()

    AUC= roc_auc_score(y_test, y_pred, average='macro')
    TN, FP, FN, TP = confusion_matrix(y_test, y_pred).ravel()
    # Sensitivity, hit rate, recall, or true positive rate
    SN = TP/(TP+FN)
    # Specificity or true negative rate
    SP = TN/(TN+FP) 
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    # Negative predictive value
    NPV = TN/(TN+FN)
    # Overall accuracy
    ACC = (TP+TN)/(TP+FP+FN+TN)
    F_score=2*(PPV*SN)/(PPV+SN)
    metrics=[round(PPV,2),round(NPV,2),round(SN,2),round(SP,2),round(ACC,2),round(F_score,2),round(AUC,2)]
    return metrics

def my_tokenizer(doc):
    tokens = lemmatizer(doc)
    return([token.lemma_ for token in tokens if len(token)>1])

def wm2df(wm, feat_names):
    # create an index for each row
    doc_names = ['Doc{:d}'.format(idx) for idx, _ in enumerate(wm)]
    df = pd.DataFrame(data=wm.toarray(), index=doc_names,columns=feat_names)
    return(df)

from sklearn.feature_extraction.text import TfidfVectorizer
X_tobe_RUS,X_test_skewed,y_tobe_RUS,y_test_skewed = train_test_split(df_st_nt_PTSD,y, test_size=0.15, random_state=40)


threshoulds=[0.10,0.15,0.20,0.25,0.3,0.35,0.4,0.45,0.50,0.55,0.6,0.65,0.7,0.75,0.80,0.85]
cols=7 # number of mertrics
rows=len(threshoulds)
sum_vector= [[0]*cols]*rows
fold=0
cv = StratifiedKFold(n_splits=10)
for train,test in cv.split(X_RUS_train,y_RUS_train):
  fold+=1

  print("----------",fold,"-----------")


  X_nt_train=X_RUS_train.iloc[train].note
  X_st_test=X_RUS_train.iloc[test].drop(['idx','note'],axis=1)
  X_nt_test=X_RUS_train.iloc[test].note

  X_nt_test_extra_neg = extra_negative_samples_for_test_crossvalidation.note
  y_nt_test_extra_neg=extra_negative_samples_for_test_crossvalidation.PTSD


  X_test_st_skewed=pd.concat([X_st_test, X_st_test_extra_neg])
  idx_test_extra=pd.concat([X_RUS_train.iloc[test].idx,extra_negative_samples_for_test_crossvalidation.idx])

  X_test_nt_skewed=pd.concat([X_nt_test, X_nt_test_extra_neg])
  print('********* X_test_st_skewed',X_test_st_skewed.shape)
  print('********* X_test_nt_skewed',X_test_nt_skewed.shape)

  

  y_train=y_RUS_train.iloc[train].PTSD
  y_test=y_RUS_train.iloc[test].PTSD
  y_nt_test_extra_neg_i = y_nt_test_extra_neg
  y_test=pd.concat([y_test, y_nt_test_extra_neg_i],axis=0)


  print('X_test_skewed',X_test_skewed.shape)
  print('y_test_skewed',y_test_skewed.shape)


  custom_vec = TfidfVectorizer(tokenizer=my_tokenizer,
                            analyzer='word',
                            ngram_range=(1,3),
                            stop_words='english',
                            min_df=0.05)
  wm_train = custom_vec.fit_transform(X_nt_train)
  wm_test = custom_vec.transform(X_test_nt_skewed)
  tokens = custom_vec.get_feature_names()
  df_wm_train=wm2df(wm_train,tokens)
  df_wm_test=wm2df(wm_test,tokens)
  print("len of tokens: ",len(tokens))

  from sklearn.ensemble import RandomForestClassifier
  RF= RandomForestClassifier(n_estimators=100, random_state=96)

  RF_model_nt = RF.fit(df_wm_train, y_train)
  y_pred = RF_model_nt.predict_proba(df_wm_test)[::,1]

  for i in range(len(threshoulds)):
        threshould=threshoulds[i]
        print("------------------------------ threshould: ",threshould,"--------------------",fold,"-------------------")
        #print(y_pred)
        y_pred_rounded = (y_pred > threshould)
        cm=confusion_matrix(y_test, y_pred_rounded)  
        metrics=show_metrics(y_test, y_pred_rounded, "RF-(1,1)")
        sum_vector[i]=sum_vector[i]+np.array(metrics)
        print(cm)
        print(metrics)
#----------------------- calculate the average metrics for each threshould -----------
print("\n-------------------- average for each threshold -----------------")
print("\tPPV \t NPV \t SN \t SP \t ACC \t F1 \t AUC")
for loop in range(len(threshoulds)):
  v=sum_vector[loop]
  print(threshoulds[loop],end=':\t')
  for n in v:
    print(round(n/fold,2),end='\t')
  print()

threshould=0.75


print('\n\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ apply the model on the HOLDOUT test data @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
X_test_holdout=X_test_skewed.note
X_test_holdout=[note for note in  X_test_holdout]
y_test_holdout=y_test_skewed

wm_test_holdout = custom_vec.transform(X_test_holdout)
df_wm_test_holdout=wm2df(wm_test_holdout,tokens)
y_pred_rf = RF_model_nt.predict_proba(df_wm_test_holdout)[::,1]

y_pred_rounded = (y_pred_rf > threshould)
cm=confusion_matrix(y_test_holdout, y_pred_rounded)  
metrics=show_metrics(y_test_holdout, y_pred_rounded, "RF_(1,1)")
print(cm)
print(metrics)



**********************************************************************************************************************
*                                                 Note_Data CNN model                                                *
**********************************************************************************************************************


cnn_score_train = pd.DataFrame(columns=['fold', 'idx', 'score'])

fold=0
threshoulds=[0.15,0.20,0.25,0.30,0.35,0.4,0.45,0.50,0.55,0.60,0.65,0.7,0.75,0.8,0.85,0.90,0.95]
cols=7 # number of mertrics
rows=len(threshoulds)

sum_vector= [[0]*cols]*rows

cv = StratifiedKFold(n_splits=10)
for train,test in cv.split(X_RUS_train,y_RUS_train):

  print("----------",fold,"-----------")
  print(test)

  pid=X_RUS_train.iloc[train].idx
  #print('pid: ',pid)
  #print('train: ',train)
  #print('test: ',test)
  X_st_train=X_RUS_train.iloc[train].drop(['idx','note'],axis=1)
  X_nt_train=X_RUS_train.iloc[train].note
  X_st_test=X_RUS_train.iloc[test].drop(['idx','note'],axis=1)
  X_nt_test=X_RUS_train.iloc[test].note

  X_st_test_extra_neg = extra_negative_samples_for_test_crossvalidation.drop(['idx','note','PTSD'],axis=1)
  #X_nt_test_extra_neg=df_st_nt_just_negative_PTSD.note
  X_nt_test_extra_neg = extra_negative_samples_for_test_crossvalidation.note
  y_nt_test_extra_neg=extra_negative_samples_for_test_crossvalidation.PTSD

  print('********** X_st_test',X_st_test.shape)
  print('********** X_st_test_extra_neg',X_st_test_extra_neg.shape)

  X_test_st_skewed=pd.concat([X_st_test, X_st_test_extra_neg])
  idx_test_extra=pd.concat([X_RUS_train.iloc[test].idx,extra_negative_samples_for_test_crossvalidation.idx])

  X_test_nt_skewed=pd.concat([X_nt_test, X_nt_test_extra_neg])
  print('********* X_test_st_skewed',X_test_st_skewed.shape)
  print('********* X_test_nt_skewed',X_test_nt_skewed.shape)
  

  y_train=y_RUS_train.iloc[train].PTSD
  y_test=y_RUS_train.iloc[test].PTSD
  y_nt_test_extra_neg_i = y_nt_test_extra_neg
  y_test_skewed=pd.concat([y_test, y_nt_test_extra_neg_i],axis=0)


  print('X_test_skewed',X_test_skewed.shape)
  print('y_test_skewed',y_test_skewed.shape)

  ########################################### three kernels ###############################
  max_len = max([len(s.split()) for s in X_nt_train])
  max_len=80000
  print('Maximum length: %d' % max_len)
  tokenizer = Tokenizer()
  tokenizer.fit_on_texts(X_nt_train)
  vocab_size = len(tokenizer.word_index) + 1
  X_train_encoded = tokenizer.texts_to_sequences(X_nt_train)
  X_test_encoded = tokenizer.texts_to_sequences(X_test_nt_skewed)
  # pad sequences
  X_train_padded = pad_sequences(X_train_encoded, maxlen=max_len, padding='post')
  X_test_padded = pad_sequences(X_test_encoded, maxlen=max_len, padding='post')

  #--------------------------------------- note data: kernel size=1 -------------------------------
  inputs1 = Input(shape=(max_len,))
  embedding1 = Embedding(vocab_size, 100)(inputs1)
  conv1 = Conv1D(filters=64, kernel_size=1, activation='relu')(embedding1)
  drop1 = Dropout(0.5)(conv1)
  pool1 =layers.GlobalMaxPool1D()(drop1)
  #flat1 = Flatten()(pool1)
  #--------------------------------------- note data: kernel size=2 -------------------------------
  # channel 2
  inputs2 = Input(shape=(max_len,))
  embedding2 = Embedding(vocab_size, 100)(inputs2)
  conv2 = Conv1D(filters=64, kernel_size=2, activation='relu')(embedding2)
  drop2 = Dropout(0.5)(conv2)
  #pool2 = MaxPooling1D(pool_size=2)(drop2)
  pool2 =layers.GlobalMaxPool1D()(drop2)
  #flat2 = Flatten()(pool2)
  #--------------------------------------- note data: kernel size=3 -------------------------------
  # channel 3
  inputs3 = Input(shape=(max_len,))
  embedding3 = Embedding(vocab_size, 100)(inputs3)
  conv3 = Conv1D(filters=64, kernel_size=3, activation='relu')(embedding3)
  drop3 = Dropout(0.5)(conv3)
  #pool3 = MaxPooling1D(pool_size=2)(drop3)
  pool3 =layers.GlobalMaxPool1D()(drop3)
  #flat3 = Flatten()(pool3)
  # merge
  #--------------------------------------- merge all of the channels -------------------------------
  merged = concatenate([pool1, pool2, pool3])
  # interpretation
  dense1 = Dense(10, activation='relu')(merged)
  outputs = Dense(1, activation='sigmoid')(dense1)
  model_cnn_3k = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)
  # compile
  model_cnn_3k.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

  #########################################################################################


  model_cnn_3k.fit([X_train_padded,X_train_padded,X_train_padded], array(y_train), epochs=9, batch_size=16 ,verbose=1)
  
  y_pred_cnn = model_cnn_3k.predict([X_test_padded,X_test_padded,X_test_padded], verbose=False)

  pid=X_RUS_train.iloc[test]['idx'].to_list()
  note_score = pd.DataFrame(
    {'fold':[fold for i in y_pred_cnn.tolist()],
     'idx': idx_test_extra,
     'score': [i[0] for i in y_pred_cnn.tolist()]})
  cnn_score_train=cnn_score_train.append(note_score)




  for loop in range(len(threshoulds)):
    threshould=threshoulds[loop]
    print("------------------------------ threshould: ",threshould,"---------------------------------",fold,"-----------")
    y_pred_cnn_rounded = (y_pred_cnn > threshould)
    cm=confusion_matrix(y_test_skewed, y_pred_cnn_rounded)  
    metrics=show_metrics(y_test_skewed, y_pred_cnn_rounded, "CNN")
    sum_vector[loop]=sum_vector[loop]+np.array(metrics)
    print(cm)

    metrics_rounded = [round(num, 2) for num in metrics]
    print(metrics_rounded)

  fold+=1
#----------------------- calculate the average metrics for each threshould -----------
print("\n-------------------- average for each threshold -----------------")
print("\tPPV \t NPV \t SN \t SP \t ACC \t F1 \t AUC")
for loop in range(len(threshoulds)):
  v=sum_vector[loop]
  print(threshoulds[loop],end=':\t')
  for n in v:
    print(round(n/fold,2),end='\t')
  print()

#-------TEST with hold out data-------------------- only CNN model with three kernels

threshoulds=0.85 # the threshold that maximized the F1 in the training dataset


X_nt_holdout=X_test_skewed.note
X_st_holdout=X_test_skewed.drop(['idx','note'],axis=1)

X_test_encoded = tokenizer.texts_to_sequences(X_nt_holdout)
X_test_padded = pad_sequences(X_test_encoded, maxlen=max_len, padding='post')

y_pred_cnn_nt_holdout = model_cnn_3k.predict([X_test_padded,X_test_padded,X_test_padded], verbose=False)

y_pred_cnn_nt_holdout_rounded = (y_pred_cnn_nt_holdout > threshould)
cm=confusion_matrix(y_test_skewed, y_pred_cnn_nt_holdout_rounded)  
metrics=show_metrics(y_test_skewed, y_pred_cnn_nt_holdout_rounded, "CNN")
print(metrics)
print(cm)



**********************************************************************************************************************
*                                               Mixed_Data MLNN (Parallel)                                           *
**********************************************************************************************************************
threshoulds=[0.15,0.20,0.25,0.30,0.35,0.4,0.45,0.50,0.55,0.60,0.65,0.7,0.75,0.8,0.85,0.90,0.95]
cols=7 # number of mertrics
rows=len(threshoulds)

sum_vector= [[0]*cols]*rows

fold=0


cv = StratifiedKFold(n_splits=2)
for train,test in cv.split(X_RUS_train,y_RUS_train):

  print("----------",fold,"-----------")
  print(test)
  pid=X_RUS_train.iloc[train].idx
  X_st_train=X_RUS_train.iloc[train].drop(['idx','note','recNo','rand'],axis=1)
  X_nt_train=X_RUS_train.iloc[train].note
  X_st_test=X_RUS_train.iloc[test].drop(['idx','note','recNo','rand'],axis=1)
  X_nt_test=X_RUS_train.iloc[test].note
  X_st_test_extra_neg = extra_negative_samples_for_test_crossvalidation.drop(['idx','note','recNo','rand','PTSD'],axis=1)
  X_nt_test_extra_neg = extra_negative_samples_for_test_crossvalidation.note
  y_nt_test_extra_neg=extra_negative_samples_for_test_crossvalidation.PTSD

  print('********** X_st_test',X_st_test.shape)
  print('********** X_st_test_extra_neg',X_st_test_extra_neg.shape)

  X_test_st_skewed=pd.concat([X_st_test, X_st_test_extra_neg])
  X_test_nt_skewed=pd.concat([X_nt_test, X_nt_test_extra_neg])
  print('********* X_test_st_skewed',X_test_st_skewed.shape)
  print('********* X_test_nt_skewed',X_test_nt_skewed.shape)
  

  y_train=y_RUS_train.iloc[train].PTSD
  y_test=y_RUS_train.iloc[test].PTSD
  y_nt_test_extra_neg_i = y_nt_test_extra_neg
  y_test_skewed=pd.concat([y_test, y_nt_test_extra_neg_i],axis=0)


  print('X_test_skewed',X_test_skewed.shape)
  print('y_test_skewed',y_test_skewed.shape)

  #--------------------------------------- st data -------------------------------
  seed(1)
  random.set_seed(1)
  input_st = Input(shape=(399,))
  dense_layer_1 = Dense(50, activation='relu')(input_st)
  dense_layer_2 = Dense(50, activation='relu')(dense_layer_1)
  dense_layer_3 = Dense(50, activation='relu')(dense_layer_2)
  dense20_st = Dense(20, activation='relu')(dense_layer_3)

  #--------------------------------------- note data: kernel size=1 -------------------------------
  max_len = max([len(s.split()) for s in X_nt_train])
  max_len=70000
  print('Maximum length: %d' % max_len)
  tokenizer = Tokenizer()
  tokenizer.fit_on_texts(X_nt_train)
  vocab_size = len(tokenizer.word_index) + 1
  X_train_encoded = tokenizer.texts_to_sequences(X_nt_train)
  X_test_encoded = tokenizer.texts_to_sequences(X_test_nt_skewed)
  # pad sequences
  X_train_padded = pad_sequences(X_train_encoded, maxlen=max_len, padding='post')
  X_test_padded = pad_sequences(X_test_encoded, maxlen=max_len, padding='post')
  
  seed(1)
  random.set_seed(1)

  #######################################################
   #--------------------------------------- note data: kernel size=1 -------------------------------
  inputs1 = Input(shape=(max_len,))
  embedding1 = Embedding(vocab_size, 100)(inputs1)
  conv1 = Conv1D(filters=64, kernel_size=1, activation='relu')(embedding1)
  drop1 = Dropout(0.5)(conv1)
  pool1 =layers.GlobalMaxPool1D()(drop1)
  dense20_p1 = Dense(20, activation='relu')(pool1)
  #flat1 = Flatten()(pool1)
  flat1 = Flatten()(dense20_p1)
  #--------------------------------------- note data: kernel size=2 -------------------------------
  # channel 2
  inputs2 = Input(shape=(max_len,))
  embedding2 = Embedding(vocab_size, 100)(inputs2)
  conv2 = Conv1D(filters=64, kernel_size=2, activation='relu')(embedding2)
  drop2 = Dropout(0.5)(conv2)
  #pool2 = MaxPooling1D(pool_size=2)(drop2)
  pool2 =layers.GlobalMaxPool1D()(drop2)
  dense20_p2 = Dense(20, activation='relu')(pool2)
  flat2 = Flatten()(dense20_p2)
  #--------------------------------------- note data: kernel size=3 -------------------------------
  # channel 3
  inputs3 = Input(shape=(max_len,))
  embedding3 = Embedding(vocab_size, 100)(inputs3)
  conv3 = Conv1D(filters=64, kernel_size=3, activation='relu')(embedding3)
  drop3 = Dropout(0.5)(conv3)
  #pool3 = MaxPooling1D(pool_size=2)(drop3)
  pool3 =layers.GlobalMaxPool1D()(drop3)
  dense20_p3 = Dense(20, activation='relu')(pool3)
  #flat3 = Flatten()(pool3)
  flat3 = Flatten()(dense20_p3)
  # merge
  #--------------------------------------- merge all of the channels -------------------------------
  #merged = concatenate([dense20_st,pool1, pool2, pool3])
  #merged = concatenate([dense20_st,dense20_p1, dense20_p2, dense20_p3])
  merged = concatenate([dense20_st,flat1, flat2, flat3])
  # interpretation
  dense1 = Dense(20, activation='relu')(merged)
  #dense2 = Dense(10, activation='relu')(dense1)
  outputs = Dense(1, activation='sigmoid')(dense1)
  model_cnn_3k_st = Model(inputs=[input_st,inputs1, inputs2, inputs3], outputs=outputs)
  # compile
  model_cnn_3k_st.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

  X_st_train = np.asarray(X_st_train).astype(np.float32)
  X_test_st_skewed = np.asarray(X_test_st_skewed).astype(np.float32)
  

  model_cnn_3k_st.fit([X_st_train,X_train_padded,X_train_padded,X_train_padded], array(y_train), epochs=7, batch_size=16 ,verbose=1)
  
  #------------------
    
  y_pred_cnn_3k_st = model_cnn_3k_st.predict([X_test_st_skewed,X_test_padded,X_test_padded,X_test_padded], verbose=False)

  ##########################





  for loop in range(len(threshoulds)):
    threshould=threshoulds[loop]
    print("------------------------------ threshould: ",threshould,"---------------------------------",fold,"-----------")
    y_pred_mixed_keras_rounded = (y_pred_cnn_3k_st > threshould)
    cm=confusion_matrix(y_test_skewed, y_pred_mixed_keras_rounded)  
    metrics=show_metrics(y_test_skewed, y_pred_mixed_keras_rounded, "mixed parallel")
    sum_vector[loop]=sum_vector[loop]+np.array(metrics)
    print(cm)
    print(metrics)

  fold+=1
#----------------------- calculate the average metrics for each threshould -----------
print("\n-------------------- average for each threshold -----------------")
print("\tPPV \t NPV \t SN \t SP \t ACC \t F1 \t AUC")
for loop in range(len(threshoulds)):
  v=sum_vector[loop]
  print(threshoulds[loop],end=':\t')
  for n in v:
    print(round(n/5,2),end='\t')
  print()

########################################## test the model with the hold-out dataset ############################################

threshoulds=0.8

from sklearn.model_selection import train_test_split
#----------------------------------------- set aside a subset of the data as hold-out test dataset --------------------
X_tobe_RUS,X_test_skewed,y_tobe_RUS,y_test_skewed = train_test_split(df_st_nt_PTSD,y, test_size=0.15, random_state=40)
print(X_tobe_RUS.shape)
print(X_test_skewed.shape)
print(y_tobe_RUS.shape)
print(y_test_skewed.shape)

X_nt_holdout=X_test_skewed.note
X_st_holdout=X_test_skewed.drop(['idx','note','recNo','rand','PTSD'],axis=1)




#--------------------------------------- note data: kernel size=1 -------------------------------
X_test_encoded = tokenizer.texts_to_sequences(X_nt_holdout)
X_test_padded = pad_sequences(X_test_encoded, maxlen=max_len, padding='post')

X_st_holdout = np.asarray(X_st_holdout).astype(np.float32)


y_pred_cnn = model_cnn_3k_st.predict([X_st_holdout,X_test_padded,X_test_padded,X_test_padded], verbose=False)

y_pred_cnn_rounded = (y_pred_cnn > threshould)
cm=confusion_matrix(y_test_skewed, y_pred_cnn_rounded)  
metrics=show_metrics(y_test_skewed, y_pred_cnn_rounded, "CNN")
print(metrics)
print(cm)

**********************************************************************************************************************
*                                              Mixed_Data RF (Serial)                                                *
**********************************************************************************************************************
from sklearn.ensemble import RandomForestClassifier

cnn_score_train = pd.DataFrame(columns=['fold', 'idx', 'score'])

fold=0
threshoulds=[0.15,0.20,0.25,0.30,0.35,0.4,0.45,0.50,0.55,0.60,0.65,0.7,0.75,0.8,0.85,0.90,0.95]
cols=7 # number of mertrics
rows=len(threshoulds)

sum_vector= [[0]*cols]*rows

cv = StratifiedKFold(n_splits=10)
for train,test in cv.split(X_RUS_train,y_RUS_train):

  print("----------",fold,"-----------")
  print(test)

  pid=X_RUS_train.iloc[train].idx
  X_st_train=X_RUS_train.iloc[train].drop(['idx','note'],axis=1)
  X_nt_train=X_RUS_train.iloc[train].note
  X_st_test=X_RUS_train.iloc[test].drop(['idx','note'],axis=1)
  X_nt_test=X_RUS_train.iloc[test].note

  X_st_test_extra_neg = extra_negative_samples_for_test_crossvalidation.drop(['idx','note','PTSD'],axis=1)
  X_nt_test_extra_neg = extra_negative_samples_for_test_crossvalidation.note
  y_nt_test_extra_neg=extra_negative_samples_for_test_crossvalidation.PTSD

  print('********** X_st_test',X_st_test.shape)
  print('********** X_st_test_extra_neg',X_st_test_extra_neg.shape)

  X_test_st_skewed=pd.concat([X_st_test, X_st_test_extra_neg])
  idx_test_extra=pd.concat([X_RUS_train.iloc[test].idx,extra_negative_samples_for_test_crossvalidation.idx])

  X_test_nt_skewed=pd.concat([X_nt_test, X_nt_test_extra_neg])
  print('********* X_test_st_skewed',X_test_st_skewed.shape)
  print('********* X_test_nt_skewed',X_test_nt_skewed.shape)
  

  y_train=y_RUS_train.iloc[train].PTSD
  y_test=y_RUS_train.iloc[test].PTSD
  y_nt_test_extra_neg_i = y_nt_test_extra_neg
  y_test_skewed=pd.concat([y_test, y_nt_test_extra_neg_i],axis=0)


  print('X_test_skewed',X_test_skewed.shape)
  print('y_test_skewed',y_test_skewed.shape)

  ########################################### three kernels ###############################
  max_len = max([len(s.split()) for s in X_nt_train])
  max_len=80000
  print('Maximum length: %d' % max_len)
  tokenizer = Tokenizer()
  tokenizer.fit_on_texts(X_nt_train)
  vocab_size = len(tokenizer.word_index) + 1
  X_train_encoded = tokenizer.texts_to_sequences(X_nt_train)
  X_test_encoded = tokenizer.texts_to_sequences(X_nt_test)
  # pad sequences
  X_train_padded = pad_sequences(X_train_encoded, maxlen=max_len, padding='post')
  X_test_padded = pad_sequences(X_test_encoded, maxlen=max_len, padding='post')

  #--------------------------------------- note data: kernel size=1 -------------------------------
  inputs1 = Input(shape=(max_len,))
  embedding1 = Embedding(vocab_size, 100)(inputs1)
  conv1 = Conv1D(filters=64, kernel_size=1, activation='relu')(embedding1)
  drop1 = Dropout(0.5)(conv1)
  pool1 =layers.GlobalMaxPool1D()(drop1)
  #flat1 = Flatten()(pool1)
  #--------------------------------------- note data: kernel size=2 -------------------------------
  # channel 2
  inputs2 = Input(shape=(max_len,))
  embedding2 = Embedding(vocab_size, 100)(inputs2)
  conv2 = Conv1D(filters=64, kernel_size=2, activation='relu')(embedding2)
  drop2 = Dropout(0.5)(conv2)
  #pool2 = MaxPooling1D(pool_size=2)(drop2)
  pool2 =layers.GlobalMaxPool1D()(drop2)
  #flat2 = Flatten()(pool2)
  #--------------------------------------- note data: kernel size=3 -------------------------------
  # channel 3
  inputs3 = Input(shape=(max_len,))
  embedding3 = Embedding(vocab_size, 100)(inputs3)
  conv3 = Conv1D(filters=64, kernel_size=3, activation='relu')(embedding3)
  drop3 = Dropout(0.5)(conv3)
  #pool3 = MaxPooling1D(pool_size=2)(drop3)
  pool3 =layers.GlobalMaxPool1D()(drop3)
  #flat3 = Flatten()(pool3)
  # merge
  #--------------------------------------- merge all of the channels -------------------------------
  merged = concatenate([pool1, pool2, pool3])
  # interpretation
  dense1 = Dense(10, activation='relu')(merged)
  outputs = Dense(1, activation='sigmoid')(dense1)
  model_cnn_3k = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)
  # compile
  model_cnn_3k.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

  #########################################################################################


  model_cnn_3k.fit([X_train_padded,X_train_padded,X_train_padded], array(y_train), epochs=9, batch_size=16 ,verbose=1)
  
  y_pred_cnn = model_cnn_3k.predict([X_test_padded,X_test_padded,X_test_padded], verbose=False)

  pid=X_RUS_train.iloc[test]['idx'].to_list()
  note_score = pd.DataFrame(
    {'idx': idx_test_extra,
     'note_score': [i[0] for i in y_pred_cnn.tolist()]})
  cnn_score_train=cnn_score_train.append(note_score)
  
  X_test_st_skewed_note_score = pd.concat([X_test_st_skewed, cnn_score_train],axis=1, join="inner")


  RF= RandomForestClassifier(n_estimators=100, random_state=96)

  RF_model_nt = RF.fit(X_st_train, y_train)
  y_pred = RF_model_nt.predict_proba(X_test_st_skewed)[::,1]



#==================================================================================================================
  for loop in range(len(threshoulds)):
    threshould=threshoulds[loop]
    print("------------------------------ threshould: ",threshould,"---------------------------------",fold,"-----------")
    y_pred_cnn_rounded = (y_pred_cnn > threshould)
    cm=confusion_matrix(y_test_skewed, y_pred_cnn_rounded)  
    metrics=show_metrics(y_test_skewed, y_pred_cnn_rounded, "CNN")
    sum_vector[loop]=sum_vector[loop]+np.array(metrics)
    print(cm)

    metrics_rounded = [round(num, 2) for num in metrics]
    print(metrics_rounded)

  fold+=1
#----------------------- calculate the average metrics for each threshould -----------
print("\n-------------------- average for each threshold -----------------")
print("\tPPV \t NPV \t SN \t SP \t ACC \t F1 \t AUC")
for loop in range(len(threshoulds)):
  v=sum_vector[loop]
  print(threshoulds[loop],end=':\t')
  for n in v:
    print(round(n/fold,2),end='\t')
  print()