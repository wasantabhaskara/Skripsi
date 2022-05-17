"""
Random Forest ini akan membaca Trained Model rfModel.pkl dan melakukan prediksi
terhadap 10 file sample yang dibuat secara random dari dataset utama yang 
berjumlah 1.046.845 Row. 
"""

#import random
import pandas as pd
import numpy as np
import pickle

#Load Trained Model RF
rfModel = 'TM1K_n10.pkl'
print ("Trained Model :" + rfModel)

#Load  10 file testing uji akurasi
dfTest1 = pd.read_csv('02-21-2018_normal_test_100r.csv')
dfTest2 = pd.read_csv('02-21-2018_normal_test_200r.csv')
dfTest3 = pd.read_csv('02-21-2018_normal_test_300r.csv')
dfTest4 = pd.read_csv('02-21-2018_normal_test_400r.csv')
dfTest5 = pd.read_csv('02-21-2018_normal_test_500r.csv')
dfTest6 = pd.read_csv('02-21-2018_normal_test_600r.csv')
dfTest7 = pd.read_csv('02-21-2018_normal_test_700r.csv')
dfTest8 = pd.read_csv('02-21-2018_normal_test_800r.csv')
dfTest9 = pd.read_csv('02-21-2018_normal_test_900r.csv')
dfTest10 = pd.read_csv('02-21-2018_normal_test_1000r.csv')

features = ['Init Bwd Win Byts','Dst Port','Fwd Pkt Len Max','Fwd Pkt Len Std','Fwd Seg Size Avg','Fwd Pkt Len Mean',
'ACK Flag Cnt','Pkt Len Mean','Pkt Len Max','Pkt Size Avg','PSH Flag Cnt','Pkt Len Std','Pkt Len Var','RST Flag Cnt',
'ECE Flag Cnt','Init Fwd Win Byts','Flow Byts/s','Bwd Seg Size Avg','Bwd Pkt Len Mean','Bwd Pkts/s','Tot Bwd Pkts',
'Subflow Bwd Pkts','Down/Up Ratio','Flow Pkts/s','Bwd Pkt Len Std','Bwd Header Len','Fwd Pkts/s','Bwd IAT Min',
'TotLen Bwd Pkts','Subflow Bwd Byts','Bwd Pkt Len Max','Bwd IAT Mean','Protocol','Flow Duration','Fwd IAT Tot',
'Fwd Seg Size Min','Idle Max','Fwd IAT Max','Flow IAT Max','Flow IAT Std','Fwd IAT Std','Idle Mean','Active Mean',
'Idle Std','Active Max','Subflow Fwd Byts','TotLen Fwd Pkts','Active Min','Fwd Header Len','Tot Fwd Pkts',
'Subflow Fwd Pkts','Fwd Act Data Pkts','Fwd IAT Mean','Flow IAT Mean','Idle Min','Pkt Len Min','Fwd Pkt Len Min',
'Active Std','Fwd IAT Min','Flow IAT Min','Bwd IAT Max','Bwd IAT Tot','URG Flag Cnt','SYN Flag Cnt','Fwd PSH Flags',
'Bwd IAT Std','Bwd Pkt Len Min','FIN Flag Cnt','Fwd Blk Rate Avg','Bwd Blk Rate Avg','Bwd Pkts/b Avg','Bwd Byts/b Avg',
'CWE Flag Count','Fwd Byts/b Avg', 'Bwd PSH Flags','Fwd URG Flags','Bwd URG Flags','Fwd Pkts/b Avg']

#Merubah Format Nilai Y: Benign menjadi 0 dan DDOS attack-HOIC menjadi 1
dfTest1['Label'] = np.where( dfTest1['Label'] == 'Benign', 0, 1)
dfTest2['Label'] = np.where( dfTest2['Label'] == 'Benign', 0, 1)
dfTest3['Label'] = np.where( dfTest3['Label'] == 'Benign', 0, 1)
dfTest4['Label'] = np.where( dfTest4['Label'] == 'Benign', 0, 1)
dfTest5['Label'] = np.where( dfTest5['Label'] == 'Benign', 0, 1)
dfTest6['Label'] = np.where( dfTest6['Label'] == 'Benign', 0, 1)
dfTest7['Label'] = np.where( dfTest7['Label'] == 'Benign', 0, 1)
dfTest8['Label'] = np.where( dfTest8['Label'] == 'Benign', 0, 1)
dfTest9['Label'] = np.where( dfTest9['Label'] == 'Benign', 0, 1)
dfTest10['Label'] = np.where( dfTest10['Label'] == 'Benign', 0, 1)

#Memfilter dataframe berdasarkan fitur CFS untuk diprediksi dan memisahkan target Y
dfTest1_x = dfTest1[features]
dfTest1_y = dfTest1['Label'].values

dfTest2_x = dfTest2[features]
dfTest2_y = dfTest2['Label'].values

dfTest3_x = dfTest3[features]
dfTest3_y = dfTest3['Label'].values

dfTest4_x = dfTest4[features]
dfTest4_y = dfTest4['Label'].values

dfTest5_x = dfTest5[features]
dfTest5_y = dfTest5['Label'].values

dfTest6_x = dfTest6[features]
dfTest6_y = dfTest6['Label'].values

dfTest7_x = dfTest7[features]
dfTest7_y = dfTest7['Label'].values

dfTest8_x = dfTest8[features]
dfTest8_y = dfTest8['Label'].values

dfTest9_x = dfTest9[features]
dfTest9_y = dfTest9['Label'].values

dfTest10_x = dfTest10[features]
dfTest10_y = dfTest10['Label'].values

# Load Trained RF Model
with open(rfModel, 'rb') as file:
    model = pickle.load(file)
 
#Fungsi Prediksi dan Komponennya
def predict_tree(tree, X_test):
    feature_idx = tree['feature_idx']
    
    if X_test[feature_idx] <= tree['split_point']:
        if type(tree['left_split']) == dict:
            return predict_tree(tree['left_split'], X_test)
        else:
            value = tree['left_split']
            return value
    else:
        if type(tree['right_split']) == dict:
            return predict_tree(tree['right_split'], X_test)
        else:
            return tree['right_split']

def predict_rf(tree_ls, X_test):
    pred_ls = list()
    for i in range(len(X_test)):
        ensemble_preds = [predict_tree(tree, X_test.values[i]) for tree in tree_ls]
        final_pred = max(ensemble_preds, key = ensemble_preds.count)
        pred_ls.append(final_pred)
    return np.array(pred_ls)
   
    
#Memprediksi Hasil dan membandingkannya dengan Y untuk melihat akurasi
preds1 = predict_rf(model, dfTest1_x)
acc1 = sum(preds1 == dfTest1_y) / len(dfTest1_y)
print("Sample Uji Akurasi 1 (100)Row: {}%".format(100*np.round(acc1,3)))
print("Confusion Matrix Test 1")
confusion_matrix1 = pd.crosstab(dfTest1_y, preds1, rownames=['Actual'], colnames=['Predicted'])
print (confusion_matrix1)
print(" ")

preds2 = predict_rf(model, dfTest2_x)
acc2 = sum(preds2 == dfTest2_y) / len(dfTest2_y)
print("Sample Uji Akurasi 2 (200)Row: {}%".format(100*np.round(acc2,3)))
print("Confusion Matrix Test 2")
confusion_matrix2 = pd.crosstab(dfTest2_y, preds2, rownames=['Actual'], colnames=['Predicted'])
print (confusion_matrix2)
print(" ")

preds3 = predict_rf(model, dfTest3_x)
acc3 = sum(preds3 == dfTest3_y) / len(dfTest3_y)
print("Sample Uji Akurasi 3 (300)Row: {}%".format(100*np.round(acc3,3)))
print("Confusion Matrix Test 3")
confusion_matrix3 = pd.crosstab(dfTest3_y, preds3, rownames=['Actual'], colnames=['Predicted'])
print (confusion_matrix3)
print(" ")

preds4 = predict_rf(model, dfTest4_x)
acc4 = sum(preds4 == dfTest4_y) / len(dfTest4_y)
print("Sample Uji Akurasi 4 (400)Row: {}%".format(100*np.round(acc4,3)))
print("Confusion Matrix Test 4")
confusion_matrix4 = pd.crosstab(dfTest4_y, preds4, rownames=['Actual'], colnames=['Predicted'])
print (confusion_matrix4)
print(" ")

preds5 = predict_rf(model, dfTest5_x)
acc5 = sum(preds5 == dfTest5_y) / len(dfTest5_y)
print("Sample Uji Akurasi 5 (500)Row: {}%".format(100*np.round(acc5,3)))
print("Confusion Matrix Test 5")
confusion_matrix5 = pd.crosstab(dfTest5_y, preds5, rownames=['Actual'], colnames=['Predicted'])
print (confusion_matrix5)
print(" ")

preds6 = predict_rf(model, dfTest6_x)
acc6 = sum(preds6 == dfTest6_y) / len(dfTest6_y)
print("Sample Uji Akurasi 6 (600)Row: {}%".format(100*np.round(acc6,3)))
print("Confusion Matrix Test 6")
confusion_matrix6 = pd.crosstab(dfTest6_y, preds6, rownames=['Actual'], colnames=['Predicted'])
print (confusion_matrix6)
print(" ")

preds7 = predict_rf(model, dfTest7_x)
acc7 = sum(preds7 == dfTest7_y) / len(dfTest7_y)
print("Sample Uji Akurasi 7 (700)Row: {}%".format(100*np.round(acc7,3)))
print("Confusion Matrix Test 7")
confusion_matrix7 = pd.crosstab(dfTest7_y, preds7, rownames=['Actual'], colnames=['Predicted'])
print (confusion_matrix7)
print(" ")

preds8 = predict_rf(model, dfTest8_x)
acc8 = sum(preds8 == dfTest8_y) / len(dfTest8_y)
print("Sample Uji Akurasi 8 (800)Row: {}%".format(100*np.round(acc8,3)))
print("Confusion Matrix Test 8")
confusion_matrix8 = pd.crosstab(dfTest8_y, preds8, rownames=['Actual'], colnames=['Predicted'])
print (confusion_matrix8)
print(" ")

preds9 = predict_rf(model, dfTest9_x)
acc9 = sum(preds9 == dfTest9_y) / len(dfTest9_y)
print("Sample Uji Akurasi 9 (900)Row: {}%".format(100*np.round(acc9,3)))
print("Confusion Matrix Test 9")
confusion_matrix9 = pd.crosstab(dfTest9_y, preds9, rownames=['Actual'], colnames=['Predicted'])
print (confusion_matrix9)
print(" ")

preds10 = predict_rf(model, dfTest10_x)
acc10 = (sum(preds10 == dfTest10_y) / len(dfTest10_y))
print("Sample Uji Akurasi 10 (1000)Row: {}%".format(100*np.round(acc10,3)))
print("Confusion Matrix Test 10")
confusion_matrix10 = pd.crosstab(dfTest10_y, preds10, rownames=['Actual'], colnames=['Predicted'])
print (confusion_matrix10)
print(" ")

avg = (acc1+acc2+acc3+acc4+acc5+acc6+acc7+acc8+acc9+acc10)/10
print("Rata-rata Akurasi: {}%".format(np.round(100*avg,3)))