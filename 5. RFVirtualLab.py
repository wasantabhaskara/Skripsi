"""
Random Forest ini akan membaca Trained Model TM50K.pkl dan melakukan prediksi
terhadap output log dari Aplikasi  yang telah di tentukan.  
"""

import pandas as pd
import numpy as np
import pickle
import os
import timeit

rfModel = 'TM1K_n10.pkl'
cekRowBaru = 0

filecsv = input('Masukkan lokasi dan nama file log (full path): ')

features = ['Init Bwd Win Byts','Dst Port','Fwd Pkt Len Max','Fwd Pkt Len Std','Fwd Seg Size Avg',
'Fwd Pkt Len Mean','ACK Flag Cnt','Pkt Len Mean','Pkt Len Max','Pkt Size Avg','PSH Flag Cnt',
'Pkt Len Std','Pkt Len Var','RST Flag Cnt','ECE Flag Cnt','Init Fwd Win Byts','Flow Byts/s',
'Bwd Seg Size Avg','Bwd Pkt Len Mean','Bwd Pkts/s','Tot Bwd Pkts','Subflow Bwd Pkts','Down/Up Ratio',
'Flow Pkts/s','Bwd Pkt Len Std','Bwd Header Len','Fwd Pkts/s','Bwd IAT Min','TotLen Bwd Pkts',
'Subflow Bwd Byts','Bwd Pkt Len Max','Bwd IAT Mean','Protocol','Flow Duration','Fwd IAT Tot',
'Fwd Seg Size Min','Idle Max','Fwd IAT Max','Flow IAT Max','Flow IAT Std','Fwd IAT Std','Idle Mean',
'Active Mean','Idle Std','Active Max','Subflow Fwd Byts','TotLen Fwd Pkts','Active Min','Fwd Header Len',
'Tot Fwd Pkts','Subflow Fwd Pkts','Fwd Act Data Pkts','Fwd IAT Mean','Flow IAT Mean','Idle Min',
'Pkt Len Min','Fwd Pkt Len Min','Active Std','Fwd IAT Min','Flow IAT Min','Bwd IAT Max','Bwd IAT Tot',
'URG Flag Cnt','SYN Flag Cnt','Fwd PSH Flags','Bwd IAT Std','Bwd Pkt Len Min','FIN Flag Cnt']

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

print("Menunggu data pada ",filecsv,"....")

while True:
    start = timeit.default_timer() # catat waktu mulai
    #Load data log
    df = pd.read_csv(filecsv)
    
    if df.shape[0] > cekRowBaru:
        #update cekRowbaru dan Bersihkan Layar
        cekRowBaru = df.shape[0]
        os.system('cls' if os.name == 'nt' else 'clear')
        
        #Memfilter dataframe berdasarkan fitur CFS untuk diprediksi
        dfTest = df[features]
        dfAsli = df['Label'].values
        print("----------------------------------------------------------------")
        print("| No\t| Packet\t| Label Asli\t| Hasil Prediksi \t|")
        print("----------------------------------------------------------------")

        #Memprediksi Hasil
        preds = predict_rf(model, dfTest)
        for x in range(len(preds)):
            prediksi = "DDOS HOIC" if preds[x] == 1 else "Benign"
            asli = "DDOS HOIC" if dfAsli[x] == "DDOS attack-HOIC" else "Benign"
            print("| ",x+1,"\t| ","{0:03}".format(x),"\t\t| ",asli,"\t|",prediksi,"\t\t|")     
        print("----------------------------------------------------------------")
        stop = timeit.default_timer() # catat waktu selesai
        lama_eksekusi = stop - start # lama eksekusi dalam satuan detik
        print("Lama eksekusi: ",lama_eksekusi,"detik")