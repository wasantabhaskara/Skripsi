"""
Random Forest adalah Metode pembelajaran ensemble yang digunakan untuk 
klasifikasi dan regresi, tetapi dalam penelitian ini saya akan berfokus 
pada klasifikasi. Fitur dari dataset dipilih dari CFS dengan jumlah 14 fitur.

Saat selesai melakukan training dan mendapatkan nilai akurasi, otomatis akan 
dibuat juga TrainedModel untuk digunakan pada LAB nantinya dengan nama file:
rfModel.pkl
"""

import random
import pandas as pd
import numpy as np
import pickle

#Load dataset
df = pd.read_csv('02-21-2018_normal_1k.csv')

#Menggunakan fitur X yang terpilih dari CFS
features = ['Init Bwd Win Byts','Dst Port','Fwd Pkt Len Max','Fwd Pkt Len Std','Fwd Seg Size Avg','Fwd Pkt Len Mean',
'ACK Flag Cnt','Pkt Len Mean','Pkt Len Max','Pkt Size Avg','PSH Flag Cnt','Pkt Len Std','Pkt Len Var','RST Flag Cnt',
'ECE Flag Cnt','Init Fwd Win Byts','Flow Byts/s','Bwd Seg Size Avg','Bwd Pkt Len Mean','Bwd Pkts/s','Tot Bwd Pkts',
'Subflow Bwd Pkts','Down/Up Ratio','Flow Pkts/s','Bwd Pkt Len Std','Bwd Header Len','Fwd Pkts/s','Bwd IAT Min',
'TotLen Bwd Pkts','Subflow Bwd Byts','Bwd Pkt Len Max','Bwd IAT Mean','Protocol','Flow Duration','Fwd IAT Tot',
'Fwd Seg Size Min','Idle Max','Fwd IAT Max','Flow IAT Max','Flow IAT Std','Fwd IAT Std','Idle Mean','Active Mean',
'Idle Std','Active Max','Subflow Fwd Byts','TotLen Fwd Pkts','Active Min','Fwd Header Len','Tot Fwd Pkts',
'Subflow Fwd Pkts','Fwd Act Data Pkts','Fwd IAT Mean','Flow IAT Mean','Idle Min','Pkt Len Min','Fwd Pkt Len Min',
'Active Std','Fwd IAT Min','Flow IAT Min','Bwd IAT Max','Bwd IAT Tot','URG Flag Cnt','SYN Flag Cnt','Fwd PSH Flags',
'Bwd IAT Std','Bwd Pkt Len Min','FIN Flag Cnt']

#Menentukan variable yang Dependen Y
dependen = 'Label'

#Merubah Format Nilai Y: Benign menjadi 0 dan DDOS attack-HOIC menjadi 1
df[dependen] = np.where( df[dependen] == 'Benign', 0, 1)

#Memasukkan nilai ke variable dependen Y dan independen X
X_train = df[features]
y_train = df[dependen].values


#Fungsi Random Forest dan Komponennya
def entropy(p):
    if p == 0:
        return 0
    elif p == 1:
        return 0
    else:
        return - (p * np.log2(p) + (1 - p) * np.log2(1-p))

def information_gain(left_child, right_child):
    parent = left_child + right_child
    p_parent = parent.count(1) / len(parent) if len(parent) > 0 else 0
    p_left = left_child.count(1) / len(left_child) if len(left_child) > 0 else 0
    p_right = right_child.count(1) / len(right_child) if len(right_child) > 0 else 0
    IG_p = entropy(p_parent)
    IG_l = entropy(p_left)
    IG_r = entropy(p_right)
    return IG_p - len(left_child) / len(parent) * IG_l - len(right_child) / len(parent) * IG_r

def draw_bootstrap(X_train, y_train):
    bootstrap_indices = list(np.random.choice(range(len(X_train)), len(X_train), replace = True))
    oob_indices = [i for i in range(len(X_train)) if i not in bootstrap_indices]
    X_bootstrap = X_train.iloc[bootstrap_indices].values
    y_bootstrap = y_train[bootstrap_indices]
    X_oob = X_train.iloc[oob_indices].values
    y_oob = y_train[oob_indices]

    return X_bootstrap, y_bootstrap, X_oob, y_oob

def oob_score(tree, X_test, y_test):
    mis_label = 0
    for i in range(len(X_test)):
        pred = predict_tree(tree, X_test[i])
        if pred != y_test[i]:
            mis_label += 1
    return mis_label / len(X_test)

def find_split_point(X_bootstrap, y_bootstrap, max_features):
    feature_ls = list()
    num_features = len(X_bootstrap[0])
    
    while len(feature_ls) <= max_features:
        feature_idx = random.sample(range(num_features), 1)
        if feature_idx not in feature_ls:
            feature_ls.extend(feature_idx)
    
    best_info_gain = -999
    node = None
    for feature_idx in feature_ls:
        for split_point in X_bootstrap[:,feature_idx]:
            left_child = {'X_bootstrap': [], 'y_bootstrap': []}
            right_child = {'X_bootstrap': [], 'y_bootstrap': []}
            
            # split children for continuous variables
            if type(split_point) in [int, float]:
                for i, value in enumerate(X_bootstrap[:,feature_idx]):
                    if value <= split_point:
                        left_child['X_bootstrap'].append(X_bootstrap[i])
                        left_child['y_bootstrap'].append(y_bootstrap[i])
                    else:
                        right_child['X_bootstrap'].append(X_bootstrap[i])
                        right_child['y_bootstrap'].append(y_bootstrap[i])
            # split children for categoric variables
            else:
                for i, value in enumerate(X_bootstrap[:,feature_idx]):
                    if value == split_point:
                        left_child['X_bootstrap'].append(X_bootstrap[i])
                        left_child['y_bootstrap'].append(y_bootstrap[i])
                    else:
                        right_child['X_bootstrap'].append(X_bootstrap[i])
                        right_child['y_bootstrap'].append(y_bootstrap[i])
            
            split_info_gain = information_gain(left_child['y_bootstrap'], right_child['y_bootstrap'])
            if split_info_gain > best_info_gain:
                best_info_gain = split_info_gain
                left_child['X_bootstrap'] = np.array(left_child['X_bootstrap'])
                right_child['X_bootstrap'] = np.array(right_child['X_bootstrap'])
                node = {'information_gain': split_info_gain, 
                        'left_child': left_child, 
                        'right_child': right_child, 
                        'split_point': split_point,
                        'feature_idx': feature_idx}
    return node

def terminal_node(node):
    y_bootstrap = node['y_bootstrap']
    pred = max(y_bootstrap, key = y_bootstrap.count)
    return pred

def split_node(node, max_features, min_samples_split, max_depth, depth):
    left_child = node['left_child']
    right_child = node['right_child']    

    del(node['left_child'])
    del(node['right_child'])
    
    if len(left_child['y_bootstrap']) == 0 or len(right_child['y_bootstrap']) == 0:
        empty_child = {'y_bootstrap': left_child['y_bootstrap'] + right_child['y_bootstrap']}
        node['left_split'] = terminal_node(empty_child)
        node['right_split'] = terminal_node(empty_child)
        return
    
    if depth >= max_depth:
        node['left_split'] = terminal_node(left_child)
        node['right_split'] = terminal_node(right_child)
        return node
    
    if len(left_child['X_bootstrap']) <= min_samples_split:
        node['left_split'] = node['right_split'] = terminal_node(left_child)
    else:
        node['left_split'] = find_split_point(left_child['X_bootstrap'], left_child['y_bootstrap'], max_features)
        split_node(node['left_split'], max_depth, min_samples_split, max_depth, depth + 1)
    if len(right_child['X_bootstrap']) <= min_samples_split:
        node['right_split'] = node['left_split'] = terminal_node(right_child)
    else:
        node['right_split'] = find_split_point(right_child['X_bootstrap'], right_child['y_bootstrap'], max_features)
        split_node(node['right_split'], max_features, min_samples_split, max_depth, depth + 1)

def build_tree(X_bootstrap, y_bootstrap, max_depth, min_samples_split, max_features):
    root_node = find_split_point(X_bootstrap, y_bootstrap, max_features)
    split_node(root_node, max_features, min_samples_split, max_depth, 1)
    return root_node

def random_forest(X_train, y_train, n_estimators, max_features, max_depth, min_samples_split):
    tree_ls = list()
    oob_ls = list()
    for i in range(n_estimators):
        X_bootstrap, y_bootstrap, X_oob, y_oob = draw_bootstrap(X_train, y_train)
        tree = build_tree(X_bootstrap, y_bootstrap, max_features, max_depth, min_samples_split)
        tree_ls.append(tree)
        oob_error = oob_score(tree, X_oob, y_oob)
        oob_ls.append(oob_error)
    print("Estimasi OOB: {:.2f}".format(np.mean(oob_ls)))
    return tree_ls

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

#Parameter MODEL
n_estimators = 100    #Jumlah Tree
max_features = 8     #Jumlah fitur yang dipertimbangkan untuk melakukan split rumus akar p, p adalah jumlah fitur yang digunakan yaitu 68 jadinya mendekati 7,8
max_depth = 10        #Kedalaman maksimum pohon
min_samples_split = 2 #Jumlah minimum sampel row yang diperlukan untuk split

#Mentraining MODEL
model = random_forest(X_train, y_train, n_estimators, max_features, max_depth, min_samples_split)

#Menyimpan Model
with open('TM1K.pkl', 'wb') as file:
    pickle.dump(model, file)
