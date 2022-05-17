"""
Correlation-based Feature Selection

Untuk mengoptimalkan kinerja dari metode klasifikasi. Tujuan utama dari seleksi
fitur adalah mengurangi kompleksitas, meningkatkan akurasi dan memilih fitur 
optimal dari suatu kumpulan fitur data.
"""

import pandas as pd
from scipy.stats import pointbiserialr
from math import sqrt
import numpy as np

#Load dataset
df = pd.read_csv('02-21-2018.csv')

#Menghapus kolom 'Timestamp'
df.drop('Timestamp',axis=1,inplace=True)

#Menentukan variable yang Dependen Y
dependen = 'Label'

#Merubah Format Nilai Y: Benign menjadi 0 dan DDOS attack-HOIC menjadi 1
df[dependen] = np.where( df[dependen] == 'Benign', 0, 1)

#Menentukan variable yang Independen X
#(Mengambil semua nama kolom dikurangi kolom Label)
features = df.columns.tolist()
features.remove(dependen)

#Fungsi Menghitung nilai Merit
def getMerit(subset, dependen):
    k = len(subset)

    # average feature-class correlation
    rcf_all = []
    for feature in subset:
        coeff = pointbiserialr( df[dependen], df[feature] )
        rcf_all.append( abs( coeff.correlation ) )
    rcf = np.mean( rcf_all )

    # average feature-feature correlation
    corr = df[subset].corr()
    corr.values[np.tril_indices_from(corr.values)] = np.nan
    corr = abs(corr)
    rff = corr.unstack().mean()

    return (k * rcf) / sqrt(k + k * (k-1) * rff)

#Menghitung Merit dan Menampilkannya
best_value = -1
for feature in features:
    coeff = pointbiserialr( df[dependen], df[feature] )
    abs_coeff = abs(coeff.correlation)
    print("Fitur %s nilai relasi %.4f"%(feature, abs_coeff))
    
    