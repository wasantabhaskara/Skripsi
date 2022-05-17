'''
IDS 2018 Intrusion CSVs (CSE-CIC-IDS2018) 
Karakteristik Dataset '02-21-2018.csv' (Training dan Testing)

Dependen Y: Label.
Nilai Y:
   Benign (360.833 Row)
   DDOS attack-HOIC (686.012 Row)
   DDOS attack-LOIC-UDP (1.730 Row)

Normalisasi akan menghapus fitur, mengatur format nilai dan data row 
yang tidak diperlukan. Yaitu:
   Fitur: Timestamp (dalam kasus ini, tidak mencari pola waktu)
   Row Nilai Y: DDOS attack-LOIC-UDP (Diluar batasan penelitian)
'''

import pandas as pd

#Load dataset
df = pd.read_csv('02-21-2018.csv')

#Menghapus kolom 'Timestamp'
df.drop('Timestamp',axis=1,inplace=True)

#Menghapus Row Nilai Y: 'DDOS attack-LOIC-UDP'
df.drop(df.index[(df['Label'] == 'DDOS attack-LOIC-UDP')],axis=0,inplace=True)

#Menentukan jumlah acak sample untuk Training (1.000)Row
dfTrain = df.sample(n=1000)

#Menentukan jumlah acak sample untuk Uji Akurasi 1 
dfTest1 = df.sample(n=100)
#Menentukan jumlah acak sample untuk Uji Akurasi 2 
dfTest2 = df.sample(n=200)
#Menentukan jumlah acak sample untuk Uji Akurasi 3 
dfTest3 = df.sample(n=300)
#Menentukan jumlah acak sample untuk Uji Akurasi 4 
dfTest4 = df.sample(n=400)
#Menentukan jumlah acak sample untuk Uji Akurasi 5 
dfTest5 = df.sample(n=500)
#Menentukan jumlah acak sample untuk Uji Akurasi 6 
dfTest6 = df.sample(n=600)
#Menentukan jumlah acak sample untuk Uji Akurasi 7 
dfTest7 = df.sample(n=700)
#Menentukan jumlah acak sample untuk Uji Akurasi 8 
dfTest8 = df.sample(n=800)
#Menentukan jumlah acak sample untuk Uji Akurasi 9 
dfTest9 = df.sample(n=900)
#Menentukan jumlah acak sample untuk Uji Akurasi 10 
dfTest10 = df.sample(n=1000)

#Menyimpan kedalam file baru
dfTrain.to_csv('02-21-2018_normal_1k.csv',index=False)

dfTest1.to_csv('02-21-2018_normal_test_100r.csv',index=False)
dfTest2.to_csv('02-21-2018_normal_test_200r.csv',index=False)
dfTest3.to_csv('02-21-2018_normal_test_300r.csv',index=False)
dfTest4.to_csv('02-21-2018_normal_test_400r.csv',index=False)
dfTest5.to_csv('02-21-2018_normal_test_500r.csv',index=False)
dfTest6.to_csv('02-21-2018_normal_test_600r.csv',index=False)
dfTest7.to_csv('02-21-2018_normal_test_700r.csv',index=False)
dfTest8.to_csv('02-21-2018_normal_test_800r.csv',index=False)
dfTest9.to_csv('02-21-2018_normal_test_900r.csv',index=False)
dfTest10.to_csv('02-21-2018_normal_test_1000r.csv',index=False)
