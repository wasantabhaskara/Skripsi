'''
IDS 2018 Intrusion CSVs (CSE-CIC-IDS2018) 
Karakteristik Dataset '02-21-2018.csv' (Training dan Testing)

Dependen Y: Label.
Nilai Y:
   Benign (360.833 Row)
   DDOS attack-HOIC (686.012 Row)
   DDOS attack-LOIC-UDP (1.730 Row)

Sistem ini melakukan injeksi ke dalam file log sesuai dengan skema yang dipilih.
User dapat memilih Benign atau DDOS HOIC atau gabungan keduanya dengan jumlah
tertentu dari dataset CSE-CIC-IDS2018 '02-21-2018.csv'.
'''

import pandas as pd

print('Loading dataset CSE-CIC-IDS2018 02-21-2018.csv ....')
#Load dataset
df = pd.read_csv('02-21-2018.csv')

#Membuat file kosong dengan header atribut
dfFeature = df.sample(n=0)
dfFeature.to_csv('ids.log',index=False)
print("Membuat file simulasi ids.log selesai...")

#Menghapus Row Nilai Y: 'DDOS attack-LOIC-UDP'
df.drop(df.index[(df['Label'] == 'DDOS attack-LOIC-UDP')],axis=0,inplace=True)

#Membuat dataframe khusus untuk menampung hasil 'Benign'
dfBenign = df.drop(df.index[(df['Label'] != 'Benign')],axis=0)
#Membuat dataframe khusus untuk menampung hasil 'DDOS attack-HOIC'
dfHoic= df.drop(df.index[(df['Label'] != 'DDOS attack-HOIC')],axis=0)
while True:
    print(" ")
    qtyBenign = int(input('Masukan jumlah row data Normal: '))
    qtyHoic = int(input('Masukan jumlah row data DDOS HOIC: '))

    #Menyimpan dataframe sesuai jumlah yang di inginkan
    dfBenignQTY = dfBenign.sample(n=qtyBenign)
    dfHoicQTY = dfHoic.sample(n=qtyHoic)

    #Menggabungkan dua dataframe dan mengacaknya.
    dfCombine=pd.concat([dfBenignQTY,dfHoicQTY])
    dfCombine=dfCombine.sample(frac=1)

    #Menyisipkan data ke ids.log
    dfCombine.to_csv('ids.log', mode='a', index=False, header=False)
    print(" ")
    print(dfCombine)
    print("Penyisipan data selesai... tekan CTRL+C untuk mengakhiri...")
