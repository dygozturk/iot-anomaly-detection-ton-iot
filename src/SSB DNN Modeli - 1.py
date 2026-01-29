#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout


# In[2]:


# Veri setini yükle
df = pd.read_csv("C:\\Users\\Lenovo\\Desktop\\OneDrive_2025-05-07\\TON_IoT datasets\\Processed_datasets\\Processed_IoT_dataset\\IoT_Fridge.csv")
df.drop(['date', 'time', 'type'], axis=1, inplace=True)
df = pd.get_dummies(df, columns=['temp_condition'], drop_first=True)

# X ve y ayır
X = df.drop('label', axis=1)
y = df['label']

# Ölçekleme (Normalizasyon)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Eğitim / test böl
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# In[3]:


model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))  # Binary classification

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# In[4]:


history = model.fit(X_train, y_train, epochs=30, batch_size=64, validation_split=0.2,
                    class_weight={0: 1, 1: 4})  # Anomalilere ağırlık veriyoruz


# In[5]:


y_pred = model.predict(X_test)
y_pred = [1 if i > 0.5 else 0 for i in y_pred]

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# Classification Report – Kritik Göstergeler
# - Accuracy:         %80	    Tüm verinin %80’ini doğru tahmin etmiş.
# - Recall (1):	    1.00	Anomalileri kaçırmıyor, bu en kritik şey.
# - Precision (1):	0.43	Tespit ettiklerinin %43'ü gerçekten anomali (diğerleri false positive).
# - F1-Score (1):	    0.60	Denge metriği olarak gayet iyi.

# Not: F1-score’un %60 olması, hem doğru bulma hem de yanlış alarmların dengeli olduğunu gösterir.

# In[6]:


from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Modelin olasılıksal tahminlerini al
y_proba = model.predict(X_test)

# ROC eğrisi için değerleri hesapla
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

# ROC eğrisini çiz
plt.figure(figsize=(8,5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC Curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.title('ROC Eğrisi (DNN Modeli)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()


# - Eğrinin şekli hızla yukarı çıkıyor ve sonra üstte yatay ilerliyor.
# - Bu, modelin anomalileri çok iyi yakalayabildiğini gösteriyor.
# - Turuncu eğri, rastgele tahmin (mavi kesikli çizgi) olan y = x'ten ne kadar uzaksa, modelin başarımı o kadar yüksek.

# AUC = 0.88 Bu da demek oluyor ki: modelin %88 oranında pozitif ve negatif örnekleri doğru ayırt edebiliyor.

# - True Positive Rate yüksek (Recall yüksek). Anomalileri kaçırmıyor.
# - False Positive Rate bir miktar var. Bazı normal verileri "anomali" sanabiliyor. Bu da çok doğal ve kabul edilebilir bir davranış, özellikle güvenlik sistemlerinde!

# - Eğri Eğimi ...	  Hızlı yukarı, sonra yatay (ideal yapı)
# - Model Başarısı ...  Güçlü, dengeli, saldırı kaçırmıyor
# - Uygun Senaryo  ...  Gerçek zamanlı IoT güvenliği, DDoS tespiti, sızma algılama sistemleri

# In[7]:


import seaborn as sns
from sklearn.metrics import confusion_matrix

# 0.5 eşik değerine göre sınıflandır
y_pred = [1 if i > 0.5 else 0 for i in y_proba]

# Confusion matrix hesapla
cm = confusion_matrix(y_test, y_pred)

# Confusion Matrix ısı haritası çiz
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix (DNN Modeli)')
plt.xlabel('Tahmin')
plt.ylabel('Gerçek')
plt.xticks([0.5, 1.5], ['Normal', 'Anomali'])
plt.yticks([0.5, 1.5], ['Normal', 'Anomali'], rotation=0)
plt.show()


# Genel Değerlendirme:
# - Anomali kaçırma (FN)...	 Yok 
# - Yanlış alarm (FP)...	     Biraz yüksek, ama kabul edilebilir (özellikle güvenlikte)
# - Genel doğruluk...     	%80 – oldukça başarılı
# - Uygunluk...           	Gerçek zamanlı saldırı tespiti, IoT güvenliği, edge AI sistemleri

# In[ ]:




