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
from tensorflow.keras.layers import LSTM, Dense, Dropout


# In[2]:


# Veri oku
df = pd.read_csv("C:\\Users\\Lenovo\\Desktop\\OneDrive_2025-05-07\\TON_IoT datasets\\Processed_datasets\\Processed_IoT_dataset\\IoT_Fridge.csv")
df.drop(['date', 'time', 'type'], axis=1, inplace=True)
df = pd.get_dummies(df, columns=['temp_condition'], drop_first=True)

# Giriş ve hedef
X = df.drop('label', axis=1)
y = df['label']

# Ölçekle
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# LSTM için input [örnek_sayısı, zaman_adımı, özellik_sayısı]
# Burada her veri noktasını 1 zaman adımı kabul ediyoruz
X_lstm = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

# Eğitim/test böl
X_train, X_test, y_train, y_test = train_test_split(X_lstm, y, test_size=0.2, random_state=42)


# In[3]:


model = Sequential()
model.add(LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))  # Binary output

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Eğit
history = model.fit(X_train, y_train, epochs=30, batch_size=64, validation_split=0.2,
                    class_weight={0: 1, 1: 4})  # Anomalilere ağırlık veriyoruz


# In[4]:


y_pred = model.predict(X_test)
y_pred = [1 if i > 0.5 else 0 for i in y_pred]

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# -Recall (Anomali) = 1.00
# Anomalilerin hepsi yakalanmış!
# Hiçbir saldırıyı kaçırmamış.
# -Precision (Anomali) = 0.43
# Tahmin ettiği “anomalilerin” sadece %43’ü gerçekten anomali.
# Yani biraz false alarm (yanlış pozitif) üretiyor.
# Ancak bu tür sistemlerde “alarm vereyim de yanlış çıksın, yeter ki gerçek saldırıyı kaçırmayayım” yaklaşımı geçerlidir
# -Accuracy = 0.80
# Verinin %80’i doğru tahmin edilmiş.
# -F1-Score (Anomali) = 0.60
# Precision ve Recall arasında bir denge.
# LSTM modeli anomaliyi yakalıyor ama biraz alarm üretiyor.

# In[5]:


from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# LSTM modeli .predict() sonucu: 0 ile 1 arasında olasılıklar verir
y_proba = model.predict(X_test)

# ROC eğrisi hesapla
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

# ROC eğrisi çiz
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Eğrisi (LSTM Modeli)')
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()


# ROC Eğrisi (Receiver Operating Characteristic)
# X Ekseni (False Positive Rate - Yanlış alarm oranı): Ne kadar "yanlış yere alarm verdi"
# Y Ekseni (True Positive Rate - Doğru tespit oranı): Ne kadar "doğru alarm verdi"
# Eğri, sol üst köşeye ne kadar yakınsa model o kadar iyi 

# AUC (Area Under Curve) = 0.88
# Bu ne demek?
# Modelin normal ile anomalileri ayırabilme yeteneği %88 oranında başarılı.
# AUC:
# 0.5: Rastgele seçim
# 0.7-0.8: İyi
# 0.8-0.9: Çok iyi (Benim Sonucum)
# 0.9+: Mükemmel (veya overfit riski olabilir)

# Grafik Gözlemleri:
# Eğri hızlıca yukarı fırlamış → model saldırıları çabuk fark ediyor.
# 0.2'nin altında FPR varken TPR 1'e ulaşmış → erken ve doğru tespit yapıyor.
# ROC eğrisindeki keskin dönüşler → net bir karar sınırı var.

# In[6]:


from sklearn.metrics import confusion_matrix
import seaborn as sns

# Confusion matrix hesapla
cm = confusion_matrix(y_test, y_pred)

# Görselleştir
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Normal', 'Anomali'], 
            yticklabels=['Normal', 'Anomali'])

plt.xlabel('Tahmin')
plt.ylabel('Gerçek')
plt.title('Confusion Matrix (LSTM Modeli)')
plt.tight_layout()
plt.show()


# 1. Sıfır Kaçırma (FN = 0)
# Gerçek anomalilerin hiçbiri kaçırılmamış.
# Bu, saldırı ya da anormal davranışları yakalama açısından güzel.

# 2. False Positive Var (23.408 adet)
# Normal bazı davranışlar “anomali” zannedilmiş.
# Bu yanlış alarm demek ama anomali tespit sistemlerinde bu genellikle kabul edilebilir (özellikle güvenlikte: yanlış alarm >> gözden kaçan saldırı).

# 3. Dengeli Dağılım ve Sağlam Öğrenme
# True Positive ve True Negative değerlerin yüksekliği LSTM’in hem öğrendiğini hem de genelleme yapabildiğini gösteriyor.

# Sonuç Olarak:
# Anomali kaçırmıyor.
# Alarm veriyor ama "önlem almak için" yeterince güvenli.
# AUC = 0.88 ve accuracy = 0.80 ile güzel bir sıralı veri modeli.

# In[ ]:




