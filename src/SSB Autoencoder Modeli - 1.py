#!/usr/bin/env python
# coding: utf-8

# Autoencoder Nedir?
# - Bir veriyi sıkıştırarak yeniden oluşturmaya çalışan sinir ağıdır.
# - Anomaliler, bu yeniden oluşturma sırasında yüksek hata üretir biz de bu kısımdan yakalayacağız.

# In[4]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import regularizers


# In[5]:


# Veri oku
df = pd.read_csv("C:\\Users\\Lenovo\\Desktop\\OneDrive_2025-05-07\\TON_IoT datasets\\Processed_datasets\\Processed_IoT_dataset\\IoT_Fridge.csv")
df.drop(['date', 'time', 'type'], axis=1, inplace=True)
df = pd.get_dummies(df, columns=['temp_condition'], drop_first=True)

# Giriş ve hedef
X = df.drop('label', axis=1)
y = df['label']

# Sadece "normal" veriyi eğitim için kullan (denetimsiz yaklaşım!)
X_normal = X[y == 0]

# Ölçekle
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_normal_scaled = scaler.transform(X_normal)

# Train/test ayır
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# In[6]:


input_dim = X_train.shape[1]

input_layer = Input(shape=(input_dim,))
encoded = Dense(32, activation="relu")(input_layer)
encoded = Dense(16, activation="relu")(encoded)
decoded = Dense(32, activation="relu")(encoded)
output_layer = Dense(input_dim, activation="linear")(decoded)

autoencoder = Model(inputs=input_layer, outputs=output_layer)
autoencoder.compile(optimizer='adam', loss='mse')


# In[7]:


autoencoder.fit(X_normal_scaled, X_normal_scaled,
                epochs=30,
                batch_size=64,
                shuffle=True,
                validation_split=0.2)


# In[8]:


# Yeniden oluştur
X_test_pred = autoencoder.predict(X_test)

# Yeniden yapılandırma hatasını hesapla (Mean Squared Error)
reconstruction_error = np.mean(np.square(X_test - X_test_pred), axis=1)

# Eşik belirle (manuel veya istatistiksel olarak)
threshold = np.percentile(reconstruction_error, 95)  # üst %5 anomali

# Tahmin yap
y_pred = [1 if err > threshold else 0 for err in reconstruction_error]

# Sonuçları yazdır
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# Accuracy ≈ %83
# - Genel doğruluk yüksek ama bu veri dengesiz olduğu için yanıltıcı olabilir!
# - Zaten normal veri çok fazla, sadece "normal" deseydi bile accuracy yüksek çıkardı.

# Ne Anladık?
# - Autoencoder modeli, normal veriyi gayet iyi öğrenmiş.
# - Ama anomalileri ayırt etme konusunda zayıf kaldı (low recall).
# - Bu, genelde threshold çok düşük seçilirse olur ya da model daha derin/kompleks olmalıydı.

# Bir Sonraki Denemeler için Öneriler:
# - Threshold’u düşürüp yeniden deneyebilirsin (mesela %90 değil, %85 percentile gibi).
# - Daha derin bir autoencoder kurabilirsin (daha fazla layer).

# In[13]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix

# Autoencoder çıktısını al
X_test_pred = autoencoder.predict(X_test)

# Rekonstrüksiyon hatasını (anomalilik skoru) hesapla
reconstruction_error = np.mean(np.square(X_test - X_test_pred), axis=1)

# ROC eğrisi için
fpr, tpr, thresholds = roc_curve(y_test, reconstruction_error)
roc_auc = auc(fpr, tpr)

# ROC eğrisi çizimi
plt.figure(figsize=(8, 5))
plt.plot(fpr, tpr, color='orange', label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Eğrisi (Autoencoder)')
plt.legend(loc='lower right')
plt.grid()
plt.show()

# Confusion matrix için threshold belirle (örnek olarak %95'lik bir eşik)
threshold = np.percentile(reconstruction_error, 95)  # ya da ROC'dan çıkan en uygun threshold'u seçebilirsin

# Tahminleri binary hale getir
y_pred_class = [1 if err > threshold else 0 for err in reconstruction_error]

# Confusion matrix çizimi
cm = confusion_matrix(y_test, y_pred_class)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix (Autoencoder)')
plt.xlabel('Tahmin')
plt.ylabel('Gerçek')
plt.xticks(ticks=[0.5, 1.5], labels=["Normal", "Anomali"])
plt.yticks(ticks=[0.5, 1.5], labels=["Normal", "Anomali"])
plt.show()


# - AUC = 0.72, modelin "anomali" ile "normal" verileri birbirinden ayırt etmede orta seviyede başarılı olduğunu gösteriyor.
# - Değer 0.5'ten yüksek olduğu için model rastgele sınıflandırmadan daha iyidir.

# - True Positive (TP): 1.518 — Anomalileri doğru tahmin ettiği örnekler.
# - False Negative (FN): 15.871 — Anomalileri normal zannetti (tehlikeli!).
# - False Positive (FP): 4.338 — Normal verileri yanlışlıkla anomali zannetti.
# - True Negative (TN): 95.689 — Normal verileri doğru bildi.
# 
# Önemli Nokta:
# - Recall (1. sınıf için) çok düşük: Anomalileri yakalama oranı %9.
# - Precision (1. sınıf için) da düşük: Yakalanan anomalilerin doğruluk oranı %26.
# - Autoencoder modeli bu veri setinde "anomali" sınıfında yüksek hata yapma eğiliminde.

# Autoencoder, verideki normal davranışları öğrenmeye çalışır. Bu nedenle anomali sınıfında yeterince temsil olmayan örneklerde düşük başarı gösterebilir. Model daha çok rekonstrüksiyon hatasına dayandığı için doğrudan sınıflandırma problemlerinde (özellikle bu kadar dengesiz sınıflarda) sınırlı performans gösterebilir.

# In[ ]:




