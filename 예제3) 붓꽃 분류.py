#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd

df = pd.read_csv('iris.csv')
df.head()


# In[5]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.pairplot(df, hue='Species')
plt.show()


# In[6]:


dataset = df.values
X = dataset[:, :4].astype(float)
Y_obj = dataset[:, 4]

from sklearn.preprocessing import LabelEncoder

e = LabelEncoder()
e.fit(Y_obj)
Y = e.transform(Y_obj)


# In[8]:


from tensorflow.keras import utils
import tensorflow as tf
Y_encoded = tf.keras.utils.to_categorical(Y)


# In[10]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(16, input_dim=4, activation='relu'))
model.add(Dense(3, activation='softmax'))


# In[13]:


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X,Y_encoded, epochs=50, batch_size=1)
print("\n Accuracy: %.4f" % (model.evaluate(X, Y_encoded)[1]))


# In[23]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import LabelEncoder


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


# seed 값 설정
np.random.seed(3)
tf.random.set_seed(3)


# 데이터 입력
df = pd.read_csv('iris.csv')


# 데이터 분류
dataset = df.values
X = dataset[:,:-1].astype(float)
Y_obj = dataset[:,-1]

print(X)
# 문자열을 숫자로 변환
e = LabelEncoder()
e.fit(Y_obj)
Y = e.transform(Y_obj)
Y_encoded = tf.keras.utils.to_categorical(Y)

print(Y_encoded)
# 모델의 설정
model = Sequential()
model.add(Dense(16, input_dim=5, activation='relu'))
model.add(Dense(3, activation='softmax'))


# 모델 컴파일
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# 모델 실행
model.fit(X, Y_encoded, epochs=50, batch_size=1)


# 결과 출력
print('\n Accuracy: %.4f' % (model.evaluate(X, Y_encoded)[1]))


# In[ ]:




