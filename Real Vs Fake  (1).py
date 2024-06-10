#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import nltk
nltk.download('punkt')
nltk.download('punkt')
nltk.download('stopwords')


# In[2]:


Fake_news = pd.read_csv('Fake.csv')
Real_news = pd.read_csv('True.csv') 


# In[3]:


Fake_news.tail(2064)


# In[4]:


# Removing rows from the Fake_news to balance the dataset
Fake_news_manual_testing = Fake_news.tail(10)
for i in range(23480, 21416, -1):
   Fake_news.drop([i], axis = 0, inplace = True)
    
    


# ### Inserting Target column

# In[5]:


Fake_news['class'] = 0
Real_news['class'] = 1


# In[6]:


Fake_news.tail(2)


# In[7]:


Real_news.tail(2)


# In[8]:


Fake_news.info()


# In[9]:


Real_news.info()


# In[10]:


Fake_news.shape, Real_news.shape


# ## Joinning the two files 

# In[11]:


Fake_Vs_Real = pd.concat([Fake_news, Real_news], axis = 0)
Fake_Vs_Real.head(5)


# In[12]:


Fake_Vs_Real.tail(5)


# In[13]:


Fake_Vs_Real.shape


# In[14]:


Fake_Vs_Real.columns


# In[15]:


# Drop irrelevant columns
F_Vs_R = Fake_Vs_Real.drop(['title', 'subject', 'date'], axis = 1)


# In[16]:


#Random shuffling of the data
F_Vs_R = F_Vs_R.sample(frac = 1)


# In[17]:


F_Vs_R.head(4)


# In[18]:


F_Vs_R.reset_index(inplace = True)
F_Vs_R.drop(['index'], axis = 1, inplace = True)


# In[19]:


F_Vs_R.head()


# In[20]:


F_Vs_R.columns


# In[21]:


class_value_counts = F_Vs_R['class'].value_counts()

# Plotting
plt.figure(figsize=(8, 6))
class_value_counts.plot(kind='bar',color=['green','red'])
plt.title('Value Counts of Class Column')
plt.xlabel('Class Values')
plt.ylabel('Counts')
plt.show()


# # Tokenization
# # Stopword removal
# # Stemming

# In[22]:


# Preprocessing using NLTK
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Tokenization, stopword removal, and stemming
def preprocess_text(text):
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
    preprocessed_text = ' '.join(stemmed_tokens)
    return preprocessed_text



# In[23]:


F_Vs_R['processed_text'] = F_Vs_R['text'].apply(preprocess_text)


# In[24]:


F_Vs_R


# In[25]:


X_train, X_test, y_train, y_test = train_test_split(F_Vs_R['processed_text'], F_Vs_R['class'], test_size = 0.25)


# In[26]:


Vectorization = TfidfVectorizer()
Xv_train = Vectorization.fit_transform(X_train)
Xv_test = Vectorization.transform(X_test)


# In[27]:


LR = LogisticRegression()
LR.fit(Xv_train, y_train)


# In[28]:


Pred_LR = LR.predict(Xv_test)


# In[29]:


print(accuracy_score(y_test, Pred_LR))


# In[30]:


from sklearn.tree import DecisionTreeClassifier


# In[31]:


DT= DecisionTreeClassifier()
DT.fit(Xv_train, y_train)


# In[32]:


Pred_DT= DT.predict(Xv_test)


# In[33]:


print('The accuracy score of DT : ', accuracy_score(y_test, Pred_DT))


# In[34]:


from sklearn.ensemble import RandomForestClassifier


# In[35]:


RF = RandomForestClassifier()
RF.fit(Xv_train, y_train)


# In[36]:


Pred_RF = RF.predict(Xv_test)


# In[37]:


print('The accuracy score of RF :', accuracy_score(y_test, Pred_RF))


# In[38]:


from sklearn.svm import SVC


# In[39]:


svm = SVC(kernel = 'linear', C = 1)


# In[40]:


svm.fit(Xv_train, y_train)


# In[41]:


Pred_SVM = svm.predict(Xv_test)


# In[42]:


print('The accuracy of SVM : ', accuracy_score(y_test, Pred_SVM))


# In[43]:


print(classification_report(y_test, Pred_SVM))


# In[44]:


from sklearn.ensemble import GradientBoostingClassifier


# In[45]:


GB = GradientBoostingClassifier(random_state = 0)

# Fit the model to your training data
GB.fit(Xv_train, y_train)


# In[46]:


Pred_GB = GB.predict(Xv_test)


# In[47]:


print('The accuracy of GB : ', accuracy_score(y_test, Pred_GB))


# In[48]:


print(classification_report(y_test, Pred_GB))


# In[ ]:


def output_label(n):
    if n == 0:
        return 'The news is fake'
    elif n == 1:
        return 'This news is true'
    
    
def fake_news_detector(news):
    texting_news = {'text':[news]}
    new_def_test = pd.DataFrame(texting_news)
    new_def_test['text'] = new_def_test['text'].apply( preprocess_text)
    new_X_test = new_def_test['text']
    new_Xv_test = Vectorization.transform(X_test)
    Pred_LR = LR.predict(new_Xv_test)
    Pred_DT = DT.predict(new_Xv_test)
    Pred_RF = RF.predict(new_Xv_test)
    Pred_GB = GB.predict(new_Xv_test)
    return print('\n\nLR prediction: {} \nDT prediction: {} \nRF prediction: {} \nGB prediction: {}'.format(output_label(Pred_LR[0]),
                                                                                                            output_label(Pred_DT[0]),
                                                                                                            output_label(Pred_RF[0]),
                                                                                                            output_label(Pred_GB[0])))


# In[ ]:


news = "Monkeypox cases confirmed in Europe and North America"
fake_news_detector(news)

