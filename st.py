#!/usr/bin/env python
# coding: utf-8

# ## Importing the required libraries

# In[68]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
import re
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score


# ## Reading and Extracting data from .csv files

# In[69]:


train_instagrams = pd.read_csv('Data/train_gunung.csv', encoding='latin-1')
test_instagrams = pd.read_csv('Data/test_gunung.csv',encoding='latin-1')


# In[70]:


train_instagrams = train_instagrams[['label','instagram']]
test = test_instagrams['instagram']


# ## Exploratory Data Analysis

# In[91]:


train_instagrams['length'] = train_instagrams['instagram'].apply(len)
fig1 = sns.barplot('label','length',data = train_instagrams,palette='PRGn')
plt.title('Average Word Length vs label')
plot = fig1.get_figure()
plot.savefig('Barplot.png')


# In[92]:


fig2 = sns.countplot(x= 'label',data = train_instagrams)
plt.title('Label Counts')
plot = fig2.get_figure()
plot.savefig('Count Plot.png')


# ## Feature Engineering

# In[77]:


def text_processing(instagram):
    
    #Generating the list of words in the instagram (hastags and other punctuations removed)
    def form_sentence(instagram):
        instagram_blob = TextBlob(instagram)
        return ' '.join(instagram_blob.words)
    new_instagram = form_sentence(instagram)
    
    #Removing stopwords and words with unusual symbols
    def no_user_alpha(instagram):
        instagram_list = [ele for ele in instagram.split() if ele != 'user']
        clean_tokens = [t for t in instagram_list if re.match(r'[^\W\d]*$', t)]
        clean_s = ' '.join(clean_tokens)
        clean_mess = [word for word in clean_s.split() if word.lower() not in stopwords.words('english')]
        return clean_mess
    no_punc_instagram = no_user_alpha(new_instagram)
    
    #Normalizing the words in instagrams 
    def normalization(instagram_list):
        lem = WordNetLemmatizer()
        normalized_instagram = []
        for word in instagram_list:
            normalized_text = lem.lemmatize(word,'v')
            normalized_instagram.append(normalized_text)
        return normalized_instagram
    
    
    return normalization(no_punc_instagram)


# In[79]:


train_instagrams['instagram_list'] = train_instagrams['instagram'].apply(text_processing)
test_instagrams['instagram_list'] = test_instagrams['instagram'].apply(text_processing)


# In[81]:


train_instagrams[train_instagrams['label']==1].drop('instagram',axis=1).head()


# ## Model Selection and Machine Learning

# In[83]:


X = train_instagrams['instagram']
y = train_instagrams['label']
test = test_instagrams['instagram']


# In[84]:


from sklearn.model_selection import train_test_split
msg_train, msg_test, label_train, label_test = train_test_split(train_instagrams['instagram'], train_instagrams['label'], test_size=0.2)


# In[85]:


#Machine Learning Pipeline
pipeline = Pipeline([
    ('bow',CountVectorizer(analyzer=text_processing)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])
pipeline.fit(msg_train,label_train)


# In[86]:


predictions = pipeline.predict(msg_test)

print(classification_report(predictions,label_test))
print ('\n')
print(confusion_matrix(predictions,label_test))
print(accuracy_score(predictions,label_test))


# In[98]:


def form_sentence(instagram):
    instagram_blob = TextBlob(instagram)
    return ' '.join(instagram_blob.words)
print(form_sentence(train_instagrams['instagram'].iloc[10]))
print(train_instagrams['instagram'].iloc[10])


# In[99]:


def no_user_alpha(instagram):
    instagram_list = [ele for ele in instagram.split() if ele != 'user']
    clean_tokens = [t for t in instagram_list if re.match(r'[^\W\d]*$', t)]
    clean_s = ' '.join(clean_tokens)
    clean_mess = [word for word in clean_s.split() if word.lower() not in stopwords.words('english')]
    return clean_mess

print(no_user_alpha(form_sentence(train_instagrams['instagram'].iloc[10])))
print(train_instagrams['instagram'].iloc[10])


# In[101]:


def normalization(instagram_list):
        lem = WordNetLemmatizer()
        normalized_instagram = []
        for word in instagram_list:
            normalized_text = lem.lemmatize(word,'v')
            normalized_instagram.append(normalized_text)
        return normalized_instagram
    
instagram_list = 'I was playing with my friends with whom I used to play, when you called me yesterday'.split()
print(normalization(instagram_list))


# In[ ]:





