import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
# Dataset from - https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection
df = pd.read_table('smsspamcollection/SMSSpamCollection',
                   sep='\t', 
                   header=None, 
                   names=['label', 'sms_message'])

# Output printing out first 5 columns
df.head()


df['label'] = df.label.map({'ham':0, 'spam':1})
print(df.shape)
df.head(5) # returns (rows, columns)



documents = ['Hello, how are you!',
                'Win money, win from home.',
                'Call me now.',
                'Hello, Call hello you tomorrow?']


count_vector = CountVectorizer()

print(count_vector)

count_vector.fit(documents)
count_vector.get_feature_names()

