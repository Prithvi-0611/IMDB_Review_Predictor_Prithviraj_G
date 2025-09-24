import re
import pandas as pd
import numpy as np

d=pd.read_csv('IMDB Dataset.csv')
d=d.dropna()
d['sentiment']=d['sentiment'].map({'positive': 1, 'negative': 0})

def cleaning(text):
    text=text.lower()
    text=re.sub(r"[^a-zA-Z0-9]", " ", text)
    words=text.split()
    processed=[]
    skip=False
    for i,w in enumerate(words):
        if skip:
            skip=False
            continue
        if w=="not" and i+1<len(words):
            processed.append("not_" + words[i + 1])
            skip=True
        else:
            processed.append(w)
    return processed

d['review']=d['review'].apply(cleaning)

positive=["good", "great", "excellent", "amazing", "fantastic", "wonderful", "superb", "brilliant","perfect", "favorite", "loved", "enjoyable", "entertaining", "fun", "best", "beautiful","inspiring", "touching", "impressive", "awesome",]
negative=["bad", "terrible", "awful", "boring", "dull", "worst", "hated", "poor", "disappointing","confusing", "annoying", "horrible", "unwatchable", "mediocre", "lame", "stupid","ridiculous", "cheesy", "overrated", "frustrating",]
negative += ["not_" + w for w in positive]

def vectorizer(a):
    vec=[]
    for i in a:
        if i in positive:
            vec.append(1)
        elif i in negative:
            vec.append(0)
    if vec.count(0)==0 and vec.count(1)==0:return None
    else:
        return [vec.count(1), vec.count(0)]

d['review']=d['review'].apply(vectorizer)
d=d.dropna()
x = np.array(d['review'].tolist())
y = d['sentiment'].values.reshape(-1, 1)

def sigmoid(z):
    return 1/(1+np.exp(-z))

a=0.01
b=0
w=np.zeros((2,1))
m = x.shape[0]

for i in range(1000):
    z=np.dot(x,w)+b
    r=sigmoid(z)
    dw=(1/m)*np.dot(x.T, (r - y))
    db=(1/m)*np.sum(r - y)
    w-=a*dw
    b-=a*db

z=np.dot(x, w) + b
y_prob=sigmoid(z)
y_pred=(y_prob>=0.5).astype(int)
accuracy=np.mean(y_pred==y)
print("Accuracy on dataset:", accuracy*100, "%")

review=input("Enter the review text: ")
review=cleaning(review)
review=vectorizer(review)
x = np.array(review)
z=np.dot(x,w)+b
r=sigmoid(z)
if r>=0.5:
    print("Positive review")
else:
    print("Negative review")


