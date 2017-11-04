# -*- coding:utf-8 -*-

import sys

import nltk
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

reload(sys)
sys.setdefaultencoding("utf-8")
pal = sns.color_palette()
mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # enable chinese

dataset = pd.read_csv("./data/mobile_dataset_top100.csv")

tokenstr = nltk.word_tokenize(" ".join(dataset.text.values).decode("utf-8"))
fdist1 = nltk.FreqDist(tokenstr)
listkey = []
listval = []
print u".........统计出现最多的前30个词..............."
for key, val in sorted(fdist1.iteritems(), key=lambda x: (x[1], x[0]), reverse=True)[:40]:
    listkey.append(key)
    listval.append(val)
    # print key, val, u' ',

df = pd.DataFrame(listval, columns=[u'次数'])
df.index = listkey
df.plot(kind='bar')
plt.title(u'词频统计')
plt.show()


#  plot class distribution
dataset.first_class.value_counts().plot(kind="bar")
plt.show()
dataset.second_class.value_counts().plot(kind="bar")
plt.show()

#  Number of words in the text ##
dataset["num_words"] = dataset["text"].apply(lambda x: len(str(x).split()))
dataset['num_words'].loc[dataset['num_words'] > 1000] = 1000  # truncation for better visuals
plt.figure(figsize=(12,8))
sns.violinplot(x='first_class', y='num_words', data=dataset)
plt.xlabel('First Class', fontsize=12)
plt.ylabel('Number of words in text', fontsize=12)
plt.title("Number of words in First Class", fontsize=15)
plt.show()

# global len distribution.
plt.figure(figsize=(12, 8))
plt.hist(dataset["num_words"], bins=200, range=[10, 1000], color=pal[1], normed=True, label='train')
plt.title('Normalised histogram of words count in text', fontsize=15)
plt.legend()
plt.xlabel('Number of words', fontsize=15)
plt.ylabel('Probability', fontsize=15)
plt.show()

# word cloud

# mask_img = imread("./data/images.jpg")
cloud = WordCloud(width=1440, height=1080, font_path="data/msyh.ttf").generate(" ".join(dataset.text.values).decode("utf-8"))
plt.figure(figsize=(20, 15))
plt.imshow(cloud)
plt.axis('off')
plt.show()
