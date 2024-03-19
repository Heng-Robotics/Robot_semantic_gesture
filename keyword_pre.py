#coding:utf-8
"""
This code is used for keyword and sentence pre-process.
It can replace all the abbr words in the 'keyword' list and 'joke' list;
It can also covert all uppercase letter into lowercase letter.

"""

import pandas as pd
from abbreviation import limits 

df = pd.read_csv('joke_all.csv')

# replace the abbr word into full spelling
def replace_abbreviations(text):
    if isinstance(text, str):
        words = text.split()
        for i, word in enumerate(words):
            if word.lower() in limits:
                words[i] = limits[word.lower()]
        return ' '.join(words)
    else:
        return text

# 替换关键词列中的缩写
df['joke'] = df['joke'].apply(replace_abbreviations)
df['semantic_word_1'] = df['semantic_word_1'].apply(replace_abbreviations)
df['semantic_word_2'] = df['semantic_word_2'].apply(replace_abbreviations)
df['semantic_word_3'] = df['semantic_word_3'].apply(replace_abbreviations)


# 将大写字母转换为小写字母
df['joke'] = df['joke'].apply(lambda x: x.lower() if isinstance(x, str) else x)
df['semantic_word_1'] = df['semantic_word_1'].apply(lambda x: x.lower() if isinstance(x, str) else x)
df['semantic_word_2'] = df['semantic_word_2'].apply(lambda x: x.lower() if isinstance(x, str) else x)
df['semantic_word_3'] = df['semantic_word_3'].apply(lambda x: x.lower() if isinstance(x, str) else x)

df.to_csv('joke_all.csv', index=False)