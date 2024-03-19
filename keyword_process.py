# coding: utf-8

'''
## Prepare Dataset for BERT.
Convert key-text recognition dataset to BIO format dataset.

'''
import pandas as pd
import argparse
from nltk.tokenize import sent_tokenize
from abbreviation import limits

parser = argparse.ArgumentParser(description='BERT semantic_word Extraction Model')

parser.add_argument('--data', type=str, default='joke_all.csv',
                    help='location of the data corpus')
# parser.add_argument('--epochs', type=int, default=4,
#                     help='upper epoch limit')
# parser.add_argument('--batch_size', type=int, default=32, metavar='N',
#                     help='batch size')
# parser.add_argument('--seq_len', type=int, default=75, metavar='N',
#                     help='sequence length')
# parser.add_argument('--lr', type=float, default=3e-5,
#                     help='initial learning rate')
# parser.add_argument('--save', type=str, default='model.pt',
#                     help='path to save the final model')
args = parser.parse_args()


dataset_name = args.data

df = pd.read_csv(dataset_name)

jokes = df['joke'].tolist()
semantic_word_1 = df['semantic_word_1'].tolist()
semantic_word_2 = df['semantic_word_2'].tolist()
semantic_word_3 = df['semantic_word_3'].tolist()


def convert():

    key_sent = []   #initialize the list named key_sent for storing the sentences latter
    labels = []   #initialize the list named labels

    #---------change the abbr word into full spelling word in the joke
    for num, joke in enumerate(jokes):
        #-------remove punction marks in the sentences
        punctuations = '''!()-[]{};:\,<>'"./?@#$%^&*_~''' #the punction marks need to be removed
        no_punct = ""  #initialize the string 
        for char in joke:
            if char not in punctuations:
                no_punct = no_punct + char

        joke=no_punct #joke without any punction mark
       
       #-------semantic_word process
        z = ['O'] * len(joke.split()) # return a initial list that can store the label order. For example, if the joke is "I am spiderman", then the list z will be ['O', 'O', 'O']
        # for semantic_word in semantic_words:
        if pd.isna(semantic_word_1[num]):
            print(str(num) + " -condition_1")
            z = z
        elif pd.isna(semantic_word_2[num]):
            print(str(num) + " -condition_2")
            z[joke.split().index(semantic_word_1[num].split()[0])] = 'F'
        elif pd.isna(semantic_word_3[num]):
            print(str(num) + " -condition_3")
            z[joke.split().index(semantic_word_1[num].split()[0])] = 'F'
            z[joke.split().index(semantic_word_2[num].split()[0])] = 'S'
        else:
            print(str(num) + " -condition_else")
            z[joke.split().index(semantic_word_1[num].split()[0])] = 'F'
            z[joke.split().index(semantic_word_2[num].split()[0])] = 'S'
            z[joke.split().index(semantic_word_3[num].split()[0])] = 'T'

        labels.append(z) 
        key_sent.append(joke)
    return key_sent, labels

if __name__ == "__main__":
    sent,label = convert()
    # print(sent)

