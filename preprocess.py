import time
import numpy as np
import nltk
import datetime

from nltk.corpus import stopwords
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
import re


class TextPreprocessor:
    '''
        Text Preprocessor
    '''

    def __init__(self):
        '''
            Initiailize
        '''
        self.sr = stopwords.words('english')
        self.characters = [' ', ',', '.', 'DBSCAN', ':', ';', '?', '_',
                           '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%', '-', '...', '^', '{', '}']
        # Download necessary libs
        nltk.download('stopwords')
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
        nltk.download('wordnet')

    def tokenize(self, sentence):
        '''
            Delete unnecessary blanks. Segment. Pos-tag.
        '''
        sentence = re.sub(r'\s+', ' ', sentence)
        token_words = word_tokenize(sentence)
        token_words = pos_tag(token_words)
        return token_words

    def stem(self, token_words):
        '''
            Stemming process
        '''
        wordnet_lematizer = WordNetLemmatizer()  # Lemmatize
        words_lematizer = []
        for word, tag in token_words:
            if tag.startswith('NN'):
                # n.
                word_lematizer = wordnet_lematizer.lemmatize(word, pos='n')
            elif tag.startswith('VB'):
                # v.
                word_lematizer = wordnet_lematizer.lemmatize(word, pos='v')
            elif tag.startswith('JJ'):
                # adj.
                word_lematizer = wordnet_lematizer.lemmatize(word, pos='a')
            elif tag.startswith('R'):
                # r
                word_lematizer = wordnet_lematizer.lemmatize(word, pos='r')
            else:
                word_lematizer = wordnet_lematizer.lemmatize(word)
            words_lematizer.append(word_lematizer)
        return words_lematizer

    def delete_stopwords(self, token_words):
        '''
            Delete stopwords
        '''
        cleaned_words = [word for word in token_words if word not in self.sr]
        return cleaned_words

    def is_number(self, s):
        '''
            Whether s is a number
        '''
        try:
            float(s)
            return True
        except ValueError:
            pass

        try:
            import unicodedata
            unicodedata.numeric(s)
            return True
        except (TypeError, ValueError):
            pass

        return False
    
    def is_spcieal(self, s):
        '''
            Whether s only contains special characters
        '''
        # Containing numbers
        if bool(re.search(r'\d', s)) is True:
            return False
        # Containing letters
        if bool(re.search(r'[A-Za-z]', s)) is True:
            return False
        return True


    def delete_characters(self, token_words):
        '''
            Delete special characters
        '''
        # words_list = [word for word in token_words if word not in self.characters and not self.is_number(word)]
        words_list = [word for word in token_words if not self.is_spcieal(word)]
        return words_list

    def to_lower(self, token_words):
        '''
            Covert to lower
        '''
        words_lists = [x.lower() for x in token_words]
        return words_lists

    def preprocess(self, text):
        '''
            Preprocess text
        '''
        token_words = self.tokenize(text)
        token_words = self.stem(token_words)
        token_words = self.delete_stopwords(token_words)
        token_words = self.delete_characters(token_words)
        token_words = self.to_lower(token_words)
        return token_words

        