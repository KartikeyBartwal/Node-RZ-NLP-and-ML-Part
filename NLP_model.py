import numpy as np 
import pandas as pd 
from vector_store import VectorStore
from nltk.tokenize import word_tokenize
import string 
import nltk
from nltk.corpus import stopwords 
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from transformers import AutoTokenizer, AutoModel 
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch 
import sys


nltk.download('stopwords')
nltk.download('punkt')


def basic_cleaning(text):
    lowercase_text = text.lower()
    no_punctuation_text = ''.join(char for char in lowercase_text if char not in string.punctuation)
    return no_punctuation_text

def remove_stopwords(text):
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word.lower() not in stop_words]
    filtered_text = ' '.join(filtered_words)
    return filtered_text


def stemming(text):
    words = word_tokenize(text)
    porter_stemmer = PorterStemmer()
    stemmed_words = [porter_stemmer.stem(word) for word in words]
    stemmed_text = ' '.join(stemmed_words)
    return stemmed_text


class NLP_Model:
    df = None 
    flow_index = {}
    website_index = {}
    cases = []
    vocabulary = set()
    vector_store = None 
    word_to_index = {}
    case_vectors = {}
    vector_store = None 

    def __init__(self):
        # loading the dataset
        df = pd.read_excel("flow samples for training.xlsx")
        # indexing
        df.dropna(axis = 0 , inplace = True)
        # print("Columns of the dataset:" , df.columns)

        # preprocessing on the training dataset
        for index, row in df.iterrows():
            details = row["Website"]
            print("TEXT BEFORE:" , details)
            details = basic_cleaning(details)
            details = remove_stopwords(details)
            details = stemming(details)

            print("TEXT AFTER:" , details)

            df.at[index, "Website"] = details


        for index, row in df.iterrows():
            self.flow_index[index] = row["flow.json"]
            self.website_index[row["Website"]] = index
            self.cases.append(row['Website'])
        

        # class for vector embedding and similarity search
        self.vector_store = VectorStore()

        # tokenization and vocabulary creation
        for curr_web in self.cases:
            try:
                tokens = curr_web.lower().split()
            except Exception as e:
                print(curr_web)
                print("*" * 100)
            
            self.vocabulary.update(tokens)
        
        # assign unique indices to words in the vocabulary
        self.word_to_index = {word: i for i, word in enumerate(self.vocabulary)}
        # vectorization
        for curr_case in self.cases:
            try:
                tokens = curr_case.lower().split()
                vector = np.zeros(len(self.vocabulary))
                for token in tokens:
                    vector[self.word_to_index[token]] += 1
                self.case_vectors[curr_case] = vector
            
            except Exception as e:
                pass 
                
        # storing in VectorStore
            for curr_web, vector in self.case_vectors.items():
                self.vector_store.add_vector(curr_web , vector)


    def GetJSON(self , prompt):
        print(self.word_to_index)
        print("GETJSON started")

        query_vector = np.zeros(len(self.vocabulary))

        print("query vector created")

        query_tokens = prompt.lower().split()

        print("Done with query tokens") 

        for token in query_tokens:
            if(token not in self.vocabulary):
                continue
            query_vector[self.word_to_index[token]] += 1


        print("Done with query_tokens")

        similar_sentences = self.vector_store.find_similar_vectors(query_vector , num_results = 5)
       
        # # general backend (if similarities are barely there)
        # if(similar_sentences[0][1] <= 31):
        #     return self.flow_index["12"] 
        
        for search in similar_sentences:
            print(search[1])
            print()
        print("similar searches are done")
        print()
        print()

        output_json = self.flow_index[self.website_index[similar_sentences[0][0]]]
        return output_json


class NLP_Model_BERT:
    df = None 
    flow_index = {}
    website_index = {}
    vector_store = None 
    word_to_index = {}
    universal_embeddings = None 
    model = None 

    def __init__(self):
        # loading the dataset
        df = pd.read_excel("file_dataset.xlsx")
        # indexing
        df.dropna(axis = 0 , inplace = True)

        # preprocessing on the training dataset
        for index, row in df.iterrows():
            details = row["Website"]
            print("TEXT BEFORE:" , details)
            details = basic_cleaning(details)
            details = remove_stopwords(details)
            details = stemming(details)

            print("TEXT AFTER:" , details)

            df.at[index, "Website"] = details

        data_samples = []
        for index, row in df.iterrows():
            self.flow_index[index] = row["flow.json"]
            self.website_index[row["Website"]] = index
            data_samples.append(row["Website"])
        
        # ATokenizers and Vectorization of natural language data
        self.model = SentenceTransformer('bert-base-nli-mean-tokens')
        #encoding
        self.universal_embeddings = self.model.encode(data_samples)
        print("Shape of the universal embedding: universal_embeddings.shape")
        print(self.universal_embeddings)


    def GetJSON(self , prompt):
        # get the vectorized representation of the input prompt
        input_prompt_vector = self.model.encode(prompt)
        similarity_scores = cosine_similarity(
            [input_prompt_vector] , 
            self.universal_embeddings
        )

        scores = similarity_scores[0]
        
        index = 0
        best_index = 0
        max_score = -sys.float_info.max
        for score in scores:
            print("Cosine Similarity Score:" , score)
            if(score > max_score):
                max_score = score
                best_index = index 
            index += 1

        best_index = best_index + 1
        print("Best Score:" , max_score)
        print("Best Index:" , best_index)
        print("SEARCH HAS BEEN PRINTED")
        
        # find the index having highest cosine similarity
        output_json = self.flow_index[best_index]
        return output_json


