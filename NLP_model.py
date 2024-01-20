import numpy as np 
import pandas as pd 

# clasas for vector embedding and similarity search
class VectorStore:
    def __init__(self):
        self.vector_data = {}
        self.vector_index = {}

    def add_vector(self, vector_id, vector):
        self.vector_data[vector_id] = vector
        self._update_index(vector_id, vector)

    def get_vector(self, vector_id):
        return self.vector_data.get(vector_id)

    def _update_index(self, vector_id, vector):
        # In this simple example, we use brute-force cosine similarity for indexing
        for existing_id, existing_vector in self.vector_data.items():
            similarity = np.dot(vector, existing_vector) / (np.linalg.norm(vector) * np.linalg.norm(existing_vector))
            if existing_id not in self.vector_index:
                self.vector_index[existing_id] = {}
            self.vector_index[existing_id][vector_id] = similarity

    def find_similar_vectors(self, query_vector, num_results=5):
        results = []
        for vector_id, vector in self.vector_data.items():
            similarity = np.dot(query_vector, vector) / (np.linalg.norm(query_vector) * np.linalg.norm(vector))
            results.append((vector_id, similarity))

        # Sort by similarity in descending order
        results.sort(key=lambda x: x[1], reverse=True)

        # Return the top N results
        return results[:num_results]



class NLP_Model:
    df = None 
    flow_index = {}
    website_index = {}
    cases = []
    vocabulary = set()
    vector_store = None 
    word_to_index = {}
    case_vectors = {}

    def __init__(self):
        # loading the dataset
        df = pd.read_excel("optimal and final website x flow.json.xlsx")
        # indexing
        df.dropna(axis = 0 , inplace = True)
        
        for index, row in df.iterrows():
            self.flow_index[index] = row["flow_json"]
            self.website_index[index] = row["website"]
            self.cases.append(row['website'])
        
        # class for vector embedding and similarity search
        vector_store = VectorStore()

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
                vector = np.zeroes(len(self.vocabulary))
                for token in tokens:
                    vector[self.word_to_index[token]] += 1
                self.case_vectors[curr_case] = vector
            
            except Exception as e:
                pass 
                
        # storing in VectorStore
            for curr_web, vector in self.case_vectors.items():
                vector_store.add_vector(curr_web , vector)


    def GetJSON(prompt):
        query_vector = np.zeroes(len(self.vocabulary))
        query_tokens = self.query_sentence.lower().split()
        for token in query_tokens:
            if(token in self.query_tokens):
                self.query_vector[self.word_to_index[token]] += 1

        similar_sentences = self.vector_store.find_similar_vectors(query_vector , num_results = 1) 
        
        print(similar_sentences)






