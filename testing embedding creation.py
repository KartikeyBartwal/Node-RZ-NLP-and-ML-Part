from transformers import AutoTokenizer, AutoModel 
import torch 

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
model = AutoModel.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')

# tokenize the sentences like before

sent = [
    "Three years later, the coffin was still full of Jello",
    "The fish dreamed of escaping the fishbowl and into the toilet where he saw his friend go",
    "The person box was packed with jelly many dozens of months later",
    "He found a leprechaun in his walnut shell."
]

#initialize dictionary, stores tokenized sentences
token = {'input_ids' : [] , 'attention_mask' : []}
for sentence in sent:
    #encode each sentence, append it to dictionary
    new_token = tokenizer.encode_plus(sentence , max_length = 128 , truncation = True , padding = 'max-length' , return_tensors = 'pt')
    token['input_ids'].append(new_token['input_ids'[0]])
    token['attention_mask'].append(new_token['attension_mask'][0])

# reformat list of tensors to a single tensor
    token['inputt_ids'] = torch.stack(token['input_ids'])
    token['attention_mask'].append(new_token['attention_mask'][0])

# reformat list of tensors to single tensor
token['input_ids'] = torch.stack(token['input_ids'])
token['attention_mask'] = torch.stack(token['attention_mask'])

#process tokens through model
output = model(**token)
print(output.keys())