import json

'''
This class should be constructed with trained tokenizer data:
vocab_file : a string path to a vocab.txt file
merges_file : a string path to a merges.json file

The class should implement two methods:
encode(string): returns a list of integer ids (tokenized text)
decode(list_of_ids): returns a string re-assembled from token ids

You may assume that only a single sample is passed in at a time (no batching).
You can add additional methods, classes, etc as you find helpful.

Important: Our vocabulary and merges may include 
punctuation. Just treat all non-space characters equally.

---

Notes on validating your solution:

A good sanity check is that decode(encode(x)) should return x.

Additionally, make sure that the tokenizer is using the merges in order.
For example, if your merges contain: ("m","o"), ("s","e"), ("u","s"), then
"mouse" should be represented as mo|u|se.

'''


class Tokenizer:
    
    def __init__(self, vocab_file, merges_file):
        self.vocab = {}
        self.id_to_token = {} 
        with open(merges_file, 'r', encoding='utf-8') as f:
            self.merges =json.load(f)
        with open(vocab_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                token = line.rstrip('\n')
                self.vocab[token]= i
                self.id_to_token[i]= token


    def doMerges(self, tokens):

        for merge_pair in self.merges:
            left,right = merge_pair
            newTokens = []
            i = 0
            while i<len(tokens):
                if (i <len(tokens) - 1 and 
                    tokens[i] == left and 
                    tokens[i+1] == right):
                    newTokens.append(left+right)
                    i+= 2
                else:
                    newTokens.append(tokens[i])
                    i+= 1
            tokens=newTokens
        
        return tokens

    def encode(self, string):
        '''
        param string : a string to be encoded
        returns a list of integers (token ids)
        '''
        if not string:
            return []
        words = string.split(' ')
        allTokens = []
        for i, word in enumerate(words):
            if not word: 
                continue
            wordTokens = list(word)
            wordTokens = self.doMerges(wordTokens)
            if i>0:
                allTokens.append(' ')
            allTokens.extend(wordTokens)
        tokenIds = []
        for token in allTokens:
            if token in self.vocab:
                tokenIds.append(self.vocab[token])
        return tokenIds

    def decode(self, list_of_integers):
        '''
        param list_of_integers : a list of token ids
        returns a string formed by decoding these ids.
        '''
        if not list_of_integers:
            return ""
        
        tokens = []
        for token_id in list_of_integers:
            if token_id in self.id_to_token:
                tokens.append(self.id_to_token[token_id])

        return ''.join(tokens)


if __name__ == "__main__":
    # example of using this class
    tok = Tokenizer("./vocab.txt", "./merges.json")
    x = tok.encode("Peter piper picked a peck of pickled peppers.")
    print(x)
    x = tok.decode(x)
    print(x)  # should be our original text.
