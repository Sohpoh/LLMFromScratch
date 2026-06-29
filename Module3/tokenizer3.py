import json

'''
This class should be constructed with trained tokenizer data:
vocab_file : a string path to a vocab.txt file
merges_file : a string path to a merges.json file

The vocab.txt file is a list of words, one per line.
The merges.json file is a list of lists which represent merges.
For example, [["m","o"],["s","e"],["u","s"]]

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
        # Load vocabulary
        with open(vocab_file, 'r', encoding='utf-8') as f:
            self.vocab = [line.strip() for line in f.readlines()]
        
        # Create token to id mapping
        self.token_to_id = {token: idx for idx, token in enumerate(self.vocab)}
        
        # Load merges
        with open(merges_file, 'r', encoding='utf-8') as f:
            self.merges = json.load(f)


    def encode(self, string):
        '''
        param string : a string to be encoded
        returns a list of integers (token ids)
        '''
        # Start with individual characters
        tokens = list(string)
        
        # Apply merges in order
        for merge_pair in self.merges:
            left, right = merge_pair
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == left and tokens[i + 1] == right:
                    # Found a merge pair, combine them
                    new_tokens.append(left + right)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        
        # Convert tokens to IDs
        token_ids = []
        for token in tokens:
            if token in self.token_to_id:
                token_ids.append(self.token_to_id[token])
            else:
                # Handle unknown tokens (fallback to individual characters)
                for char in token:
                    if char in self.token_to_id:
                        token_ids.append(self.token_to_id[char])
                    else:
                        # Skip unknown characters
                        pass
        
        return token_ids


    def decode(self, list_of_integers):
        '''
        param list_of_integers : a list of token ids
        returns a string formed by decoding these ids.
        '''
        # Convert IDs back to tokens
        tokens = []
        for token_id in list_of_integers:
            if 0 <= token_id < len(self.vocab):
                tokens.append(self.vocab[token_id])
            else:
                # Handle invalid token IDs
                tokens.append('')
        
        # Join tokens to form the final string
        return ''.join(tokens)



if __name__ == "__main__":

    # example of using this class

    tok = Tokenizer("./vocab.txt", "./merges.json")
    x = tok.encode("Peter piper picked a peck of pickled peppers.")
    print(x)
    x = tok.decode(x)
    print(x) # should be our original text.
