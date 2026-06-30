import json
from collections import defaultdict, Counter
from saving import save_vocab, save_merges
'''
Your assignment is to implement BPE in the following method. You can add
classes or other routines if you find them helpful. 

This method should save two output files:
./vocab.txt : a list of the final vocabulary in order, one entry per line
./merges.json : a list of tuples of merges, in order

NOTE: Typically these file extensions are reversed (the vocabulary is a
json file and the merge list is a txt file), but for our purposes this way seems
simplier.

Does not need to return anything.

-------

This should implement a GPT-style tokenizer which prefixes words with a space.
You can assume that the base vocabulary contains all single characters that will occur.
Treat punctuation (besides spaces) just like the other characters.

You do NOT need to worry about using a placeholder token in place of a space. 
You do NOT need to worry about special tokens (pad, bos, eos, unk, etc.). We have not covered these yet.

IMPORTANT: If there are ties while computing the merges, you should use lexigraphic order to resolve.
Points will be taken off if a different tie-break is used as it will not match the homework solution.

For example, if the pairs ('ab','out') and ('sp','ite') are tied for most occuring,
then "about" should be recorded before "spite".

'''

def train_tokenizer(txt_file, vocab_size, base_vocabulary):
    '''
    param : txt_file - a string path to a text file of data, i.e. "./data.txt"
    param : vocab_size - integer specifying the final vocab size
    param : base_vocabulary - list of strings to add to the vocabulary by default

    saves:
    ./vocab.txt : a list of the final vocabulary in order, one entry per line, ties broken alphabetically
    ./merges.json : a list of tuples of merges, in order
    '''

    with open(txt_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    text = text.replace('\n', ' ').replace('\t', ' ')
    words = text.split(' ')
    words = [word for word in words if word]
    if words:
        processed_words = [words[0]] + [' '+word for word in words[1:]]
    else:
        processed_words = []
    
    word_counts = Counter()
    
    for word in processed_words:
        token_list = list(word)
        word_counts[tuple(token_list)] += 1
    
    vocab = set(base_vocabulary)
    merges = []
    
    while len(vocab)< vocab_size:
        pair_counts = defaultdict(int)
        
        for token_list, count in word_counts.items():
            for i in range(len(token_list)-1):
                pair = (token_list[i],token_list[i+1])
                pair_counts[pair]+=count
        
        if not pair_counts:
            break
        
        max_count=max(pair_counts.values())
        candidate_pairs=[pair for pair, count in pair_counts.items() if count == max_count]
        best_pair=min(candidate_pairs)
        
        new_token = best_pair[0] + best_pair[1]
        vocab.add(new_token)
        merges.append(best_pair)
        new_word_counts = Counter()
        
        for token_list, count in word_counts.items():
            new_token_list = []
            i = 0
            while i < len(token_list):
                if (i < len(token_list) -1 and token_list[i]==best_pair[0] and token_list[i+1]==best_pair[1]):
                    new_token_list.append(new_token)
                    i +=2
                else:
                    new_token_list.append(token_list[i])
                    i +=1
            
            new_word_counts[tuple(new_token_list)]+= count
        
        word_counts =new_word_counts
    
    vocab_list=sorted(list(vocab))

    save_vocab(vocab_list)
    save_merges(merges)





if __name__ == "__main__":


    base = "abcdefghijklmnopqrstuvwxyz"
    base += "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    base += "0123456789"
    base += "!@#$%^&*()_+-=[]{}|;':,.<>/?`~ "
    base += "\\"
    base += '"'

    train_tokenizer("./data.txt", len(base)+1000, [c for c in base])
