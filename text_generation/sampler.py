
import torch
import numpy as np

'''
Class implementing a sampler for inference on a model. Given the raw logits from
an LLM model, this will sample the next token id.
'''
class Sampler:

	def __init__(
		self,
		top_k=None,
		top_p=None,
		frequency_penalty=1.0,
		presence_penalty=1.0
	):
		'''
		param top_k : (None or int)
			If specified, only the top k logits should be used during sampling
			If this is specified, top_p should be None

		param top_p : (None or int)
			If specified, only the logits representing the probability mass p should be used during sampling.
			Or, if the top token has mass greater than p, the top token is returned.
			If this is specified, top_k should be None

		If top_k and top_p are both None, sample from the whole distribution (same as top_p=1.0)

		param frequency_penalty : (float)
			A penalty applied to tokens that have previously occured in the sequence. Along with
			presence_penalty, this adjusts the per-token softmax temperature.
			A penalty of 1.0 indicates no change from normal softmax.

		param presence_penalty : (float)
			A penalty applied to tokens IF they have previously occured in the sequence. Along with
			frequency_penalty, this adjusts the per-token softmax temperature.
			A penalty of 1.0 indicates no change from normal softmax.
		'''
		# TODO
		self.top_k = top_k
		if top_p is not None:
			self.top_p = top_p
		else:
			if top_k is None:
				self.top_p = 1.0
			else:
				self.top_p = None

		self.frequency_penalty = frequency_penalty
		self.presence_penalty = presence_penalty

	def make_token_distribution(self, raw_unsorted_logits, previous_token_ids):
		'''
		param: raw_unsorted_logits (float numpy array)
			A one dimensional list of logits representing an unnormalized distribution over next tokens
			These are "unsorted" in the sense that their order aligns with vocabulary order, not with probability.

		param: previous_token_ids (int numpy array)
			A one dimensional list of ids representing the previous tokens, for calculating repetition penalties.

		returns:
			- the final probability distribution that this token is sampled from
			It should be returned back to token-id order (unsorted order) before returning.
		'''

		# TODO


		# very rough outline:
		# make temperature=1.0 for each vocabulary option
		# adjust temps as needed with penalties
		# logits = logits - np.min(logits) to make sure all are positive
		# apply temps & softmax
		# sort the distribution (and track the sort order so you can undo it later)
		# find either the top-p or top-k cutoff
		# renormalize this portion by simply dividing by the sum
		# revert back to original ordering of the distribution
			# helpful tip for this: 
			# indices = np.argsort(arr)
			# undo_indices = np.argsort(indices) # take argsort of the argsort
			# sorted_array = arr[indices]
			# put_back = sorted_array[undo_indices]
		# return distribution

		logits = raw_unsorted_logits.copy()
		temps = np.ones(len(logits))
		k = np.ones(len(logits))
		if len(previous_token_ids) > 0:
			
			occurences =np.bincount(previous_token_ids,minlength=len(logits))
			presence=(occurences>0).astype(float)
			#print(occurences.shape)
			frequencyPenalties = occurences* (self.frequency_penalty-1.0)
			presencePenalties = presence* (self.presence_penalty-1.0)
			#print(k.shape)
			#print(frequencyPenalties.shape)

			k += frequencyPenalties
			k += presencePenalties

		logits = logits - np.min(logits)

		y = logits/k
		y = np.exp(y)
		probs = y/np.sum(y)

		sortedIndices = np.argsort(-probs)
		sortedProbs = probs[sortedIndices]

		if self.top_k is not None:
			index = min(self.top_k,len(sortedProbs))
			sortedProbs[index:] = 0.0
		
		elif self.top_p is not None:
			currProbSum = 0.0
			index = len(sortedProbs)
			for i in range(len(sortedProbs)):
				currProbSum += sortedProbs[i]
				if currProbSum > self.top_p:
					index = i+1
					break
			sortedProbs[index:] = 0.0

		probSum = np.sum(sortedProbs)
		if probSum > 0:
			sortedProbs = sortedProbs/probSum
		else:
			sortedProbs = np.ones(len(logits))/len(logits)
		
		
		undo_indices = np.argsort(sortedIndices)
		resultProbs = sortedProbs[undo_indices]

		return resultProbs






	#==========================
	# for actually sampling the distribution
	def sample_one_token(self, raw_unsorted_logits, previous_token_ids):
		probs = self.make_token_distribution(raw_unsorted_logits, previous_token_ids)
		return np.random.choice(np.arange(len(raw_unsorted_logits)), p=probs)

	# for convenience, this is also callable
	def __call__(self, raw_unsorted_logits, previous_token_ids):
		return self.sample_one_token(raw_unsorted_logits, previous_token_ids)




if __name__ == "__main__":
    
    # example of using this with dummy data, keeping everything in token ids

    sampler = Sampler(top_p=0.8, frequency_penalty=1.1, presence_penalty=1.1)

    sequence = [1,2,3,4,5]

    for i in range(10):
    	# fake logits for a vocab of size 500
    	logits = np.random.randn(500)

    	# get next token in sequence
    	next_token = sampler(logits, sequence)
    	sequence.append(next_token)

    print(sequence)