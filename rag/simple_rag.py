from datasets import load_dataset
import torch
import numpy as np
from openai import OpenAI
from transformers import AutoTokenizer, AutoModel
from dotenv import load_dotenv 
import os 
load_dotenv()

class SimpleRAGNews():

	def __init__(self):

		# load the dataset "permutans/fineweb-bbc-news" in streaming mode,
		# using the subset: "CC-MAIN-2013-20"
		# and the split: "train"
		self.dataset = load_dataset("permutans/fineweb-bbc-news", "CC-MAIN-2013-20", split="train", streaming=True)

		# TODO

		# load the model "ibm-granite/granite-embedding-30m-english"
		# and corresponding tokenizer using AutoModel and AutoTokenizer
		# as per the instructions here (scroll down a bit): 
		# https://huggingface.co/ibm-granite/granite-embedding-30m-english
		
		self.tokenizer = AutoTokenizer.from_pretrained("ibm-granite/granite-embedding-30m-english")
		self.model = AutoModel.from_pretrained("ibm-granite/granite-embedding-30m-english")  

		# TODO

		self.setup_db()
		HFTOKEN = os.getenv("HFTOKEN")

		# finally, create a client to use huggingface inference
		self.client = OpenAI(
		    base_url="https://router.huggingface.co/v1",
		    api_key=HFTOKEN, # paste your key in here or load from a variable
		)


	def setup_db(self):
		# TODO
		
		# take the first 100 samples from the bbc-news dataset
		# pare down the columns and only keep the "text" column
		# then convert to a list: list(ds) for easy retrieval later

		ds = self.dataset.take(100)
		#print(ds)
		#print("-------------------------")
		columnsToRemove = ['url']
		ds = ds.remove_columns(columnsToRemove)
		self.articles = list(ds)

		# for each entry in the dataset, call self.embed() on the text
		# save these vectors for later use
		
		self.embeddings = []
		#i = 0
		for article in self.articles:
			embedding = self.embed(article["text"])
			#if i == 0: print(embedding.shape)
			self.embeddings.append(embedding)
			#i+=1

		self.embeddings = torch.cat(self.embeddings, dim=0)
		#print(self.embeddings.shape)


	def embed(self, text):
		# TODO 

		# given a passed string, tokenize the text using the ibm-granite
		# tokenizer
		
		# hint: make sure you pass the following to the tokenizer:
		# padding=True, truncation=True, return_tensors='pt'
		tokens = self.tokenizer(text, padding = True, truncation = True, return_tensors='pt')

		# then pass through the embedding model as shown on the ibm-granite page:
		# embedding = model(**tokens)[0][:, 0]
		# embedding = torch.nn.functional.normalize(embedding, dim=1)
		#if i == 0:
		#	print(tokens['input_ids'][0])
		with torch.no_grad():
			embedding = self.model(**tokens)[0][:,0]
			embedding = torch.nn.functional.normalize(embedding,dim=1)
		
		return embedding


	def get_most_relevant_news_article_text(self, user_query):
		# given a user query (string), this method should:
		# call self.embed(user_query)
		# compare the embedding against all stored embeddings using
		# torch.nn.functional.cosine_similarity
		# use the top match to return the text of most relevant article
		# TODO

		queryEmbedding = self.embed(user_query)
		#print(queryEmbedding.shape)
		similarities = torch.nn.functional.cosine_similarity(queryEmbedding,self.embeddings,dim=1)
		topMatchIdx = torch.argmax(similarities).item()
		return self.articles[topMatchIdx]["text"]


	def summarize_article(self, article):
		# Use the HF Inference API to ask "openai/gpt-oss-20b" to
		# summarize an article. 

		# This should then return the model's final response text.
		completion = self.client.chat.completions.create(
			# ...
			model = "openai/gpt-oss-20b",
			messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes news articles concisely."},
                {"role": "user", "content": f"Summarize the article:\n{article}"}],
			max_tokens = 200
		)
		return completion.choices[0].message.content

	def summary_for_query(self, query):
		# get the most relevant article
		# get a summary of the article
		# return to user

		article = self.get_most_relevant_news_article_text(query)
		#print(article)
		summary = self.summarize_article(article)
		return summary
		# In practice this could be expanded to use the article text in a more
		# complex way.

if __name__ == "__main__":
	rag = SimpleRAGNews()
	query = "california wildfires"
	news_blurb_for_user = rag.summary_for_query(query)
	print("An AI-generated summary of the most relevant article:")
	print(news_blurb_for_user)