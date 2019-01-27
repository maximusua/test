#!/usr/bin/python

from typing import List

from flair.data_fetcher import NLPTaskDataFetcher, NLPTask
from flair.data import TaggedCorpus
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, CharLMEmbeddings, CharacterEmbeddings
from flair.training_utils import EvaluationMetric
from flair.visual.training_curves import Plotter
from flair.data import TaggedCorpus
from flair.data_fetcher import NLPTaskDataFetcher
from flair.embeddings import FlairEmbeddings, BertEmbeddings, ELMoEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from flair.data import Sentence


import os
import sys, getopt
import codecs
import requests
import json
import time
import re
from requests import get
from collections import defaultdict
import warnings
import os
import pandas as pd

import torch

class ExternalMarkup:
	def __init__(self, lang_id = '1', gpu = '0'):
		self._lang_id  = lang_id
		self._gpu = gpu

	def initModel(self):
		torch.cuda.set_device(int(self._gpu))
		torch.cuda.is_available(), torch.cuda.current_device()
		self._data_path = "/home/max/nlp/ner/dataset/rnd/"
		
		self._train_path = "ner_cn_train_hotel_" + self._lang_id + ".csv"
		self._valid_path = "ner_cn_valid_hotel_" + self._lang_id + ".csv"
		self._test_path = "ner_cn_test_hotel_" + self._lang_id + ".csv"




		
		self._best_model_path="/home/max/nlp/ner/models/FLAIR/best_rnd_model_hotel" + "_" + self._lang_id
		return True

	def loadModel(self):
		# load the model you trained
		self._model = SequenceTagger.load_from_file(self._best_model_path + '/final-model.pt')

	def predict(self, sents):
		
		sp = []
		for sent in sents:
			sentence = Sentence(sent)

			# predict tags and print
			self._model.predict(sentence)
			spans = []
			for token in sentence:
				tag = token.get_tag('ner')
				tag_value = tag.value
				if tag_value != 'O':
					if tag_value[0:2] in ['B-', 'I-', 'O-', 'E-', 'S-']:
						tag_value = tag_value[2:]
				spans.append((token.text, tag_value, str(int(tag.score*100))))
			sp.append(spans)
			print(spans)
		#print(sp)

		return sp

def process(gettext, lang_id):
	work_url = str("http://travel-cms.oriekhova/syntexts/")
	work_page = "external_sgp_job"
	
	for it in range(0, 1000):
		url = work_url + '?page=' + work_page + '&format=json&lang_id=' + lang_id
		r = requests.get(url=url)
		data = r.json()
		nprocess_count = data["count"]
		print('ncount: ', nprocess_count)
		if int(nprocess_count) == 0:
			return
		train = pd.DataFrame.from_records(data["job"]["sentences"])
		print(train.head())
		spans = gettext.predict(train["Toks"])

		for i, span in enumerate(spans):
			print('|'.join([v[0] + "_" + v[1] + '(' + v[2] + ')' for v in span]))
			data_str = {
				'page': work_page, 
				'format': 'json', 
				'action': 'save', 
				'items' : {
					'model_id' : '1',
					'sentence_id': train["ID"][i], 
					'market_text' : '|'.join([v[0] + "_" + v[1] + '(' + v[2] + ')' for v in span])
				}
				}
			r = requests.post(work_url, data=data_str)
			


def main():
	try:
		opts, _ = getopt.getopt(sys.argv[1:],"l:g:",["lang-id=", "gpu="])
	except getopt.GetoptError:
		print('errorparams')

	lang_id  = '1'
	gpu = 0
	for opt, arg in opts:
		if opt in ["-l", "--lang-id"]:
			lang_id = arg
		elif opt in ["-g", "--gpu"]:
			gpu = arg

	gettext = ExternalMarkup(lang_id = lang_id, gpu = gpu)
	gettext.initModel()

	gettext.loadModel()
	process(gettext, lang_id)


if __name__ == "__main__":
	main()