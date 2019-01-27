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
from flair.data import TaggedCorpus
from flair.data_fetcher import NLPTaskDataFetcher, NLPTask
from flair.embeddings import WordEmbeddings, FlairEmbeddings, DocumentLSTMEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer
from pathlib import Path
import urllib.request
import pathlib




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

class ExternalText:
	def __init__(self, from_lang_id = '1', to_lang_id = '', gpu = '1', variable_system = '1'):
		self._from_lang_id  = from_lang_id
		self._to_lang_id = to_lang_id
		self._gpu = gpu
		self._model_version = "flair_1024_2_crf"

	def initModel(self):
		torch.cuda.set_device(int(self._gpu))
		torch.cuda.is_available(), torch.cuda.current_device()
		self._data_path = "dataset/review/"
		self._train_path = "train_" + self._from_lang_id + "_" + self._to_lang_id + ".csv"
		self._valid_path = "valid_" + self._from_lang_id + "_" + self._to_lang_id + ".csv"
		self._test_path = "test_" + self._from_lang_id + "_" + self._to_lang_id + ".csv"



		self.checkDataset()
		
		self._best_model_path="models/review_" + self._from_lang_id + "_" + self._to_lang_id

		return True

	def checkDataset(self):
		pathlib.Path(self._data_path).mkdir(parents=True, exist_ok=True)
		try:
			self._url_path = "http://travel-cms.fabrica.net.ua/data/dataset/review/"
			urllib.request.urlretrieve(self._url_path + self._train_path,  self._data_path + self._train_path)
			urllib.request.urlretrieve(self._url_path + self._valid_path,  self._data_path + self._valid_path)
			urllib.request.urlretrieve(self._url_path + self._test_path,  self._data_path + self._test_path)
		except:
			print(self._url_path + self._train_path);
			print("An exception occurred")
		return True

	def train(self, from_checkpoint = False):
		bert_ename = "bert-base-multilingual-cased"
		
		# init multilingual BERT
		bert_embedding = BertEmbeddings(bert_ename)

		# use your own data path
		data_folder = Path(self._data_path)

		# load corpus containing training, test and dev data
		corpus: TaggedCorpus = NLPTaskDataFetcher.load_classification_corpus(data_folder,
										train_file=self._train_path,
										test_file=self._test_path,
										dev_file=self._valid_path)

		# 2. create the label dictionary
		label_dict = corpus.make_label_dictionary()

		# 3. make a list of word embeddings
		# initialize embeddings

		embedding_types = [
			    bert_embedding,
#			    CharacterEmbeddings()
		]

		# 4. init document embedding by passing list of word embeddings
		document_embeddings: DocumentLSTMEmbeddings = DocumentLSTMEmbeddings(embedding_types,
                                                                     hidden_size=256,
                                                                     reproject_words=True,
                                                                     reproject_words_dimension=128,
                                                                     )
		print(corpus)

		# 5. create the text classifier
		classifier = TextClassifier(document_embeddings, label_dictionary=label_dict, multi_label=False)

		# 6. initialize the text classifier trainer
		trainer = ModelTrainer(classifier, corpus)

		# 7. start the training
		trainer.train(self._best_model_path,
				learning_rate=0.1,
				mini_batch_size=8,
				anneal_factor=0.5,
				patience=5,
				max_epochs=10,
				embeddings_in_memory=False, checkpoint=True, save_final_model=True)

		
		
		return True


	def loadModel(self):
		# load the model you trained
		self._model = TextClassifier.load_from_file(self._best_model_path + '/final-model.pt')

	def predict(self, sents):
		
		sp = []
		for sent in sents:
			sentence = Sentence(sent)

			# predict tags and print
			self._model.predict(sentence)
			#print(sentence)
			if sentence.labels[0].value == 'ok':
				sp.append('1')
			else:
				sp.append('3')
		return sp



def process(gettext, lang_id, to_lang_id):
	work_url = str("http://travel-cms.fabrica.net.ua/review/");
	work_page = "external_worker";

	for it in range(0, 1000):
		url = work_url + '?page=' + work_page + '&format=json&from_lang_id=' + lang_id + '&to_lang_id=' + to_lang_id + "&version=" + gettext._model_version
		
		r = requests.get(url=url)
		data = r.json()
		nprocess_count = data["count"]
		print('ncount:', nprocess_count)
		if int(nprocess_count) == 0:
			return;
		train = pd.DataFrame.from_records(data["job"]["texts"])
		print(train.head())
		spans = gettext.predict(train["Text"]);
		for i, span in enumerate(spans):
			data_str = {'page': work_page, 'format': 'json', 'action': 'saveone', 'version': gettext._model_version, 'id': train["ID"][i], 'status' : span}
			#r = requests.post(work_url, data=data_str)
			print(data_str)


def main():
	try:
		opts, args = getopt.getopt(sys.argv[1:],"l:t:a:g:s:p:c:",["from-lang-id=", "to-lang-id=", "action=", "gpu=", "from-checkpoint="])
	except getopt.GetoptError:
		print('errorparams')

	from_lang_id  = '1'
	to_lang_id = ''
	action   = 'process'
	gpu = 0
	from_checkpoint = False
	for opt, arg in opts:
		if opt in ["-l", "--from-lang-id"]:
			lang_id = arg;
		elif opt in ["-t", "--to-lang-id"]:
			to_lang_id = arg;
		elif opt in ["-a", "--action"]:
			action = arg;
		elif opt in ["-g", "--gpu"]:
			gpu = arg;
		elif opt in ["-c", "--from-checkpoint"]:
			if int(arg) > 0:
				from_checkpoint = True

	gettext = ExternalText(from_lang_id = from_lang_id, to_lang_id = to_lang_id, gpu = gpu)
	gettext.initModel();

	if action == 'train':
		gettext.train(from_checkpoint)
	else:
		gettext.loadModel()
		process(gettext, lang_id, to_lang_id)


if __name__ == "__main__":
	main()