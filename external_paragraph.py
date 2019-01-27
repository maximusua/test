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

class ExternalMarkup:
	def __init__(self, lang_id = '1', acc_type = '1', gpu = '1', variable_system = '1'):
		self._lang_id  = lang_id
		self._acc_type = acc_type
		self._gpu = gpu
		self._variable_system = variable_system
		self._model_version = "flair_1024_2_crf"
		if self._lang_id == '5' or self._lang_id == '20':
			self._model_version = "flair_1024_2_crf_emb"

	def initModel(self):
		torch.cuda.set_device(int(self._gpu))
		torch.cuda.is_available(), torch.cuda.current_device()
		self._type2name = {"1": "hotel", "2": "app", "3": "villa", "4": "hostel"}
		self._data_path = "dataset/own/"
		if self._variable_system == '2':
			self._data_path = "dataset/rnd/"
		self._train_path = "ner_cn_train_" + self._acc_type + "_" + self._lang_id + ".csv"
		self._valid_path = "ner_cn_valid_" + self._acc_type + "_" + self._lang_id + ".csv"
		self._test_path = "ner_cn_test_" + self._acc_type + "_" + self._lang_id + ".csv"



		self.checkDataset()
		
		self._best_model_path="/home/max/nlp/ner/models/FLAIR/best_model_" + self._type2name[self._acc_type] + "_" + self._lang_id
		if self._variable_system == '2':
			self._best_model_path="/home/max/nlp/ner/models/FLAIR/best_rnd_model_" + self._type2name[self._acc_type] + "_" + self._lang_id


		return True

	def checkDataset(self):
		pathlib.Path(self._data_path).mkdir(parents=True, exist_ok=True)
		try:
			self._url_path = "http://travel-cms.fabrica.net.ua/data/dataset/ner/own/"
			if self._variable_system == '2':
				self._url_path = "http://travel-cms.fabrica.net.ua/data/dataset/ner/rnd/"
			urllib.request.urlretrieve(self._url_path + self._train_path,  self._data_path + self._train_path)
			urllib.request.urlretrieve(self._url_path + self._valid_path,  self._data_path + self._valid_path)
			urllib.request.urlretrieve(self._url_path + self._test_path,  self._data_path + self._test_path)
		except:
			print(self._url_path + self._train_path);
			print("An exception occurred")
		return True

	def train(self, from_checkpoint = False):
		bert_ename = "bert-base-multilingual-cased"
		if self._lang_id == '1':
			bert_ename = "bert-base-cased"
		if self._lang_id == '17':
			bert_ename = "bert-base-chinese"
		
		# init multilingual BERT
		bert_embedding = BertEmbeddings(bert_ename)

		# define columns
		columns = {0: 'text', 1: 'ner'}

		# this is the folder in which train, test and dev files reside

		# retrieve corpus using column format, data folder and the names of the train, dev and test files
		corpus: TaggedCorpus = NLPTaskDataFetcher.load_column_corpus(self._data_path, columns,
                                                              train_file=self._train_path,
                                                              test_file=self._test_path,
                                                              dev_file=self._valid_path)



		print(corpus)

		# 2. what tag do we want to predict?
		tag_type = 'ner'

		# 3. make the tag dictionary from the corpus
		tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)

		# initialize embeddings

		embedding_types = [bert_embedding]
		emb_in_memory = False
		if self._lang_id == '1':
			embedding_types.append(WordEmbeddings('en-news'))
			embedding_types.append(ELMoEmbeddings('original'))
			embedding_types.append(CharacterEmbeddings())

#			    WordEmbeddings('glove'),
			    #FlairEmbeddings('news-forward'),
#			    #FlairEmbeddings('news-backward'),
			    # comment in this line to use character embeddings

			    # comment in these lines to use contextual string embeddings
			    #
			    #CharLMEmbeddings('news-forward'),
			    #
			    #CharLMEmbeddings('news-backward'),

		if self._lang_id == '4':
			embedding_types.append(WordEmbeddings('es'))
			embedding_types.append(FlairEmbeddings('spanish-forward'))
			embedding_types.append(FlairEmbeddings('spanish-backward'))
			embedding_types.append(CharacterEmbeddings())

		
		
		if self._lang_id == '2':
			embedding_types = [
			    WordEmbeddings('de'),
			    bert_embedding,
			    FlairEmbeddings('german-forward'),
			    FlairEmbeddings('german-backward'),
			    CharacterEmbeddings(),
			]
		
		if self._lang_id == '3':
			embedding_types = [
			    WordEmbeddings('fr'),
			    bert_embedding,
			    FlairEmbeddings('french-forward'),
			    FlairEmbeddings('french-backward'),
			    CharacterEmbeddings(),
			]
		
		if self._lang_id == '5':
			embedding_types.append(WordEmbeddings('it'))
			embedding_types.append(FlairEmbeddings('multi-forward'))
			embedding_types.append(FlairEmbeddings('multi-backward'))
			embedding_types.append(CharacterEmbeddings())
		if self._lang_id == '6':
			embedding_types = [
			    WordEmbeddings('nl'),
			    bert_embedding,
			    FlairEmbeddings('dutch-forward'),
			    FlairEmbeddings('dutch-backward'),
			    CharacterEmbeddings(),
			]
		if self._lang_id == '8':
			embedding_types = [
			    WordEmbeddings('pt'),
			    
			    bert_embedding,
			    ELMoEmbeddings('pt'),
			    FlairEmbeddings('portuguese-forward'),
			    FlairEmbeddings('portuguese-backward'),
			    CharacterEmbeddings(),
			]
		if self._lang_id == '9':
			embedding_types = [
			    WordEmbeddings('no'),
			    bert_embedding,
			    CharacterEmbeddings(),
			]
		if self._lang_id == '11':
			embedding_types = [
			    WordEmbeddings('sv'),
			    bert_embedding,
			    FlairEmbeddings('swedish-forward'),
			    FlairEmbeddings('swedish-backward'),
			    CharacterEmbeddings(),
			]
		if self._lang_id == '12':
			embedding_types = [
			    WordEmbeddings('da'),
			    bert_embedding,
			    CharacterEmbeddings(),
			]
		if self._lang_id == '13':
			embedding_types = [
			    WordEmbeddings('cz'),
			    bert_embedding,
			    FlairEmbeddings('czech-forward'),
			    FlairEmbeddings('czech-backward'),
			    CharacterEmbeddings(),
			]
		if self._lang_id == '14':
			embedding_types = [
			    bert_embedding,
			    
			    CharacterEmbeddings(),
			]
		if self._lang_id == '16':
			embedding_types = [
			    WordEmbeddings('ja'),
			    bert_embedding,
			    CharacterEmbeddings(),
			]
		if self._lang_id == '17':
			embedding_types = [
			    WordEmbeddings('zh'),
			    bert_embedding,
			    CharacterEmbeddings(),
			]
		if self._lang_id == '18':
			embedding_types = [
			    WordEmbeddings('pl'),
			    bert_embedding,
			    FlairEmbeddings('polish-forward'),
			    FlairEmbeddings('polish-backward'),
			    CharacterEmbeddings(),
			]
		if self._lang_id == '19':
			embedding_types = [
			    bert_embedding,
			    CharacterEmbeddings(),
			]

		
		if self._lang_id == '20':
			embedding_types.append(WordEmbeddings('ru'))

		if self._lang_id == '21':
			embedding_types = [
			    WordEmbeddings('tr'),
			    bert_embedding,
			    CharacterEmbeddings(),
			]
		if self._lang_id == '22':
			embedding_types = [
			    WordEmbeddings('ar'),
			    bert_embedding,
			    CharacterEmbeddings(),
			]
		if self._lang_id == '24':
			embedding_types = [
			    WordEmbeddings('ko'),
			    bert_embedding,
			    CharacterEmbeddings(),
			]
		if self._lang_id == '25':
			embedding_types = [
			    WordEmbeddings('he'),
			    bert_embedding,
			    CharacterEmbeddings(),
			]
		if self._lang_id == '29':
			embedding_types = [
			    bert_embedding,
			    CharacterEmbeddings(),
			]
		if self._lang_id == '31':
			embedding_types = [
			    WordEmbeddings('pt'),
			    bert_embedding,
			    elmo_embedding,
			    FlairEmbeddings('portuguese-forward'),
			    FlairEmbeddings('portuguese-backward'),
			    CharacterEmbeddings(),
			]

		embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

		# initialize sequence tagger
		tagger: SequenceTagger = SequenceTagger(hidden_size=1024,
                                        embeddings=embeddings,
                                        tag_dictionary=tag_dictionary,
                                        tag_type=tag_type,
					rnn_layers = 2,
					dropout = 0.3,
                                        use_crf=True)

		# initialize trainer
    
		trainer: ModelTrainer = ModelTrainer(tagger, corpus)
		if from_checkpoint:
			trainer = ModelTrainer.load_from_checkpoint(Path(self._best_model_path + '/checkpoint.pt'), 'SequenceTagger', corpus)

		trainer.train(self._best_model_path, learning_rate=0.1, mini_batch_size=16,
				max_epochs=10, embeddings_in_memory=emb_in_memory, checkpoint=True, save_final_model=True)

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
			#pr = sentence.to_dict(tag_type='ner')
			#print(pr)
			#for ent in pr['entities']:
			#	spans.append((ent['text'], ent['type'], str(int(ent['confidence']*100))))
			
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



def process(gettext, lang_id, acc_type, variable_system):
	work_url = str("http://travel-cms.fabrica.net.ua/syntexts/");
	work_page = "external_paragraph_job";

	
	
	for it in range(0, 1000):
		url = work_url + '?page=' + work_page + '&format=json&lang_id=' + lang_id + '&acc_type_id=' + acc_type + "&version=" + gettext._model_version
		if variable_system == '2':
			url += '&source_type=4'
		else:
			url += '&source_type=3'
		print(url)
		r = requests.get(url=url)
		data = r.json()
		nprocess_count = data["count"]
		print('ncount:', nprocess_count)
		if int(nprocess_count) == 0:
			return;
		train = pd.DataFrame.from_records(data["job"]["sentences"])
		print(train.head())
		spans = gettext.predict(train["Toks"]);
		items = []
		for i, span in enumerate(spans):
			data_str = {'page': work_page, 'format': 'json', 'action': 'saveone', 'version': gettext._model_version, 'id': train["ID"][i], 'market_text' : '|'.join([v[0] + "_" + v[1] + '(' + v[2] + ')' for v in span])}
			r = requests.post(work_url, data=data_str)
			print('|'.join([v[0] + "_" + v[1] + '(' + v[2] + ')' for v in span]))


def main():
	try:
		opts, args = getopt.getopt(sys.argv[1:],"l:t:a:g:s:p:c:",["lang-id=", "acc-type=", "action=", "gpu=", "variable-system=", "from-checkpoint="])
	except getopt.GetoptError:
		print('errorparams')

	lang_id  = '29'
	acc_type = '1'
	action   = 'process'
	gpu = 1
	variable_system = '1'
	from_checkpoint = False
	for opt, arg in opts:
		if opt in ["-l", "--lang-id"]:
			lang_id = arg;
		elif opt in ["-t", "--acc-type"]:
			acc_type = arg;
		elif opt in ["-a", "--action"]:
			action = arg;
		elif opt in ["-g", "--gpu"]:
			gpu = arg;
		elif opt in ["-s", "--variable-system"]:
			variable_system = arg;
		elif opt in ["-c", "--from-checkpoint"]:
			if int(arg) > 0:
				from_checkpoint = True

	print(variable_system)
	gettext = ExternalMarkup(lang_id = lang_id, acc_type = acc_type, gpu = gpu, variable_system = variable_system)
	gettext.initModel();

	if action == 'train':
		gettext.train(from_checkpoint)
	else:
		gettext.loadModel()
		process(gettext, lang_id, acc_type, variable_system)


if __name__ == "__main__":
	main()