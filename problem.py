
# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import rampwf as rw
import h5py
import matplotlib.pyplot as plt

import os.path
import sys

from sklearn.model_selection import StratifiedShuffleSplit,ShuffleSplit
from sklearn.metrics import accuracy_score, recall_score

from rampwf.score_types.base import BaseScoreType

from rampwf.workflows import FeatureExtractorClassifier

#-----------------------------------------------------------------------
problem_title = 'Side Channel Attack Challenge'
_prediction_label_names = [i for i in range(256)]
# A type (class) which will be used to create wrapper objects for y_pred






def _myInit(self, y_pred=None, y_true=None, n_samples=None):#we have to do that to let pass the y_true
	if y_pred is not None:
		self.y_pred = np.array(y_pred)
		self.check_y_pred_dimensions()#bypass thisone 
	elif y_true is not None:
		self.y_pred = np.array(y_true)
	elif n_samples is not None:
		self.y_pred = np.empty((n_samples, self.n_columns), dtype=float)
		self.y_pred.fill(np.nan)
		self.check_y_pred_dimensions()#bypass thisone 

	else:
		raise ValueError(
			'Missing init argument: y_pred, y_true, or n_samples')



def make_multiclass(label_names=[]):
	Predictions = type(
		'Predictions',
		(rw.prediction_types.make_multiclass(_prediction_label_names),),
		{
		 '__init__': _myInit,
		})
	return Predictions





Predictions = make_multiclass(_prediction_label_names)


#Predictions = rw.prediction_types.make_multiclass(_prediction_label_names)

# An object implementing the workflow

class OurFeClf(FeatureExtractorClassifier):
	"""docstring for OurFeClf"""
	def __init__(self, workflow_element_names=['feature_extractor', 'classifier']):
		super(OurFeClf, self).__init__(workflow_element_names)

	def train_submission(self, module_path, X_df, y_array, train_is=None):
		if train_is is None:
			train_is = slice(None, None, None)
		#y_array2 = y_array[:,0] 
		fe = self.feature_extractor_workflow.train_submission(module_path, X_df, y_array[:,0] , train_is)
		X_train_array = self.feature_extractor_workflow.test_submission(fe, X_df.iloc[train_is])
		clf = self.classifier_workflow.train_submission(module_path, X_train_array, y_array[train_is,0])
		return fe, clf

	def test_submission(self, trained_model, X_df):
		fe, clf = trained_model
		X_test_array = self.feature_extractor_workflow.test_submission(
			fe, X_df)
		y_proba = self.classifier_workflow.test_submission(clf, X_test_array)
		return y_proba


workflow = OurFeClf()

#-----------------------------------------------------------------------
# Define custom score metrics for the churner class
class AUTR(BaseScoreType):
	def __init__(self, name='autr'):
		self.name = name
		self.accuracy = None
		self.rankings = None
		self.max_autr = 256 * 1000
		self.precision = 2
		self.is_lower_the_better = True 
		self.maximum = 1.0
		self.minimum = 0.0
		self.worse = 1.0


	def score_function(self, ground_truths, predictions, valid_indexes=None):
		if valid_indexes is None:
			valid_indexes = slice(None, None, None)
		y_true = ground_truths.y_pred[valid_indexes]
		#print(ground_truths)
		y_pred = predictions.y_pred[valid_indexes]
		return self.__call__(y_true, y_pred)


	def __call__(self, y_true, y_pred):
		#(_, Metadata_attack) = load_ascad( os.path.join(".", 'data', _file),load_metadata=True)[2]
		autr_score, self.rankings = rannking(y_pred, y_true, len(y_true))
		#self.accuracy = accuracy_score(y_true, np.argmax(y_pred, axis=1))
		return autr_score/self.max_autr


score = AUTR()

score_types = [score]

#-----------------------------------------------------------------------
def get_cv(X, y):
	"""Returns stratified randomized folds."""
	cv = ShuffleSplit(n_splits=10, test_size=0.25, random_state=57)
	#cv =ShuffleSplit(n_splits=5)
	k= cv.split(X,y)
	return k


def _read_data(path, filename = "ASCAD.h5"):
	pass

_file = "ASCAD.h5"


def concat(y, metadata):
	return np.stack((y, metadata["key"][:,2] ,metadata["plaintext"][:,2]), axis=-1)

def get_train_data(path='.'):
	(X, y), _, (meta,_) =  load_ascad(os.path.join(path, 'data', _file),load_metadata=True)
	return pd.DataFrame(X), concat(y, meta)

def get_test_data(path='.'):
	_, (X_test, y_test) , (_,meta) = load_ascad( os.path.join(path, 'data', _file), load_metadata=True)
	return pd.DataFrame(X_test), concat(y_test,meta)

AES_Sbox = np.array([
			0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
			0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
			0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
			0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
			0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
			0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
			0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
			0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
			0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
			0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
			0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
			0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
			0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
			0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
			0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
			0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16
			])



def check_file_exists(file_path):
	if os.path.exists(file_path) == False:
		print("Error: provided file path '%s' does not exist!" % file_path)
		sys.exit(-1)
	return

# Compute the rank of the real key for a give set of predictions

def load_ascad(ascad_database_file, load_metadata=False):
	check_file_exists(ascad_database_file)
	# Open the ASCAD database HDF5 for reading
	try:
		in_file  = h5py.File(ascad_database_file, "r")
	except:
		print("Error: can't open HDF5 file '%s' for reading (it might be malformed) ..." % ascad_database_file)
		sys.exit(-1)
	# Load profiling traces
	X_profiling = np.array(in_file['Profiling_traces/traces'], dtype=np.int8)
	# Load profiling labels
	Y_profiling = np.array(in_file['Profiling_traces/labels'])
	# Load attacking traces
	X_attack = np.array(in_file['Attack_traces/traces'], dtype=np.int8)
	# Load attacking labels
	Y_attack = np.array(in_file['Attack_traces/labels'])
	if load_metadata == False:
		return (X_profiling, Y_profiling), (X_attack, Y_attack)
	else:
		return (X_profiling, Y_profiling), (X_attack, Y_attack), (in_file['Profiling_traces/metadata'], in_file['Attack_traces/metadata'])


def rannking(predictions, metadata, num_traces=2000):
	# Load profiling and attack data and metadata from the ASCAD database
	# Load model
	# We test the rank over traces of the Attack dataset, with a step of 10 traces
	ranks = full_ranks(predictions, metadata, 0, num_traces, 10)
	return sum([ranks[i][1] for i in range(0, ranks.shape[0])]), ranks


def full_ranks(predictions, metadata, min_trace_idx =0 , max_trace_idx=200, rank_step=10):
	# Real key byte value that we will use. '2' is the index of the byte (third byte) of interest.
	real_key = metadata[0][1]
	# Check for overflow

	index = np.arange(min_trace_idx+rank_step, max_trace_idx, rank_step)
	f_ranks = np.zeros((len(index), 2), dtype=np.uint32)
	key_bytes_proba = []
	for t, i in zip(index, range(0, len(index))):
		real_key_rank, key_bytes_proba = rank(predictions[t-rank_step:t], metadata, real_key, t-rank_step, t, key_bytes_proba)
		f_ranks[i] = [t - min_trace_idx, real_key_rank]
	return f_ranks


def rank(predictions, metadata, real_key, min_trace_idx, max_trace_idx, last_key_bytes_proba):
	# Compute the rank
	if len(last_key_bytes_proba) == 0:
		key_bytes_proba = np.zeros(256)
	else:
		# This is not the first rank we compute: we optimize things by using the
		# previous computations to save time!
		key_bytes_proba = last_key_bytes_proba

	for p in range(0, max_trace_idx-min_trace_idx):
		# Go back from the class to the key byte. '2' is the index of the byte (third byte) of interest.
		plaintext = metadata[min_trace_idx + p][2]
		for i in range(0, 256):
			# Our candidate key byte probability is the sum of the predictions logs
			proba = predictions[p][AES_Sbox[int(plaintext) ^ i]]
			if proba != 0:
				key_bytes_proba[i] += np.log(proba)
			else:
				# We do not want an -inf here, put a very small epsilon
				# that correspondis to a power of our min non zero proba
				min_proba_predictions = predictions[p][np.array(predictions[p]) != 0]
				if len(min_proba_predictions) == 0:
					print("Error: got a prediction with only zeroes ... this should not happen!")
					sys.exit(-1)
				min_proba = min(min_proba_predictions)
				key_bytes_proba[i] += np.log(min_proba/2)
	# Now we find where our real key candidate lies in the estimation.
	# We do this by sorting our estimates and find the rank in the sorted array.
	sorted_proba = np.array(list(map(lambda a : key_bytes_proba[a], key_bytes_proba.argsort()[::-1])))
	real_key_rank = np.where(sorted_proba == key_bytes_proba[real_key])[-1][-1]
	#print(np.where(sorted_proba == key_bytes_proba[real_key]))
	#print("aa")
	return (real_key_rank, key_bytes_proba)
