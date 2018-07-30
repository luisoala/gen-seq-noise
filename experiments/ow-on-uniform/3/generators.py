"""
This code is adapted from the batch generator example of Shervine Amidi at https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html
"""

import numpy
np = numpy
import keras
from keras.preprocessing import text, sequence

class ContAllGenerator(keras.utils.Sequence):
	'Generates data for Keras'
	def __init__(self, list_IDs, labels, tokenizer, embedding_matrix, maxlen_text, maxlen_summ, batch_size, dim,
				shuffle, data_dir, sample_info):
		'Initialization'
		self.dim = dim
		self.batch_size = batch_size
		self.labels = labels
		self.list_IDs = list_IDs
		self.shuffle = shuffle
		self.on_epoch_end()
		
		self.tokenizer = tokenizer
		self.embedding_matrix = embedding_matrix
		self.maxlen_text = maxlen_text
		self.maxlen_summ = maxlen_summ
		self.data_dir = data_dir

	def __len__(self):
		'Denotes the number of batches per epoch'
		return int(np.floor(len(self.list_IDs) / self.batch_size))

	def __getitem__(self, index):
		'Generate one batch of data'
		# Generate indexes of the batch
		indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

		# Find list of IDs
		list_IDs_temp = [self.list_IDs[k] for k in indexes]

		# Generate data
		X, y = self.__data_generation(list_IDs_temp)

		return X, y

	def on_epoch_end(self):
		'Updates indexes after each epoch'
		self.indexes = np.arange(len(self.list_IDs))
		if self.shuffle == True:
			np.random.shuffle(self.indexes)

	def __data_generation(self, list_IDs_temp):
		'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
		# Initialization
		X_one = np.empty((self.batch_size, *self.dim[0]))
		X_two = np.empty((self.batch_size, *self.dim[1]))
		y = np.empty((self.batch_size), dtype=int)

		# Generate data
		for i, ID in enumerate(list_IDs_temp):
			# Store sample
			
			with open(self.data_dir + ID, 'r') as file:
				data_point = file.read()
			
			text, summ = data_point.split('\n')
			
			text = self.tokenizer.texts_to_sequences(numpy.array([text], dtype=object))
			summ = self.tokenizer.texts_to_sequences(numpy.array([summ], dtype=object))
			
			text = sequence.pad_sequences(text, maxlen=self.maxlen_text, truncating = 'post', padding = 'pre')
			summ = sequence.pad_sequences(summ, maxlen=self.maxlen_summ, truncating = 'post', padding = 'post')
			
			X_one[i] = self.embedding_matrix[text[numpy.newaxis,:,:]]
			X_two[i] = self.embedding_matrix[summ[numpy.newaxis,:,:]]
			
			# Store class
			y[i] = self.labels[ID]

		return [X_one, X_two], y

class ThreeQuartGenerator(keras.utils.Sequence):
	'Generates data for Keras'
	def __init__(self, list_IDs, labels, tokenizer, embedding_matrix, maxlen_text, maxlen_summ, batch_size, dim,
				shuffle, data_dir, sample_info):
		'Initialization'
		self.dim = dim
		self.batch_size = batch_size
		self.labels = labels
		self.list_IDs = list_IDs
		self.shuffle = shuffle
		self.on_epoch_end()
		
		self.tokenizer = tokenizer
		self.embedding_matrix = embedding_matrix
		_, self.embedding_dim = embedding_matrix.shape
		self.maxlen_text = maxlen_text
		self.maxlen_summ = maxlen_summ
		self.sample_func, self.stat_A, self.stat_B = sample_info #sample is tuple (sample_func, stat_A, stat_B)
		#if sample_func = numpy.random.uniform stat_A = min and stat_B = max
		#if sample_func = numpy.random.normal stat_A = loc and stat_B = scale (the standard deviations)
		self.data_dir = data_dir

	def __len__(self):
		'Denotes the number of batches per epoch'
		return int(np.floor(len(self.list_IDs) / self.batch_size))

	def __getitem__(self, index):
		'Generate one batch of data'
		# Generate indexes of the batch
		indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

		# Find list of IDs
		list_IDs_temp = [self.list_IDs[k] for k in indexes]

		# Generate data
		X, y = self.__data_generation(list_IDs_temp)

		return X, y

	def on_epoch_end(self):
		'Updates indexes after each epoch'
		self.indexes = np.arange(len(self.list_IDs))
		if self.shuffle == True:
			np.random.shuffle(self.indexes)

	def __data_generation(self, list_IDs_temp):
		'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
		# Initialization
		X_one = np.empty((self.batch_size, *self.dim[0]))
		X_two = np.empty((self.batch_size, *self.dim[1]))
		y = np.empty((self.batch_size), dtype=int)

		# Generate data
		for i, ID in enumerate(list_IDs_temp):
			# Store sample
			
			with open(self.data_dir + ID, 'r') as file:
				data_point = file.read()
			
			text, summ = data_point.split('\n')
			
			# Store class
			y[i] = self.labels[ID]
			
			if y[i] == 1:
				text = self.tokenizer.texts_to_sequences(numpy.array([text], dtype=object))
				summ = self.tokenizer.texts_to_sequences(numpy.array([summ], dtype=object))
			
				text = sequence.pad_sequences(text, maxlen=self.maxlen_text, truncating = 'post', padding = 'pre')
				summ = sequence.pad_sequences(summ, maxlen=self.maxlen_summ, truncating = 'post', padding = 'post')
				
				X_one[i] = self.embedding_matrix[text[numpy.newaxis,:,:]]
				X_two[i] = self.embedding_matrix[summ[numpy.newaxis,:,:]]
			
			else: #if the current point is a noise point -> we sample from the dist for the summ
				text = self.tokenizer.texts_to_sequences(numpy.array([text], dtype=object))
				
				text = sequence.pad_sequences(text, maxlen=self.maxlen_text, truncating = 'post', padding = 'pre')
			
				X_one[i] = self.embedding_matrix[text[numpy.newaxis,:,:]]
				X_two[i] = self.sample_func(self.stat_A, self.stat_B).reshape((self.maxlen_summ, self.embedding_dim))
			

		return [X_one, X_two], y

class TwoQuartGenerator(keras.utils.Sequence):
	'Generates data for Keras'
	def __init__(self, list_IDs, labels, tokenizer, embedding_matrix, maxlen_text, maxlen_summ, batch_size, dim,
				shuffle, data_dir, sample_info):
		'Initialization'
		self.dim = dim
		self.batch_size = batch_size
		self.labels = labels
		self.list_IDs = list_IDs
		self.shuffle = shuffle
		self.on_epoch_end()
		
		self.tokenizer = tokenizer
		self.embedding_matrix = embedding_matrix
		_, self.embedding_dim = embedding_matrix.shape
		self.maxlen_text = maxlen_text
		self.maxlen_summ = maxlen_summ
		self.sample_func, self.stat_A, self.stat_B = sample_info #sample is tuple (sample_func, stat_A, stat_B)
		#if sample_func = numpy.random.uniform stat_A = min and stat_B = max
		#if sample_func = numpy.random.normal stat_A = loc and stat_B = scale (the standard deviations)
		self.data_dir = data_dir

	def __len__(self):
		'Denotes the number of batches per epoch'
		return int(np.floor(len(self.list_IDs) / self.batch_size))

	def __getitem__(self, index):
		'Generate one batch of data'
		# Generate indexes of the batch
		indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

		# Find list of IDs
		list_IDs_temp = [self.list_IDs[k] for k in indexes]

		# Generate data
		X, y = self.__data_generation(list_IDs_temp)

		return X, y

	def on_epoch_end(self):
		'Updates indexes after each epoch'
		self.indexes = np.arange(len(self.list_IDs))
		if self.shuffle == True:
			np.random.shuffle(self.indexes)

	def __data_generation(self, list_IDs_temp):
		'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
		# Initialization
		X_one = np.empty((self.batch_size, *self.dim[0]))
		X_two = np.empty((self.batch_size, *self.dim[1]))
		y = np.empty((self.batch_size), dtype=int)

		# Generate data
		for i, ID in enumerate(list_IDs_temp):
			# Store class
			y[i] = self.labels[ID]
			
			if y[i] == 1:
				# Store sample
			
				with open(self.data_dir + ID, 'r') as file:
					data_point = file.read()
			
				text, summ = data_point.split('\n')
				
				text = self.tokenizer.texts_to_sequences(numpy.array([text], dtype=object))
				summ = self.tokenizer.texts_to_sequences(numpy.array([summ], dtype=object))
			
				text = sequence.pad_sequences(text, maxlen=self.maxlen_text, truncating = 'post', padding = 'pre')
				summ = sequence.pad_sequences(summ, maxlen=self.maxlen_summ, truncating = 'post', padding = 'post')
				
				X_one[i] = self.embedding_matrix[text[numpy.newaxis,:,:]]
				X_two[i] = self.embedding_matrix[summ[numpy.newaxis,:,:]]
			
			else: #if the current point is a noise point -> we sample from the dist for the text AND summ
				data_point = self.sample_func(self.stat_A, self.stat_B).reshape((self.maxlen_text + self.maxlen_summ, self.embedding_dim))
			
				X_one[i] = data_point[:self.maxlen_text]
				X_two[i] = data_point[self.maxlen_text:]
			

		return [X_one, X_two], y
