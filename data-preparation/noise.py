import numpy
import nltk
import nltk.data

def noise_randomDGP(orig_data_2d, fractions, separate, nc_dist):
	"""
	Args
		orig_data_2d (numpy array): Nx2 array of clean (text, summ) tuples
		fractions (dict): dict where keys are the different noise types and values are the respective fractions of the noise types
							vis a vis the total noise
		N_noise (int): integer specifying the number of noise points
		separate (bool): whether to separate noise creation data from clean train data
		nc_dist ((float, float)): (clean_ratio, noise_ratio) tuple describing the desired noise-clean 
									distribution in the output dataset, sum(nc_dist) = 1
	Return
		noise_data_2d (numpy array): Nx2 array of noise (text, summ) tuples
	"""
	###get all sentences and all words###
	N = orig_data_2d.shape[0]
	sentences = []
	summ_sentences = []
	text_sentences = []
	text_words = []
	summ_words = []
	all_words = []
	removes = []
	sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
	for i in range(N):
		#sentences += sent_detector.tokenize(orig_data_2d[i,0].strip())
		#sentences += sent_detector.tokenize(orig_data_2d[i,1].strip())
		text_list = sent_detector.tokenize(orig_data_2d[i,0].strip())
		summ_list = sent_detector.tokenize(orig_data_2d[i,1].strip())
		text_words.append(orig_data_2d[i,0].split(" "))
		summ_words.append(orig_data_2d[i,1].split(" "))
		removes.append(summ_list[-1])
		summ_list = summ_list[0:-1]
		new_summ_list = []
		if len(summ_list) > 0:
			for j in range(len(summ_list)):
				if j == 0:
					new_summ_list.append(summ_list[j]+" </s>")
				else:
					new_summ_list.append(summ_list[j][5:]+" </s>")
		else:
			new_summ_list.append(orig_data_2d[i,1])
		new_text_list = []
		for sentence in text_list:
			new_text_list.append("<s> "+sentence+" </s> ")
		sentences += new_text_list + new_summ_list
		summ_sentences.append(new_summ_list)
		text_sentences.append(new_text_list)
	all_words = summ_words + text_words
	all_words = set([item for sublist in all_words for item in sublist])
	all_words.discard('<s>')
	all_words.discard('</s>')
	all_words = list(all_words)
	
	if separate:
		N_clean = orig_data_2d.shape[0]
		clean_ratio, noise_ratio = nc_dist
		N_noise = int((N_clean*noise_ratio)) #calculate N_noise
		clean_full_2d = numpy.copy(orig_data_2d) #keep copy of original clean data
		delete_mask = numpy.random.choice(N_clean, size=N_noise, replace=False) # select indices for noise generation pool
		orig_data_2d = clean_full_2d[delete_mask] # reduce orig data pool for noise generation to disjunct subset of orig data
		#TODO: is there an issue with splitting the clean data this way?
		
		###noise: switch pairs###
		rel_fact = fractions["switch-pairs"]
		N = orig_data_2d.shape[0]
		N_noise1 = int(N_noise*rel_fact)
		noise_data1_2d = numpy.ndarray((N_noise1,2), dtype=object)
		#get regular indices and randomized indices
		if type(rel_fact) == int:
			indices_in_order = numpy.array(list(range(N))*rel_fact)
		if type(rel_fact) == float:
			start = int(rel_fact//1)
			indices_in_order = list(range(N))*start
			on_top = N_noise1 - len(indices_in_order)
			#relative_index = int((N*(rel_fact%1))/100)
			indices_in_order += list(range(N))[0:on_top]
		random_indices = numpy.random.choice(list(range(N)), size=N_noise1, replace=True)
		#quick fix noise indices that would create signal data
		#TODO: make this noise fixing truly random
		random_indices[indices_in_order == random_indices] += 1
		random_indices[random_indices > N] -= 2
		#fill noised array
		noise_data1_2d[:,0] = orig_data_2d[indices_in_order,0]
		noise_data1_2d[:,1] = orig_data_2d[random_indices,1]
	
		###noise: sentence switch (entire bank)###
		rel_fact = fractions["sentence-switch-entire-bank"]
		N = orig_data_2d.shape[0]
		N_noise2 = int(N_noise*rel_fact)
		#initialize noise 2d array
		noise_data2_2d = numpy.ndarray((N_noise2,2), dtype=object)
		#get total number of sentences
		num_sentences = len(sentences)
		#track avg number of changes across all changes
		avg_num_changes = 0
		for i in range(N_noise2): #iterate through the number of noisy points we want to create
			textID = numpy.random.randint(0,N)
			#summID = numpy.random.randint(0,N)
			summID = textID
			len_summ = len(summ_sentences[summID])
			if len_summ == 0:
				print('summID: ', summID)
				print('textID: ', textID)
				print(orig_data_2d[summID,1])
				print(summ_sentences[summID])
			#pick number of changes
			num_changes = numpy.random.randint(1,len_summ+1)
			avg_num_changes += num_changes/N_noise2
			#pick positions for changes
			pos_changes = numpy.random.randint(0,len_summ, size=num_changes)
			old_summ = summ_sentences[summID][:]
			#new_summ = ""
			for j in pos_changes:
				old_summ[j] = sentences[numpy.random.randint(0,num_sentences)]
			new_summ = " ".join(old_summ)
			if textID == summID:
				if old_summ == new_summ:
					print('ouch')
			noise_data2_2d[i,0] = orig_data_2d[textID,0]
			noise_data2_2d[i,1] = new_summ
	
		###noise: sentence switch (same text bank)###
		rel_fact = fractions["sentence-switch-same-text-bank"]
		N = orig_data_2d.shape[0]
		N_noise3 = int(N_noise*rel_fact)
		#initialize noise 2d array
		noise_data3_2d = numpy.ndarray((N_noise3,2), dtype=object)
		#track avg number of changes across all changes
		avg_num_changes = 0
		for i in range(N_noise3): #iterate through the number of noisy points we want to create
			textID = numpy.random.randint(0,N)
			#summID = numpy.random.randint(0,N)
			summID = textID
			len_summ = len(summ_sentences[summID])
			#get total number of sentences in the corresponding text
			num_sentences = len(text_sentences[textID])
			if len_summ == 0:
				print('summID: ', summID)
				print('textID: ', textID)
				print(orig_data_2d[summID,1])
				print(summ_sentences[summID])
			#pick number of changes
			num_changes = numpy.random.randint(1,len_summ+1)
			avg_num_changes += num_changes/N_noise3
			#pick positions for changes
			pos_changes = numpy.random.randint(0,len_summ, size=num_changes)
			old_summ = summ_sentences[summID][:]
			#new_summ = ""
			for j in pos_changes:
				old_summ[j] = text_sentences[textID][numpy.random.randint(0,num_sentences)]
			new_summ = " ".join(old_summ)
			if textID == summID:
				if old_summ == new_summ:
					print('ouch')
			noise_data3_2d[i,0] = orig_data_2d[textID,0]
			noise_data3_2d[i,1] = new_summ
		
		###noise: word switch (entire bank)###
		rel_fact = fractions["word-switch-entire-bank"]
		N = orig_data_2d.shape[0]
		N_noise4 = int(N_noise*rel_fact)
		#initialize noise 2d array
		noise_data4_2d = numpy.ndarray((N_noise4,2), dtype=object)
		#get total number of sentences
		num_sentences = len(sentences)
		num_words = len(all_words)
		#track avg number of changes across all changes
		avg_num_changes = 0
		for i in range(N_noise4): #iterate through the number of noisy points we want to create
			textID = numpy.random.randint(0,N)
			#summID = numpy.random.randint(0,N)
			summID = textID
			len_summ = len(summ_words[summID])
			if len_summ == 0:
				print('summID: ', summID)
				print('textID: ', textID)
				print(orig_data_2d[summID,1])
				print(summ_sentences[summID])
			#pick number of changes
			num_changes = numpy.random.randint(1,len_summ+1)
			avg_num_changes += num_changes/N_noise4
			#pick positions for changes
			pos_changes = numpy.random.randint(0,len_summ, size=num_changes)
			old_summ = summ_words[summID][:]
			#new_summ = ""
			for j in pos_changes:
				if old_summ[j] == '<s>' or old_summ[j] == '</s>':
					pass
				else:
					old_summ[j] = all_words[numpy.random.randint(0,num_words)]
			new_summ = " ".join(old_summ)
			if textID == summID:
				if old_summ == new_summ:
					print('ouch')
			noise_data4_2d[i,0] = orig_data_2d[textID,0]
			noise_data4_2d[i,1] = new_summ
	
		###combine all noise into one###
		N_noise = N_noise1 + N_noise2 + N_noise3 + N_noise4
		noise_data_2d = numpy.concatenate((noise_data1_2d,noise_data2_2d,noise_data3_2d, noise_data4_2d))
	
		###clean out <s> and </s>
		#preprocess clean data, i.e. remove <s> and </s>
		for i in range(N_noise):
			noise_data_2d[i,1] = noise_data_2d[i,1].replace('<s> ', '')
			noise_data_2d[i,1] = noise_data_2d[i,1].replace(' </s>', '')
		
		clean_data_2d = numpy.delete(clean_full_2d, delete_mask, axis=0)
		return clean_data_2d, noise_data_2d
	
	else:
		N_clean = orig_data_2d.shape[0]
		clean_ratio, noise_ratio = nc_dist
		N_noise = int((N_clean - N_clean*clean_ratio)/clean_ratio)
	
	
		###noise: switch pairs###
		rel_fact = fractions["switch-pairs"]
		N = orig_data_2d.shape[0]
		N_noise1 = int(N_noise*rel_fact)
		noise_data1_2d = numpy.ndarray((N_noise1,2), dtype=object)
		#get regular indices and randomized indices
		if type(rel_fact) == int:
			indices_in_order = numpy.array(list(range(N))*rel_fact)
		if type(rel_fact) == float:
			start = int(rel_fact//1)
			indices_in_order = list(range(N))*start
			on_top = N_noise1 - len(indices_in_order)
			#relative_index = int((N*(rel_fact%1))/100)
			indices_in_order += list(range(N))[0:on_top]
		random_indices = numpy.random.choice(list(range(N)), size=N_noise1, replace=True)
		#quick fix noise indices that would create signal data
		#TODO: make this noise fixing truly random
		#print(len(indices_in_order))
		#print(random_indices.shape)
		#print(orig_data_2d.shape)
		#print(noise_data1_2d.shape)
		
		random_indices[indices_in_order == random_indices] += 1
		random_indices[random_indices > N] -= 2
		#fill noised array
		noise_data1_2d[:,0] = orig_data_2d[indices_in_order,0]
		noise_data1_2d[:,1] = orig_data_2d[random_indices,1]
	
		###noise: sentence switch (entire bank)###
		rel_fact = fractions["sentence-switch-entire-bank"]
		N = orig_data_2d.shape[0]
		N_noise2 = int(N_noise*rel_fact)
		#initialize noise 2d array
		noise_data2_2d = numpy.ndarray((N_noise2,2), dtype=object)
		#get total number of sentences
		num_sentences = len(sentences)
		#track avg number of changes across all changes
		avg_num_changes = 0
		for i in range(N_noise2): #iterate through the number of noisy points we want to create
			textID = numpy.random.randint(0,N)
			#summID = numpy.random.randint(0,N)
			summID = textID
			len_summ = len(summ_sentences[summID])
			if len_summ == 0:
				print('summID: ', summID)
				print('textID: ', textID)
				print(orig_data_2d[summID,1])
				print(summ_sentences[summID])
			#pick number of changes
			num_changes = numpy.random.randint(1,len_summ+1)
			avg_num_changes += num_changes/N_noise2
			#pick positions for changes
			pos_changes = numpy.random.randint(0,len_summ, size=num_changes)
			old_summ = summ_sentences[summID][:]
			#new_summ = ""
			for j in pos_changes:
				old_summ[j] = sentences[numpy.random.randint(0,num_sentences)]
			new_summ = " ".join(old_summ)
			if textID == summID:
				if old_summ == new_summ:
					print('ouch')
			noise_data2_2d[i,0] = orig_data_2d[textID,0]
			noise_data2_2d[i,1] = new_summ
	
		###noise: sentence switch (same text bank)###
		rel_fact = fractions["sentence-switch-same-text-bank"]
		N = orig_data_2d.shape[0]
		N_noise3 = int(N_noise*rel_fact)
		#initialize noise 2d array
		noise_data3_2d = numpy.ndarray((N_noise3,2), dtype=object)
		#track avg number of changes across all changes
		avg_num_changes = 0
		for i in range(N_noise3): #iterate through the number of noisy points we want to create
			textID = numpy.random.randint(0,N)
			#summID = numpy.random.randint(0,N)
			summID = textID
			len_summ = len(summ_sentences[summID])
			#get total number of sentences in the corresponding text
			num_sentences = len(text_sentences[textID])
			if len_summ == 0:
				print('summID: ', summID)
				print('textID: ', textID)
				print(orig_data_2d[summID,1])
				print(summ_sentences[summID])
			#pick number of changes
			num_changes = numpy.random.randint(1,len_summ+1)
			avg_num_changes += num_changes/N_noise3
			#pick positions for changes
			pos_changes = numpy.random.randint(0,len_summ, size=num_changes)
			old_summ = summ_sentences[summID][:]
			#new_summ = ""
			for j in pos_changes:
				old_summ[j] = text_sentences[textID][numpy.random.randint(0,num_sentences)]
			new_summ = " ".join(old_summ)
			if textID == summID:
				if old_summ == new_summ:
					print('ouch')
			noise_data3_2d[i,0] = orig_data_2d[textID,0]
			noise_data3_2d[i,1] = new_summ
		
		###noise: word switch (entire bank)###
		rel_fact = fractions["word-switch-entire-bank"]
		N = orig_data_2d.shape[0]
		N_noise4 = int(N_noise*rel_fact)
		#initialize noise 2d array
		noise_data4_2d = numpy.ndarray((N_noise4,2), dtype=object)
		#get total number of sentences
		num_sentences = len(sentences)
		num_words = len(all_words)
		#track avg number of changes across all changes
		avg_num_changes = 0
		for i in range(N_noise4): #iterate through the number of noisy points we want to create
			textID = numpy.random.randint(0,N)
			#summID = numpy.random.randint(0,N)
			summID = textID
			len_summ = len(summ_words[summID])
			if len_summ == 0:
				print('summID: ', summID)
				print('textID: ', textID)
				print(orig_data_2d[summID,1])
				print(summ_sentences[summID])
			#pick number of changes
			num_changes = numpy.random.randint(1,len_summ+1)
			avg_num_changes += num_changes/N_noise4
			#pick positions for changes
			pos_changes = numpy.random.randint(0,len_summ, size=num_changes)
			old_summ = summ_words[summID][:]
			#new_summ = ""
			for j in pos_changes:
				if old_summ[j] == '<s>' or old_summ[j] == '</s>':
					pass
				else:
					old_summ[j] = all_words[numpy.random.randint(0,num_words)]
			new_summ = " ".join(old_summ)
			if textID == summID:
				if old_summ == new_summ:
					print('ouch')
			noise_data4_2d[i,0] = orig_data_2d[textID,0]
			noise_data4_2d[i,1] = new_summ
	
		###combine all noise into one###
		N_noise = N_noise1 + N_noise2 + N_noise3 + N_noise4
		noise_data_2d = numpy.concatenate((noise_data1_2d,noise_data2_2d,noise_data3_2d, noise_data4_2d))
	
		###clean out <s> and </s>
		#preprocess clean data, i.e. remove <s> and </s>
		for i in range(N_noise):
			noise_data_2d[i,1] = noise_data_2d[i,1].replace('<s> ', '')
			noise_data_2d[i,1] = noise_data_2d[i,1].replace(' </s>', '')
		return orig_data_2d, noise_data_2d
