# Creates features from reddit comment data

import json
import sys
import nltk
import string
import pickle
import re
import numpy as np
from nltk.tokenize import word_tokenize

# list of smileys and their sentiments, according to https://en.wikipedia.org/wiki/List_of_emoticons
smileys_pos = [":-)",":)",":-]",":]",":-3",":3",":->",":>","8-)","8)",":-}",":}",":o)",":c)",":^)","=]","=)",":-D",":D","8-D","8D","x-D","xD","X-D","XD","=D","=3","B^D",":-))",":'-)",":\')", ":-*",":*",":×",";‑)",";)","*-)","*)",";‑]",";]",";^)",":-,",";D",":‑P",":P","X‑P","XP","x-p","xp",":-p",":p",":-Þ",":Þ",":-þ",":þ",":-b",":b","d:","=p",">:P","O:-)","O:)","0:-3","0:3","0:-)","0:)","0;^)",">:-)",">:)","}:-)","}:)","3:-)","3:)","^_^","(°o°)","(^_^)/","(^O^)／","(^o^)／","(^^)/","(≧∇≦)/","(/◕ヮ◕)/","(^o^)丿","∩(·ω·)∩","(·ω·)","^ω^",">;)","#-)","%-)","<3","\o/","( ͡° ͜ʖ ͡°)","(=^·^=)","(=^··^=)","=_^=","%)",">^_^<","<^!^>","^/^","（*^_^*）","§^.^§","(^<^)","(^.^)","(^ム^)","(^·^)","(^.^)","(^_^.)","(^_^)","(^^)","(^J^)","(*^.^*)","^_^","(#^.^#)","（^—^）"]
smileys_neg = [":-(",":(",":-c",":c",":-<",":<",":-[","(>_<)","(>_<)>",":[",":-||",">:[",":{","<:-|","(ー_ー)!!","(-.-)","(-_-)","(一一)","(；一_一)",":@",">:(",":'-(","(=_=)","(..)","(._.)",":'(","D-':","D:<","D:","D8","D;","D=","DX",":-/",":/",":-.",">:\\",":\\",
"=/","=\\",":L","=L",":S",":-|",":|","v.v","(^^ゞ","(^_^;)","(-_-;)","(~_~;)","(・.・;)","(・_・;)","(・・;)","^^;","^_^;","(#^.^#)","(^^;)",":$",":-X",":X",":-#","('_')","(/_;)","(T_T)","(;_;)","(;_;","(;_:)","(;O;)","(:_;)","(ToT)","(Ｔ▽Ｔ)",";_;",";-;",";n;",";;","Q.Q","T.T","QQ","Q_Q",":#",":-&",":&",":-###..","</3","<\3","O_O","o‑o","O_o","o_O","o_o","O-O",">.<","((+_+))","(+o+)","(°°)","(°-°)","(°.°)","(°_°)","(°_°>)","(°レ°)",":###.."]
smileys_neut = [":-O",":O",":-o",":o",":-0","8-0",">:O","|;-)","|-O",":-J"]

# regexp for standalone numbers
numeral = re.compile('^[-+]?[0-9]+$')

comments = [] # make list of comments

# opening relevant subreddit data
with open('RC_2017-06/exportTIL.json', 'r', encoding="utf-8") as file:
	comments = json.load(file)

def make_pos_tags(tokenized_sentence):
	''' input: tokenized comment as list of tokens
		returns: pos-tagged comment as list of sets (token, tag)'''
	return nltk.pos_tag(tokenized_sentence)

def make_num_alpha(tokenized_sentence):
	'''	input: tokenized comment as list of tokens
		returns: number of alphabetic characters'''
	alphas = 0
	for token in tokenized_sentence:
		for char in token:
			if char.isalpha():
				alphas += 1
	return alphas
	
def make_upper(tokenized_sentence):
	'''	input: tokenized comment as list of tokens
		returns: number of upper case characters'''
	upper = 0
	for token in tokenized_sentence:
		for char in token:
			if char.isupper():
				upper += 1
	return upper
	
def make_digits(tokenized_sentence):
	'''	input: tokenized comment as list of tokens
		returns: number of digits'''
	digits = 0
	for token in tokenized_sentence:
		for char in token:
			if char.isdigit():
				digits += 1
	return digits
	
def make_spaces(string):
	'''	input: tokenized comment as list of tokens
		returns: number of spaces'''
	spaces = 0
	for char in string:
		if char == " " or char == "\n":
			spaces += 1
	return spaces
	
def make_tabs(string):
	'''	input: comment as string
		returns: number of tabs'''
	tabs = 0
	for char in string:
		if char == "\t":
			tabs += 1
	return tabs
	
def make_short_ratio(tokenized_sentence):
	'''	input: tokenized comment as list of tokens
		returns: ratio of tokens <= 4 characters to all tokens'''
	shorts = 0
	for token in tokenized_sentence:
		if token not in string.punctuation:
			if len(token) <= 4:
				shorts += 1
	if len(tokenized_sentence) > 0:
		return float(shorts/len(tokenized_sentence))
	else:
		return 0

def make_min_word_length(tokenized_sentence):
	'''	input: tokenized comment as list of tokens
		returns: length of shortest token'''
	min = 900000000
	for token in tokenized_sentence:
		if token not in string.punctuation:
			if len(token) < min:
				min = len(token)
	return min
	
def make_max_word_length(tokenized_sentence):
	'''	input: tokenized comment as list of tokens
		returns: length of longest token'''
	maximum = 0
	for token in tokenized_sentence:
		if token not in string.punctuation:
			if len(token) < maximum:
				maximum = len(token)
	return maximum
	
def make_hapax_legomena(tokenized_sentence):
	'''	input: tokenized comment as list of tokens
		returns: number of hapax legomena'''
	hapaxes = 0
	words = {}
	for token in tokenized_sentence:
		if token in words:
			words[token] += 1
		else:
			words[token] = 1
	for word in words:
		if words[word] == 1:
			hapaxes += 1
	return hapaxes
	
def make_hapax_dislegomena(tokenized_sentence):
	'''	input: tokenized comment as list of tokens
		returns: number of hapax dislegomena'''
	hapaxes = 0
	words = {}
	for token in tokenized_sentence:
		if token in words:
			words[token] += 1
		else:
			words[token] = 1
	for word in words:
		if words[word] == 2:
			hapaxes += 1
	return hapaxes

def make_nouns(tagged_sentence):
	'''	input: tagged comment as list of sets (token, tag)
		returns: number of nouns'''
	nouns = 0
	for tag in tagged_sentence:
		if tag[1] == 'NN' or tag[1] == 'NNS':
			nouns += 1
	return nouns
	
def make_proper_nouns(tagged_sentence):
	'''	input: tagged comment as list of sets (token, tag)
		returns: number of proper nouns'''
	proper_nouns = 0
	for tag in tagged_sentence:
		if tag[1] == 'NNP' or tag[1] == 'NNPS':
			proper_nouns += 1
	return proper_nouns
	
def make_adjectives(tagged_sentence):
	'''	input: tagged comment as list of sets (token, tag)
		returns: number of adjectives'''
	adjectives = 0
	for tag in tagged_sentence:
		if tag[1] == 'JJ' or tag[1] == 'JJR' or tag[1] == 'JJS':
			adjectives += 1
	return adjectives
	
def make_adverbs(tagged_sentence):
	'''	input: tagged comment as list of sets (token, tag)
		returns: number of adverbs'''
	adverbs = 0
	for tag in tagged_sentence:
		if tag[1] == 'RB' or tag[1] == 'RBR' or tag[1] == 'RBS' or tag[1] == 'WRB':
			adverbs += 1
	return adverbs
	
def make_prepositions(tagged_sentence):
	'''	input: tagged comment as list of sets (token, tag)
		returns: number of prepositions'''
	prepositions = 0
	for tag in tagged_sentence:
		if tag[1] == 'IN':
			prepositions += 1
	return prepositions
	
def make_verbs(tagged_sentence):
	'''	input: tagged comment as list of sets (token, tag)
		returns: number of verbs'''
	verbs = 0
	for tag in tagged_sentence:
		if tag[1] == 'VB' or tag[1] == 'VBD' or tag[1] == 'VBG' or tag[1] == 'VBN' or tag[1] == 'VBP' or tag[1] == 'VBZ':
			verbs += 1
	return verbs

def make_pronouns(tagged_sentence):
	'''	input: tagged comment as list of sets (token, tag)
		returns: number of pronouns'''
	pronouns = 0
	for tag in tagged_sentence:
		if tag[1] == 'PRP' or tag[1] == 'PRP$' or tag[1] == 'WP' or tag[1] == 'WP$':
			pronouns += 1
	return pronouns

def make_interjections(tagged_sentence):
	'''	input: tagged comment as list of sets (token, tag)
		returns: number of interjections'''
	interjections = 0
	for tag in tagged_sentence:
		if tag[1] == 'UH':
			interjections += 1
	return interjections
	
def make_word_length(tokenized_sentence):
	'''	input: tokenized comment as list of tokens
		returns: length of comment in tokens'''
	word_count = 0
	for token in tokenized_sentence:
		if token not in string.punctuation:
			word_count += 1
		else:
			continue
	return word_count
	
def make_punct(tokenized_sentence):
	'''	input: tokenized comment as list of tokens
		returns: number of punctuation symbols in comment'''
	punct_count = 0
	for token in tokenized_sentence:
		if token in string.punctuation:
			punct_count += 1
		else:
			continue
	return punct_count

def make_char_length(string):
	'''	input: tokenized comment as list of tokens
		returns: length of comment in characters'''
	char_count = 0
	for character in string:
		char_count += 1
	return char_count
	
def make_avg_word_length(tokenized_sentence):
	'''	input: tokenized comment as list of tokens
		returns: average word length in comments'''
	word_lengths = []
	for token in tokenized_sentence:
		word_length = 0
		if token not in string.punctuation:
			for character in token:
				word_length += 1
			word_lengths.append(word_length)
		else:
			continue
	if len(word_lengths) > 0:
		avg_length = sum(word_lengths)/float(len(word_lengths))
	else:
		avg_length = 0
	return avg_length
	
def make_long_ratio(tokenized_sentence):
	'''	input: tokenized comment as list of tokens
		returns: ratio of tokens > 5 characters to all tokens'''
	word_lengths = []
	for token in tokenized_sentence:
		word_length = 0
		if token not in string.punctuation:
			for character in token:
				word_length += 1
			word_lengths.append(word_length)
		else:
			continue
	long_words = 0
	for length in word_lengths:
		if length > 5:
			long_words += 1
		else:
			continue
	if len(word_lengths) > 0:
		ratio = long_words/float(len(word_lengths))
	else:
		ratio = 0
	return ratio
	
def make_num_positive_smileys(tokenized_sentence):
	'''	input: tokenized comment as list of tokens
		returns: number of smileys rated as positive'''
	happy = 0
	for token in tokenized_sentence:
		if token in smileys_pos:
			happy += 1
		else:
			continue
	if len(tokenized_sentence) > 0:
		return float(happy/len(tokenized_sentence))
	else:
		return 0

def make_num_negative_smileys(tokenized_sentence):
	'''	input: tokenized comment as list of tokens
		returns: number of smileys rated as negative'''
	sad = 0
	for token in tokenized_sentence:
		if token in smileys_neg:
			sad += 1
		else:
			continue
	if len(tokenized_sentence) > 0:
		return float(sad/len(tokenized_sentence))
	else:
		return 0
	
def make_num_neutral_smileys(tokenized_sentence):
	'''	input: tokenized comment as list of tokens
		returns: number of smileys rated as neutral'''
	meh = 0
	for token in tokenized_sentence:
		if token in smileys_neut:
			meh += 1
		else:
			continue
	if len(tokenized_sentence) > 0:
		return float(meh/len(tokenized_sentence))
	else:
		return 0
		
def make_standalone_number(tokenized_sentence):
	'''	input: tokenized comment as list of tokens
		returns: number of standalone numbers'''
	standalone_numbers = 0
	for token in tokenized_sentence:
		if numeral.match(token):
			standalone_numbers += 1
	if len(tokenized_sentence) > 0:
		return float(standalone_numbers/len(tokenized_sentence))
	else:
		return 0
	
def make_words_to_superlatives_ratio(tagged_sentence):
	'''	input: tagged comment as list of sets (token, tag)
		returns: ratio of superlatives to all words'''
	superlatives = 0
	words = 0
	for item in tagged_sentence:
		words += 1
		if item[1] == 'JJS' or item[1] == 'RBS': # POS tags for adjective, superlative and adverb, superlative
			superlatives += 1
	if words > 0:
		return float(superlatives/words)
	else:
		return 0
	
def make_words_to_plurals_ratio(tagged_sentence):
	'''	input: tagged comment as list of sets (token, tag)
		returns: ratio of plurals to all words'''
	plurals = 0
	words = 0
	for item in tagged_sentence:
		words += 1
		if item[1] == 'NNS' or item[1] == 'NNPS': # POS tags for noun, plural and proper noun, plural
			plurals += 1
	if words > 0:
		return float(plurals/words)
	else:
		return 0
	
def rescaling_features(feature_vector):
	'''	input: vector of features collected over entire dataset
		returns: vector with rescaled features in range 0.0-1.0 for each feature '''
	normalized_vector = []
	maximum = max(feature_vector)
	minimum = min(feature_vector)
	if maximum == 0:
		normalized_vector = feature_vector
	else:
		for i in range(0, len(feature_vector)):
			value = feature_vector[i]
			normalized_vector.append(float((value - minimum)/(maximum - minimum)))
	return normalized_vector

def make_test(comments):
	# text mining
	num_alphas = []
	uppers = []
	digits = []
	spaces = []
	tabs = []
	short_ratios = []
	min_word_lengths = []
	max_word_lengths = []
	hapax_legomenas = []
	hapax_dislegomenas = []
	nouns = []
	proper_nouns = []
	adjectives = []
	adverbs = []
	prepositions = []
	verbs = []
	pronouns = []
	interjections = []
	word_lengths = []
	puncts = []
	char_lengths = []
	avg_word_lengths = []
	
	# style tags
	long_ratios = []
	num_positive_smileyss = []
	num_negative_smileyss = []
	num_neutral_smileyss = []
	standalone_numbers = []
	words_to_superlatives_ratios = []
	words_to_plurals_ratios = []
	
	# metadata
	post_times = []
	gildeds = []
	
	# score
	scores = []

	for i in range(0, len(comments)):
		# tokenizing
		tokens = word_tokenize(comments[i]["body"])
		# pos-tagging
		pos_tags = nltk.pos_tag(tokens)
		
		# text mining
		num_alphas.append(make_num_alpha(tokens))
		uppers.append(make_upper(tokens))
		digits.append(make_digits(tokens))
		spaces.append(make_spaces(comments[i]["body"]))
		tabs.append(make_tabs(comments[i]["body"]))
		short_ratios.append(make_short_ratio(tokens))
		min_word_lengths.append(make_min_word_length(tokens))
		max_word_lengths.append(make_max_word_length(tokens))
		hapax_legomenas.append(make_hapax_legomena(tokens))
		hapax_dislegomenas.append(make_hapax_dislegomena(tokens))
		nouns.append(make_nouns(pos_tags))
		proper_nouns.append(make_proper_nouns(pos_tags))
		adjectives.append(make_adjectives(pos_tags))
		adverbs.append(make_adverbs(pos_tags))
		prepositions.append(make_prepositions(pos_tags))
		verbs.append(make_verbs(pos_tags))
		pronouns.append(make_pronouns(pos_tags))
		interjections.append(make_interjections(pos_tags))
		word_lengths.append(make_word_length(tokens))
		puncts.append(make_punct(tokens))
		char_lengths.append(make_char_length(comments[i]["body"]))
		avg_word_lengths.append(make_avg_word_length(tokens))
		
		# style
		long_ratios.append(make_long_ratio(tokens))
		num_positive_smileyss.append(make_num_positive_smileys(tokens))
		num_negative_smileyss.append(make_num_negative_smileys(tokens))
		num_neutral_smileyss.append(make_num_neutral_smileys(tokens))
		standalone_numbers.append(make_standalone_number(tokens))
		words_to_superlatives_ratios.append(make_words_to_superlatives_ratio(pos_tags))
		words_to_plurals_ratios.append(make_words_to_plurals_ratio(pos_tags))
		
		# meta
		post_times.append(int(comments[i]["created_utc"]))
		gildeds.append(int(comments[i]["gilded"]))
		
		# score
		scores.append(int(comments[i]["score"]))
		print(i)
	# rescaling
	
	num_alphas = rescaling_features(num_alphas)
	uppers = rescaling_features(uppers)
	digits = rescaling_features(digits)
	spaces = rescaling_features(spaces)
	tabs = rescaling_features(tabs)
	short_ratios = rescaling_features(short_ratios)
	min_word_lengths = rescaling_features(min_word_lengths)
	max_word_lengths = rescaling_features(max_word_lengths)
	hapax_legomenas = rescaling_features(hapax_legomenas)
	hapax_dislegomenas = rescaling_features(hapax_dislegomenas)
	nouns = rescaling_features(nouns)
	proper_nouns = rescaling_features(proper_nouns)
	adjectives = rescaling_features(adjectives)
	adverbs = rescaling_features(adverbs)
	prepositions = rescaling_features(prepositions)
	verbs = rescaling_features(verbs)
	pronouns = rescaling_features(pronouns)
	interjections = rescaling_features(interjections)
	word_lengths = rescaling_features(word_lengths)
	puncts = rescaling_features(puncts)
	char_lengths = rescaling_features(char_lengths)
	avg_word_lengths = rescaling_features(avg_word_lengths)
	
	# style tags
	long_ratios = rescaling_features(long_ratios)
	num_positive_smileyss = rescaling_features(num_positive_smileyss)
	num_negative_smileyss = rescaling_features(num_negative_smileyss)
	num_neutral_smileyss = rescaling_features(num_neutral_smileyss)
	standalone_numbers = rescaling_features(standalone_numbers)
	words_to_superlatives_ratios = rescaling_features(words_to_superlatives_ratios)
	words_to_plurals_ratios = rescaling_features(words_to_plurals_ratios)

	super_vector = [scores, num_alphas, uppers, digits, spaces, tabs, short_ratios,	min_word_lengths, max_word_lengths, hapax_legomenas, hapax_dislegomenas, nouns, proper_nouns, adjectives, adverbs, prepositions, verbs, pronouns, interjections, word_lengths, puncts, char_lengths, avg_word_lengths, long_ratios, num_positive_smileyss, num_negative_smileyss, num_neutral_smileyss, standalone_numbers, words_to_superlatives_ratios, words_to_plurals_ratios, post_times, gildeds]
	
	return super_vector
	
complete = make_test(comments)

# save result to file
pickle.dump(complete, open("TIL-2-feats.p","wb"))