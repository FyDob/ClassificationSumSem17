# Uses features made by feature_extraction.py to train several classifiers

import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn import linear_model
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn_pandas import DataFrameMapper, cross_val_score

comments = [] # make list of comments

with open('RC_2017-06/exportAR.json', 'r', encoding="utf-8") as file: # import subreddit json file
	comments = json.load(file)
	
strings = [comment["body"] for comment in comments] # make strings
scores = [int(comment["score"]) for comment in comments] # make scores
categories = [] # initialize categories

# fill categories according to score threshold
for score in scores:
	if score > 6: # 6/7 for AskReddit, 8 for Showerthoughts, 9 for til
		categories.append(1) # popular
	else:
		categories.append(2) # unpopular
		
feats = pickle.load(open("AR-2-feats.p","rb")) # load pre-made features
# fill dictionary with features
d = {"categories" : categories, "strings" : strings, "num_alphas" : feats[1], "uppers" : feats[2], "digits" : feats[3], "spaces" : feats[4], "tabs" : feats[5], "short_ratios" : feats[6], "min_word_lengths" : feats[7], "max_word_lengths" : feats[8], "hapax_legomenas" : feats[9], "hapax_dislegomenas" : feats[10], "nouns" : feats[11], "proper_nouns" : feats[12], "adjectives" : feats[13], "adverbs" : feats[14], "prepositions" : feats[15], "verbs" : feats[16], "pronouns" : feats[17], "interjections" : feats[18], "word_lengths" : feats[19], "puncts" : feats[20], "char_lengths" : feats[21], "avg_word_lengths" : feats[22], "long_ratios" : feats[23], "num_positive_smileyss" : feats[24], "num_negative_smileyss" : feats[25], "num_neutral_smileyss" : feats[26], "standalone_numbers" : feats[27], "words_to_superlatives_ratios" : feats[28], "words_to_plurals_ratios" : feats[29], "post_times" : feats[30], "gildeds" : feats[31]}

# make pandas DataFrame from dictionary
data = pd.DataFrame(d)
# subsampling
pop_data = data[data.categories == 1] # len(data[data.categories == 1]) is amount of popular comments, subsample to that
unpop_data = data[data.categories == 2][:len(pop_data)]
data = pd.DataFrame()
data = data.append([pop_data, unpop_data]) # subsampled data to work on
print(data)

# create mappers from sklearn_pandas to work with DataFrame
mapper1 = DataFrameMapper([('strings', [CountVectorizer(), TfidfTransformer()]),('num_alphas', None), ('uppers', None), ('digits',None), ('spaces',None), ('tabs',None), ('short_ratios',None), ('min_word_lengths',None), ('max_word_lengths',None), ('hapax_legomenas',None), ('hapax_dislegomenas',None), ('nouns',None), ('proper_nouns',None), ('adjectives',None), ('adverbs',None), ('prepositions',None), ('verbs',None), ('pronouns',None), ('interjections',None), ('word_lengths',None), ('puncts',None), ('char_lengths',None), ('avg_word_lengths',None), ('long_ratios',None), ('num_positive_smileyss',None), ('num_negative_smileyss',None), ('num_neutral_smileyss',None), ('standalone_numbers',None), ('words_to_superlatives_ratios',None), ('words_to_plurals_ratios',None), ('post_times',None), ('gildeds',None)],sparse=True) # mapper for TfxIdf + Features
# mapper for just features
mapper2 = DataFrameMapper([('num_alphas', None), ('uppers', None), ('digits',None), ('spaces',None), ('tabs',None), ('short_ratios',None), ('min_word_lengths',None), ('max_word_lengths',None), ('hapax_legomenas',None), ('hapax_dislegomenas',None), ('nouns',None), ('proper_nouns',None), ('adjectives',None), ('adverbs',None), ('prepositions',None), ('verbs',None), ('pronouns',None), ('interjections',None), ('word_lengths',None), ('puncts',None), ('char_lengths',None), ('avg_word_lengths',None), ('long_ratios',None), ('num_positive_smileyss',None), ('num_negative_smileyss',None), ('num_neutral_smileyss',None), ('standalone_numbers',None), ('words_to_superlatives_ratios',None), ('words_to_plurals_ratios',None), ('post_times',None), ('gildeds',None)],sparse=True)


# lists for score evaluation
complete_nb_results = []
complete_svm_results = []
combined_nb = []
combined_nb_prec = []
tfidf_nb = []
tfidf_nb_prec = []
tm_nb = []
tm_nb_prec = []
combined_svm = []
combined_svm_prec = []
tfidf_svm = []
tfidf_svm_prec = []
tm_svm = []
tm_svm_prec = []

# 10-fold cross validation
for i in range(0,10):
	data = data.sample(frac=1) # shuffle data
	pipe = Pipeline([('featurize',mapper1),('clf', MultinomialNB())]) # TfxIdf + Features MNNB classifier
	tf_clf = Pipeline([('vect',CountVectorizer()),('tfidf', TfidfTransformer()),('cl', MultinomialNB())]) # TfxIdf MNNB Classifier
	pipe_no_tf = Pipeline([('featurize',mapper2),('clf', MultinomialNB())]) # Features MNNB Classifier

	svm1 =  Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42))]) # TfxIdf SVM
	svm2 = Pipeline([('featurize',mapper1),('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42))]) # TfxIdf + Features SVM
	svm3 = Pipeline([('featurize',mapper2),('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42))]) # Features SVM

	# splitting X and y for training and testing
	X_train = data[:-(int(len(data)/10))] 
	X_train_tf = list(data.strings)[:-(int(len(data)/10))]
	y_train = list(data.categories)[:-(int(len(data)/10))]
	
	X_test = data[-(int(len(data)/10)):]
	X_test_tf = list(data.strings)[-(int(len(data)/10)):]
	y_test = list(data.categories)[-(int(len(data)/10)):]
	
	# fitting classifiers
	pipe.fit(X_train, y_train)
	tf_clf.fit(X_train_tf, y_train)
	pipe_no_tf.fit(X_train, y_train)
	
	svm1.fit(X_train_tf, y_train)
	svm2.fit(X_train, y_train)
	svm3.fit(X_train, y_train)

	# text output
 	print("----------- Trial " + str(i) + " -----------")
	nb_results = []
	svm_results = []
	print("With text mining features:")
	print(pipe.score(X_test, y_test))
	y_pred = pipe.predict(X_test)
	target_names = ["popular", "unpopular"]
	print(classification_report(y_test, y_pred, target_names=target_names))
	print(precision_score(y_test, y_pred, pos_label=1))
	nb_results.append(pipe.score(X_test, y_test))
	combined_nb.append(pipe.score(X_test, y_test))
	combined_nb_prec.append(precision_score(y_test, y_pred, pos_label=1))

	print("Without:")
	print(tf_clf.score(X_test_tf, y_test))
	y_pred = tf_clf.predict(X_test_tf)
	target_names = ["popular", "unpopular"]
	print(classification_report(y_test, y_pred, target_names=target_names))
	nb_results.append(tf_clf.score(X_test_tf, y_test))
	tfidf_nb.append(tf_clf.score(X_test_tf, y_test))
	tfidf_nb_prec.append(precision_score(y_test, y_pred, pos_label=1))
	
	print("Just text mining features:")
	print(pipe_no_tf.score(X_test, y_test))
	y_pred = pipe_no_tf.predict(X_test)
	target_names = ["popular", "unpopular"]
	print(classification_report(y_test, y_pred, target_names=target_names))
	nb_results.append(pipe_no_tf.score(X_test, y_test))
	tm_nb.append(pipe_no_tf.score(X_test, y_test))
	tm_nb_prec.append(precision_score(y_test, y_pred, pos_label=1))
	
	print("SVM with TM:")
	print(svm2.score(X_test, y_test))
	svm_results.append(svm2.score(X_test, y_test))
	combined_svm.append(svm2.score(X_test, y_test))
	combined_svm_prec.append(precision_score(y_test, y_pred, pos_label=1))
	
	print("SVM just BOW:")
	print(svm1.score(X_test_tf, y_test))
	svm_results.append(svm1.score(X_test_tf, y_test))
	tfidf_svm.append(svm1.score(X_test_tf, y_test))
	tfidf_svm_prec.append(precision_score(y_test, y_pred, pos_label=1))
	
	print("SVM just text mining:")
	print(svm3.score(X_test, y_test))
	svm_results.append(svm3.score(X_test, y_test))
	tm_svm.append(svm3.score(X_test, y_test))
	tm_svm_prec.append(precision_score(y_test, y_pred, pos_label=1))
	
	print("\n NB: ")
	print(nb_results)
	print(max(nb_results), nb_results.index(max(nb_results))) # 0 = with text mining + tf, 1 = just tf, 2 = just text mining 
	complete_nb_results.append((max(nb_results), nb_results.index(max(nb_results))))
	print("\n SVM: ")
	print(svm_results)
	print(max(svm_results), svm_results.index(max(svm_results))) # 0 = with text mining + tf, 1 = just tf, 2 = just text mining 
	complete_svm_results.append((max(svm_results), svm_results.index(max(svm_results))))
	
#print(complete_results)
# more text output
c = 0
for item in complete_nb_results:
	if item[1] == 0 or item[1] == 2:
		c += 1

print(c/100)
print("Average Accuracy NB:")
print("Tfidf only:" + str(sum(tfidf_nb)/len(tfidf_nb)))
print("TM + TFIDF:" + str(sum(combined_nb)/len(combined_nb)))
print("Just TM:" + str(sum(tm_nb)/len(tm_nb)))
print("%.4f" % (sum(tfidf_nb)/len(tfidf_nb)))
print("%.4f" % (sum(tm_nb)/len(tm_nb)))
print("%.4f" % (sum(combined_nb)/len(combined_nb)))

print("Average Precision score NB:")
print("Tfidf only:" + str(sum(tfidf_nb_prec)/len(tfidf_nb_prec)))
print("TM + TFIDF:" + str(sum(combined_nb_prec)/len(combined_nb_prec)))
print("Just TM:" + str(sum(tm_nb_prec)/len(tm_nb_prec)))
print("%.4f" % (sum(tfidf_nb_prec)/len(tfidf_nb_prec)))
print("%.4f" % (sum(tm_nb_prec)/len(tm_nb_prec)))
print("%.4f" % (sum(combined_nb_prec)/len(combined_nb_prec)))

c = 0
for item in complete_svm_results:
	if item[1] == 0 or item[1] == 2:
		c += 1
print(c/100)
print("Average Accuracy SVM:")
print("Tfidf only:" + str(sum(tfidf_svm)/len(tfidf_svm)))
print("TM + TFIDF:" + str(sum(combined_svm)/len(combined_svm)))
print("Just TM:" + str(sum(tm_svm)/len(tm_svm)))
print("%.4f" % (sum(tfidf_svm)/len(tfidf_svm)))
print("%.4f" % (sum(tm_svm)/len(tm_svm)))
print("%.4f" % (sum(combined_svm)/len(combined_svm)))

print("Average Precision score SVM:")
print("Tfidf only:" + str(sum(tfidf_svm_prec)/len(tfidf_svm_prec)))
print("TM + TFIDF:" + str(sum(combined_svm_prec)/len(combined_svm_prec)))
print("Just TM:" + str(sum(tm_svm_prec)/len(tm_svm_prec)))
print("%.4f" % (sum(tfidf_svm_prec)/len(tfidf_svm_prec)))
print("%.4f" % (sum(tm_svm_prec)/len(tm_svm_prec)))
print("%.4f" % (sum(combined_svm_prec)/len(combined_svm_prec)))
