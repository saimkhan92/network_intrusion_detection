from __future__ import division
import argparse
from naive_bayes import Data
from naive_bayes import convert_to_float
import numpy as np
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--limit', type=int, default=-1,
	                    help="Restrict training to this many examples")
	parser.add_argument('--split', type=float, default=0.2,
	                    help="Percent of data to use for validation")
	args = parser.parse_args()

	if args.limit < 0:
		dataset = Data("corrected.gz",1-args.split)
		# train = Data("kddcup.data_10_percent.gz", 1)
		# valid = Data("corrected.gz", 1)

	else:
		dataset = Data("kddcup.data_10_percent.gz", 1 - args.split, args.limit)

	# convert continuous data to floats and store it in np arrays
	cts_train_x = np.array(convert_to_float(dataset.cont_trainx), dtype='int')
	cts_valid_x = np.array(convert_to_float(dataset.cont_validx), dtype='int')

	# encode y labels
	le = LabelEncoder()
	labels = le.fit_transform(np.hstack((dataset.train_y, dataset.valid_y)))
	train_y = labels[0:len(dataset.train_y)]
	valid_y = labels[len(dataset.train_y):]

	# break categorical xtrain data into 2 sets: strings and numbers
	cat_str = np.asarray(dataset.multi_trainx)[:, 0:3]
	cat_num = np.asarray(dataset.multi_trainx)[:, 3:]
	cat_num = cat_num.astype(np.float)
	# encode the columns of string data into numbers
	cat_enc = np.zeros(np.shape(cat_str))
	for k in range(np.shape(cat_str)[1]):
		lab = LabelEncoder()
		cat_enc[:, k] = lab.fit_transform(cat_str[:, k])

	# recombine the categorical xtrain data
	cat_train_x = np.zeros(np.shape(dataset.multi_trainx))
	cat_train_x[:, 0:3] = cat_enc
	cat_train_x[:, 3:] = cat_num

	# break categorical xvalid data into 2 sets: strings and numbers
	cs = np.asarray(dataset.multi_validx)[:, 0:3]
	cn = np.asarray(dataset.multi_validx)[:, 3:]
	cn = cn.astype(np.float)
	# encode the columns into numbers
	ce = np.zeros(np.shape(cs))
	for kk in range(np.shape(cs)[1]):
		ll = LabelEncoder()
		ce[:, kk] = ll.fit_transform(cs[:, kk])

	# recombine categorical xvalid data
	cat_valid_x = np.zeros(np.shape(dataset.multi_validx))
	cat_valid_x[:, 0:3] = ce
	cat_valid_x[:, 3:] = cn

	# initialize models
	mod_cts = tree.DecisionTreeRegressor()
	mod_cat = tree.DecisionTreeClassifier(criterion='entropy')

	# use pipeline to combine continuous and categorical classifiers
	pipe = Pipeline([('continuous', mod_cts), ('categorical', mod_cat)])
	pipe.fit(np.hstack((cat_train_x, cts_train_x)), train_y)
	train_acc = pipe.score(np.hstack((cat_train_x, cts_train_x)), train_y)
	print 'train accuracy: %f' % train_acc
	pred = pipe.predict(np.hstack((cat_valid_x, cts_valid_x)))
	acc = pipe.score(np.hstack((cat_valid_x, cts_valid_x)), valid_y)
	print 'test accuracy: %f' % acc

