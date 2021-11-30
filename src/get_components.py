import pickle
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def shape_pca():
	"""
	Perform PCA (99% of var) on the landmark positions, i.e. shape dimensions.
	"""
	df = pickle.load(open('../data/feret100_flat.p','rb'))

	df_train = df.loc[df[0].str.contains('_0')].reset_index(drop=True)
	df_test = df.loc[~df[0].str.contains('_0')].reset_index(drop=True)


	pca = PCA(n_components=25)
	y = df_train[0]
	X = pca.fit_transform(df_train.drop(0,axis=1))
	print("PCA shape: {}".format(X.shape[1]))

	test_y = df_test[0]
	test_X = pca.transform(df_test.drop(0,axis=1))

	return(X,y,test_X,test_y)

def shapeless_pca():
	"""
	Perform PCA (99% of var) on the landmark positions, i.e. appearance dimensions.
	"""
	df = pickle.load(open('../data/shapeless_intensities.p','rb'))

	df_train = df.loc[df[0].str.contains('_0')].reset_index(drop=True)
	df_test = df.loc[~df[0].str.contains('_0')].reset_index(drop=True)


	pca = PCA(n_components=25)
	y = df_train[0]
	X = pca.fit_transform(df_train.drop(0,axis=1))
	print("PCA shape: {}".format(X.shape[1]))

	test_y = df_test[0]
	test_X = pca.transform(df_test.drop(0,axis=1))

	return(X,y,test_X,test_y)

def main():
	stX,y,sTX,test_y = shape_pca()
	sltX,_,slTX,_ = shape_pca()

	X = np.concatenate([stX,sltX],axis=1)
	test_X = np.concatenate([sTX,slTX],axis=1)

	w = 0
	correct_ranks = []
	for row in test_X:
		distances = []; people = []

		for i in range(X.shape[0]): # other in X:
			distances.append(np.linalg.norm(X[i,:] - row))
			people.append(y[i])


		
		dd = pd.DataFrame({'Person':people,'Distance':distances})

		# print('Test Y: ' + test_y[w])

		sorted = dd.sort_values(by='Distance')

		correct_rank = np.where(sorted.Person.str.contains(test_y[w][:5]))[0][0]
		correct_ranks.append(correct_rank)
		# print('\t Correct answer rank: ' + str(correct_rank))
		w += 1

	# import pdb; pdb.set_trace()
	vc = pd.Series(correct_ranks).value_counts()
	print("Top 1: {:.3f} Top 3: {:.3f} Top 5: {:.3f} Top 10: {:.3f}".format(vc[0]/len(test_y),
																		sum([vc[i] for i in range(3)])/len(test_y),
																		sum([vc[i] for i in range(5)])/len(test_y),
																		sum([vc[i] for i in range(10)])/len(test_y)))

	# Get counts for all possible ranks, including those that are zero. Lazily exploit try/except
	vc_vec = np.zeros(len(X))

	for i in range(len(X)):
		try:
			vc_vec[i] = vc[i]
		except KeyError:
			vc_vec[i] = 0

	# import pdb; pdb.set_trace()
	plt.plot(np.cumsum(vc_vec)/np.sum(vc_vec))
	plt.title('Correct Labels in Top N Guesses')
	plt.xlabel('Top N')
	plt.ylabel('Proportion of Total Test Images')
	plt.ylim((0,1.05))


	plt.axhline(y=0.25, color='b', linestyle='-',label='N=1',alpha=0.2)
	plt.axhline(y=0.5, color='g', linestyle='-',label='N=8',alpha=0.2)
	plt.axhline(y=0.8, color='orange', linestyle='-',label='N=38',alpha=0.2)
	plt.axvline(x=1, color='b', linestyle='-',alpha=0.2)
	plt.axvline(x=8, color='g', linestyle='-',alpha=0.2)
	plt.axvline(x=38, color='orange', linestyle='-',alpha=0.2)
	plt.legend()
	plt.show()

if __name__ == "__main__": main()


















			