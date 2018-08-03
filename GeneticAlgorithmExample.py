#!/usr/bin/env python
#RMS 2018 
#An example of using the genetic algorithm code

from GeneticAlgorithm import GeneticAlgorithm
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd


def main():

	feature_df = pd.read_csv('Features_test.dat')

	Y = pd.get_dummies(feature_df['activityID'],drop_first=True)

	X = feature_df.drop(['activityID'],axis=1)

	RF = RandomForestClassifier(n_estimators=1,min_samples_split=2,min_samples_leaf=1)

	GA = GeneticAlgorithm(X,Y,RF)

	GA.fit()

	print(GA.best_fitness)
	print(GA.best_individual)

	X_subset = GA.feature_selection

	print(X_subset.head())


if __name__ == '__main__':

	main()

