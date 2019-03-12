import yaml
import json
import sys
import math
import copy
import user_based_collaborative_filtering as ub_filtering
import item_based_collaborative_filtering as ib_filtering
import recommendation_system as rec

# return weighted average for predictions from all configurations
def weightedAverage(tempPredictions, users, cfgs):
	weightedPredictions = {}

	for user, l in users.items():
		weightedPredictions[user] = {}
		for movie, _ in l[1].items():
			numerator = denominator = 0
			for cfgNum, predictions in tempPredictions.items():
				numerator += predictions[user][movie] * cfgs[cfgNum][1]
				denominator += cfgs[cfgNum][1]
			weightedPredictions[user][movie] = round(numerator/denominator)

	return weightedPredictions


'''
Emperical Weights for different configurations
cosine = 0.802946971
cosine_k = 0.802864883
euclidean_distance = 0.790674766
pearson = 0.81969299
pearson_IUF = 0.81772287
pearson_amplify = 0.838491217
pearson_IUF_amplify = 0.836110655
item-based = 0.817107207

'''

def customAlgorithm(trainingData, users, cfg):
	# set up all the configurations
	cfgs = []

	cfg["filtering_algorithm"] = "user-based"
	
	
	# cosine similarity
	cfg_cosine = copy.deepcopy(cfg)
	cfg_cosine["user-based"]["cosine_similarity"] = True 
	cfg_cosine["user-based"]["pearson_correlation"] = \
		cfg_cosine["user-based"]["IUF"] = cfg_cosine["user-based"]["case_amplification"] = False
	cfgs.append((cfg_cosine, 1-0.802946971))
	

	# cosine similarity k = 40
	cfg_cosine_k = copy.deepcopy(cfg_cosine)
	cfg_cosine_k["user-based"]["k"] = 40
	cfgs.append((cfg_cosine_k, 1-0.802864883))

	
	# pearson correlation
	cfg_pearson = copy.deepcopy(cfg_cosine)
	cfg_pearson["user-based"]["cosine_similarity"] = False
	cfg_pearson["user-based"]["pearson_correlation"] = True
	cfgs.append((cfg_pearson, 1-0.81969299))

	# pearson correlation with IUF
	cfg_pearson_IUF = copy.deepcopy(cfg_pearson)
	cfg_pearson_IUF["user-based"]["IUF"] = True
	cfgs.append((cfg_pearson_IUF, 1-0.81772287))
	

	''' Better performance when not using case-amplification
	# pearson correlation with case amplification
	cfg_pearson_amplify = copy.deepcopy(cfg_pearson)
	cfg_pearson_amplify["user-based"]["case_amplification"] = True
	cfgs.append((cfg_pearson_amplify, 1-0.838491217))

	# pearson correlation with IUF and case amplification
	cfg_pearson_IUF_amplify = copy.deepcopy(cfg_pearson_IUF)
	cfg_pearson_IUF_amplify["user-based"]["case_amplification"] = True
	cfgs.append((cfg_pearson_IUF_amplify, 1-0.836110655))
	'''
	
	# item based collaborative filtering
	cfg_item = copy.deepcopy(cfg)
	cfg_item["filtering_algorithm"] = "item-based"
	cfgs.append((cfg_item, 1-0.817107207))
	
	# euclidean distance
	cfg_euclidean = copy.deepcopy(cfg_cosine)
	cfg_euclidean["user-based"]["cosine_similarity"] = False
	cfg_euclidean["user-based"]["euclidean_distance"] = True
	cfgs.append((cfg_euclidean, 1-0.790674766))

	predictions = {}	# final predictions
	tempPredictions = {}	# {cfg1: {userID: {movieID: rating. . .}, cfg2: . . . }}
	count = 0
	# collect predictions for all configurations
	for c in cfgs:
		tempPredictions[count] = {}
		if(c[0]["filtering_algorithm"]=="user-based"):
			tempPredictions[count] = ub_filtering.userBasedCollaborativeFiltering(trainingData, users, c[0])
		elif(c[0]["filtering_algorithm"]=="item-based"):
			tempPredictions[count] = ib_filtering.itemBasedCollaborativeFiltering(trainingData, users, c[0])
		#print("{} \n MAE: {}".format(json.dumps(c, indent=3), rec.meanAbsoluteError(tempPredictions[count], trainingData)))
		#print("{} \n MAE: {}".format(json.dumps(c, indent=3), rec.meanAbsoluteError(tempPredictions[count], trainingData)))
		count += 1
	# get weighted average of all the predictions for different configurations
	predictions = weightedAverage(tempPredictions, users, cfgs)
	return predictions





	