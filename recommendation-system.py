import yaml
import sys
import math
import parse

# a is dict, b is a list of size 1000
def cosineSimilarity(a, b): 
	dotproduct = 0
	size_a = 0
	size_b = 0
	count = 0 		# number of common ratings

	# k: movie , v: rating
	for k, v in a.items():
		if b[k-1] == 0:
			continue

		dotproduct += (v * b[k-1])
		size_a += math.pow(v,2)
		size_b += math.pow(b[k-1],2)
		count += 1

	# do not count metric if less than two in common ratings
	if count < 2:
		return 0
	'''
	if(size_a == 0 or size_b == 0):
		print("ERROR, size_a or size_b = 0!!!"),
		print(a)
		return 0
	'''	
	answer = 0
	try:
		answer = (dotproduct)/(math.sqrt(size_a)*math.sqrt(size_b))
	except ZeroDivisionError:
		return 0
	return answer
	#return (dotproduct)/(math.sqrt(size_a)*math.sqrt(size_b))

# target: target movie we are trying to assign a rating to for a user
# training: complete training data (list of lists)
# neighbors: top neighbors (list of tuples (userID, cos_sim))
def weightedAverage(target, training, neighbors):
	numerator = 0
	denominator = 0

	for uID, cos_sim in neighbors:
		numerator += training[uID-1][target-1]*cos_sim
		denominator += cos_sim

	# debug
	if denominator == 0:
		return None
	#print(numerator, denominator, target)
	return round(numerator/denominator)

# Returns average of values in a dictionary or list
# dictionary: used for users
# list: used for training data
def computeAverage(x):
	sum = 0
	count = 0
	if(type(x) == dict):
		for value in x.values():
			if value > 0:
				sum += value
				count += 1
	elif(type(x) == list):
		for value in x:
			if value > 0:
				sum += value
				count += 1
	else:
		print("Wrong parameter type for average() (dict or list)")
		sys.exit(-1)
	
	return sum/count

# a: dict, b: list, avgA: avg rating for user A, avgB: avg rating for user B
def pearsonCorrelation(a, b, avgA, avgB, cfg):
	dotproduct = 0
	size_a = 0
	size_b = 0
	count = 0
	
	# k: movie , v: rating
	for k, v in a.items():
		if b[k-1] == 0:
			continue
		if(cfg["IUF"]==True):
			dotproduct += (v*IUF_array[k-1] - avgA) * (b[k-1]*IUF_array[k-1] - avgB)
			size_a += math.pow((v*IUF_array[k-1] - avgA),2)
			size_b += math.pow((b[k-1]*IUF_array[k-1] - avgB),2)
		else:
			dotproduct += (v - avgA) * (b[k-1] - avgB)
			size_a += math.pow((v - avgA),2)
			size_b += math.pow((b[k-1] - avgB),2)
		count += 1

	# do not count metric if less than two in common ratings
	if count < 2:
		return 0

	# this is not useful information
	if(size_a == 0 or size_b == 0):
		return 0	

	return (dotproduct)/(math.sqrt(size_a)*math.sqrt(size_b))

# target: target movie we are trying to assign a rating to for a user
# training: complete training data (list of lists)
# neighbors: top neighbors (list of tuples (userID, cos_sim))
def pearsonWeightedAverage(target, training, neighbors, avgUser, avgTraining, cfg):
	numerator = 0
	denominator = 0

	for uID, pearson_cor in neighbors:
		if(cfg["case_amplification"]==True):
			pearson_cor *= pow(pearson_cor, 1.5)
		numerator += (training[uID-1][target-1] - avgTraining[uID-1])*pearson_cor
		denominator += abs(pearson_cor)

	# debug
	if denominator == 0:
		return None
	#print(numerator, denominator, target)
	return round(avgUser + numerator/denominator)

def userBasedCollaborativeFiltering(trainingData, users, cfg):
	numUsers = len(trainingData) # number of users in training data
	numMovies = len(trainingData[0]) # number of movies in training data
	# get average rating for all users in training data
	avgTraining = [0]*numUsers
	for i in range(numUsers):
		avgTraining[i] = computeAverage(trainingData[i])

	# calculate IUF for every movie
	IUF_array = [1]*numMovies
	if(cfg["IUF"] == True):
		for j in range(numMovies):
			count = 0
			for i in range(numUsers):
				if(trainingData[i][j] > 0):
					count += 1
			try:
				IUF_array[j] = log(numUsers/count)
			except ZeroDivisionError:
				continue
	#x = 0
	#y = 0
	predictions = {}
	#k=40
	for user, l in users.items():
		predictions[user] = {}
		# calculate user's average rating
		avgUser = computeAverage(l[0])
		# calculate similarity metric between new user and users in training data
		similarities = {}
		for i in range(len(trainingData)):
			if(cfg["cosine_similarity"]==True):
				similarities[i+1] = cosineSimilarity(l[0],trainingData[i])
			elif(cfg["pearson_correlation"]==True):
				similarities[i+1] = pearsonCorrelation(l[0],trainingData[i], avgUser, avgTraining[i],cfg)
		# sort by similarity metric (use absolute value here!)
		neighbors = sorted(similarities.items(), key=lambda kv:abs(kv[1]), reverse=True)
		# make a prediction for each movie
		for targetMovie, rating in l[1].items():
			topNeighbors = neighbors.copy()
			remove = []

			# remove neighbors which don't have a rating for target movie or whose pearson correlation is 0
			for i in range(len(topNeighbors)):
				if trainingData[topNeighbors[i][0]-1][targetMovie-1] == 0 or topNeighbors[i][1] < 0.8:
					remove.append(i)

			# delete neighbors
			offset = 0
			for i in remove:
				del topNeighbors[i-offset]
				offset += 1
			
			#topNeighbors = topNeighbors[:min(k,len(topNeighbors))]

			# handle case when there are no neighbors to compare with!
			# need to figure out a better way of approaching this
			if len(topNeighbors) == 0:
				#y += 1
				predictions[user][targetMovie] = round(avgUser)
				#predictions[user][targetMovie] = 3
			else:
				#if(topNeighbors[len(topNeighbors)-1][1] < 0.8):
					#x += 1
				if(cfg["cosine_similarity"]==True):
					predictions[user][targetMovie] = weightedAverage(targetMovie,trainingData,topNeighbors)
				elif(cfg["pearson_correlation"]==True):
					p = pearsonWeightedAverage(targetMovie,trainingData,topNeighbors, avgUser, avgTraining,cfg)
					# pearson prediction is unbounded, so account for this
					if p < 1: 
						p = 1
					elif p > 5:
						p = 5
					predictions[user][targetMovie] = p
	return predictions

# avgTraining: average ratings for all users
def adjustedCosineSimilarity(a, b, avgUserRatings):
	dotproduct = 0
	size_a = 0
	size_b = 0
	count = 0

	for i in range(len(a)):
		if(a[i] == 0 or b[i] == 0):
			continue
		dotproduct += (a[i] - avgUserRatings[i])*(b[i] - avgUserRatings[i])
		size_a += pow((a[i] - avgUserRatings[i]), 2)
		size_b += pow((b[i] - avgUserRatings[i]), 2)
		count += 1

	if(count < 2):
		return 0

	if(size_a == 0 or size_b == 0):
		return 0

	return (dotproduct)/(math.sqrt(size_a)*math.sqrt(size_b))

def itemBasedWeightedAverage(target, userRatings, neighbors, avgUser):
	numerator = 0
	denominator = 0

	for i in range(len(neighbors)):
		numerator += neighbors[i][1] * (userRatings[neighbors[i][0]] - avgUser)
		denominator += abs(neighbors[i][1])

	return round(avgUser + numerator/denominator)

def itemBasedCollaborativeFiltering(trainingData, users, cfg):
	numUsers = len(trainingData) # number of users in training data
	numMovies = len(trainingData[0]) # number of movies in training data
	# get average rating for all users in training data
	avgUserRatings = [0]*numUsers
	for i in range(numUsers):
		avgUserRatings[i] = computeAverage(trainingData[i])
	print("averages of all user's ratings: {}".format(avgUserRatings))
	# get transpose of training matrix
	trainingTranspose = [ [] for i in range(numMovies)]
	for i in range(numUsers):
		for j in range(numMovies):
			trainingTranspose[j].append(trainingData[i][j])

	# predictions: dictionary of dictionaries {userID:{movieID:rating. . . }}
	predictions = {} 		
	#k = 40
	for user, l in users.items():
		predictions[user] = {}
		# calculate user's average rating
		avgUser = computeAverage(l[0])
		print("average new user ratings: {}".format(avgUser))
		
		for targetMovie, rating in l[1].items():
			# find similar movies to target movie
			# calculated adjusted cosine similarity between target movie and all other movies the user has rated
			similarities = {}	# {movieID: similarity}
			for movie, rating in l[0].items():
				print("calculating similarity between {} and {}".format(trainingTranspose[targetMovie-1],trainingTranspose[movie-1]))
				similarities[movie] = adjustedCosineSimilarity(trainingTranspose[targetMovie-1],trainingTranspose[movie-1], avgUserRatings)
			print("similarities: {}".format(similarities))
			# sort by similarity metric
			neighbors = sorted(similarities.items(), key=lambda kv:abs(kv[1]), reverse=True)
			print("neighbors for movie {}: {}".format(targetMovie, neighbors))
			remove = []
			for i in range(len(neighbors)):
				if(neighbors[i][1] == 0):
					remove.append(i)

			# delete neighbors
			offset = 0
			for i in remove:
				del neighbors[i-offset]
				offset += 1

			if len(neighbors) == 0:
				predictions[user][targetMovie] = round(avgUser)
			else:
				predictions[user][targetMovie] = itemBasedWeightedAverage(targetMovie, l[0], neighbors, avgUser)
		
	
	return predictions

def test():
	'''
	# SAMPLE TRAINING DATA TO TEST COSINE SIMILARITY AND WEIGHTED AVERAGE
	sampleTrainingData = [[9,5,2,9], [2,9,8,1]]
	sampleUser = {1:7, 2:6, 3:3}		# users will be represented by dictionary : {movieID:rating} , remember the offset for movieID and userID!

	similarities = {}		# dict or {uID:cos_sim}
	for i in range(len(sampleTrainingData)):
		similarities[i+1] = cosineSimilarity(sampleUser,sampleTrainingData[i])

	k = 2
	topNeighbors = sorted(similarities.items(), key=lambda kv:kv[1], reverse=True)		# returns a list of tuples (userID, cos_sim)
	print(topNeighbors)
	topNeighbors = topNeighbors[:k]
	print(topNeighbors)
	# need to make sure neighbors have a rating for the target movie before slicing the top k neighbors

	target = 4
	prediction = weightedAverage(target, sampleTrainingData, topNeighbors)
	print(prediction)
	# expected results:
		# prediction: 5.448 -> 5
	'''

	#SAMPLE TRAINING DATA TO TEST PEARSON CORRELATION AND PEARSON WEIGHTED AVERAGE
	#sampleTrainingData = [[7,8,9,10]]
	#sampleUser = {201:[{1:4, 2:3, 3:2}, {4:0}]}
	
	#sampleTrainingData = [[9,5,2,9], [2,9,8,1]]
	#sampleUser = {201: [{1:7, 2:6, 3:3}, {4:0}]}
	#predictions = pearsonCorrelationMethod(sampleTrainingData, sampleUser)
	#print(predictions)
	# Expected Results:
		# prediction: 8.557267 --> 9

	#SAMPLE TRAINING DATA TO TEST INDEX-BASED FILTERING
	sampleTrainingData = [[2,1,5,4],[4,2,5,5]]
	sampleUser = {201:[{1:1, 2:1, 3:5}, {4:0}]}
	predictions = itemBasedCollaborativeFiltering(sampleTrainingData,sampleUser, None)
	print(predictions)
	# Expected results: 
	#similarities: {1: -0.7071067811865475, 2: -0.9999999999999998, 3: 0.9486832980505138}
    #neighbors for movie 4: [(3, 0.9486832980505138), (1, -0.7071067811865475), (2, -0.9999999999999998)]
    # prediction: {201: {4: 1}}


def main():

	# Check if enough arguments supplied
	if len(sys.argv) != 5:
		print("Usage: {} <config file> <training-data> <test file> <output file>".format(sys.argv[0]))
		sys.exit(-1)

	# Open files
	try:
		cfg_file = open(sys.argv[1], 'r')
		training_file = open(sys.argv[2], 'r')
		test_file = open(sys.argv[3], 'r')
		output_file = open(sys.argv[4], 'w')
	except IOError as e:
		print("Error opening files")
		sys.exit(-1)

	# Represent yaml file as python object
	cfg = yaml.load(cfg_file)
	cfg_file.close()

	# Parse training file
	trainingData = parse.parseTrainingData(training_file)
	training_file.close()
	
	# Parse test file
	users = parse.parseTestData(test_file)
	test_file.close()

	# Get predictions with a specific method and write them to output file
	#predictions = cosineSimilarityMethod(trainingData, users)
	#predictions = pearsonCorrelationMethod(trainingData, users)
	predictions = userBasedCollaborativeFiltering(trainingData, users, cfg)
	for user, d in predictions.items():
		for movie, prediction in d.items():
			output_file.write("{} {} {}\n".format(user,movie,prediction))
	
	
	output_file.close()
	sys.exit(0)
	
	

if __name__ == "__main__":
	main()