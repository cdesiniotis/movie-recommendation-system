import yaml
import sys
import math
import parse
import user_based_collaborative_filtering as ub_filtering

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
	predictions = ub_filtering.userBasedCollaborativeFiltering(trainingData, users, cfg)
	for user, d in predictions.items():
		for movie, prediction in d.items():
			output_file.write("{} {} {}\n".format(user,movie,prediction))
	
	
	output_file.close()
	sys.exit(0)
	
	

if __name__ == "__main__":
	main()