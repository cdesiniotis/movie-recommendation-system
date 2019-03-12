import math
import user_based_collaborative_filtering as ub_filtering

# Return euclidean distance between a and b
# a: list, b: list
def euclidianDistance(a, b):
	distance = 0
	count = 0

	for i in range(len(a)):
		if(a[i]==0 or b[i]==0):
			continue
		distance += math.pow(a[i] - b[i], 2)
		count += 1

	distance = math.sqrt(distance)
	similarity = (1/(distance + 1))
	if count == 0:
		return 0
	elif count == 1:
		return 0.2*similarity
	else:
		return similarity

# Return adjusted cosine similarity between a and b
# a: list, b: list, avgUserRatings: average ratings for all users in training data
def adjustedCosineSimilarity(a, b, avgUserRatings, cfg):
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
	
	# throw away information that is not useful
	if count == 0:
		return 0
	# custom algorithm: use euclidean distance when there is only 1 similar rating or size_a or size_b = 0
	elif cfg["filtering_algorithm"] == "custom" and (count == 1 or size_a == 0 or size_b == 0):
		return 0.25*euclidianDistance(a,b)
	# throw away information that is not useful
	elif count == 1 or size_a == 0 or size_b == 0:
		return 0
	# return adjusted-cosine-similarity
	else:
		return (dotproduct)/(math.sqrt(size_a)*math.sqrt(size_b))


# Return weighted average of ratings for similar movies
# target: movie we are trying to predict the rating for
# userRatings: known ratings for user we are trying to predict for
# neighbors: similar movies (list of tuples (movieID, similarity))
# avgUser: average rating for user we are trying to predict for
def itemBasedWeightedAverage(target, userRatings, neighbors, avgUser):
	numerator = 0
	denominator = 0

	for i in range(len(neighbors)):
		numerator += neighbors[i][1] * (userRatings[neighbors[i][0]] - avgUser)
		denominator += abs(neighbors[i][1])

	return round(avgUser + numerator/denominator)


# Returns a dictionary of rating predictions
# Performs the appropriate item-based collaborative filtering method
# specified in the configuration file
def itemBasedCollaborativeFiltering(trainingData, users, cfg):
	numUsers = len(trainingData) # number of users in training data
	numMovies = len(trainingData[0]) # number of movies in training data
	
	# get average rating for all users in training data
	avgUserRatings = [0]*numUsers
	for i in range(numUsers):
		avgUserRatings[i] = ub_filtering.computeAverage(trainingData[i])
	
	# get transpose of training matrix
	trainingTranspose = [ [] for i in range(numMovies)]
	for i in range(numUsers):
		for j in range(numMovies):
			trainingTranspose[j].append(trainingData[i][j])

	# predictions: dictionary of dictionaries {userID:{movieID:rating. . . }}
	predictions = {} 
	y = 0	# debug
	for user, l in users.items():
		predictions[user] = {}
		# calculate user's average rating
		avgUser = ub_filtering.computeAverage(l[0])
		
		for targetMovie, rating in l[1].items():
			# find similar movies to target movie
			# calculate adjusted cosine similarity between target movie and all other movies the user has rated
			similarities = {}	# {movieID: similarity}
			for movie, rating in l[0].items():
				similarities[movie] = adjustedCosineSimilarity(trainingTranspose[targetMovie-1],trainingTranspose[movie-1], avgUserRatings, cfg)
			# sort by similarity metric
			neighbors = sorted(similarities.items(), key=lambda kv:abs(kv[1]), reverse=True)
			
			# remove neighboring movies whose similarity metric = 0
			remove = []
			for i in range(len(neighbors)):
				if(neighbors[i][1] == 0):
					remove.append(i)

			# delete neighbors
			offset = 0
			for i in remove:
				del neighbors[i-offset]
				offset += 1

			# get prediction by calculating weighted average
			# handle case when there are no neighbors to compare with
			if len(neighbors) == 0:
				y += 1
				predictions[user][targetMovie] = round(avgUser)
			else:
				p = itemBasedWeightedAverage(targetMovie, l[0], neighbors, avgUser)
				if p < 1: 
					p = 1
				elif p > 5:
					p = 5
				predictions[user][targetMovie] = p
	# debug	
	print("Times there were no similar users: {}".format(y))
	return predictions
