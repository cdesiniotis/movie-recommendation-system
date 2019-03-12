import math

# Return cosine similarity between a and b
# a: dict, b: list
def cosineSimilarity(a, b, cfg): 
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
	
	# throw away information that is not useful
	if count == 0:
		return 0
	# custom-algorithm: use euclidean distance when there is only 1 similar rating
	elif count == 1 and cfg["filtering_algorithm"] == "custom":
		# euclidean distance times a small factor (experimenting for a good value)
		return 0.2*euclideanDistance(a, b)
	# throw away information that is not useful
	elif count == 1:
		return 0
	# return cosine similarity
	else:
		answer = 0
		try:
			answer = (dotproduct)/(math.sqrt(size_a)*math.sqrt(size_b))
		except ZeroDivisionError:
			return 0
		return answer

# Return euclidean distance between a and b
# a: dict, b: list
def euclideanDistance(a, b):
	distance = 0
	count = 0
	# k: movie, v: rating
	for k, v in a.items():
		if b[k-1] == 0:
			continue
		distance += math.pow(v - b[k-1], 2)
		count += 1
	
	distance = math.sqrt(distance)
	similarity = (1/(distance + 1))
	if count == 0:
		return 0
	elif count == 1:
		return 0.2*similarity
	else:
		return similarity
	
# Return weighted average of ratings from similar users
# target: movie we are trying to predict the rating for
# training: complete training data (list of lists)
# neighbors: similar users (list of tuples (userID, cos_sim))
def weightedAverage(target, training, neighbors):
	numerator = 0
	denominator = 0

	for uID, cos_sim in neighbors:
		numerator += training[uID-1][target-1]*cos_sim
		denominator += cos_sim

	return round(numerator/denominator)

# Return average of values in a dictionary or list
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

# Return pearson correlation between a and b
# a: dict, b: list, avgA: avg rating for user A, avgB: avg rating for user B
# IUF_array: inverse user frequency for each movie
def pearsonCorrelation(a, b, avgA, avgB, IUF_array, cfg):
	dotproduct = 0
	size_a = 0
	size_b = 0
	count = 0
	
	# k: movie , v: rating
	for k, v in a.items():
		if b[k-1] == 0:
			continue
		if(cfg["user-based"]["IUF"]==True):
			dotproduct += IUF_array[k-1]*(v - avgA) * (b[k-1] - avgB)*IUF_array[k-1]
			size_a += math.pow((v - avgA)*IUF_array[k-1],2)
			size_b += math.pow((b[k-1]- avgB)*IUF_array[k-1] ,2)
		else:
			dotproduct += (v - avgA) * (b[k-1] - avgB)
			size_a += math.pow((v - avgA),2)
			size_b += math.pow((b[k-1] - avgB),2)
		count += 1
	
	# throw away information that is not useful
	if count == 0:
		return 0
	# custom algorithm: use euclidean distance when there is only 1 similar rating or size_a or size_b = 0
	elif cfg["filtering_algorithm"] == "custom" and (count == 1 or size_a == 0 or size_b == 0):
		# euclidean distance times a small factor (experimenting for a good value)
		return 0.2*euclideanDistance(a, b)
	# throw away information that is not useful
	elif count == 1 or size_a == 0 or size_b == 0:
		return 0
	# return pearson correlation
	else:
		return (dotproduct)/(math.sqrt(size_a)*math.sqrt(size_b))

# Return weighted average of ratings from similar users
# target: target movie we are trying to predict a rating for
# training: complete training data (list of lists)
# neighbors: top neighbors (list of tuples (userID, cos_sim))
def pearsonWeightedAverage(target, training, neighbors, avgUser, avgTraining, cfg):
	numerator = 0.0
	denominator = 0.0

	for uID, pearson_cor in neighbors:
		if(cfg["user-based"]["case_amplification"]==True):
			pearson_cor *= math.pow(abs(pearson_cor), 1.5)
		numerator += (training[uID-1][target-1] - avgTraining[uID-1])*pearson_cor
		denominator += abs(pearson_cor)

	return round(avgUser + numerator/denominator)


# Returns a dictionary of rating predictions
# Performs the appropriate user-based collaborative filtering method
# specified in the configuration file 
def userBasedCollaborativeFiltering(trainingData, users, cfg):
	numUsers = len(trainingData) # number of users in training data
	numMovies = len(trainingData[0]) # number of movies in training data
	# get average rating for all users in training data
	avgTraining = [0]*numUsers
	for i in range(numUsers):
		avgTraining[i] = computeAverage(trainingData[i])

	# calculate IUF for every movie
	IUF_array = [1]*numMovies
	if(cfg["user-based"]["IUF"] == True):
		for j in range(numMovies):
			count = 0
			for i in range(numUsers):
				if(trainingData[i][j] > 0):
					count += 1
			try:
				IUF_array[j] = math.log(numUsers/count)
			except ZeroDivisionError:
				continue

	y = 0	# for debugging
	predictions = {}
	k = int(cfg["user-based"]["k"]) 
	similarityThreshold = float(cfg["user-based"]["similarity_threshold"])
	for user, l in users.items():
		predictions[user] = {}
		
		# calculate user's average rating
		avgUser = computeAverage(l[0])
		
		# calculate similarity metric between new user and users in training data
		similarities = {}
		for i in range(len(trainingData)):
			if(cfg["user-based"]["cosine_similarity"]==True):
				similarities[i+1] = cosineSimilarity(l[0],trainingData[i],cfg)
			elif(cfg["user-based"]["pearson_correlation"]==True):
				similarities[i+1] = pearsonCorrelation(l[0],trainingData[i], avgUser, avgTraining[i], IUF_array, cfg)
			elif(cfg["user-based"]["euclidean_distance"]==True):
				similarities[i+1] = euclideanDistance(l[0],trainingData[i])
		# sort by similarity metric (use absolute value here!)
		neighbors = sorted(similarities.items(), key=lambda kv:abs(kv[1]), reverse=True)
		
		# make a prediction for each movie
		for targetMovie, rating in l[1].items():
			topNeighbors = neighbors.copy()
			remove = []

			# remove neighbors which don't have a rating for target movie or whose similarity metric is 0
			for i in range(len(topNeighbors)):
				if trainingData[topNeighbors[i][0]-1][targetMovie-1] == 0 or topNeighbors[i][1] == similarityThreshold:
					remove.append(i)

			# delete neighbors
			offset = 0
			for i in remove:
				del topNeighbors[i-offset]
				offset += 1
			
			# only use top-k neighbors
			if(k>0):
				topNeighbors = topNeighbors[:min(k,len(topNeighbors))]
			
			# get prediction by calculating weighted average
			# handle case when there are no neighbors to compare with
			if len(topNeighbors) == 0:
				y += 1
				predictions[user][targetMovie] = round(avgUser) # use average rating for user
			else:
				if(cfg["user-based"]["cosine_similarity"]==True):
					predictions[user][targetMovie] = weightedAverage(targetMovie,trainingData,topNeighbors)
				elif(cfg["user-based"]["pearson_correlation"]==True):
					p = pearsonWeightedAverage(targetMovie,trainingData,topNeighbors, avgUser, avgTraining,cfg)
					# pearson prediction is unbounded, so account for this
					if p < 1: 
						p = 1
					elif p > 5:
						p = 5
					predictions[user][targetMovie] = p
				elif(cfg["user-based"]["euclidean_distance"]==True):
					predictions[user][targetMovie] = weightedAverage(targetMovie, trainingData, topNeighbors)
	# debug
	print("Times there were no similar users: {}".format(y))
	return predictions