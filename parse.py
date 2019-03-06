
# Parse and represent training data as a multidimensional array
def parseTrainingData(trainingFile):
	#trainingData: (row=user, column=movie) 
	trainingData = []					
	for line in trainingFile:
		line = line.split() 			# convert to list of characters
		line = [int(i) for i in line] 	# convert each element to an int
		trainingData.append(line)		# append row to our training data
	return trainingData

# Parse and represent test data 
def parseTestData(testFile):
	#users: dictionary of lists
	#users[275] --> list of dictionaries associated with userID 275
	#users[275][0] --> dictionary for movies userID 275 has rated {movieID:rating}
	#user[275][1] --> dictionary with target movies for userID 275
	users = {}
	for line in testFile:
		line = line.split()
		line = [int(i) for i in line]
		
		uID = line[0]
		mID = line[1]
		rating = line[2]

		if uID not in users:
			users[line[0]] = [ {}, {} ] 

		if rating != 0:
			users[uID][0][mID] = rating
		else:
			users[uID][1][mID] = rating
	return users