import yaml
import sys
import math
import parse
import user_based_collaborative_filtering as ub_filtering
import item_based_collaborative_filtering as ib_filtering


# Create test data from a subsection of the training data (for testing purposes)
# testData: last 20 rows from training data (users 181-200)
# knownRatings: number of ratings to use for predictions
def createTestData(testData, knownRatings):
	userID = 181
	users = {}
	for user in testData:
		users[userID] = [ {}, {} ]
		count = 0
		for i in range(len(user)):
			if(user[i] > 0 and count < knownRatings):
				users[userID][0][i+1] = user[i]		# prior knowledge
				count += 1
			elif(user[i] > 0 and count == knownRatings):
				users[userID][1][i+1] = 0			# movies we need to predict for
		userID += 1
	return users

# Calculate MAE for a set of predictions against the ground truth values
def meanAbsoluteError(predictions, trainingData):
	numerator = 0
	denominator = 0
	for user, d in predictions.items():
		for movie, prediction in d.items():
			numerator += abs(prediction - trainingData[user-1][movie-1])
			denominator += 1
	return numerator/denominator


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

	# Test mode: create test data from training data
	# Non-test mode: use test file for test data 
	if(cfg["mode"]=="test"):
		trainingData = parse.parseTrainingData(training_file)
		training_file.close()
		test_file.close()

		numUsers = len(trainingData)
		testData = trainingData[(numUsers-20):]		# test data: users 181-200
		newTrainingData = trainingData[:(numUsers-20)]	# training data : users 1-180
		users = createTestData(testData, 5)
		#for user, l in users.items():
			#print("User: {} Known ratings: {} Predictions: {}".format(user, len(l[0]), len(l[1])))
		
		# Get predictions with a specific method and write them to output file
		if(cfg["filtering_algorithm"]=="user-based"):
			predictions = ub_filtering.userBasedCollaborativeFiltering(trainingData, users, cfg["user-based"])
		elif(cfg["filtering_algorithm"]=="item-based"):
			predictions = ib_filtering.itemBasedCollaborativeFiltering(trainingData, users, cfg)
		
		
		# Output predictions to output file
		for user, d in predictions.items():
			for movie, prediction in d.items():
				output_file.write("{} {} {}\n".format(user,movie,prediction))

		# Print out MAE for predictions
		print(meanAbsoluteError(predictions, trainingData))

	else:
		# Parse training file
		trainingData = parse.parseTrainingData(training_file)
		training_file.close()
		
		# Parse test file
		users = parse.parseTestData(test_file)
		test_file.close()

		# Get predictions with a specific method and write them to output file
		if(cfg["filtering_algorithm"]=="user"):
			predictions = ub_filtering.userBasedCollaborativeFiltering(trainingData, users, cfg)
		elif(cfg["filtering_algorithm"]=="item"):
			predictions = ib_filtering.itemBasedCollaborativeFiltering(trainingData, users, cfg)
		
		# Output predictions to output file
		for user, d in predictions.items():
			for movie, prediction in d.items():
				output_file.write("{} {} {}\n".format(user,movie,prediction))
		
		
	output_file.close()
	sys.exit(0)
	

if __name__ == "__main__":
	main()