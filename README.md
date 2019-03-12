Movie Recommendation System
=============================

This project is inspired by the [Netflix Prize](https://www.netflixprize.com/) competition from 2006. The competition sought to improve the accuracy of predictions for how much a user will like a movie based on his/her previous movie ratings/preferences.

In this project, a subset of the original training data, train.txt, is used to make predictions for users in the various test files (test5.txt, test10.txt, test20.txt). Various user-based/index-based collaborative filtering algorithms are used for making predictions.

Running this project
----------------------
> python3 recommendation_system.py config.yml train.txt <test file> <output file>
	