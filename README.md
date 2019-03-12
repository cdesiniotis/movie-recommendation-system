Movie Recommendation System
=============================

This project is inspired by the [Netflix Prize](https://www.netflixprize.com/) competition from 2006. The competition sought to improve the accuracy of predictions for how much a user will like a movie based on his/her previous movie ratings/preferences.

In this project, a subset of the original training data, train.txt, is used to make predictions for users in the various test files (test5.txt, test10.txt, test20.txt). The training data consists of a set of movie ratings by 200 users (userID: 1-200) on 1000 movies (movieID: 1-1000). The test data consists of a pool of new users (userid > 200) who have rated a set of movies and need predictions for others.

Various user-based/index-based collaborative filtering algorithms are used for making predictions. Implementation of user-based collaborative filtering algorithms are in `user_based_collaborative_filtering.py`. Implementation of item-based collaborative filtering algorithms are in `item_based_collaborative_filtering.py`. A custom algorithm, which uses an ensemble of all the collaborative filtering approaches, is in `custom_algorithm.py`.

Running this project
----------------------
`python3 recommendation_system.py config.yml train.txt <test file> <output file>`

Edit `config.yml` to specify the collaborative filtering algorithm to run.

Testing
----------------------
Set `mode` to "test" in `config.yml`. Now, a subset of train.txt will represent the test data. The specified collaborative filtering algorithm will be run and the Mean Absolute Error (MAE) will be calculated/returned, indicating the performance of the recommendation system. 

Results
----------------------
The custom ensemble algorithm achieved the best results, with an MAE of 0.7642 when run against the test files.