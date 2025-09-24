# IMDB_Review_Predictor_Prithviraj_G
MIC Recruitment Project

Hi I am Prithviraj G (25BEC1227), a first year student with real passion for ML and this project is my take on sentiment analysis using the IMDB movie reviews dataset. I thought of a simple intuitive method of counting the number of positive and negative words in a review and calculated weight for one positive and one negative word. This allowed for nearly 80 percent accuracy while I do think much better results can be acheived if I had used other vectorisation methods which I didn't due to time constraints.

What I did:
  • I cleaned the data set by removing empty reviews, html markups, made everything lower case.
	• Negation Handling: I made sure phrases like "not good" are treated differently from "good"
	• Custom Bag of Words: I counted positive and negative words to create a simple [positive_count, negative_count] feature for each           review
	•	Manual Logistic Regression: Implemented gradient descent from scratch to predict sentiment
	•	Interactive Prediction: You can give it your own review, and it will tell you if it thinks it’s positive or negative

Results:
On the IMDB dataset, this simple approach gets around 80% (79.04) accuracy

Future Plans
	•	Implement TF–IDF for better feature weighting
	•	Expand the vocabulary and include bi-grams for context


	


