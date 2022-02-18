# Read Me #
To compute the sentiment analysis run 'main.py' with any of the opts mentioned 
(check file) and three input files (training data, dev data, test data). 

A list of stop words has been applied from the nltk  library, to install these
first open a python terminal, then enter these commands:
> import nltk
> 
> nltk.download('stopwords')

The following text files are included to aid the implementation
* adjectives.txt - a list of adjectives for feature selection
* negation.txt - a list of negation words (such as 'not') for altering likelihoods
* strengthen.txt - a list of intensifier words (such as 'very') for altering likelihoods
