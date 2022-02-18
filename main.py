"""
Author: Callum Clear

USE: python <PROGNAME> (options) csv_datafile
OPTIONS:
    -3: use 3 class sentiment
    -a: use adjective list as features
    -s: filter features by stop words list
    -i: use intensifiers
    -n: word negation
"""

import sys, getopt, csv, nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import itertools
import numpy as np

opts, args = getopt.getopt(sys.argv[1:], 'casino')
opts = dict(opts)

CLASS_SIZE = 3 if '-c' in opts else 5
feat_adjectives = '-a' in opts
feat_stop_list = '-s' in opts
intensifier = '-i' in opts
negation = '-n' in opts
output_data = '-o' in opts


def read_word_list(text_file):
    with open(text_file) as f:
        words = f.read().splitlines()
    return words


adjectives = read_word_list('adjectives.txt') if feat_adjectives else []
intens_words = read_word_list('strengthen.txt') if intensifier else []
negate_words = read_word_list('negation_words.txt') if negation else []

if feat_stop_list:
    # define stop words
    stop_list = stopwords.words('english')

if len(args) < 3:
    print("\n** ERROR: no arg files provided **", file=sys.stderr)

if len(args) > 3:
    print("\n** ERROR: too many arg files provided **", file=sys.stderr)

if not args[0].endswith(".tsv"):
    print("\n** ERROR: arg file should be in tsv format **", file=sys.stderr)

# tokenization pattern
pattern = r'''(?x)          # set flag to allow verbose regexps
        (?:[A-Z]\.)+        # abbreviations, e.g. U.S.A.
      | \w+(?:-\w+)*        # words with optional internal hyphens
      | \$?\d+(?:\.\d+)?%?  # currency and percentages, e.g. $12.40, 82%
      | \.\.\.              # ellipsis
      | [][.,;"'?():_`-]    # these are separate tokens; includes ], [
    '''


# Read file method
def read_file(file, data, method, data_type):
    with open(file, 'r') as data_file:
        reader = csv.reader(data_file, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        next(reader)
        # convert tsv file to lists
        for row in reader:
            # convert data to lower case
            text = row[1].lower().split()

            if feat_adjectives:
                # filter by adjectives
                text = [w for w in text if w in adjectives]
            elif feat_stop_list:
                # filter by stop words list
                text = [w for w in text if w not in stop_list]

            if data_type != 'test':
                label = int(row[2])
                # map 5 sentiment to 3
                if CLASS_SIZE == 3:
                    label = map_sentiment(label)

            if data_type == 'train':
                data += 1
                method(text, label)
            else:
                # classify for dev and test data
                classification = method(text)
                if data_type == 'dev':
                    data.append([label, classification])
                else:
                    data.append([row[0], classification])
    return data


#############################################################################
# TRAINING
feature_prob = {}
label_freq = np.zeros(CLASS_SIZE)
feature_freq = np.zeros(CLASS_SIZE)


# Map 5 class sentiment to 3
def map_sentiment(x):
    return int(x / 2) if x % 2 == 0 else x - 1


# Find sentiment
def update_prob(features, sentiment):
    label_freq[sentiment] += 1
    # iterate through each word
    for feat in features:
        feature_freq[sentiment] += 1

        # if word isn't in dictionary add it
        if feat not in feature_prob:
            feature_prob[feat] = np.full(CLASS_SIZE, 1)
        feature_prob[feat][sentiment] += 1


def likelihood(freq):
    # get vocabulary size and add to count
    v = len(feature_prob)
    freq += v
    # loop through each word and divide by sentiment frequency
    for w in feature_prob:
        feature_prob[w] = feature_prob[w] / feature_freq


# Find prior probabilities of each class
def prior_prob(sentiment, total_count):
    _, sentiment_count = np.unique(sentiment, return_counts=True)
    return sentiment_count / total_count


# Read training data
filename = args[0]
total = read_file(filename, 0, update_prob, 'train')

# Naive Bayes pre-calculations
likelihood(feature_freq)
# sentiment probabilities
sentiment_prob = label_freq / total


#############################################################################
# DEVELOPING
def classify(text):
    # compute sentiment
    nb = np.copy(sentiment_prob)
    previous_word = ''
    # iterate through each word in review
    for word in text:
        # find product of prior probabilities and likelihood
        if word in feature_prob:
            if previous_word in negate_words and negation:
                # flip probability if negation word occurs
                nb *= np.flip(feature_prob[word])
            elif previous_word in intens_words and intensifier:
                fb = feature_prob[word]
                # get indexes of sorted feature probability
                sort = np.argsort(feature_prob[word])
                scale = 1
                # iterate through each sorted index
                for i in sort:
                    # increase likelihood by increasing scale
                    fb[i] *= scale
                    scale += 0.5
                nb *= fb
            else:
                nb *= feature_prob[word]

        # save previous word
        previous_word = word
    return np.argmax(nb)


# Read developing data
filename = args[1]
dev_data = read_file(filename, [[]], classify, 'dev')
dev_data.remove([])
dev_data = np.asarray(dev_data)


def plot_confusion_matrix(cm, target_names, normalize=False):
    title = 'Confusion matrix'
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if normalize:
        cm = cm.astype('float') / cm.sum()
    cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]), horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]), horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.grid(False)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()


# Calculate the f1 score of the dev data
def calc_f1_score(cm):
    tp = np.diag(cm)
    fp = np.sum(cm, axis=0) - tp
    fn = np.sum(np.swapaxes(cm, 0, 1), axis=0) - tp
    score = 2 * tp / (2 * tp + fp + fn)
    return np.sum(score) / CLASS_SIZE


# Create confusion matrix
confusion_matrix = np.zeros((CLASS_SIZE, CLASS_SIZE))
for d in dev_data:
    confusion_matrix[d[0]][d[1]] += 1

# define class labels
class_labels = ['negative', 'somewhat negative', 'neutral', 'somewhat positive', 'positive']
# reduce to three labels
if CLASS_SIZE == 3:
    class_labels = list(filter(lambda x: 'somewhat' not in x, class_labels))

plot_confusion_matrix(confusion_matrix, class_labels)
f1 = calc_f1_score(confusion_matrix)
print("F1-measure: ", f1)

#############################################################################
# TESTING
filename = args[2]
test_data = read_file(filename, [[]], classify, 'test')
test_data.remove([])


#############################################################################
# OUTPUT
def output_results(data_type, data):
    output = "{}_predictions_{}classes_Callum_CLEAR.tsv".format(data_type, CLASS_SIZE)
    # open output file
    with open(output, 'a') as out:
        # write headers
        print('SentenceId', 'Sentiment', file=out)
        for row in data:
            print(row[0], row[1], file=out)
    out.close()


if output_data:
    output_results('dev', dev_data)
    output_results('test', test_data)


