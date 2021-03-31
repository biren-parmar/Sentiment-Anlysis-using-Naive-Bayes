import sys
import getopt
import os
import math
import operator
import collections
import re
import string
import spacy
nlp = spacy.load('en_core_web_sm')


class NaiveBayes:
    class TrainSplit:
        """
        * Represents a set of training and test data. Both self.test and self.train are a list of Examples.
        """
        def __init__(self):
            self.train = []
            self.test = []

    class Example:
        """
        * Represents a document with a label. sentiment is 'pos' or 'neg' by convention; words is a list of strings.
        """
        def __init__(self):
            self.sentiment = ''
            self.words = []

    def __init__(self):
        """
        * NaiveBayes initialization
        """
        self.FILTER_STOP_WORDS = False
        self.BOOLEAN_NB = False
        self.stopList = set(self.readFile('data/english.stop'))
        self.numFolds = 10
        self.posDict = {}
        self.negDict = {}
        self.vocabulary = set()
        self.positiveWordCount = dict()
        self.negativeWordCount = dict()
        self.positiveCount = 0
        self.negativeCount = 0
        self.zeroPosProb = 0.0
        self.zeroNegProb = 0.0
        self.posClassCount = 0
        self.negClassCount = 0

    def classify(self, words):
        """
        * Classification based on the NB, return - neg/pos
        """
        probNegative = 0.0
        probPositive = 0.0
        guess = ''
        if self.FILTER_STOP_WORDS:
            words = self.filterStopWords(words)
        for token in words:
            probNegative += math.log10(self.negDict.get(token, self.zeroNegProb))
            probPositive += math.log10(self.posDict.get(token, self.zeroPosProb))
        probPositive += math.log10((self.posClassCount / float(self.posClassCount + self.negClassCount)))
        probNegative += math.log10((self.negClassCount / float(self.posClassCount + self.negClassCount)))
        guess = 'pos' if probPositive > probNegative else 'neg'
        return guess

    def addExample(self, sentiment, words):
        """
        * Takes in a document and its sentiment and count the word frequency and create a vocabulary.
        """
        if self.FILTER_STOP_WORDS:
            words = self.filterStopWords(words)
        if sentiment == 'pos':
            self.posClassCount += 1
        else:
            self.negClassCount += 1

        if self.BOOLEAN_NB:
            posWordBooleanCountDoc = {}
            negWordBooleanCountDoc = {}
            for word in words:
                if sentiment == 'pos':
                    posWordBooleanCountDoc[word] = True
                elif sentiment == 'neg':
                    negWordBooleanCountDoc[word] = True
                self.vocabulary.add(word)
            if sentiment == 'pos':
                for word, bool in posWordBooleanCountDoc.items():
                    self.positiveWordCount[word] = self.positiveWordCount.get(word, 0) + 1
            else:
                for word, bool in negWordBooleanCountDoc.items():
                    self.negativeWordCount[word] = self.negativeWordCount.get(word, 0) + 1
        else:
            for word in words:
                if sentiment == 'pos':
                    self.positiveWordCount[word] = self.positiveWordCount.get(word, 0) + 1
                    self.positiveCount += 1
                elif sentiment == 'neg':
                    self.negativeWordCount[word] = self.negativeWordCount.get(word, 0) + 1
                    self.negativeCount += 1
                self.vocabulary.add(word)

    def prepareDictionaries(self):
        """
        * Updating log likelihood words in vocabulary.
        """
        vocab_size = len(self.vocabulary)
        if self.BOOLEAN_NB:
            for word, count in self.positiveWordCount.items():
                self.positiveCount += self.positiveWordCount[word]
            for word, count in self.negativeWordCount.items():
                self.negativeCount += self.negativeWordCount[word]

            for token in self.vocabulary:
                self.posDict[token] = float(
                    (self.positiveWordCount.get(token, 0) + 1) / float((self.positiveCount + vocab_size + 1)))
                self.negDict[token] = float(
                    (self.negativeWordCount.get(token, 0) + 1) / float((self.negativeCount + vocab_size + 1)))
            self.zeroNegProb = float(1 / float((self.negativeCount + vocab_size + 1)))
            self.zeroPosProb = float(1 / float((self.positiveCount + vocab_size + 1)))
        else:
            for token in self.vocabulary:
                self.posDict[token] = float(
                    (self.positiveWordCount.get(token, 0) + 1) / float((self.positiveCount + vocab_size + 1)))
                self.negDict[token] = float(
                    (self.negativeWordCount.get(token, 0) + 1) / float((self.negativeCount + vocab_size + 1)))
            self.zeroNegProb = float(1 / float((self.negativeCount + vocab_size + 1)))
            self.zeroPosProb = float(1 / float((self.positiveCount + vocab_size + 1)))

    def readFile(self, fileName):
        """
         * Code for reading a file and creates a bag of words (BOW) representation of the text.
        """
        contents = []
        f = open(fileName)
        for line in f:
            contents.append(line)
        f.close()
        contents_ = self.textClean('\n'.join(contents))
        result = self.tokenWords(contents_)
        return result

    def textClean(self, s):
        """
        * Code for clean up of text
        """
        # Removing the html tags from text
        result1 = re.compile(r'<[^>]+>').sub('', s)
        '''
        # to lower
        result2 = result1.lower()
        # remove special chars
        result3 = re.sub(r'[^a-zA-z0-9.,!?/:;\"\'\s]', ' ', result2)
        # removing numbers
        result4 = re.sub(r'[^a-zA-z.,!?/:;\"\'\s]', '', result3)
        # replace period with space
        result5 = re.sub(r"\.", " ", result4)
        # to remove punctuation
        result6 = ''.join([c for c in result5 if c not in string.punctuation])
        # white space removal
        result7 = re.sub(r'^\s*|\s\s*', ' ', result6).strip()
        '''

        '''
        with open("text_debug.txt", "a") as file:
            file.write(result7)
            file.write("\n")
            file.write("======")
            file.write("\n")
        '''
        return result1

    def tokenWords(self, s):
        """
         * Code for tokenization and lemmatization - Splits lines on whitespace for file reading.
        """
        #return [token.lemma_ for token in nlp(s)]
        return s.split()

    def trainSplit(self, trainDir):
        """
        * Takes in a trainDir, returns one TrainSplit with train set.
        """
        split = self.TrainSplit()
        posTrainFileNames = os.listdir('%s/pos/' % trainDir)
        negTrainFileNames = os.listdir('%s/neg/' % trainDir)
        for fileName in posTrainFileNames:
            example = self.Example()
            example.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
            example.sentiment = 'pos'
            split.train.append(example)
        for fileName in negTrainFileNames:
            example = self.Example()
            example.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
            example.sentiment = 'neg'
            split.train.append(example)
        return split

    def train(self, split):
        """
        * For each document in the train path, performs training.
        """
        for example in split.train:
            words = example.words
            if self.FILTER_STOP_WORDS:
                words = self.filterStopWords(words)
            self.addExample(example.sentiment, words)

    def crossValidationSplits(self, trainDir):
        """
        * Returns a list of TrainSplits corresponding to the cross validation splits.
        """
        splits = []
        posTrainFileNames = os.listdir('%s/pos/' % trainDir)
        negTrainFileNames = os.listdir('%s/neg/' % trainDir)
        # for fileName in trainFileNames:
        for fold in range(0, self.numFolds):
            split = self.TrainSplit()
            for fileName in posTrainFileNames:
                example = self.Example()
                example.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
                example.sentiment = 'pos'
                if int(fileName.split("_")[0]) in range(int((len(posTrainFileNames)/10)*fold), int((len(posTrainFileNames)/10)*(fold+1))):
                    split.test.append(example)
                else:
                    split.train.append(example)
            for fileName in negTrainFileNames:
                example = self.Example()
                example.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
                example.sentiment = 'neg'
                if int(fileName.split("_")[0]) in range(int((len(negTrainFileNames)/10)*fold), int((len(negTrainFileNames)/10)*(fold+1))):
                    split.test.append(example)
                else:
                    split.train.append(example)
            splits.append(split)
        return splits

    def filterStopWords(self, words):
        """
        * Filters stop words.
        """
        filtered = []
        for word in words:
            if not word in self.stopList and word.strip() != '':
                filtered.append(word)
        return filtered


def CV_NFold(dirpath, FILTER_STOP_WORDS, BOOLEAN_NB):
    nb = NaiveBayes()
    splits = nb.crossValidationSplits(dirpath)
    avgAccuracy = 0.0
    avgF1score = 0.0
    fold = 0
    for split in splits:
        classifier = NaiveBayes()
        classifier.FILTER_STOP_WORDS = FILTER_STOP_WORDS
        classifier.BOOLEAN_NB = BOOLEAN_NB
        accuracy = 0.0
        precision = 0.0
        recall = 0.0
        f1score = 0.0
        true_negative = 0.0
        true_positive = 0.0
        false_positive = 0.0
        false_negative = 0.0
        for example in split.train:
            words = example.words
            classifier.addExample(example.sentiment, words)

        classifier.prepareDictionaries()

        for example in split.test:
            words = example.words
            guess = classifier.classify(words)
            if example.sentiment == guess:
                accuracy += 1.0
            if example.sentiment == guess and guess == 'neg':
                true_negative += 1.0
            if example.sentiment == guess and guess == 'pos':
                true_positive += 1.0
            if example.sentiment == "pos" and guess == 'neg':
                false_negative += 1.0
            if example.sentiment == "neg" and guess == 'pos':
                false_positive += 1.0

        print(accuracy, len(split.test))
        accuracy = accuracy / len(split.test)
        precision = true_positive / (true_positive+false_positive)
        recall = true_positive / (true_positive+false_negative)
        f1score = (2*precision*recall)/(precision+recall)
        avgAccuracy += accuracy
        avgF1score +=f1score
        print('[INFO]\tFold %d Accuracy: %f, Precision: %f, Recall: %f, F1score: %f' % (fold, accuracy,precision,recall,f1score))
        fold += 1
    avgAccuracy = avgAccuracy / fold
    avgF1score = avgF1score / fold
    print('[INFO]\tAccuracy: %f, F1score: %f' % (avgAccuracy,avgF1score))


def classifyDir(FILTER_STOP_WORDS, BOOLEAN_NB, trainDir, testDir):
    classifier = NaiveBayes()
    classifier.FILTER_STOP_WORDS = FILTER_STOP_WORDS
    classifier.BOOLEAN_NB = BOOLEAN_NB
    trainSplit = classifier.trainSplit(trainDir)
    classifier.train(trainSplit)
    testSplit = classifier.trainSplit(testDir)
    accuracy = 0.0
    precision = 0.0
    recall = 0.0
    f1score = 0.0
    true_negative = 0.0
    true_positive = 0.0
    false_positive = 0.0
    false_negative = 0.0
    classifier.prepareDictionaries()
    for example in testSplit.train:
        words = example.words
        guess = classifier.classify(words)
        if example.sentiment == guess:
            accuracy += 1.0
        if example.sentiment == guess and guess == 'neg':
            true_negative += 1.0
        if example.sentiment == guess and guess == 'pos':
            true_positive += 1.0
        if example.sentiment == "pos" and guess == 'neg':
            false_negative += 1.0
        if example.sentiment == "neg" and guess == 'pos':
            false_positive += 1.0
    accuracy = accuracy / len(testSplit.train)
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    f1score = (2 * precision * recall) / (precision + recall)

    print('[INFO]\tAccuracy: %f, Precision: %f, Recall: %f, F1score: %f' % (accuracy, precision, recall, f1score))


def main():
    # USER defined variables - START ###
    # Please modify them as needed   ###

    # If any one of the FILTER_STOP_WORDS and BOOLEAN_NB flags is on, the
    # other one is meant to be off.
    FILTER_STOP_WORDS = False
    # Multinomial Naive Bayes classifier and the Naive Bayes Classifier with Boolean (Binarized) features.
    # If the BOOLEAN_NB flag is true, Boolean (Binarized) Naive Bayes (that relies on feature presence/absence)
    # instead of the usual algorithm that relies on feature counts is used
    BOOLEAN_NB = True
    # Path to the data for training
    train_path = './data/'
    test_path = './data/'
    run = 'CV'     # CV for N fold Cross validation on the current total data. #
    # run = 'test' # test_path provides the path to the dir containing test data. Currently it is set to train data.
    # USER defined variables - END ###

    if run == "CV":
        CV_NFold(train_path, FILTER_STOP_WORDS, BOOLEAN_NB)
    elif run == "test":
        classifyDir(FILTER_STOP_WORDS, BOOLEAN_NB, train_path, test_path)


if __name__ == "__main__":
    main()