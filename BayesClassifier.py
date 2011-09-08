"""
Erik Hope
BayesClassifier.py


This file contains the definition of the Bayes Classifier class.

Use of the class is simple. Feed in tokenized texts (i.e. lists of words),
along with their classification. Once all of the training data has been input,
call compute_probabilities(). After that, call "classify(instance)" to obtain
the classification of a new instnace. Testing instances should be in the same format as 
training instances. 

"""

import sys, os, decimal, math, re

with open("common_words.txt") as f:
    common_words_list=[x.lower() for x in re.findall("[a-zA-Z]+", "".join(f.readlines()))]
    common_words = {}
    for word in common_words_list: common_words[word] = True


def mean(values):
    return float(sum(values))/len(values)

def stdev(values):
    total = sum(values)
    count = len(values)
    mean = float(total)/count
    return math.sqrt((float(1)/(count-1))*sum([(x-mean)**2 for x in values]))



class Text(object):
    """This class has a few rules about what it does to the data data. Mainly,
    it strips out occurrences of the most words from the file common_words.txt"""
    def __init__(self, text):
        self.data = text
        self.data_length = len(text)
        self.token_frequencies = {}
        for token in text:
            if token.lower() not in common_words:
                if token in self.token_frequencies:
                    self.token_frequencies[token] += 1
                else:
                    self.token_frequencies[token] = 1

class Instance(object):
    def __init__(self, data):
        self.data = data
        self.data_length = len(data)
        self.token_frequencies = {}
        for token in data:
            if token in self.token_frequencies:
                self.token_frequencies[token] += 1
            else:
                self.token_frequencies[token] = 1

class Attribute(object):
    pass
        
        


class BayesClassifier(object):
    def __init__(self):
        """Initialize the classifications dictionary, the document count,
        the tokens dictionary, the classification probability dictionary, 
        the token probability dictionary, the number of tokens count and
        the number of tokens per classification dictionary"""
        self._classifications = {}
        self._number_of_instances = 0
        self._tokens = {}
        self._classification_probabilities = {}
        self._token_probabilities = {}
        self._number_of_tokens = 0
        self._number_of_tokens_in_classification = {}
        self._attribute_numeric = {}

    def add_text_instance(self, text, classification):
        """Adds a new text instance for text classification learning"""
        text = Text(text)
        for token in text.data:
            self._tokens[token] = True
        if classification in self._classifications:
            self._classifications[classification].append(text)
        else:
            self._classifications[classification] = [text]
        self._number_of_instances += 1


    def _setup_attribute_numeric(self, instance):
        for index, value in enumerate(instance):
            try:
                x = float(value)
                self._attribute_numeric[index] = True
            except ValueError:
                self._attribute_numeric[index] = False

    def _convert_numerics_to_floats(self, instance):
        converted_instance = []
        for attribute, value in enumerate(instance):
            if self._attribute_numeric[attribute]:
                converted_instance.append(float(value))
            else:
                converted_instance.append(value)
        return converted_instance

    def add_instance(self, instance, classification):
        """Adds a new numeric isntance for numeric classification learning"""
        if not self._attribute_numeric:
            self._setup_attribute_numeric(instance)
        instance = self._convert_numerics_to_floats(instance)
        if classification in self._classifications:
            self._classifications[classification].append(instance)
        else:
            self._classifications[classification] = [instance]
        self._number_of_instances += 1

    def compute_probabilities(self):
        """Either compute mean and standard deviation of numeric attributes, or 
        prior probabilities of non-numeric probabilities"""
        self._classification_attribute_data = {}
        nbr_of_attributes = len(self._attribute_numeric)
        for classification,instances in self._classifications.items():            
            attribute_data = {}
            nbr_of_instances = len(instances)
            for attribute_id in range(nbr_of_attributes):
                if self._attribute_numeric[attribute_id]:
                    attribute = Attribute()
                    attribute.numeric = True
                    attribute.mean = mean([instance[attribute_id] for instance in instances])
                    attribute.stdev = stdev([instance[attribute_id] for instance in instances])
                else:
                    attribute = Attribute()
                    attribute.numeric = False
                    attribute_values = [instance[attribute_id] for instance in instances]
                    value_counts = {}
                    for value in attribute_values:
                        if value in value_counts:
                            value_counts[value] += 1
                        else:
                            value_counts[value] = 1
                    value_probabilities = {}
            
                    for value, count in value_counts.items():
                        value_probabilities[value] = float(count)/nbr_of_instances
                    attribute.value_probabilities = value_probabilities

                attribute_data[attribute_id] = attribute
            self._classification_attribute_data[classification] = attribute_data
            self._classification_probabilities[classification] = (float(nbr_of_instances)/self._number_of_instances)
            

    def compute_probabilities_text(self):
        """Computes the probability of each of the tokens from the training instances
        appearing in each classification"""
        self._number_of_tokens = len(self._tokens)
        for classification, texts in self._classifications.items():
            self._classification_probabilities[classification] = (float(len(texts))/
                                                  float(self._number_of_instances))
            concatenated_texts = []
            total_token_frequency = {}
            for text in texts:
                concatenated_texts += text.data
                for token, frequency in text.token_frequencies.items():
                    if token in total_token_frequency:
                        total_token_frequency[token] += frequency
                    else:
                        total_token_frequency[token] = frequency
            tokens_in_concatenated_texts = len(concatenated_texts)
            for token, frequency in total_token_frequency.items():
                self._token_probabilities[(token, classification)] = (
                    float(frequency+1)/(self._number_of_tokens + tokens_in_concatenated_texts))
            self._number_of_tokens_in_classification[classification] = tokens_in_concatenated_texts


    
    def _density(self, attribute, value):
        """Density function for computing dependent probabilities of numeric values"""

        density = float(1)/(math.sqrt(math.pi*2)*attribute.stdev)
        exp = ((value - attribute.mean)**2)/(2*(attribute.stdev**2))
        #Log of density and of e^-exp
        return math.log(density) + (-exp)



    def classify(self, data):
        """Classifies a non_text instance, based on the probabilities computed prior"""
        probabilities = {}
        instance = self._convert_numerics_to_floats(data)
        for classification, probability in self._classification_probabilities.items():
            probability_of_this_classification = math.log(probability)
            for attribute,value in enumerate(instance):
                attribute_data = self._classification_attribute_data[classification][attribute]
                if self._attribute_numeric[attribute]:
                    if attribute_data.stdev != 0:
                        probability_of_this_classification += (self._density(attribute_data,value))
                else:
                    if value in attribute_data.value_probabilities:
                        probability_of_this_classification += math.log(attribute_data.value_probabilities[value])
            probabilities[classification] = math.exp(probability_of_this_classification)
        total_probability = sum(probabilities.values())
        for classification, probability in probabilities.items():
            if total_probability > 0:
                probabilities[classification] = probability/total_probability
        return probabilities

    def classify_text(self, text):
        """Classifies a new instance based on the probabilities determined above.
        Instead of multiplying probabilities to compute the dependent probabilities of each
        classification, logs are added. This is to avoid floating point underflow"""
        new_text = Text(text)
        probabilities = {}
        for classification, probability in self._classification_probabilities.items():
            probability_of_this_classification = math.log(probability)
            for token, frequency in new_text.token_frequencies.items():
                if token in self._tokens:
                    if (token, classification) in self._token_probabilities:
                        token_classification_probability = math.log(self._token_probabilities[(token, classification)])
                    else:
                        token_classification_probability = math.log(float(1)/self._number_of_tokens_in_classification[classification])
                    probability_of_this_classification += token_classification_probability * frequency
            #Divide by 10000 to prevent underflow!
            probabilities[classification] = math.exp(probability_of_this_classification/10000)
        total_probability = sum(probabilities.values())
        for classification,probability in probabilities.items():
                probabilities[classification] = probability/total_probability
        return probabilities

