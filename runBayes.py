import os,sys
import BayesClassifier
import optparse
import re
import statlib

def get_tokens(filename):
    text = []
    with open(filename, "r") as f:
        for line in f:
            text += filter(None, [x.strip() for x in re.findall("\W+", line)]) + re.findall("\w+", line)
    return text



def train_bayes(classifier, classifications, training_path):
    for classification in classifications:
        classification_directory = os.path.join(training_path,classification)
        for file_name in [os.path.join(classification_directory,x) 
                          for x in os.listdir(classification_directory) if x[-4:] == ".txt"]:
            classifier.add_text_instance(get_tokens(file_name),classification)
    classifier.compute_probabilities_text()    


def test_bayes(classifier, classifications, testing_path):
    print "TEXT".ljust(30) + "CLASSIFICATION".ljust(30) + "TRUE CLASSIFICATION".ljust(30) + "PROBABILITY".ljust(15) + "OTHER PROBABILITIES"
    total = 0
    correct = 0   
    for classification in classifications:
        classification_directory = os.path.join(testing_path,classification)
        for file_name in [os.path.join(classification_directory,x) 
                          for x in os.listdir(classification_directory) if x[-4:] == ".txt"]:
            result = classifier.classify_text(get_tokens(file_name))
            result = sorted(result.items(),key=lambda x: -x[1])
            output_file_name = os.path.basename(file_name).replace(".txt","").strip().replace("_"," ")
            mark = "*"
            if result[0][0].strip() == classification.strip():
                correct += 1
                mark = ""
            print mark+output_file_name.ljust(30-len(mark)) + result[0][0].ljust(30) + \
                classification.ljust(30) + str(100*result[0][1]).ljust(15) + str(["%s:%s" % (r[0],r[1]*100) for r in result[1:]])
            total += 1
    print "accuracy: %s/%s: %s%%" % (correct,total,float(correct)*100/total)


def run_bayes(data_path):
    training_path = os.path.join(data_path,"TRAINING")
    classifications = [x for x in os.listdir(training_path)
                       if os.path.isdir(os.path.join(training_path,x))]
    classifier = BayesClassifier.BayesClassifier()
    train_bayes(classifier, classifications, training_path)
    testing_path = os.path.join(data_path,"TESTING")
    print "Running on Training Data (asterisk means incorrect)..."
    test_bayes(classifier, classifications, training_path)
    print "Running on Testing Data (asterisk means incorrect)..."
    test_bayes(classifier, classifications, testing_path)



if __name__ == "__main__":
    parser = optparse.OptionParser()
    (options,args) = parser.parse_args()
    if len(args) == 0:
        print "please input the name of the directory containing the training and testing data"
        sys.exit(0)
    elif not os.path.isdir(args[0]):
        print "%s is not a valid directory" % args[0]
        sys.exit(0)
    data_directory = args[0]
    run_bayes(data_directory)
