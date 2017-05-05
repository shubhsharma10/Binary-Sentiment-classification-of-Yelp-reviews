from naive_bayes import process_naive_bayes
from svm import process_svm
from CNN import process_cnn
from RNN import process_recnn
from data_processor.process_data import DataProcessor
from data_processor.process_data_deep_learning import DeepDataProcessor
import pickle
import getopt, sys

def load(filePath):
    with open(filePath, 'rb') as data:
        return pickle.load(data)

def select_data_file(size=1):
    if size == 1:
        return 'data/yelp_academic_dataset_review_small.csv'
    elif size == 2:
        return 'data/yelp_academic_dataset_review_medium.csv'
    elif size == 3:
        return 'data/yelp_academic_dataset_review_large.csv'

def get_data(data_file,reloadData=False):
    if reloadData:
        data_processor = DataProcessor(data_file)
        data_processor.load_data()
        data_processor.save('data/data_processor.data')
    else:
        data_processor = load('data/data_processor.data')
    return data_processor

def get_data_deep_learning(data_file,reloadData=False):
    if reloadData:
        deep_data_processor = DeepDataProcessor(data_file)
        deep_data_processor.load_data()
        deep_data_processor.save('data/deep_data_processor.data')
    else:
        deep_data_processor = load('data/deep_data_processor.data')
    return deep_data_processor

def process(algo=1, data_file=1, reloadData=False, model_type="non-static"):
    if algo <=2:
        data_processor = get_data(select_data_file(data_file),reloadData)
        if algo == 1:
            process_naive_bayes.run_naive_bayes(data_processor)
        elif algo == 2:
            process_svm.run_svm(data_processor)
    else:
        deep_data_processor = get_data_deep_learning(select_data_file(data_file),reloadData)
        if algo == 3:
            process_cnn.run_cnn(deep_data_processor, model_type)
        elif algo == 4:
            process_recnn.run_rnn(deep_data_processor, model_type)


if __name__ == '__main__':
    try:
        opts, args = getopt.getopt(sys.argv[1:], " ")
        if len(args) == 4:
            process(int(args[0]), int(args[1]), bool(args[2]), args[3])
        elif len(args) == 3:
            process(int(args[0]), int(args[1]), bool(args[2]))
        elif len(args) == 2:
            process(int(args[0]), int(args[1]))
        elif len(args) == 1:
            process(int(args[0]))
        else:
            process()
    except getopt.GetoptError as err:
        print("Incorrect Input arguments")
