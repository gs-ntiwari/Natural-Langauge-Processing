import time
import BuildModel as starter

import numpy

MAX_TRAIN_TIME = 30*60  # 3 minutes.

reader = starter.DatasetReader()
#it_isdt_train_tagged
#ja_gsd_train_tagged
train_filename = '/Users/nishatiwari/Documents/deep learning nlp/hmm-training-data/ja_gsd_train_tagged.txt'
test_filename = train_filename.replace('_train_', '_dev_')
term_index, tag_index, train_data, test_data = reader.ReadData(train_filename, test_filename)
(train_terms, train_tags, train_lengths) = train_data

(test_terms, test_tags, test_lengths) = test_data
num_terms = max(train_terms.max(), test_terms.max()) + 1
model = starter.SequenceModel(train_terms.shape[1], num_terms, train_tags.max() + 1)
model.train_data=train_data
language_name = "Italian"


num_terms = max(train_terms.max(), test_terms.max()) + 1
model = starter.SequenceModel(train_terms.shape[1], num_terms, train_tags.max() + 1)


#
def get_test_accuracy():
    predicted_tags = model.run_inference(test_terms, test_lengths)
    if predicted_tags is None:
        print('Is your run_inference function implented?')
        return 0
    test_accuracy = numpy.sum(
        numpy.cumsum(numpy.equal(test_tags, predicted_tags), axis=1)[
            numpy.arange(test_lengths.shape[0]), test_lengths - 1]) / numpy.sum(test_lengths + 0.0)
    return test_accuracy


model.build_inference()
model.build_training()


start_time_sec = time.clock()
train_more = True
num_iters = 0
while train_more:
    print('  Test accuracy for %s after %i iterations is %f' % (language_name, num_iters, get_test_accuracy()))
    train_more = model.train_epoch(train_terms, train_tags, train_lengths)
    train_more = train_more and (time.clock() - start_time_sec) < MAX_TRAIN_TIME
    num_iters += 1

# Done training. Measure test.
print('  Final accuracy for %s is %f' % (language_name, get_test_accuracy()))

