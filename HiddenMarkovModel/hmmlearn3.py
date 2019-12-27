import data
import sys
import numpy as np
    
def main(input_path):
    model_file="hmmmodel.txt"
    tags, words, transition_prob_matrix, emission_prob_matrix, most_common_tags=data.processData(input_path)
    with open(model_file, 'w') as f:
        #print('file_created')
        #file.writelines( "%s\n" % item for item in list )
        #print(tags)
        f.writelines("%s " % item for item in tags)              
        f.write('\n')
        f.writelines("%s " % item for item in words)
        f.write('\n')
        for i in range(len(tags)):
            f.writelines( "%s " % item for item in transition_prob_matrix[i] )
            f.write('\n') 
        for i in range(len(words)+1):
            f.writelines( "%s " % item for item in emission_prob_matrix[i] )
            f.write('\n')
        f.writelines( "%s " % item for item in most_common_tags )

if __name__ == "__main__":
    input_path = str(sys.argv[1])
    #input_path='/Users/nishatiwari/Downloads/op_spam_training_data/'
    main('/Users/nishatiwari/Downloads/hmm-training-data/it_isdt_train_tagged.txt') 
    #TODO
    ##write model to nbbmodel file
    
    
    
