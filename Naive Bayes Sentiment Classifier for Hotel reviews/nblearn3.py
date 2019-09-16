import data
import utils
import sys


def learn(path='/Users/nishatiwari/Downloads/op_spam_training_data/'):
    
    Train_data, Train_labela, Train_labelb, bag_of_words, bag_of_words_for_each_class= data.new_data_processing(path, False)
    
    #print(Train_labela, Train_labelb)
    
    Train_data_dict=data.convertToDictionary(Train_data)
    #Val_data_dict=data.convertToDictionary(Val_data)
    
    #list of dictionary which hold term frequency for each word for a document
    Train_data_dict=data.convertToDictionary(Train_data)
    #Val_data_dict=data.convertToDictionary(Val_data)
    
    #List of words and the number of documents they appear on
    #print(bag_of_words)
    train_words_idf=data.calculateIdfforEachWord(Train_data_dict, bag_of_words)
    
    #val_words_idf=data.calculateIdfforEachWord(Val_data_dict, bag_of_words)
    
    Train_words_tf_idf=data.calculateTfIDFforEachWord(Train_data_dict, train_words_idf,Train_data)
    #Val_words_tf_idf=data.calculateTfIDFforEachWord(Val_data_dict,val_words_idf,Val_data)
    
        #print(dict_classes)
    classes=['truthful_positive', 'truthful_negative', 'deceptive_positive','deceptive_negative']
    
    classes_data_set=dict()
    dict_of_words_for_each_class=dict()
    
    for i in range(len(Train_data)):
        #print(Train_labela[i], Train_labelb[i])
        current_class=None;
        if Train_labela[i]=='truthful' and Train_labelb[i]=='positive':
            current_class='truthful_positive'
        elif Train_labela[i]=='truthful' and Train_labelb[i]=='negative':
            current_class='truthful_negative'
        elif Train_labela[i]=='deceptive' and Train_labelb[i]=='positive':
            current_class='deceptive_positive'
        else:
            current_class='deceptive_negative'
              
        classes_data_set, dict_of_words_for_each_class=utils.updateClassDictionary(current_class, Train_data_dict[i],classes_data_set,dict_of_words_for_each_class)
        #print(len(dict_of_words_for_each_class.keys()), current_class)  
    
    
    probability_for_each_class=utils.calculateProbabiltyForEachClass(classes, classes_data_set)
    
    count_of_each_word_matrix, count_of_total_words=utils.calculateTheCountOfWords(classes, classes_data_set, bag_of_words)
    
    count_of_each_word_matrix= utils.applyingLaplacesmoothing(count_of_each_word_matrix, count_of_total_words, len(bag_of_words))
    
    utils.findoutTopWords(count_of_each_word_matrix,bag_of_words)
    #print(classes_data_set.get('deceptive_negative')[0])
    #print(len(bag_of_words), len(count_of_each_word_matrix), len(count_of_each_word_matrix[0]), len(probability_for_each_class) )
    #print(bag_of_words)
    return bag_of_words, count_of_each_word_matrix, probability_for_each_class,classes

def main(input_path):
    bag_of_words, count_of_each_word_matrix, probability_for_each_class,classes=learn(input_path)
    model_file = "nbmodel.txt"
    with open(model_file, 'w') as f:
        #print('file_created')
        #file.writelines( "%s\n" % item for item in list )
        f.writelines("%s " % item for item in bag_of_words)
                       
        f.write('\n')
        for i in range(len(classes)):
            f.writelines( "%s " % item for item in count_of_each_word_matrix[i] )
            f.write('\n')
        
        for key,value in probability_for_each_class.items():
            f.write(key+str(" ")+str(value))
            f.write('\n')

if __name__ == "__main__":
    input_path = str(sys.argv[1])
    #input_path='/Users/nishatiwari/Downloads/op_spam_training_data/'
    main(input_path)

    
    #TODO
    ##write model to nbbmodel file
    
    
    
