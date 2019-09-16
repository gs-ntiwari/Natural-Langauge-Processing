import nblearn3
import data
import utils
import math
import sys

def classify(path, model_file):
    
    classes=['truthful_positive', 'truthful_negative', 'deceptive_positive','deceptive_negative']
    bag_of_words, count_of_each_word_matrix, probability_for_each_class= utils.readFromModelFile(model_file, classes)
    #bag_of_words, count_of_each_word_matrix, probability_for_each_class, classes= nblearn3.learn(path)
    #print(len(bag_of_words), len(count_of_each_word_matrix), len(count_of_each_word_matrix[0]), len(probability_for_each_class) )
    
    Val_data, Val_labela, Val_labelb, filepaths= data.new_data_processing(path, True)
    
    ##Val_data, filepaths= data.new_data_processing_test(path)
    
    class_indexes=utils.createIndexMappingForClass(classes)
    words_indexes=utils.createIndexMappingForClass(bag_of_words)
    
    #print(Val_data)
    Val_data_dict=data.convertToDictionary(Val_data)
    
    predicted_labela=list()
    predicted_labelb=list()
    predicted_lables=list()
    for i in range(len(Val_data_dict)): 
        max_probabilty=float("-inf")
        max_probability_class=None
        for j in range(len(classes)): 
            total_probability=0
            posterior_probability=0
            for key, value in Val_data_dict[i].items():
                if words_indexes.get(key)!=None:
                    #print(key, words_indexes.get(key),j, count_of_each_word_matrix[j][words_indexes.get(key)])
                    posterior_probability+=math.log(count_of_each_word_matrix[j][words_indexes.get(key)])
                    #print(filepaths[i],key, posterior_probability)
            prior_probability=probability_for_each_class.get(classes[j])
            total_probability= posterior_probability+math.log(prior_probability)
            #print(total_probability)
            if total_probability>max_probabilty:
                max_probabilty=total_probability
                max_probability_class=classes[j]
        #print(max_probability_class)        
        predicted_labela.append(utils.extractLabelA(max_probability_class)) 
        predicted_labelb.append(utils.extractLabelB(max_probability_class)) 
        predicted_lables.append(max_probability_class)
    
    #print(predicted_labela, predicted_labelb)
    joined_label=list()
    
    ##To Test F1 score locally
    for i in range(len(Val_labela)):
        #print(Val_labela[i], predicted_labela[i], Val_labelb[i], predicted_labelb[i])
        joined_label.append(Val_labela[i]+str('_')+Val_labelb[i])
    print(utils.f1_score_from_sklearn(Val_labela, predicted_labela, classes))
    print(utils.f1_score_from_sklearn(Val_labelb, predicted_labelb, classes))
    print(utils.f1_score_from_sklearn(joined_label, predicted_lables, classes))
    
    return predicted_labela, predicted_labelb, filepaths

def main(input_path):
    model_file = 'nbmodel.txt'
    output_file = 'nboutput.txt'
    labela, labelb, filepaths =classify(input_path, model_file)
    with open(output_file, 'w') as f:
        #print('file created')
        for i in range(len(labela)):
            f.write(labela[i]+" "+labelb[i]+" "+ filepaths[i])
            f.write('\n')
            #print(labela[i], labelb[i], filepaths[i])
    
if __name__ == "__main__":
    input_path = str(sys.argv[1])
    #input_path='/Users/nishatiwari/Downloads/op_spam_training_data/'
    main(input_path)
   
    
    ##write predictions to outputfile