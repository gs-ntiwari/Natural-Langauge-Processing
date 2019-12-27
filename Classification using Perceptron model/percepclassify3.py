import perceplearn3
import data
import utils
import math
import sys
import numpy

def classify(path, model_file):
    
    classes_labela=['deceptive', 'truthful']
    classes_labelb=['negative', 'positive']
    learned_weights_labela, selected_features_labela, learned_weights_labelb, selected_features_labelb= utils.readFromModelFile(model_file)
        ##Val_data, filepaths= data.new_data_processing_test(path)
    ##learned_weights_labela, averaged_weights_labela,selected_features_labela,learned_weights_labelb, averaged_weights_labelb,selected_features_labelb= perceplearn3.learn(path)
        #print(len(bag_of_words), len(count_of_each_word_matrix), len(count_of_each_word_matrix[0]), len(probability_for_each_class) )
    
    Val_data, Val_labela, Val_labelb, filepaths= data.new_data_processing(path, True)
    
    ##Val_data, filepaths= data.new_data_processing_test(path)
    
    #print(Val_data)
    Val_data_dict=data.convertToDictionary(Val_data)
    
    selected_features_labela_dict=utils.createIndexMappingForClass(selected_features_labela)
    
    selected_features_labelb_dict=utils.createIndexMappingForClass(selected_features_labelb)

    X_labela=utils.populateFeatureValuesWithoutLabelsForEachDocument(Val_data_dict, selected_features_labela_dict)
    
    y_labela=utils.predicted_labels(X_labela, classes_labela, learned_weights_labela)
    
    finalOutput=dict()
    
    finalOutput=utils.prepareFinalOutput(filepaths, y_labela, finalOutput,"a")
    
    X_labelb=utils.populateFeatureValuesWithoutLabelsForEachDocument(Val_data_dict, selected_features_labelb_dict)
    
    y_labelb=utils.predicted_labels(X_labelb, classes_labelb, learned_weights_labelb)
    
    finalOutput=utils.prepareFinalOutput(filepaths, y_labelb, finalOutput,"b")
    
    labela, labelb=utils.extractLabelsFromDict(finalOutput)

    score1, score2, score3=utils.f1_score_from_sklearn(Val_labela, labela, classes_labela)
    score1a, score2a, score3a=utils.f1_score_from_sklearn(Val_labelb, labelb, classes_labelb)
    
    print((score1+score1a)/2,(score2+score2a)/2,(score3+score3a)/2)
    
    return labela, labelb, filepaths

def main(input_path, model_file):
    output_file = "percepoutput.txt"
    labela, labelb, filepaths =classify(input_path, model_file)
    with open(output_file, 'w') as f:
        #print('file created')
        for i in range(len(labela)):
            f.write(labela[i]+" "+labelb[i]+" "+ filepaths[i])
            f.write('\n')
            #print(labela[i], labelb[i], filepaths[i])
    
if __name__ == "__main__":
    #input_path='/Users/nishatiwari/Downloads/op_spam_training_data/'
    model_file = str(sys.argv[1])
    input_path = str(sys.argv[2])
    main(input_path, model_file)
   
    
    ##write predictions to outputfile