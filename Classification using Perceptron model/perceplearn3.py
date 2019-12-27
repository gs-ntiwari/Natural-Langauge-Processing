import data
import utils
import sys
import numpy as np


def learn(path='/Users/nishatiwari/Downloads/op_spam_training_data/'):
    
    Train_data, Train_labela, Train_labelb, bag_of_words, bag_of_words_for_each_file= data.new_data_processing(path, False)
    
    #list of dictionary which hold term frequency for each word for a document 
    Train_data_dict=data.convertToDictionary(Train_data)
    #print(bag_of_words_for_each_file)
    #print(len(Train_labela), len(Train_labelb), len(Train_data_dict))
    #Val_data_dict=data.convertToDictionary(Val_data)
    
    word_counts_for_classes=utils.countWordsForEachClass(Train_data_dict, Train_labela, Train_labelb)
    
    word_counts_for_classes=utils.keepHighFrequencyWords(word_counts_for_classes)
    
    #List of words and the number of documents they appear on

    #train_words_idf=data.calculateIdfforEachWord(Train_data_dict, bag_of_words)
    
    #val_words_idf=data.calculateIdfforEachWord(Val_data_dict, bag_of_words)
    
    #Train_words_tf_idf=data.calculateTfIDFforEachWord(Train_data_dict, train_words_idf,Train_data)
    #Val_words_tf_idf=data.calculateTfIDFforEachWord(Val_data_dict,val_words_idf,Val_data)
    
    #print(dict_classes)
    classes_labela=['deceptive','truthful']
    classes_labelb=['negative','positive']
    
    selected_features_labela=utils.createFeaturesMatrix(classes_labela, word_counts_for_classes)
    
    selected_features_labela_dict=utils.createIndexMappingForClass(selected_features_labela)
    
    selected_features_labelb=utils.createFeaturesMatrix(classes_labelb, word_counts_for_classes)
    
    selected_features_labelb_dict=utils.createIndexMappingForClass(selected_features_labelb)
    
    X_labela=utils.populateFeratureValuesForEachDocument(Train_data_dict, selected_features_labela_dict, Train_labela, classes_labela)
    
    X_labelb=utils.populateFeratureValuesForEachDocument(Train_data_dict, selected_features_labelb_dict, Train_labelb, classes_labelb)

    w=np.zeros(len(selected_features_labela_dict)+1)
    u=np.zeros(len(selected_features_labela_dict)+1)
    c=1
    maxIter=120
    for i in range(1, maxIter):
        np.random.seed(i)
        np.random.shuffle(X_labela)
        y=X_labela[:,len(X_labela[0])-1]
        X_label=X_labela[:,0:len(X_labela[0])-1]
        X_label=np.insert(X_label, 0, 1, axis=1)
        w, u, c=utils.trainDataGivenWeights(w, u, y, X_label, c)
        #print("vanilla labela",i,utils.calculateF1scoreOnValidationSet(path, selected_features_labela, w, classes_labela,"a"))
        #print("average labela",i,utils.calculateF1scoreOnValidationSet(path, selected_features_labela, u, classes_labela,"a"))
        i+=1
    
    w=np.zeros(len(selected_features_labelb_dict)+1)
    u=np.zeros(len(selected_features_labelb_dict)+1)
    c=1
    maxIter=100

    for i in range(1, maxIter):
        np.random.seed(i)
        np.random.shuffle(X_labelb)
        y=X_labelb[:,len(X_labelb[0])-1]
        X_label=X_labelb[:,0:len(X_labelb[0])-1]
        X_label=np.insert(X_label, 0, 1, axis=1)
        w, u, c=utils.trainDataGivenWeights(w, u,y, X_label, c)
        #print("vanilla labelb",i,utils.calculateF1scoreOnValidationSet(path,selected_features_labelb, w,classes_labelb,"b"))
        #print("average labelb",i,utils.calculateF1scoreOnValidationSet(path, selected_features_labelb, u,classes_labelb,"b"))
        i+=1
    
    learned_weights_labela, averaged_weights_labela=utils.trainData(selected_features_labela_dict, X_labela)
    
    learned_weights_labelb, averaged_weights_labelb=utils.trainData(selected_features_labelb_dict, X_labelb)
    
    return learned_weights_labela, averaged_weights_labela,selected_features_labela, learned_weights_labelb,averaged_weights_labelb,selected_features_labelb

def main(input_path):
    learned_weights_labela, averaged_weights_labela,selected_features_labela, learned_weights_labelb,averaged_weights_labelb,selected_features_labelb=learn(input_path)
    model_file = "vanillamodel.txt"
    with open(model_file, 'w') as f:
        #print('file_created')
        #file.writelines( "%s\n" % item for item in list )
        f.writelines("%s " % item for item in selected_features_labela)              
        f.write('\n')
        f.writelines("%s " % item for item in learned_weights_labela)
        f.write('\n')
        f.writelines("%s " % item for item in selected_features_labelb)              
        f.write('\n')
        f.writelines("%s " % item for item in learned_weights_labelb)
        
    model_file = "averagemodel.txt"
    with open(model_file, 'w') as f:
        #print('file_created')
        #file.writelines( "%s\n" % item for item in list )
        f.writelines("%s " % item for item in selected_features_labela)              
        f.write('\n')
        f.writelines("%s " % item for item in averaged_weights_labela)
        f.write('\n')
        f.writelines("%s " % item for item in selected_features_labelb)              
        f.write('\n')
        f.writelines("%s " % item for item in averaged_weights_labelb)
        

if __name__ == "__main__":
    input_path = str(sys.argv[1])
    #input_path='/Users/nishatiwari/Downloads/op_spam_training_data/'
    main(input_path) 
    #TODO
    ##write model to nbbmodel file
    
    
    
