from sklearn.metrics import f1_score
import numpy as np
import operator
import random
import data

def updateClassDictionary(current_class, training_instance, classes_data_set, classes_bag_of_words_dict):

    if classes_data_set.get(current_class)==None:
        newList= list()
        newList.append(training_instance)
        classes_data_set[current_class]=newList
    else:
        newList= list()
        newList=classes_data_set.get(current_class)
        newList.append(training_instance)
        classes_data_set[current_class]=newList
        
    for key,value in training_instance.items():
        if classes_bag_of_words_dict.get(key)==None:
            classes_bag_of_words_dict[key]=value
        else:
            classes_bag_of_words_dict[key]+=value
    
    return classes_data_set, classes_bag_of_words_dict

def f1_score_from_sklearn(real_labels, predicted_labels, classes):
    return f1_score(real_labels, predicted_labels, average='macro'), f1_score(real_labels, predicted_labels, average='micro'), f1_score(real_labels, predicted_labels, average='weighted') 

def keepHighFrequencyWords(train_data_dict):
    for key, value in train_data_dict.items():
        if key in ('deceptive','truthful'):
            noOfParameters=2500
        else:
            noOfParameters=1000
        sorted_d=a = {k: v for k, v in sorted(value.items(), key=lambda x: x[1], reverse=True)[:noOfParameters]}
        train_data_dict[key]=sorted_d
    return train_data_dict

def countWordsForEachClass(Train_data_dict, Train_labela, Train_labelb):
    countWordsForEachClassDict=dict()
    for i in range(len(Train_data_dict)):
        if countWordsForEachClassDict.get(Train_labela[i])==None:
            countWordsForEachClassDict[Train_labela[i]]=dict(Train_data_dict[i])
        else:
            countWordsForEachClassDict=updatedictionaryWithWords(Train_labela[i], Train_data_dict[i], countWordsForEachClassDict)
            
        if countWordsForEachClassDict.get(Train_labelb[i])==None:
            countWordsForEachClassDict[Train_labelb[i]]=dict(Train_data_dict[i])
        else:
            countWordsForEachClassDict=updatedictionaryWithWords(Train_labelb[i], Train_data_dict[i], countWordsForEachClassDict)
    return countWordsForEachClassDict
            
def updatedictionaryWithWords(label, dictionary, countWordsForEachClassDict):
    for key,value in dictionary.items():
        if countWordsForEachClassDict.get(label).get(key)==None:
            countWordsForEachClassDict.get(label)[key]=value
        else:
            countWordsForEachClassDict.get(label)[key]+=value 
    return countWordsForEachClassDict

def calculateProbabiltyForEachClass(classes, train_data_dict):
    #print(train_data_dict)
    dictToreturn=dict()
    total=0
    for current_class in classes:
        total+=len(train_data_dict.get(current_class))
    
    for current_class in classes:
        dictToreturn[current_class]=len(train_data_dict.get(current_class))/total
    return dictToreturn

def extractLabelA(max_probability_class):
    return max_probability_class.split('_')[0]

def extractLabelB(max_probability_class):
    return max_probability_class.split('_')[1]

def readFromModelFile(filename):
  
    lines = tuple(open(filename, 'r'))
    
    #print(len(lines))
    features_labela=list(lines[0].strip().split(" "))
    #print('line0',len(bag_of_words),bag_of_words[0], bag_of_words[1],bag_of_words[len(bag_of_words)-1])
    
    weights_matrix_labela=list()
    
    weights_matrix_labelb=list()

    current_line=list(lines[1].strip().split(" "))
        #print('line',i, len(current_line),current_line[0],current_line[len(current_line)-1])
    for j in range(len(current_line)):
            #print(j, i-1, current_line[j])
        weights_matrix_labela.append(float(current_line[j]))
    
    features_labelb=list(lines[2].strip().split(" "))
    
    current_line=list(lines[3].strip().split(" "))
        #print('line',i, len(current_line),current_line[0],current_line[len(current_line)-1])
    for j in range(len(current_line)):
            #print(j, i-1, current_line[j])
        weights_matrix_labelb.append(float(current_line[j]))   
        
    return weights_matrix_labela,features_labela,weights_matrix_labelb,features_labelb

def calculateF1scoreOnValidationSet(path, selected_features_label, learned_weights_label, classes_label, label):
    
    Val_data, Val_labela, Val_labelb, filepaths= data.new_data_processing(path, True)
    
    ##Val_data, filepaths= data.new_data_processing_test(path)
    
    #print(Val_data)
    Val_data_dict=data.convertToDictionary(Val_data)
    
    selected_features_label_dict=createIndexMappingForClass(selected_features_label)
    

    X_label=populateFeatureValuesWithoutLabelsForEachDocument(Val_data_dict, selected_features_label_dict)
    
    y_label=predicted_labels(X_label, classes_label, learned_weights_label)
    
    finalOutput=dict()
    
    finalOutput=prepareFinalOutput(filepaths, y_label, finalOutput,label)
    
    labela, labelb=extractLabelsFromDict(finalOutput)
    
    if label=="a":
        label=labela
        Val_label=Val_labela
    else:
        label=labelb
        Val_label=Val_labelb
    score1, score2, score3=f1_score_from_sklearn(Val_label, label, classes_label)
    
    return score1, score2, score3

def createIndexMappingForClass(classes):
    indexMapping=dict()
    for i in range(len(classes)):
        indexMapping[classes[i]]=i
    return indexMapping

def createFeaturesMatrix(classes_labels, Train_data_dict):
    words_as_features= set()
    for i in range(len(classes_labels)):
        # print(Train_data_dict.get(classes_labels[i]))
        words_as_features.update(list(Train_data_dict.get(classes_labels[i]).keys()))
    return list(words_as_features)
    

def calculateTheCountOfWords(classes, dict_classes, bag_of_words):
    #print(dict_classes)
    count_matrix=np.zeros((len(classes), len(bag_of_words)))
    count_of_words_for_each_class=np.zeros(len(classes))
                                           
    for i in range(len(bag_of_words)):
        for j in range(len(classes)):
            documents_for_class=dict_classes.get(classes[j])
            for k in range(len(documents_for_class)):
                if documents_for_class[k].get(bag_of_words[i]) !=None:
                    current_count=documents_for_class[k].get(bag_of_words[i])
                    count_matrix[j][i]+=current_count
                    count_of_words_for_each_class[j]+=current_count
                                           
    return count_matrix, count_of_words_for_each_class

def populateFeratureValuesForEachDocument(Train_data_dict, selected_features_labela_dict, Train_labela, classes):
    result_matrix=np.zeros((len(Train_data_dict), len(selected_features_labela_dict)+1))
    for i in range(len(Train_data_dict)):
        for key,value in selected_features_labela_dict.items():
            if Train_data_dict[i].get(key)!=None:
                result_matrix[i][value]=Train_data_dict[i].get(key)
        if Train_labela[i]==classes[0]:
            result_matrix[i][len(selected_features_labela_dict)]=-1
        else:
            result_matrix[i][len(selected_features_labela_dict)]=1
    return result_matrix

def populateFeatureValuesWithoutLabelsForEachDocument(Train_data_dict, selected_features_labela_dict):
    result_matrix=np.zeros((len(Train_data_dict), len(selected_features_labela_dict)))
    for i in range(len(Train_data_dict)):
        for key,value in selected_features_labela_dict.items():
            if Train_data_dict[i].get(key)!=None:
                result_matrix[i][value]=Train_data_dict[i].get(key)
    return result_matrix

def trainDataGivenWeights(w, u, y, X_label, c):
    for j in range(len(X_label)):
            #print(y[j], np.dot(X_label[j], w))
        if (y[j]*np.dot(X_label[j], w))<= 0:
            w = w + y[j]*X_label[j].T
            u = u + y[j]*c*X_label[j].T
        c=c+1
    return w,np.subtract(w,(np.divide(u,c))), c 
    
def trainData(selected_features_label_dict, X_label):
    w=np.zeros(len(selected_features_label_dict)+1)
    u=np.zeros(len(selected_features_label_dict)+1)
    c=1
    maxIter=100
    np.random.shuffle(X_label)
    y=X_label[:,len(X_label[0])-1]
    X_label=X_label[:,0:len(X_label[0])-1]
    X_label=np.insert(X_label, 0, 1, axis=1)
    for i in range(1, maxIter):
        for j in range(len(X_label)):
            #print(y[j], np.dot(X_label[j], w))
            if (y[j]*np.dot(X_label[j], w)) <= 0:
                w = w + y[j]*X_label[j].T
                u = u + y[j]*c*X_label[j].T
            c=c+1
    #print(w,np.subtract(w,(np.divide(u,c))))
    return w,np.subtract(w,(np.divide(u,c))) 

def predicted_labels(X, classes_label, weights):  
    predicted_labels=list();
    X=np.insert(X, 0, 1, axis=1)
    for i in range(len(X)):
        y=np.dot(X[i],weights)
        if y<0:
            predicted_labels.append(-1) 
        else:
            predicted_labels.append(1) 
    return predicted_labels
    
def prepareFinalOutput(filepaths, y_label,finalOutput,label):
    for i in range(len(filepaths)):
        if finalOutput.get(filepaths[i])==None:
            temp=dict()
            if label=="a":
                temp["labela"]=findLabel(y_label[i], "a")
            else:
                temp["labelb"]=findLabel(y_label[i], "b")
            finalOutput[filepaths[i]]=temp
        else:
            temp=finalOutput.get(filepaths[i])
            if label=="a":
                temp["labela"]=findLabel(y_label[i], "a")
            else:
                temp["labelb"]=findLabel(y_label[i], "b")
    return finalOutput

def extractLabelsFromDict(dictionary):
    labela=list()
    labelb=list()
    for key,value in dictionary.items():
        labela.append(value.get("labela"))
        labelb.append(value.get("labelb"))
    return labela, labelb

def findLabel(y_label, label):
    if label=="a":
        if y_label==-1:
            return "deceptive"
        else:
            return "truthful"
    else:
        if y_label==-1:
            return "negative"
        else:
            return "positive"
    
def applyingLaplacesmoothing(count_of_each_word_matrix, count_of_words_for_each_class, vocabulary_count):
    for i in range(len(count_of_each_word_matrix)):
        for j in range(len(count_of_each_word_matrix[0])):
            count_of_each_word_matrix[i][j]=(count_of_each_word_matrix[i][j]+1)/(count_of_words_for_each_class[i]+vocabulary_count)
    return count_of_each_word_matrix
    
    
        