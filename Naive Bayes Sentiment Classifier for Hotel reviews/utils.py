from sklearn.metrics import f1_score
import numpy as np

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

def readFromModelFile(filename, classes):
    bag_of_words=list()
    probability_for_each_class=dict()
    
    lines = tuple(open(filename, 'r'))
    
    #print(len(lines))
    bag_of_words=list(lines[0].strip().split(" "))
    print('line0',len(bag_of_words),bag_of_words[0], bag_of_words[1],bag_of_words[len(bag_of_words)-1])
    
    count_of_each_word_matrix=np.zeros((len(classes), len(bag_of_words)))
    
    for i in range(1, 5):
        current_line=list(lines[i].strip().split(" "))
        print('line',i, len(current_line),current_line[0],current_line[len(current_line)-1])
        for j in range(len(current_line)):
            #print(j, i-1, current_line[j])
            count_of_each_word_matrix[i-1][j]=float(current_line[j])
    
    #print(count_of_each_word_matrix)
    
    for i in range(5, 9):
        current_line=lines[i].strip().split(" ")
        #print('line',i,current_line[0],current_line[len(current_line)-1])
        probability_for_each_class[current_line[0]]=float(current_line[1])
        
    return bag_of_words, count_of_each_word_matrix, probability_for_each_class
        

def createIndexMappingForClass(classes):
    indexMapping=dict()
    for i in range(len(classes)):
        indexMapping[classes[i]]=i
    return indexMapping

def findoutTopWords(count_of_words_for_each_class, bag_of_words):
    for i in range(0,4):
        indices=(-1*count_of_words_for_each_class[i]).argsort()[:100]
        print("class_no>>>>>>",i)
        totalsum=0
        for j in indices:
            totalsum+=count_of_words_for_each_class[i][j]
            print(bag_of_words[j], count_of_words_for_each_class[i][j])
        print('total sum>>>>>',totalsum)
    
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

def applyingLaplacesmoothing(count_of_each_word_matrix, count_of_words_for_each_class, vocabulary_count):
    for i in range(len(count_of_each_word_matrix)):
        for j in range(len(count_of_each_word_matrix[0])):
            count_of_each_word_matrix[i][j]=(count_of_each_word_matrix[i][j]+1)/(count_of_words_for_each_class[i]+vocabulary_count)
    return count_of_each_word_matrix
    
    
        