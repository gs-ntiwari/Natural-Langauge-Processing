import os
import math
import glob
import collections
import re

def new_data_processing(path, isTest):
    ## List all files, given the root of training data.
    labela=list()
    labelb=list()
    data_dict=list()
    all_files = glob.glob(os.path.join(path, '*/*/*/*.txt'))
    bag_of_words=set()
    bag_of_words_for_each_class=dict()
    test_by_class = collections.defaultdict(list)  # In Vocareum, this will never get populated
    train_by_class = collections.defaultdict(list)

    for f in all_files:
        
        #positive_polarity deceptive_from_MTurk fold2 d_hardrock_18.txt
        class1, class2, fold, fname = f.split('/')[-4:]
        #print(class1, class2, fold, fname)
        
        if fold == 'fold4':
            test_by_class[class2.split("_")[0]+"_"+class1.split("_")[0]].append(f)
        else:
            #print(class1, class2, fold, fname)
            train_by_class[class2.split("_")[0]+"_"+class1.split("_")[0]].append(f)
            
    if isTest: 
        #print('test_data')
        return processData(test_by_class, isTest)
    else:
        #print('train_data')
        return processData(train_by_class, isTest)
    
def new_data_processing_test(path):
    ## List all files, given the root of training data.
    labela=list()
    labelb=list()
    data_dict=list()
    all_files = glob.glob(os.path.join(path, '*/*/*/*.txt'))
    bag_of_words=set()
    bag_of_words_for_each_class=dict()
    data=list()
    for f in all_files:
        temp_list=new_process_words(f)
        data.extend(temp_list)
    return data, all_files
        
def new_process_words(filepath):
    #print(filepath)
    temp_list=list()
    f = open(filepath, 'rb')
    contents = f.read().decode("UTF-8")
    f.close()
    contents=contents.replace("-- ","").replace(" --","")
    all_words = contents.strip().split(" ")
    data = [words.lower().strip()
       .replace(",", "").replace(".", "").replace("!", "").replace("?", "")
       .replace(";", "").replace(":", "").replace("*", "")
       .replace("(", "").replace(")", "")
       .replace("/", "").replace("\n","").replace('"',"").replace("'","").replace("$","").replace("[","")
       .replace("]","").replace("}","").replace("{","").strip()
    for words in all_words]

    
    data=remove_stop_words(data)
    data=remove_common_words(data)
    data=replace_numbers(data)
    temp_list.append(data)
    return temp_list

def replace_numbers(words):
    new_words = []
    for word in words:
        if not word.isdigit():
            new_words.append(word)
    return new_words

def processData(data_by_class, isTest):
    labela=list()
    labelb=list()
    data_dict=list()
    filepaths=list()
    bag_of_words=set()
    bag_of_words_for_each_file=list()
    data=list()
    for key, value in data_by_class.items(): 
        temp_bag_words=set()
        labels=key.split("_")
        #print(labels)
        for filepath in value:
            filepaths.append(filepath)
            labela.append(str(labels[0]))
            labelb.append(str(labels[1]))
            tempData=new_process_words(filepath)
            data.extend(tempData)
            for word in tempData:
                bag_of_words.update(word)
                temp_bag_words.update(word)
            bag_of_words_for_each_file.append(tempData)

    if  isTest:
        return data, labela, labelb, filepaths
        #return data, filepaths
    else:
        return data, labela, labelb, list(bag_of_words), bag_of_words_for_each_file


def convertToDictionary(data):
    list_of_dict=list()
    for currentList in data:
        temp=dict()
        for word in currentList:
            if temp.get(word)==None:
                temp[word]=1
            else:
                temp[word]+=1
        list_of_dict.append(temp)
    return list_of_dict
            
def calculateIdfforEachWord(Train_data_dict, bag_of_words):
    words_idf=dict()
    for word in bag_of_words:
        #print('word',word)
        words_idf[word]=0
        #print(len(Train_data_dict))
        for currentdict in Train_data_dict:
            if currentdict.get(word)!=None:
                words_idf[word]+=1
    return words_idf

def calculateTfIDFforEachWord(Train_data_dict, words_idf, train_data):
    tf_idf_list=list()
    for i in range(len(Train_data_dict)):
        current_dict = dict();
        current_word_count=len(train_data[i])
        for key,value in Train_data_dict[i].items():
            tf=value/current_word_count
            idf=math.log(len(train_data)/words_idf.get(key))
            tf_idf=tf*idf
            current_dict[key]=tf_idf
        tf_idf_list.append(current_dict)
    return tf_idf_list

def remove_stop_words(data): 
    listOfStopWords={'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'itself', 'other', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through',  'yourselves', 'then', 'that',  'what',  'why', 'so', 'can', 'did', 'now', 'he', 'you', 'herself', 'has','me', 'were', 'her', 'himself', 'this', 'should', 'our', 'their', 'while',  'up', 'to', 'ours', 'had', 'she', 'all', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does',  'just', 'where', 'only', 'myself', 'which', 'those', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', '',' ', 'hers', 'yourself', 'there', 'about', 'once', 'during', 'out', 'having', 'with', 'they','ourselves'}#,'i','but', 'again','most', 'against' 'too','very','not', 'nor','no','off','down','more', 'both','because','over','above', 'under', 'don','between','than',}
    #, 'i','but', 'again','most', 'against' 'too','very', 'down'
    filtered_words = [w for w in data if not w in listOfStopWords]
    return filtered_words

def remove_common_words(data): 
    listOfStopWords={'chicago','hotel','hotels','room','rooms','michigan'}
    #'us','stay','hyatt','could','told','say','went','get','got','go','would',
    filtered_words = [w for w in data if not w in listOfStopWords]
    return filtered_words
