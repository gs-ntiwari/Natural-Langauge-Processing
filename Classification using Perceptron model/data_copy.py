import os
import math
import glob
import collections

def new_data_processing(path, isTest):
    ## List all files, given the root of training data.
    labela=list()
    labelb=list()
    data_dict=list()
    all_files = glob.glob(os.path.join(path, '*/*/*/*.txt'))
    bag_of_words=set()
    bag_of_words_for_each_class=dict()
    # defaultdict is analogous to dict() [or {}], except that for keys that do not
    # yet exist (i.e. first time access), the value gets contructed using the function
    # pointer (in this case, list() i.e. initializing all keys to empty lists).
    test_by_class = collections.defaultdict(list)  # In Vocareum, this will never get populated
    train_by_class = collections.defaultdict(list)

    for f in all_files:
      # Take only last 4 components of the path. The earlier components are useless
      # as they contain path to the classes directories.
        
        #positive_polarity deceptive_from_MTurk fold2 d_hardrock_18.txt
        class1, class2, fold, fname = f.split('/')[-4:]
        #print(class1, class2, fold, fname)
        
        if fold == 'fold1':
        # True-clause will not enter in Vocareum as fold1 wont exist, but useful for your own code.
            test_by_class[class2.split("_")[0]+"_"+class1.split("_")[0]].append(f)
        else:
            print(class1, class2, fold, fname)
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
    all_words = contents.split(" ")
    data = [words.lower()
       .replace(",", "").replace(".", "").replace("!", "").replace("?", "")
       .replace(";", "").replace(":", "").replace("*", "")
       .replace("(", "").replace(")", "")
       .replace("/", "").replace("\n","").replace('"',"").replace("'","")
    for words in all_words]

    data=remove_stop_words(data)

    temp_list.append(data)
    return temp_list

def processData(data_by_class, isTest):
    labela=list()
    labelb=list()
    data_dict=list()
    filepaths=list()
    bag_of_words=set()
    bag_of_words_for_each_class=dict()
    data=list()
    for key, value in data_by_class.items(): 
        temp_bag_words=set()
        labels=key.split("_")
        #print(labels)
        for filepath in value:
            filepaths.append(filepath)
            if not isTest:
                labela.append(str(labels[0]))
                labelb.append(str(labels[1]))
            tempData=new_process_words(filepath)
            data.extend(tempData)
            for word in tempData:
                bag_of_words.update(word)
                temp_bag_words.update(word)

            if bag_of_words_for_each_class.get(key)==None:
                bag_of_words_for_each_class[key]=temp_bag_words
            else: 
                temp_bag_words.update(bag_of_words_for_each_class.get(key))
                bag_of_words_for_each_class[key]=temp_bag_words
        
    if  isTest:
        #return data, labela, labelb, filepaths
        return data, filepaths
    else:
        return data, labela, labelb, list(bag_of_words), bag_of_words_for_each_class
        
##Data Processing Method 
def data_processing(path, isTest):
    Train_data=list()
    Val_data=list()
    Train_data_dict=list()
    Val_data_dict=list()
    Train_labela=list()
    Train_labelb=list()
    Val_labela=list()
    Val_labelb=list()
    bag_of_words=set()
    bag_of_words_for_each_class=dict()
    for (dirpath, dirnames, filenames) in os.walk(path+str('/positive_polarity')):
        #print(dirnames)
        for dirname in dirnames:
            #print('dirname', dirname)
            for (subdirpath, subdirnames, subfilenames) in walk(dirpath+dirname):
                for subdirname in subdirnames:
                    #print(3)
                    temp_bag_words=set()
                    if subdirname=='fold1' and isTest:
                        for (currentdirpath, currentdirnames, currentfilenames) in walk(subdirpath+str('/')+subdirname):
                            val_data, val_labela, val_labelb=process_words(currentfilenames, currentdirpath,'positive', subdirname)
                            Val_data.extend(val_data)
                            Val_labela.extend(val_labela)
                            Val_labelb.extend(val_labelb)
                    else:
                        for (currentdirpath, currentdirnames, currentfilenames) in walk(subdirpath+str('/')+subdirname):
                            train_data,train_labela,train_labelb= process_words(currentfilenames,currentdirpath,'positive',dirname)
                            Train_data.extend(train_data)
                            Train_labela.extend(train_labela)
                            Train_labelb.extend(train_labelb)
                            
                            #to collect unique words acroos the documents
                            for word in train_data:
                                bag_of_words.update(word)
                                temp_bag_words.update(word)
                            if subdirname.startswith("truth"):
                                var='truthful'
                            else:
                                var='deceptive'
                               
                            
                            current_class=var+'_positive'
                            if bag_of_words_for_each_class.get(current_class)==None:
                                bag_of_words_for_each_class[current_class]=temp_bag_words
                            else: 
                                temp_bag_words.update(bag_of_words_for_each_class.get(current_class))
                                bag_of_words_for_each_class[current_class]=temp_bag_words

    for (dirpath, dirnames, filenames) in os.walk(path+str('/negative_polarity')):
        #print(dirnames)
        for dirname in dirnames:
            for (subdirpath, subdirnames, subfilenames) in walk(dirpath+dirname):
                for subdirname in subdirnames:
                    if subdirname=='fold1':
                        for (currentdirpath, currentdirnames, currentfilenames) in walk(subdirpath+str('/')+subdirname):
                            val_data, val_labela, val_labelb=process_words(currentfilenames, currentdirpath,'negative', dirname)
                            Val_data.extend(val_data)
                            Val_labela.extend(val_labela)
                            Val_labelb.extend(val_labelb)
                    else:
                        temp_bag_words=set()
                        for (currentdirpath, currentdirnames, currentfilenames) in walk(subdirpath+str('/')+subdirname):
                            train_data,train_labela,train_labelb=process_words(currentfilenames,currentdirpath,'negative',subdirname)
                            Train_data.extend(train_data)
                            Train_labela.extend(train_labela)
                            Train_labelb.extend(train_labelb)
                            
                            #to collect unique words acroos the documents
                            for word in train_data:
                                bag_of_words.update(word)
                                temp_bag_words.update(word)
                            
                            if subdirname.startswith("truth"):
                                var='truthful'
                            else:
                                var='deceptive'
                                
                            current_class=var+'_negative'
                            if bag_of_words_for_each_class.get(current_class)==None:
                                bag_of_words_for_each_class[current_class]=temp_bag_words
                            else:
                                temp_bag_words.update(bag_of_words_for_each_class.get(current_class))
                                bag_of_words_for_each_class[current_class]=temp_bag_words
                                
    if  isTest:
        #print('val_data',len(val_data))
        return Val_data, Val_labela, Val_labelb
    else:
        return Train_data, Train_labela, Train_labelb, bag_of_words, bag_of_words_for_each_class
    

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
    listOfStopWords={'ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than', '',' '}
    filtered_words = [w for w in data if not w in listOfStopWords]
    return filtered_words
    
def process_words(currentfilenames, currentdirpath, polarity, dirname):
    temp_list=list()
    temp_labela_list=list()
    temp_labelb_list=list()
    for currentfilename in currentfilenames:
        if not currentfilename.startswith('.'):
        #print(currentdirpath, currentfilename)
            f = open(currentdirpath+str('/')+currentfilename, 'rb')
            contents = f.read().decode("UTF-8")
            f.close()
            all_words = contents.split(" ")
            data = [words.lower().strip()
               .replace(",", "").replace(".", "").replace("!", "").replace("?", "")
               .replace(";", "").replace(":", "").replace("*", "")
               .replace("(", "").replace(")", "")
               .replace("/", "").replace("\n","").replace('"',"").replace("'","").strip()
            for words in all_words]

            data=remove_stop_words(data)

            if dirname.startswith("truth"):
                var='truthful'
            else:
                var='deceptive'
            temp_list.append(data)
            temp_labela_list.append(var)
            temp_labelb_list.append(polarity)
    return temp_list,temp_labela_list,temp_labelb_list
