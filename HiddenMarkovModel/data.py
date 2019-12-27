import os
import math
import numpy as np
from collections import Counter
    
def processData(file):
    tags=dict()
    prev_tags=dict()
    all_words=dict()
    transition_dict=dict()
    lines = tuple(open(file, mode = "r", encoding = "utf-8"))
    for i in range(len(lines)):
        prevTag="S0"
        if tags.get(prevTag)==None:
            tags[prevTag]=1
        else:
            tags[prevTag]+=1 
        words = lines[i].strip().split(" ")
        for word in words:
            array=word.rsplit("/", 1)
            all_words, tags, transition_dict=update_dict(all_words, tags, array, prevTag, transition_dict)
            prevTag=array[1]
            
        transition= prevTag+str("->")+"E0"
        if transition_dict.get(transition)==None:
            transition_dict[transition]=1
        else:
            transition_dict[transition]+=1 
            
        '''if tags.get("E0")==None:
            tags["E0"]=1
        else:
            tags["E0"]+=1 '''
            
    transition_prob_matrix,emission_prob_matrix, most_common_tags= createProbabilitiesMatrices(all_words, tags,transition_dict)        
    return tags.keys(), all_words.keys(), transition_prob_matrix,emission_prob_matrix, most_common_tags

def update_dict(all_words, tags, array,prevTag,transition_dict):
    if all_words.get(array[0])==None:
        tempDict=dict()
        tempDict[array[1]]=1
        all_words[array[0]]=tempDict
    else:
        currentDict=all_words.get(array[0])
        if currentDict.get(array[1])==None:
            currentDict[array[1]]=1
        else:
            currentDict[array[1]]+=1
        
    if tags.get(array[1])==None:
        tags[array[1]]=1
    else:
        tags[array[1]]+=1 
  
    transition= prevTag+str("->")+array[1]
    
    if transition_dict.get(transition)==None:
        transition_dict[transition]=1
    else:
        transition_dict[transition]+=1 
        
    return all_words, tags, transition_dict

def createProbabilitiesMatrices(all_words, tags, transition_dict):
    transition_prob_matrix=np.zeros((len(tags), len(tags)))
    emission_prob_matrix=np.zeros((len(all_words)+1, len(tags)))
    
    #print('olddict',tags)
    #print(len(tags))

    tag_indexes=mapTagsToIndex(tags)
    word_indexes=mapWordsToIndex(all_words)
    #print('word_indexes',word_indexes)
    
    copy_tags=dict()
    for key,value in all_words.items():
        for key1, value1 in value.items():
            '''if key1 in newList:
                emission_prob_matrix[word_indexes.get(key)][tag_indexes.get(key1)]=value1/(tags.get(key1))
            else:
                emission_prob_matrix[word_indexes.get(key)][tag_indexes.get(key1)]=value1/(tags.get(key1))'''
            emission_prob_matrix[word_indexes.get(key)][tag_indexes.get(key1)]=value1/(tags.get(key1))
            if copy_tags.get(key1)==None:
                copy_tags[key1]=1
            else:
                copy_tags[key1]+=1 
                
    c = Counter(copy_tags)
    ordered_list = c.most_common(int(0.1*(len(copy_tags)))) 
    #print(ordered_list)
    newList=list()
    for k, f in ordered_list:
        newList.append(k)
    #print('newlist',newList)
    total_sum=0
    for key,value in tags.items():
        total_sum+=value

    for i in range(len(newList)):
        #print(newList[i],tag_indexes.get(newList[i]))
        emission_prob_matrix[len(all_words)][tag_indexes.get(newList[i])]=1/(tags.get(newList[i])+1)
        #print(tag_indexes.get(newList[i]))
    
    total_count_for_tags=dict()
    for key,value in transition_dict.items():
        tags=key.split("->")
        #print(tags[0], tags[1], key, tag_indexes)
        if total_count_for_tags.get(tag_indexes.get(tags[0]))==None:
            total_count_for_tags[tag_indexes.get(tags[0])]=value
        else:
            total_count_for_tags[tag_indexes.get(tags[0])]+=value
        #print(len(tags),tag_indexes.get(tags[0]), tag_indexes.get(tags[1]))    
        transition_prob_matrix[tag_indexes.get(tags[0])][tag_indexes.get(tags[1])]+=value
    
    for i in range(len(transition_prob_matrix)):
        for j in range(len(transition_prob_matrix[0])):
            #transition_prob_matrix[i][j]=transition_prob_matrix[i][j]/total_count_for_tags.get(i)
            if total_count_for_tags.get(i)!=None:
                transition_prob_matrix[i][j]=(transition_prob_matrix[i][j]+1)/(len(tags)+total_count_for_tags.get(i))
            else:
                transition_prob_matrix[i][j]=(transition_prob_matrix[i][j]+1)/(len(tags))
     
    #print(emission_prob_matrix[len(all_words)])
    #print(tags)
    return  transition_prob_matrix,emission_prob_matrix, newList
        
def mapWordsToIndex(all_words):
    indexes =dict()
    count=0
    for key in all_words:
        indexes[key]=count
        count+=1
    return indexes

def mapTagsToIndex(all_words):
    indexes =dict()
    count=0
    for key in all_words:
        indexes[key]=count
        count+=1
    return indexes

def readFromModelFile(filename):
    all_words=list()
    tags=list()
    
    count=0
    lines = tuple(open(filename, mode = "r", encoding = "utf-8"))
    
    #print('lines',len(lines))
    tags=list(lines[count].strip().split(" "))
    
    count+=1
    all_words=list(lines[count].strip().split(" "))
    count+=1
    #print(len(tags), len(all_words))
    transition_prob_matrix=np.zeros((len(tags), len(tags)))
    emission_prob_matrix=np.zeros((len(all_words)+1, len(tags)))
    
    #print(count,len(tags)+count, lines[count])
    lines_so_far=count
    for i in range(lines_so_far,len(tags)+lines_so_far):
        current_line=list(lines[i].strip().split(" "))
        #print('line',i, len(current_line),current_line[0],current_line[len(current_line)-1])
        for j in range(len(current_line)):
            #print(j, i-count, current_line[j])
            transition_prob_matrix[i-lines_so_far][j]=float(current_line[j])
        count+=1
        
    #print(count, len(lines), transition_prob_matrix[40])    
    lines_so_far=count
    for i in range(lines_so_far, len(lines)-1):
        current_line=list(lines[i].strip().split(" "))
        #print('line',i, len(current_line),current_line[0],current_line[len(current_line)-1])
        for j in range(len(current_line)):
            #print(j, i-count, current_line[j])
            emission_prob_matrix[i-lines_so_far][j]=float(current_line[j])
            count+=1
            
    most_common_tags=list(lines[len(lines)-1].strip().split(" "))     
       
    return tags, all_words, transition_prob_matrix, emission_prob_matrix,most_common_tags
    