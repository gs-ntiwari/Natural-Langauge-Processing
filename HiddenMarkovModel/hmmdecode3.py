import data
import math
import sys
import numpy as np

def decodeTags(filename, words_indexes,tags_indexes, transition_prob_matrix, emission_prob_matrix, most_common_tags):
    output_file="hmmoutput.txt"
    #print(emission_prob_matrix)
    lines = tuple(open(filename, mode = "r", encoding = "utf-8"))
    #print(len(lines))
    with open(output_file, 'w') as f:
        for line in lines:
            arrayOfWords=line.strip().split(" ")
            sequenceOfTokens=runViterbiAlgo(arrayOfWords, words_indexes,tags_indexes, transition_prob_matrix, emission_prob_matrix, words_indexes, most_common_tags)
            #print(arrayOfWords, sequenceOfTokens)
            string=""
            for i in range(len(arrayOfWords)):
                string+=arrayOfWords[i]+str("/")+sequenceOfTokens[i]+" "
            f.write(string.strip())
            f.write('\n')

def runViterbiAlgo(arrayOfWords,words_indexes,tags_indexes, transition_prob_matrix, emission_prob_matrix, all_words, most_common_tags):
    
    max_probability=np.zeros((len(arrayOfWords), len(tags_indexes)))
    backpointers=np.zeros((len(arrayOfWords), len(tags_indexes)))
    backpointers.fill(-1)

    tag_array=list()
    
    indexes_to_tags=reverseDict(tags_indexes)
    
    #print('tags_indexes',tags_indexes)
    for key,value in tags_indexes.items():
        backpointers[0][value]=0
                            
    for key, value in tags_indexes.items():
        if all_words.get(arrayOfWords[0])!=None:
            if emission_prob_matrix[words_indexes.get(arrayOfWords[0])][value]==0:
                max_probability[0][value]=float("-inf")
            else:
                max_probability[0][value]=math.log(emission_prob_matrix[words_indexes.get(arrayOfWords[0])][value])+math.log(transition_prob_matrix[0][value])
        else:
            if key in most_common_tags:
                max_probability[0][value]=math.log(transition_prob_matrix[0][value])+math.log(emission_prob_matrix[len(all_words)][value]) 
            else:
                max_probability[0][value]=float("-inf")
                             
    tags_indexes.pop('S0', None) 
    
    for i in range(1, len(arrayOfWords)):
        #print("word ",i, arrayOfWords[i])
        for key,value in tags_indexes.items():
            max_value=float("-inf")
            backpointer=-1
            if (all_words.get(arrayOfWords[i])==None and key not in most_common_tags) or (all_words.get(arrayOfWords[i])!=None and emission_prob_matrix[words_indexes.get(arrayOfWords[i])][value]==0):
                 max_value=float("-inf")
            else: 
                for key1,value1 in tags_indexes.items():
                    temp=float("-inf")
                    if max_probability[i-1][value1]!=float("-inf"):
                        if all_words.get(arrayOfWords[i])!=None:
                            temp=max_probability[i-1][value1]+math.log(emission_prob_matrix[words_indexes.get(arrayOfWords[i])][value])+math.log(transition_prob_matrix[value1][value])
                        else:
                            #print(value1, value, indexes_to_tags.get(value), emission_prob_matrix[len(all_words)][value],transition_prob_matrix[value1][value])
                            temp=max_probability[i-1][value1]+math.log(transition_prob_matrix[value1][value])+math.log(emission_prob_matrix[len(all_words)][value])                      
                    if max_value<temp:
                        max_value=temp
                        backpointer=value1
                    
            max_probability[i][value]=max_value
            backpointers[i][value]=backpointer
            #print(arrayOfWords[i], max_probability[i][value],backpointers[i][value])
            
    last_state=None
    max_value=float("-inf")
    for i in range(1, len(tags_indexes)):
        if max_probability[len(arrayOfWords)-1][i]>max_value:
            max_value=max_probability[len(arrayOfWords)-1][i]
            last_state=i
            
    tag_array.append(indexes_to_tags.get(last_state))
    
    i=len(arrayOfWords)-1
        
    #print(backpointers)
    #print(max_probability)
    while i>0:  
        last_state=int(backpointers[i][last_state])
        tag_array.append(indexes_to_tags.get(last_state))
        i-=1
    tags_indexes['S0']=0   
    #tags_indexes['E0']=endIndex  
    return tag_array[::-1]
    
                            
def reverseDict(tags_indexes):
    return dict((y,x) for x,y in tags_indexes.items())
                            
def main(input_path):
    model_file="hmmmodel.txt"
    output_file="hmmoutput.txt"
    tags, words, transition_prob_matrix, emission_prob_matrix, most_common_tags =data.readFromModelFile(model_file)
    #print(tags)
    #print(emission_prob_matrix[len(words)])
    #print(most_common_tags)
    #print(transition_prob_matrix)
    words_indexes=data.mapWordsToIndex(words)
    tags_indexes=data.mapWordsToIndex(tags)
    #print(emission_prob_matrix)
    decodeTags(input_path, words_indexes,tags_indexes, transition_prob_matrix, emission_prob_matrix, most_common_tags)
    lines_tagged = tuple(open(output_file, mode = "r", encoding = "utf-8"))
    lines_correct = tuple(open('/Users/nishatiwari/Downloads/hmm-training-data/ja_gsd_dev_tagged.txt', mode = "r", encoding = "utf-8"))
    total_count=0
    correct_count=0
    for i in range(len(lines_correct)):     
        word_array_tagged=lines_tagged[i].split(" ")
        word_array_correct=lines_correct[i].split(" ")
        for j in range(len(word_array_correct)):
            if word_array_tagged[j]==word_array_correct[j]:
                correct_count+=1
            total_count+=1
    print("Accuracy", correct_count/total_count)
        
if __name__=="__main__":
    input_file = sys.argv[1]
    #input_file='/Users/nishatiwari/Downloads/hmm-training-data/it_isdt_dev_raw.txt'
    main(input_file)
    #print(input_file)
   