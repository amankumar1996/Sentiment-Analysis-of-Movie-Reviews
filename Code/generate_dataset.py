#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 11:30:14 2018

@author: amankumar
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 11:22:47 2018

@author: amankumar
"""
import numpy as np
import re
from copy import deepcopy

# Automatically create vocabulary from all training datasets
dataSetKey = 'raw_IMDB-train.txt'
reference = {} 
dictionary = {}

print("Creating vocabulary for",dataSetKey)
# Open, Read, Split and save to the list
textfile = open(dataSetKey,"r",encoding="utf8")    
content = textfile.read().split('\n')
content_list = deepcopy(content)
content_list[:] = [row.split('\t') for row in content][:-1]

for entry in content_list:
    # entry[0] is the review/comment and entry[1] is the rating/sentiment
    entry[0] = ' '.join(re.compile(r'[A-Za-z ]+').findall(entry[0])).lower()       
    entry[0] = re.compile(r'[A-Za-z]+').findall(entry[0])
    for word in entry[0]:
        if word in dictionary:
            dictionary[word] += 1
        else:
            dictionary[word] = 1

vocabulary = [[word,dictionary[word]] for word in sorted(dictionary, key=dictionary.get, reverse=True)][0:10000]
    
vocab_file = open('IMDB_vocab.txt',"w")  
for i in range(len(vocabulary)):
    vocabulary[i].insert(1,i+1)
    vocab_file.write(vocabulary[i][0]+"\t"+str(vocabulary[i][1])+"\t"+str(vocabulary[i][2])+"\n")
    reference[vocabulary[i][0]] = vocabulary[i][1]        


print("Saving the vocabulary for",dataSetKey)


dataset_path={
    
    "IMDB Training":{
        "Raw":"raw_IMDB-train.txt",
        "Dataset":"IMDB-train.txt"
    },
    "IMDB Validation":{
        "Raw":"raw_IMDB-valid.txt",
        "Dataset":"IMDB-valid.txt",
    },
    "IMDB Test":{
        "Raw":"raw_IMDB-test.txt",
        "Dataset":"IMDB-test.txt",
    }
}
for setType in dataset_path:
    print("Processing",setType)
    rawdatafile = open(dataset_path[setType]["Raw"],"r",encoding="utf8")    
    datasetfile = open(dataset_path[setType]["Dataset"],"w",encoding="utf8") 
    content = rawdatafile.read().split('\n')
    content_list = content
    content_list[:] = [row.split('\t') for row in content][:-1]
    for entry in content_list:
        # entry[0] is the review/comment and entry[1] is the rating/sentiment
        data = []
        entry[0] = ' '.join(re.compile(r'[A-Za-z ]+').findall(entry[0])).lower()  
        entry[0] = re.compile(r'[A-Za-z]+').findall(entry[0])
        for word in entry[0]:
            if word in reference: 
                data.append(str(reference[word]))
        entry[0] = ' '.join(data)                
        datasetfile.write('\t'.join(entry) + "\n") 
    datasetfile.close()
    rawdatafile.close()
    
vocab_file.close()            