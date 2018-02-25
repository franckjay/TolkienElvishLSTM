# -*- coding: utf-8 -*-
"""
Spyder Editor
written_by: J Franck

All Data scraped from: http://eldamo.org/
Inspired by: https://github.com/keras-team/keras/blob/master/examples/lstm_seq2seq.py
"""
import numpy as np
import pandas as pd

def raw_data_reader(raw_file_path,out_path):
    dat=open(raw_file_path,"r",encoding="utf-8")
    out=open(out_path,"w",encoding="utf-8")
    for i in dat:
        try:
            tmp=i.split("“")
        except:
            print ("****BAD***LINE***********")
        tmp[0]=tmp[0].strip("”*[]†²\n\t")
        tmp[1]=tmp[1].strip("”*[]†²\n\t")
        out.write(tmp[1]+",\t"+tmp[0]+"\n")
    out.close()

def encodeData(csvFilePath):
    inSeq=[]#Input phrases/sequences
    outSeq=[]
    inChars=set()#Characters used in input
    outChars=set()
    
    csvFile=open(csvFilePath,"r",encoding="utf-8")
    for i in csvFile:
        tmp=(i.split(","))
        inSeq.append(tmp[0])#Append sentences to list
        outSeq.append(tmp[1])#Elvish phrases list
        #Add to the list of English Characters
        for j in tmp[0]:
            for tmpChar in j:
                if tmpChar not in inChars:
                    inChars.add(tmpChar)
        #...significantly more elvish Characters            
        for j in tmp[1]:
            for tmpChar in j:
                if tmpChar not in outChars:
                    outChars.add(tmpChar)
                    
    return sorted(list(inChars)),sorted(list(outChars)),inSeq,outSeq
            
        
    
    

process_raw=True
if process_raw:
    inPath,outPath="raw_data/","processed_data/"
    raw_data_reader(inPath+"Sindarin_Phrases.dat",outPath+"Sindarin_Phrases.csv")
    raw_data_reader(inPath+"Quenya_Phrases.dat",outPath+"Quenya_Phrases.csv")

encodeData(outPath+"Sindarin_Phrases.csv")