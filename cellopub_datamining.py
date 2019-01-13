
import re
import sys
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk import metrics, stem, tokenize 
import string
from nltk.corpus import wordnet as wn
from nltk.stem import *
from nltk.stem.porter import *
import pandas as pd
import gensim
from gensim import corpora
from gensim.test.utils import common_texts
from gensim.corpora.dictionary import Dictionary

import pickle

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


with open('./cellopub.txt', 'r') as infile, open ('./cellopub_edit.txt', 'w') as outfile:
    for line in infile:
        line = line.replace("NeferiNE   reverses","NeferiNE reverses")
        outfile.write(line)

#Diccionarios que se van a rellenar con la información de las referencias y líneas celulares
ids ={}
titles={}
abts={}

abstracts=[] 
title=[]
a='No'
file_cell = open ('./cellopub_edit.txt', 'r')
file_cell.seek(0)
cell_num=0

#Almacenaje de la información del archivo de texto en los diccionarios

for line in file_cell.readlines():
    line = line.strip()
    if line == '//':
        if len(title) != 0:
            total_tit = " ".join(title) #todo el título se junta en una variable tipo string
            titles[cell_num] = total_tit
        
        if a == 'Yes':
            total_abs = " ".join(abstracts) #todo el abstract se junta en una variable tipo string
            abts[cell_num] = total_abs
            
        a='No' 
        abstracts=[] 
        title=[]
        cell_num+=1
        
    elif re.search(r'^ID\s{3}\S+', line):
        header, identif = line.split("   ")
        ids[cell_num] = identif
        
            
    elif re.search(r'^RT\s{3}\S+', line):
        header, identif = line.split("   ")
        title.append(identif)
            
    elif re.search(r'^AB\s{3}\S+', line):
        header, identif = line.split("   ")
        if identif == 'Yes':
            a=identif
            n=0
        else:
            a='No'

    elif a == 'Yes':
        abstracts.append(line)
    
    else:
        continue     

print("Número de lineas celulares con publicación en cellopub.txt: %s, \nNúmero de títulos de las publicaciones en cellopub.txt: %s, \nNúmero de abstract de publicaciones en cellopub.txt: %s." % (len(ids.keys()), len(titles.keys()), len(abts.keys())))

lines=[]
with open('./umls_corpus.csv', 'r') as infile, open ('./umls_corpus.tsv', 'w') as outfile:
    for line in infile:
        line = line.replace('","','"\t\t"',1)
        line = line.replace(',,,,,',' ')
        line = line.replace(',,,,',' ')
        line = line.replace(',,,',' ')
        line = line.replace(',,',' ')
        line = line.replace('^',' ')
        line = line.replace('\\',' ')
        line = line.replace('+',' ')
        outfile.write(line)
#El archivo umls_corpus.tsv tiene el código y la expresión de umls separados por dos tabuladores

			#### FUNCIONES ###

#Función para obtener únicamente el morfema de la palabra
def get_lemma(word):
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma

#Function that drops conjunctions, preposition, uppercases and only returns the morfem of the word
standard_words=['protein','enzyme','cell','structure','product','location', 'observable','without']
def prepare_text_for_lda(text):
    tokens = nltk.word_tokenize(text) #obtener palabras por separado
    tokens = [token for token in tokens if len(token) > 3] #obtener solo palabras de más de 3 letras
    tokens = [token for token in tokens if token not in stopwords.words('english')] #se eliminan stopwords
    tokens = [token for token in tokens if token not in standard_words] #se quitan palabras con poca relevancia
    tokens = [get_lemma(token) for token in tokens] #se obtiene únicamente el morfema
    tokens = [token.strip('+') for token in tokens] #eliminar signos +
    tokens = [token.strip('*') for token in tokens] #eliminar signos *
    return tokens

#Función para crear modelo con algoritmo lda
def lda_model_topics(texts):
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = 1, id2word=dictionary, passes=5)
    list_top = (ldamodel.print_topics(num_topics=1, num_words=6)) #generar una lista de 6 palabras que describan el artículo
    return(list_top)

#Función para obtener los topics como palabras separadas
def clean_topic(list_lda):
    string=''.join(str(e) for e in list_lda)
    topics=[]
    sing_topics=[]
    total_list_top = []
    topics = string.split("+")
    for topic in topics:
        val, top = topic.split("*")
        top = top.strip('"')
        top = top.strip('" ')
        top = top.strip('"\')')
        sing_topics.append(top)
    for word in sing_topics:
        if word not in total_list_top:
            total_list_top.append(word)
    return (total_list_top)

#Función para transformar una lista en dos
def split_two(lst):
    l_a=[]
    l_b=[]
    lmax = len(lst)
    lmin = int(lmax/2)
    for i in range(lmin):
        l_a.append(lst[i])
    for i in range(lmin,lmax):
        l_b.append(lst[i])
    return(l_a,l_b)

			### CÓDIGO ###

#Generar códigos umls estándares
umls_stand=[]
umls_keys=[]
for k in umls_codes.keys():
    expression = umls_codes[k]
    expression = re.sub('\((\w+)\)',' ', expression)
    expression = prepare_text_for_lda(expression)
    umls_stand.append(expression)
    umls_keys.append(k)

#Obtener topics de cada publicación
total_info=[]
topics={} #diccionario que va a contener los topics de cada línea celular

for key in titles.keys():
    title = titles[key]
    title = prepare_text_for_lda(title)
    total_info.append(title)
    if key in abts.keys():                 #topics generados con la información del título y el abstract 
        abstr = abts[key]
        abstr = prepare_text_for_lda(abstr)
        total_info.append(abstr)
        
    else:
        text_1, text_2 = split_two(title)   #topics generados con información del título (abstract no disponible)
        total_info.append(text_1)
        total_info.append(text_2)
    
    list_top=lda_model_topics(total_info)
    list_top = clean_topic(list_top)
    topics[key]=list_top
    total_info=[]

#Función que asigna a cada tipo celular un código umls
count = 0
searches = 0
list_ids_cells=[] #lista con los identificadores de las células de cellopub
list_codes_umls=[] #lista con los códigos umls encontrados asociados a cada tipo celular

for key in topics.keys():
    topic = ' '.join(str(e) for e in topics[key])
    for n in range(len(umls_stand)):
        expression = umls_stand[n]
        count=0
        if expression != []:
            for i in range(len(expression)):
                word=expression[i]
                if re.search(word, topic):    
                    count += 1
                    
            if len(expression)<4:
                if count == len(expression):
                    if umls_keys[n] not in list_codes_umls:
                        list_codes_umls.append(umls_keys[n])
                        list_ids_cells.append(ids[key])
                        print(topic,expression)
                        searches += 1
            else:
                if count > (len(expression)-2):
                    if umls_keys[n] not in list_codes_umls:
                        list_codes_umls.append(umls_keys[n])
                        list_ids_cells.append(ids[key])
                        print(topic,expression)
                        searches += 1
        else:
            continue   

total_number_cells=[]
for n in range(len(list_ids_cells)):
    if list_ids_cells[n] not in total_number_cells:
        total_number_cells.append(list_ids_cells[n])

print("Número de términos de códigos umls encontrados: %s" % (len(list_codes_umls)))
print("Número de términos de líneas celulares diferentes para las que se ha encontrado término umls: %s" % (len(total_number_cells)))

#Dataframe con las líneas celulares y los códigos umls con los que se relacionan
df_cell_umls = pd.DataFrame()
df_cell_umls = pd.DataFrame(columns=['Cell_ID','UMLS_code'])
df_umls.Cell_ID=umls_keys
df_umls.UMLS_code=umls_stand

print(df_cell_umls) #dataframe que contiene los códigos umls de todas las las referencias de las líneas celulares

df_cell_umls.to_csv('cell_umls.csv', sep = '\t', index = False) #exportar dataframe a csv
