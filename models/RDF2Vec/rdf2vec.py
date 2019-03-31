#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 21:58:05 2019

@author: salihemredevrim
"""

#codes from: Lab 7
# import modules

import gensim, csv
import pandas as pd
import random

#%%
## create data structure for knowledge graph
def addTriple(net, source, target, edge):
    if source in net:
        if  target in net[source]:
            net[source][target].add(edge)
        else:
            net[source][target]= set([edge])
    else:
        net[source]={}
        net[source][target] =set([edge])
            
def getLinks(net, source):
    if source not in net:
        return {}
    return net[source]

#%%
# Generate paths (entity->relation->entity) by radom walks
def randomWalkUniform(triples, startNode, max_depth=5):
    next_node =startNode
    path = str(startNode)+'->'
    for i in range(max_depth):
        neighs = getLinks(triples,next_node)
        #print (neighs)
        if len(neighs) == 0: break
       # weights = []
        queue = []
        for neigh in neighs:
            for edge in neighs[neigh]:
                queue.append((edge,neigh))
        edge, next_node = random.choice(queue)
        path = path +str(edge)+'->'
        path = path +str(next_node)+'->'
    path =path.split('->')
    return path

#%%
def randomNWalkUniform(triples, n, walks, path_depth):
    path=[]
    for k in range(walks):
        walk = randomWalkUniform(triples, n, path_depth)
        path.append(walk)
    return path

#%%
# Build the knowledge graph structure
def preprocess(fname):
    triples = {}

    train_counter = 0

    print (fname)

    for line in csv.reader(open(fname), delimiter=' ', quotechar='"'):
        
        h = line[0]
        r = line[1]
        t = line[2]
        t = t.replace('>.','>')
        
        train_counter +=1

        addTriple(triples, h, t, r)
        train_counter+=1
    print ('Triple:',train_counter)
    return triples    

#%%
    
file = 'rdf_distinct.nt'
triples = preprocess(file)
entities = list(triples.keys())
vocabulary = entities

#%%
walks = 150
path_depth = 4

sentences =[]
for word in vocabulary:
    sentences.extend( randomNWalkUniform(triples, word, walks, path_depth) )
    
#%% skip-gram
model1 = gensim.models.Word2Vec(size=50, workers=4, window=5, sg=1, negative=25, iter=10)
model1.build_vocab(sentences)

corpus_count = model1.corpus_count
model1.train(sentences, total_examples = corpus_count, epochs = 10)

#%%

my_dict = dict({})
for idx, key in enumerate(model1.wv.vocab):
    my_dict[key] = model1.wv[key]

df = pd.DataFrame.from_dict(my_dict).T

#keep only books and users
df2 =df.reset_index()

df_books = df2[df2['index'].str.contains('www.goodreads.com/book')].reset_index(drop=True)

df_books['goodreads_book_id'] = df_books['index'].str.split('show/', 1).str[1]
df_books['goodreads_book_id'] = df_books['goodreads_book_id'].str.split('>', 1).str[0].astype(int)
df_books = df_books.drop(['index'], axis=1)

df_users = df2[df2['index'].str.contains('www.goodreads.com/user')].reset_index(drop=True)
df_users['user_id'] = df_users['index'].str.split('show/', 1).str[1]
df_users['user_id'] = df_users['user_id'].str.split('>', 1).str[0].astype(int)
df_users = df_users.drop(['index'], axis=1)

#%%save all
writer = pd.ExcelWriter('user_embeddings_sg.xlsx', engine='xlsxwriter');
df_users.to_excel(writer, sheet_name= 'user_embeddings');
writer.save();

writer = pd.ExcelWriter('book_embeddings_sg.xlsx', engine='xlsxwriter');
df_books.to_excel(writer, sheet_name= 'book_embeddings');
writer.save();

#%% cbow
model2 = gensim.models.Word2Vec(size=50, workers=4, window=5, sg=0, negative=25, iter=10, cbow_mean=1, alpha = 0.05)
model2.build_vocab(sentences)

corpus_count = model2.corpus_count
model2.train(sentences, total_examples = corpus_count, epochs = 10)

#%%

my_dict2 = dict({})
for idx, key in enumerate(model2.wv.vocab):
    my_dict2[key] = model2.wv[key]

df = pd.DataFrame.from_dict(my_dict2).T

#keep only books and users
df2 =df.reset_index()

df_books = df2[df2['index'].str.contains('www.goodreads.com/book')].reset_index(drop=True)

df_books['goodreads_book_id'] = df_books['index'].str.split('show/', 1).str[1]
df_books['goodreads_book_id'] = df_books['goodreads_book_id'].str.split('>', 1).str[0].astype(int)
df_books = df_books.drop(['index'], axis=1)

df_users = df2[df2['index'].str.contains('www.goodreads.com/user')].reset_index(drop=True)
df_users['user_id'] = df_users['index'].str.split('show/', 1).str[1]
df_users['user_id'] = df_users['user_id'].str.split('>', 1).str[0].astype(int)
df_users = df_users.drop(['index'], axis=1)

#%%save all
writer = pd.ExcelWriter('user_embeddings_cbow.xlsx', engine='xlsxwriter');
df_users.to_excel(writer, sheet_name= 'user_embeddings');
writer.save();

writer = pd.ExcelWriter('book_embeddings_cbow.xlsx', engine='xlsxwriter');
df_books.to_excel(writer, sheet_name= 'book_embeddings');
writer.save();

#%%
#took names for books for tensorflow projector 

#df_books = pd.read_excel('book_embeddings_sg.xlsx');
#vlookup = pd.read_excel('books_last.xlsx')
#
#projector = pd.merge(df_books, vlookup[['goodreads_book_id', 'title']], how='left', on='goodreads_book_id')
#projector = projector.drop(['goodreads_book_id'], axis=1)
#
#writer = pd.ExcelWriter('projector.xlsx', engine='xlsxwriter');
#projector.to_excel(writer, sheet_name= 'projector');
#writer.save();
