#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 20:33:55 2019

@author: salihemredevrim
"""
#Data source: http://fastml.com/goodbooks-10k-a-new-dataset-for-book-recommendations/

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from fuzzywuzzy import fuzz

#%%
#take datasets
ratings = pd.read_csv("ratings.csv")
books = pd.read_csv("books.csv")
tags = pd.read_csv("tags.csv")
book_tags = pd.read_csv("book_tags.csv")
pd.set_option('precision', 0)
#%%

#check distinct number of book_ids and goodreads_book_ids
count1 = books['book_id'].value_counts()
check1 = count1[count1 != 1]

max_book_id = books.book_id.max()
max_book_id2 = ratings.book_id.max()

count2 = books['goodreads_book_id'].value_counts()
check2 = count2[count2 != 1]

count3 = books.groupby('goodreads_book_id').book_id.nunique()
check3 = count3[count3 != 1]
#all distinct!

#Action 0
#Books with info in Arabic alphabet were eliminated
books = books[books['language_code'] != 'ara']

#Same language code for all English books
books['language_code'] = books.apply(lambda x: 'eng' if x['language_code'] in ('en-US', 'en-GB', 'en-CA', 'en') else x['language_code'], axis=1)

#language distribution
count11 = books['language_code'].value_counts().reset_index(drop=False)

count12 = books[books['language_code'] == 'tur']

#%% clear
del count1, check1, max_book_id, max_book_id2, count2, check2, count3, check3, count11, count12
#%%

#Action 1
#so, I can replace book_id in ratings dataset by goodreads_book_id
books2 = books[['book_id', 'goodreads_book_id']]
ratings2 = pd.merge(ratings, books2, how='inner', on='book_id')
ratings2 = ratings2[['user_id', 'goodreads_book_id', 'rating']]
ratings2 = ratings2[ratings2.goodreads_book_id.isnull() == 0]

#ratings distribution (around ~70% is 4 or 5)
plt.hist(x=ratings2['rating'])
plt.show()

#ratings distributions per book and user
#per book
gb_books = ratings2.groupby('goodreads_book_id').user_id.nunique().reset_index(name= 'num_of_ratings')

plt.hist(x=gb_books['num_of_ratings'])
plt.show()

#I've done the neutral rating (3) elimination here, because I'll make it classification problem as binary
ratings2 = ratings2[ratings2['rating'] != 3]

#per user
gb_users = ratings2.groupby('user_id').goodreads_book_id.nunique().reset_index(name= 'num_of_ratings')
plt.hist(x=gb_users['num_of_ratings'])
plt.show()

#%%
#Action 2

#y classification version 
ratings2['rating_binary45'] = ratings2.apply(lambda x: 1 if x['rating'] > 3 else 0 , axis=1)
ratings2['rating_binary5'] = ratings2.apply(lambda x: 1 if x['rating'] > 4 else 0 , axis=1)

gb_users2 = ratings2.groupby(['user_id','rating_binary45'])['goodreads_book_id'].count().reset_index()

#to obtain a robust model, first I filtered users having less than 20 ratings for each class
gb_users3 = gb_users2[gb_users2.rating_binary45 == 0]
gb_users0 = gb_users3[gb_users3.goodreads_book_id >= 20]

gb_users33 = gb_users2[gb_users2.rating_binary45 == 1]
gb_users11 = gb_users33[gb_users33.goodreads_book_id >= 20]

gb_users11 = gb_users11[gb_users11.user_id.isin(gb_users0.user_id)] 

ratings3 = ratings2[ratings2.user_id.isin(gb_users11.user_id)] 
#6057 readers left

#From those unbalanced classes, I've decided to take equal sizes from each (can be calibrated in real world later)
#Except rating 3 because it means "neutral": So, it will be like 1 and 2 vs 4 and 5 
#Sum of ratings in (1,2) is around 174K, I'll take 87K per remaining ratings

rating12 = ratings3[ratings3.rating < 3].reset_index(drop=True)

rating4 = ratings3[ratings3.rating == 4] 
rating4 = rating4.sample(n=87000, random_state=1905)
data1 = rating12.append(rating4, ignore_index=True)

rating5 = ratings3[ratings3.rating == 5] 
rating5 = rating5.sample(n=87000, random_state=1905)
data2 = data1.append(rating5, ignore_index=True) 

ratings_last = data2

#the number of user - book pairs drop to 348K from 6M

#%%
del books2, data1, data2, gb_users, gb_users2, gb_users3, gb_users0, gb_users33, gb_users11, rating12, rating4, rating5, ratings2

#%%  
#Action 3
#Train test split with 30%
#I won't use training ratings info in RDF triples

X = ratings_last[['user_id', 'goodreads_book_id']]
y = ratings_last[['rating','rating_binary45', 'rating_binary5', 'user_id', 'goodreads_book_id']]

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y[['rating_binary45','user_id']], test_size=0.3) #stratified wrt ratings and users

#variables will be added after embeddings!!

#%%
del X, y

#%%
#Action 4
#columns for RDFs from books

#just filtered the books in my sample sets

books_last = pd.merge(books, pd.DataFrame(ratings_last['goodreads_book_id'].unique()), how='inner', left_on='goodreads_book_id', right_on = 0) 

books_last = books_last[['goodreads_book_id', 'title', 'authors', 'language_code',
                'original_publication_year']]

#keep only the first writer and fix names as one word (However I let the title as it was since maybe a word in the title may trigger user's interest)
books_last['authors'] = books_last.authors.str.split(',').str[0]  
books_last['authors'] = books_last.authors.str.replace(' ','_')

#author distribution
authors = books_last['authors'].value_counts().reset_index(drop=False)

#%%
#Action 4.2
#columns for RDFs from users
#For each user in the training set I will create triples if s/he rated a book 4 or 5 (as user-tagged-book)

keep_y = y_train[y_train['rating_binary45'] == 1].reset_index()
keep_y = keep_y['index']

keep_X = X_train[X_train.index.isin(keep_y)]

#%% 
#Action 5
#Analysis of tags data

#how many times those tags used for different books before
book_tags2 = book_tags.groupby('tag_id').goodreads_book_id.nunique().reset_index(name= 'num_of_books')

#distribution
plt.hist(x=book_tags2['num_of_books'])
plt.show()

#at least 50 different books should be tagged with this tag
book_tags3 = book_tags2[book_tags2['num_of_books'] >= 50]
book_tags4 = pd.merge(book_tags3, tags, how='left', on='tag_id')
book_tags5 = book_tags4[['tag_id', 'tag_name']]

#split words
df2 = book_tags5
df3 = df2.tag_name.astype('str').str.split('-',expand=True).stack()
book_tags6 = df2.join(pd.Series(index=df3.index.droplevel(1), data=df3.values, name='new_var1'))
book_tags6 = book_tags6[['tag_id', 'new_var1']].rename(columns={'new_var1': 'tag_name'}).reset_index(drop=True)
 
#merge to books 
book_tags7 = pd.merge(book_tags, book_tags6, how='inner', on='tag_id')
book_tags7 = book_tags7[['goodreads_book_id', 'tag_id', 'tag_name']]


#Action 5.2
#manual elimination (Expert opinion!)
#group by again by new tag_name
book_tags8 = book_tags7.groupby('tag_name').goodreads_book_id.nunique().reset_index(name= 'num_of_books')
    
#manual ignore (some common tags are not quite related to books)    
ignore = [
'',
'01',
'02',
'04',
'1',
'100',
'1000',
'1001',
'1001books',
'16th',
'17th',
'18th',
'1945',
'1994',
'1996',
'1997',
'1998',
'1999',
'1st',
'2',
'200',
'2000',
'2001',
'2002',
'2003',
'2004',
'2005',
'2006',
'2007',
'2008',
'2009',
'2010',
'2011',
'2012',
'2013',
'2014',
'2015',
'2016',
'2017',
'3',
'30',
'307',
'311',
'314',
'33',
'3601',
'4',
'40',
'5',
'50',
'501',
'6',
'65',
'66',
'a',
'abandoned',
'abbi',
'about',
'absolute',
'acceptance',
'age',
'ago',
'all',
'alphas',
'already',
'alt',
'always',
'amazon',
'and',
'ap',
'archer',
'arcs',
'arthur',
'arthuriana',
'as',
'ashley',
'at',
'audible',
'audio',
'audiobook',
'audiobooks',
'author',
'authors',
'b',
'bd',
'be',
'book',
'bookclub',
'bookgroup',
'booklist',
'books',
'bookshelf',
'borrowed',
'bought',
'buy',
'c',
'calibre',
'can',
'cd',
'challenge',
'chances',
'club',
'collection',
'coming',
'completed',
'couldn',
'cs',
'currently',
'd',
'dc',
'default',
'did',
'didn',
'die',
'do',
'dr',
'e',
'en',
'f',
'faeries',
'faves',
'favorite',
'favorites',
'favourite',
'favourites',
'fi',
'finish',
'finished',
'first',
'for',
'free',
'g',
'gave',
'get',
'gg',
'gn',
'h',
'hardcover',
'have',
'hold',
'home',
'horror',
'hp',
'hq',
'hr',
'i',
'ii',
'in',
'it',
'j',
'jd',
'jk',
'jr',
'k',
'ka',
'ku',
'l',
'library',
'list',
'listened',
'lit',
'lj',
'll',
'loved',
'm',
'me',
'meh',
'mf',
'mg',
'mm',
'more',
'must',
'my',
'n',
'na',
'need',
'never',
'next',
'nf',
'no',
'non',
'nonfic',
'nook',
'not',
'nr',
'of',
'on',
'once',
'our',
'own',
'owned',
'p',
'paperback',
'part',
'pd',
'r',
'ra',
're',
'read',
'reading',
'reads',
'recommended',
'release',
'reviewed',
's',
'self',
'sf',
'shelf',
'sk',
'stars',
'suspense',
't',
'tbr',
'than',
'the',
'time',
'to',
'true',
'tv',
'u',
'uf',
'unread',
'up',
'us',
've',
'want',
'we',
'wish',
'wishlist',
'y',
'ya',
'Ýa', 
'favs',
'fav',
'rereads',
'reread',
'favoritos', 
'kindle', 
'booker',
'ebook', 
'ebooks', 
'ibooks', 
'bookcrossing',
'maybe',
'reader', 
'readers',
'goodreads', 
'selection', 
'selections',
'best', 
'was',
'yet',
'when',
'where', 
'downloaded', 
'collections', 
'this', 
'that',
'stuff', 
'stopped',
'start',
'started', 
'dnf',
'released',
'releases',
'review', 
'soon',
'new',
'giveaways',
'interested'
'interest',
'interesting'
]

#%%
book_tags9 = book_tags7[~book_tags7['tag_name'].isin(ignore)].reset_index(drop=True)
book_tags9 = book_tags9.groupby(['tag_id', 'tag_name']).goodreads_book_id.count().reset_index()
book_tags9 = book_tags9.sort_values(by='goodreads_book_id', ascending=False).reset_index(drop=True)

final_tag_count = book_tags9.tag_name.value_counts()

#Action 5.3
#Similarity check and update (Levinsthein distance)
for k in range(len(book_tags9)):
    
    str1 = book_tags9['tag_name'].iloc[k]
    for l  in range(k+1, len(book_tags9)):
        
        str2 = book_tags9['tag_name'].iloc[l]
        comp = fuzz.ratio(str1, str2)
        
        if (comp >= 90 and comp < 100):
            print(str1,' VS ', str2, comp, k, l)
            book_tags9['tag_name'].iloc[l] = book_tags9['tag_name'].iloc[k];
            

#%%   
#Action 5.4            
#Lastly, non-latin tags were eliminated
def isEnglish(s):
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
 
        return True
    
book_tags9['check'] = 1
for k in range(len(book_tags9)):
    s = book_tags9['tag_name'].iloc[k]
    book_tags9['check'].iloc[k] = isEnglish(s)

book_tags9 = book_tags9[book_tags9['check'] == 1]

#%%

#take updated names 
book_tags_last0 = pd.merge(book_tags7[['goodreads_book_id', 'tag_id']], book_tags9[['tag_id', 'tag_name']], how='inner', on='tag_id')
book_tags_last = book_tags_last0[['goodreads_book_id', 'tag_name']].drop_duplicates()

#%%
del ignore, book_tags_last0, book_tags5, book_tags4, book_tags3, book_tags2, keep_y, final_tag_count
del df2, df3, book_tags6, book_tags7, book_tags8, book_tags9

#%%save all
writer = pd.ExcelWriter('ratings_last.xlsx', engine='xlsxwriter');
ratings_last.to_excel(writer, sheet_name= 'ratings');
writer.save();

writer = pd.ExcelWriter('X_train.xlsx', engine='xlsxwriter');
X_train.to_excel(writer, sheet_name= 'x_s');
writer.save();

writer = pd.ExcelWriter('X_test.xlsx', engine='xlsxwriter');
X_test.to_excel(writer, sheet_name= 'x_s');
writer.save();

writer = pd.ExcelWriter('y_train.xlsx', engine='xlsxwriter');
y_train.to_excel(writer, sheet_name= 'y_s');
writer.save();

writer = pd.ExcelWriter('y_test.xlsx', engine='xlsxwriter');
y_test.to_excel(writer, sheet_name= 'y_s');
writer.save();

#these will be RDFs
writer = pd.ExcelWriter('book_tags_last.xlsx', engine='xlsxwriter');
book_tags_last.to_excel(writer, sheet_name= 'tags');
writer.save();

writer = pd.ExcelWriter('books_last.xlsx', engine='xlsxwriter');
books_last.to_excel(writer, sheet_name= 'books');
writer.save();

writer = pd.ExcelWriter('user_rates_training.xlsx', engine='xlsxwriter');
keep_X.to_excel(writer, sheet_name= 'user_rating');
writer.save();