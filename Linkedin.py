#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 11:13:19 2021

@author: benjamin.guigon
"""

import numpy as np
import pandas as pd 
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud,STOPWORDS
import nltk
import os
from nltk.corpus import stopwords
from spacy.lang.fr.stop_words import STOP_WORDS as fr_stop
from nltk.probability import FreqDist
from collections import Counter
import re
import string
from nltk.tokenize import word_tokenize
import datetime as dt
import time
import streamlit as st
import gender_guesser.detector as gender
from PIL import Image
st.set_option('deprecation.showPyplotGlobalUse', False)
#st.set_page_config(layout='wide')

d = gender.Detector()

stop = set(stopwords.words("french"))
hashtags = re.compile(r"^#\S+|\s#\S+")
mentions = re.compile(r"^@\S+|\s@\S+")
urls = re.compile(r"https?://\S+")

def process_text(text):
    text = re.sub(r'http\S+', '', text)
    text = hashtags.sub(' hashtag', text)
    text = mentions.sub(' entity', text)
    return text.strip().lower()

def remove_URL(text):
    url = re.compile(r"https?://\S+|www\.\S+")
    return url.sub(r"", text)


def remove_html(text):
    html = re.compile(r"<.*?>")
    return html.sub(r"", text)


def remove_emoji(string):
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub(r"", string)

def remove_punct(text):
    table = str.maketrans("", "", string.punctuation)
    return text.translate(table)




def remove_stopwords(text):
    text = [word.lower() for word in text.split() if word.lower() not in stop]
    return " ".join(text)


def net(words):
    
    
    words = words.replace('Ã©', 'é')
    words = words.replace('ã§', 'ç')
    words = words.replace('ã¨', 'è')
    words = words.replace('â\x80\x99', "\'")
    words = words.replace('ã©', 'é')
    words = words.replace('ã', 'à')
    words = words.replace('ã´', 'ô')
    words = words.replace('â\x80¦', '...')
    words = words.replace('à´', 'ô')

    words = words.replace('benjamin', '')
    #words = words.replace('Bonjour', '')
    
    words = words.replace('guigon', '')
    words = words.replace("j'espère", '')
    words = words.replace("allez", '')
    words = words.replace("bien", '')
    words = words.replace("bonne", '')
    words = words.replace("journée", '')
    words = words.replace("remercie", '')
    words = words.replace("spinmail", '')
    
    words = words.replace("x80", '')
    words = words.replace("à", '')
    words = words.replace(",", '')
    words = words.replace("?", '')
    words = words.replace("!", '')
    words = words.replace(":", '')
    
    words = words.replace("â", '')
    words = words.replace("ca", '')
    words = words.replace("ça", '')
    
    return words

def function_nettoyage(df,col):

    df = df.copy()
    df[col] = df[col].astype('str')
    df[col] = df[col].apply(process_text)
    df[col] = df[col].apply(remove_URL)
    df[col] = df[col].apply(remove_html)
    df[col] = df[col].apply(remove_emoji)
    df[col] = df[col].apply(remove_punct)
    df[col] = df[col].apply(remove_stopwords)
    df[col] = df[col].apply(net)
    
    return df

def wordBarGraphFunction(df,column,title,num_word):
    topic_words = [ z.lower() for y in
                       [ x.split() for x in df[column] if isinstance(x, str)]
                       for z in y]
    
    word_count_dict = dict(Counter(topic_words))
    popular_words = sorted(word_count_dict, key = word_count_dict.get, reverse = True)
    popular_words_nonstop = [w for w in popular_words if w not in stopwords.words("french")]
    
    #st.bar_chart(range(num_word), [word_count_dict[w] for w in reversed(popular_words_nonstop[0:num_word])])
    
    plt.barh(range(num_word), [word_count_dict[w] for w in reversed(popular_words_nonstop[0:num_word])])
    plt.yticks([x + 0.5 for x in range(num_word)], reversed(popular_words_nonstop[0:num_word]))
    plt.title(title)
    #plt.show()
    st.pyplot()

def wordBarGraphFunction2(df,column,title):
    topic_words = [ z.lower() for y in
                       [ x.split() for x in df[column] if isinstance(x, str)]
                       for z in y]
    word_count_dict = dict(Counter(topic_words))
    popular_words = sorted(word_count_dict, key = word_count_dict.get, reverse = True)
    popular_words_nonstop = [w for w in popular_words if w not in stopwords.words("french")]
    return popular_words_nonstop

# Function to plot wordcloud
def plot_wordcloud(data,max_words):
    words = '' 
    stopwords = set(STOPWORDS) 
    for val in data.values: 
        val = str(val) 
        tokens = val.split() 
        for i in range(len(tokens)):
            tokens[i] = tokens[i].lower() 

        words += " ".join(tokens)+" "
  
    wordcloud = WordCloud(width = 800, height = 800, 
                    background_color ='white', 
                    stopwords = stopwords, 
                    min_font_size = 10,
                    max_words=max_words).generate(words) 
    
    plt.figure(figsize = (2,4), facecolor = None) 
    plt.imshow(wordcloud) 
    plt.axis("off") 
    plt.tight_layout(pad = 0)
    st.pyplot()
    #plt.show()
    
# Count unique words
def counter_word(text):
    count = Counter()
    for i in text.values:
        for word in i.split():
            count[word] += 1
    return count


df_connection = pd.read_csv('/Users/benjamin.guigon/Desktop/PSB/Semestre 3/Hadoop/Streamlit/Linkedin/Connections2.csv',sep=',',encoding='utf-8')
df_contact = pd.read_csv('/Users/benjamin.guigon/Desktop/PSB/Semestre 3/Hadoop/Streamlit/Linkedin/Contacts.csv',sep=',',encoding='utf-8')
df_message = pd.read_csv('/Users/benjamin.guigon/Desktop/PSB/Semestre 3/Hadoop/Streamlit/Linkedin/messages.csv',sep=',',encoding='utf-8')


df_connection['Position'] = df_connection['Position'].fillna('')
df_connection['Connected On'] = df_connection['Connected On'].fillna('01 Dec 2015')
df_connection['Connected On'] = df_connection['Connected On'].astype('str')
df_connection['Date'] = pd.to_datetime(df_connection['Connected On'].apply(lambda x: dt.datetime.strptime(x, '%d %b %Y')))
df_connection['Genre'] = df_connection['First Name'].apply(lambda x: d.get_gender(x))
df_connection['Company'] = df_connection['Company'].replace('Societe Generale Corporate and Investment Banking - SGCIB','SGCIB')
df_connection['Company'] = df_connection['Company'].replace("ESILV - Ecole Supérieure d'Ingénieurs Léonard de Vinci",'ESILV')
df_connection['Name'] = df_connection['First Name'] + ' ' + df_connection['Last Name']
df_connection.drop(columns = ['Email Address'],inplace=True)
df_connection.dropna(inplace=True)
df_tot = pd.merge(df_message,df_connection, left_on='FROM',right_on='Name',how='inner')



#st.title("My Linkedin Statistics")
#st.write("Test .write")
st.title("My Linkedin Statistics")
st.markdown('The dashboard is an extraction of my activity on Linkedin')
st.markdown('You will find some statistics about my connections and some NLP insights of my messages')
#st.markdown('Coronavirus disease (COVID-19) is an infectious disease caused by a newly discovered coronavirus. Most people infected with the COVID-19 virus will experience mild to moderate respiratory illness and recover without requiring special treatment.')
#st.sidebar.title("Visualization Selector")
#st.sidebar.markdown("Select the Charts/Plots accordingly:")

st.write('---')


choix = st.sidebar.radio("Categories :",options=['Viz','NLP'])

if choix == 'Viz':
    '''
        ### Some insights of my network
    '''
    
    choix_3 = st.sidebar.selectbox(
     'Which page do you want',
     ('Page 1', 'Page 2','Page 3'))
    if choix_3 == 'Page 1':
        ## Pie chart
        max_company = st.slider("Choose the number of company", min_value=5, max_value=20, step=1)
        fig, ax = plt.subplots()
        x = df_connection.groupby(by=['Company']).count().sort_values(by='First Name',ascending=False)['Position'][:max_company]
        y = x.index
        ax.pie(x,labels=y,autopct='%1.1f%%')
        st.pyplot(fig)
        
        fig1, ax1 = plt.subplots()
        
        x1 = df_connection.groupby(['Genre']).count()['Company']
        
        y1 = x1.index
        ax1.pie(x1,labels=y1,autopct='%1.1f%%')
        st.pyplot(fig1)
        
        
        # Affichage de la liste
        #mot = st.text_area("Show The people i'm connecting with only with a word", "__",
        #                    help='Word like "Data" or "Sales"')
        #df_1 = df_connection.loc[df_connection['Position'].str.contains(mot),['First Name','Last Name','Company','Position','Connected On']]
        #st.write("Il y a " +str(df_1.shape[0])+ " personne dans mon réseau avec le mot "+str(mot))
        #st.table(df_1.head(20).reset_index(drop=True))
        
        
        mot = st.text_area("Show The people i'm connecting with only with a word", "",
                            help='Word like "Data" or "Sales" or "ALL" to see all the people of the company')
        
        list_company = list(df_connection.sort_values(['Company'])['Company'].dropna().unique())
        list_company.append('ALL')
        list_company = sorted(list_company)
        company = st.selectbox("Company: ",list_company) 
        
        if mot =='ALL':
            df_1 = df_connection.loc[(df_connection['Company'] == company)]
            st.table(df_1[['Name','Company','Position','Connected On']].head(20))
        
        
         
        elif company == 'ALL':
            df_1 = df_connection.loc[(df_connection['Position'] == mot)]
            st.table(df_1[['Name','Company','Position','Connected On']].head(20))
        
        else:
            if len(company) >1:
                
                df_1 = df_connection.loc[(df_connection['Position'].str.contains(mot)) & (df_connection['Company'] == company)]
                
                st.write("Il y a " +str(df_1.shape[0])+ " personne dans mon réseau avec le mot "+str(mot)+" from the company :" +str(company))
                
                st.table(df_1[['Name','Company','Position','Connected On']].head(20))#.loc[(df_connection['Position'].str.contains(mot)) & (df_connection['Company'] == company)].head())
            
          
                
            else:
                
                 st.write('vide ?')
                 
        
        #mot = st.text_area("Show The people i'm connecting with only with a word", "",
        #                    help='Word like "Data" or "Sales" or "ALL" to see all the people of the company')
        
        list_nom = list(df_connection.sort_values(['Name'])['Name'].unique())
        nom = st.selectbox("Name: ",list_nom) 


        df_1 = df_connection.loc[df_connection['Name'].str.contains(nom)]
        st.table(df_1[['Name','Company','Position','Connected On']].head(20))
        
        
        
        
        
        
        ## Affichage avec la date
        date_start = st.date_input("Date start", value = pd.to_datetime(df_connection['Date'].tail(1).values[0]),min_value=pd.to_datetime(df_connection['Date'].tail(1).values[0]),
                                   max_value=pd.to_datetime(df_connection['Date'].head(1).values[0]),)
        
        
        plt.figure(figsize = (10,4))
        df_date = df_connection.loc[df_connection['Date'] > pd.to_datetime(date_start) ]
        st.line_chart(df_date.groupby(['Date']).count()['Position'])
        
        
        
        
        
        
        
    elif choix_3 == 'Page 2':
        
        
        
        
        df_message2 = df_tot.drop(columns=['CONVERSATION ID','SENDER PROFILE URL','CONVERSATION TITLE','SUBJECT','FOLDER','First Name','DATE','CONTENT','Last Name','Connected On','Date','Name'])
        A = df_message.groupby(['FROM']).count().shape[0]
        B = df_connection.shape[0]
        D = round(100*A/B,2)
        C = df_message.groupby(['TO']).count().shape[0]
        E = round(100*C/B,2)
        
        st.write('J\'ai actuellement ' + str(B) + ' personnes dans mon réseau')
        
        st.write('---')
        
        st.write('Nombre de personne de mon réseau qui m\'ont envoyé un message : '+str(A))
        st.write('Ce qui fait : '+str(D)+'%')
    
        st.write('---')
        
        st.write('Nombre de personne de mon réseau à qui j\'ai envoyé un message : '+str(C))
        st.write('Ce qui fait : '+str(E)+'%')
        
        st.write('---')
        st.write('Ce qui donne un taux de réponse approximatif de : ' + str(round(100*D/E,2))+'%')
        
        
        st.write('---')
        
        
        df_message2 = df_message.drop(columns=['CONVERSATION ID','SENDER PROFILE URL','CONVERSATION TITLE','CONTENT','SUBJECT','FOLDER'])
        B = df_message2.loc[df_message2['TO'] == 'Benjamin Guigon'].groupby(['FROM']).count()['TO'].index
        df_no_response = df_message2.loc[(df_message2['FROM'] == 'Benjamin Guigon') & (~df_message2['TO'].isin(B))].reset_index(drop=True)
        #st.write(df_no_response.drop(columns=['FROM','DATE']).shape[0])
        num = len(df_no_response.drop(columns=['FROM','DATE']).groupby(['TO']).count())
        st.write('After such complicated calculs there are '+ str(num) + ' people that did not answer to my message. Not really cool...' )
        
        button =st.radio('Is is good ?',('Good !','Hm not bad','Be more impactful'))
        
    
        if button == 'Good !':
            image = Image.open('thanks you.png')
            st.image(image, caption='Youhoo')
            
        elif button == 'Hm not bad':
            image = Image.open('bof.jpeg')
            st.image(image, caption=':/')
            
        elif button == 'Be more impactful':
            image = Image.open('mauvais.jpeg')
            st.image(image, caption='Sad...')
            
        else:
            pass
        
        st.write('WAIT, there is an issue... How is it possible that there is '+str(num) +
         ' people that did not answer you whereas you said that you received messages from '+str(A) + ' different people and you send '+str(C)+ ' messages to different people.???')
        st.write('Actually the answer is quit simple, it\'s because of some message that came from group in Linkedin that distort the calcul and obviously there is some outlayer that are quit hard to separe from the rest...')

        
  
        

elif choix =='NLP':
    
    st.subheader("Let's start some NLP analyses of my Linkedin message.")
    
    stop_words = set(stopwords.words('english'))

    send_message = df_message.loc[(df_message['FROM'] == 'Benjamin Guigon')]
    receive_message = df_message.loc[(df_message['TO'] == 'Benjamin Guigon')& (df_message['FROM'] != 'LinkedIn Premium')]
    
    receive_message_clean = function_nettoyage(receive_message,'CONTENT')#['CONTENT']
    send_message_clean = function_nettoyage(send_message,'CONTENT')#['CONTENT']

    counter_r = counter_word(receive_message_clean['CONTENT'])
    counter_e = counter_word(send_message_clean['CONTENT'])

    
    #choix_2 = st.sidebar.radio("Categories :",options=['Message Send','Message Received'])
    choix_2 = st.sidebar.selectbox(
     'Which type do you want',
     ('Message Send', 'Message Received'))

    st.write('You selected:', choix_2)

    if choix_2 == 'Message Send':
    
        st.write("Messages send : %s words." % len(counter_e))
        num_word_s = st.slider("Choose the number of word send", value=50, min_value=5, max_value=200, step=1)
        
        '''
        ## Wordcloud of the messages a send
        '''
        plot_wordcloud(send_message_clean['CONTENT'],num_word_s)
        
        
        num_word_s_2 = st.slider("Choose the number of word send", value=10, min_value=10, max_value=20, step=1)
        
        '''
        ## Most common word Send
        '''
        wordBarGraphFunction(send_message_clean,'CONTENT','',num_word_s_2)
        
    
    elif choix_2 == 'Message Received':
    
        
        st.write("Messages received : %s word." % len(counter_r))
        
        num_word_r = st.slider("Choose the number of word received", value=50, min_value=5, max_value=200, step=1)
        
        '''
        ## Wordcloud of the messages a I received
        '''
        plot_wordcloud(receive_message_clean['CONTENT'],num_word_r)
        wordBarGraphFunction(receive_message_clean,'CONTENT','Most common word Reveived',20)
        
       






