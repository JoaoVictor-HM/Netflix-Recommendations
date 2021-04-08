import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import streamlit as st
import requests
from googlesearch import search
import webbrowser as wb


data = pd.read_csv('netflix_titles.csv')

# funções de suporte

tv_show = data['type'] == 'TV Show'
tv_show_data = data[tv_show].reset_index()
tv_show_data = tv_show_data.drop(['show_id', 'date_added', 'release_year', 'type', 'director'], axis=1)

movie = data['type'] == 'Movie'
movie_data = data[movie].reset_index()
movie_data = movie_data.drop(['show_id', 'type', 'date_added', 'release_year'], axis=1)

def get_movie_directors():
    movie_data['director']
    directors = []
    unique_directors = []
    for director in movie_data['director']:
        directors.append(director)
    for director in directors:
        if director not in unique_directors:
            unique_directors.append(director)
    return unique_directors

def get_movie_actors():
    cast = movie_data['cast']
    cast.astype('string')
    print(cast[0])
    actors = []
    for i in cast:
        try:
            i = i.split(', ')
            for actor in i:
                actors.append(actor)
        except Exception as e:
            pass

    unique_actors = []
    for actor in actors:
        if actor not in unique_actors:
            unique_actors.append(actor)
    return unique_actors


def get_tv_show_actors():
    cast = tv_show_data['cast']
    cast.astype('string')
    print(cast[0])
    actors = []
    for i in cast:
        try:
            i = i.split(', ')
            for actor in i:
                actors.append(actor)
        except Exception as e:
            pass

    unique_actors = []
    for actor in actors:
        if actor not in unique_actors:
            unique_actors.append(actor)
    return unique_actors

def busca_genero(genero):
  condicao = data['listed_in'].str.find(genero) != -1
  resultado = pd.DataFrame(data[condicao])
  return resultado

def busca_genero_TvShow(genero):
  condicao = tv_show_data['listed_in'].str.find(genero) != -1
  resultado = pd.DataFrame(tv_show_data[condicao])
  return resultado 

def busca_genero_Movie(genero):
  condicao = movie_data['listed_in'].str.find(genero) != -1
  resultado = pd.DataFrame(movie_data[condicao])
  return resultado

# funções para recomendação de filmes e séries

#FILMES

movie_features = ['director', 'cast', 'listed_in', 'description', 'title']

def combine_movie_features(row):
    return row['director']+' '+row['cast']+' '+row['listed_in']+' '+row['description']+' '+row['title']

for feature in movie_features:
    movie_data[feature] = movie_data[feature].fillna('')
movie_data['combined_features'] = movie_data.apply(combine_movie_features, axis=1)

movie_cv = CountVectorizer()
movie_count_matrix = movie_cv.fit_transform(movie_data['combined_features'])
movie_cosine_similarity = cosine_similarity(movie_count_matrix)

def find_title_from_index(index):
    return movie_data[movie_data.index == index]['title'].values[0]
def find_index_from_title(title):
    return movie_data[movie_data.title == title].index.values[0]

def movie_recommendation(movie_title):
    movie_index = find_index_from_title(movie_title)
    similar_movies = list(enumerate(movie_cosine_similarity[movie_index]))
    sorted_similar_movies = sorted(similar_movies, key = lambda x:x[1], reverse=True)[1:]
    rec_indexes = [tupla[0] for tupla in sorted_similar_movies[0:5]]
    rec_titles = []
    for index in rec_indexes:
        rec_titles.append(find_title_from_index(index))
    return rec_titles

#SERIES 

tv_show_features = ['description', 'listed_in', 'cast']

def combine_tv_show_features(row):
    return row['description']+' '+row['cast']+' '+row['listed_in']

for feature in tv_show_features:
    tv_show_data[feature] = tv_show_data[feature].fillna('')
tv_show_data['combined_features'] = tv_show_data.apply(combine_tv_show_features, axis=1)

tv_show_cv = CountVectorizer()
tv_show_count_matrix = tv_show_cv.fit_transform(tv_show_data['combined_features'])
tv_show_cosine_similarity = cosine_similarity(tv_show_count_matrix)

def tv_show_index_to_title(index):
    return tv_show_data[tv_show_data.index == index]['title'].values[0]
def tv_show_title_to_index(title):
    return tv_show_data[tv_show_data.title == title].index.values[0]

def tv_show_recommendation(tv_show_title):
    tv_show_index = tv_show_title_to_index(tv_show_title)
    similar_tv_shows = list(enumerate(tv_show_cosine_similarity[tv_show_index]))
    sorted_similar_tv_shows = sorted(similar_tv_shows, key=lambda x:x[1], reverse=True)[1:]
    indexes = [tupla[0] for tupla in sorted_similar_tv_shows[0:5]]
    titles = []
    for index in indexes:
        titles.append(tv_show_index_to_title(index))
    return titles

# Montando a aplicação





    

        












