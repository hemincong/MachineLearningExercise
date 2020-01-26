#!/usr/bin/env python
# -*- coding: utf-8 -*-


def loadMovieList():
    # GETMOVIELIST reads the fixed movie list in movie.txt and returns a
    # cell array of the words
    # movieList = GETMOVIELIST() reads the fixed movie list in movie.txt
    # and returns a cell array of the words in movieList.

    movieList = {}
    index = 0
    with open('resource/movie_ids.txt', "r", encoding="latin-1") as fp:
        line = fp.readline()
        while line:
            s = line.split(' ', maxsplit=1)
            movieList[index] = s[1].strip()
            index += 1
            line = fp.readline()

    return movieList
