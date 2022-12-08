
# import packages
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

from textblob import TextBlob
import matplotlib.pyplot as plt


def lemmatize_documents(documents:pd.Series):
    '''
    Takes a dataframe Series of documents and returns a list of the lemmatized form of documents.
    '''

    # turn each title into a TextBlob object, then for each word in it, lemmatize that word
    # then rejoin the sentence with spaces between each word and store the new document in a new column

    lem_docs = [" ".join([w.lemmatize() for w in TextBlob(doc).words]) for doc in documents]
    return lem_docs


def get_min_df(start:int, end:int, documents:pd.Series):
    '''
    Given a range for the minimum document frequency parameter and the documents to analyze, this function returns
    a plot of the number of features including in a document term matrix when the particular min_df threshold is set
    to match each value in the range. Outputs a plot and returns the results in a dataframe.
    '''

    # create list for storing loop results
    min_df_list = []
    # for each value in user-provided range
    for min in range(start, end+1, 1):
        # create a document term matrix with the iterative min_df value, dropping stop words and considering bigrams
        dtm = CountVectorizer(min_df=(min), stop_words="english", ngram_range=(1,2)).fit_transform(documents)
        # store number of features
        features = dtm.shape[1]
        # store min_df value and features to list
        min_df_list.append( (min, features) )

    # transform list of loop results into dataframe
    min_df = pd.DataFrame(min_df_list, columns=['min_df', 'features'])

    # plot results and display, returning the df for reference
    plt.figure(figsize=(16,8))
    plt.plot(min_df['min_df'], min_df['features'], marker='o')
    plt.xlabel('Minimum Document Frequency')
    plt.ylabel('Number of Features')
    plt.title('Number of Features in Document Term Matrix, Min Freq.')
    plt.show()  

    return min_df


def get_model_perplexity(documents:pd.Series, min_df:int, start:int, end:int):
    '''
    Taking in a Series of documents, a pre-decided minimum document frequency and a range of component values to test,
    this calculates and plots the Perplexity value for a LDA model using a lemmatized tfidf implementation.
    '''

    # call helper function to lemmatize documents
    lem_docs = lemmatize_documents(documents)

    # store perplexity and value in list
    perplexity_list = []

    for n_comp in range(start, end+1, 1):
        # build models and return the model and tfidf
        LDA, tfidf, _ = get_model_features(lem_docs, min_df, n_components=n_comp)
        # store perplexity and the range value from the created LDA model
        perplexity = round(LDA.perplexity(tfidf),2)
        perplexity_list.append( (n_comp, perplexity) )

    per_df = pd.DataFrame(perplexity_list, columns=['n_components', 'perplexity'])

    # plot Perplexity, output plot, and return df for reference
    plt.figure(figsize=(16,8))
    plt.plot(per_df['n_components'], per_df['perplexity'], marker='o')
    plt.xlabel('Number of Components')
    plt.ylabel('Perplexity')
    plt.title('Number of Components, Perplexity')
    plt.show()  

    return per_df


def get_model_features(documents:pd.Series, min_df:int, n_components:int):
    '''
    Taking in a series of documents and predetermined parameters for the minimum document frequency
    and number of components, returns the LDA model, the tf-idf, and the features.
    '''

    # transform documents into term frequency inverse document frequency matrix
    # consider bigrams, remove stopwords, and transform all text to lowercase
    tfidf_v = TfidfVectorizer(min_df=min_df, stop_words='english', ngram_range=(1,2), lowercase=True)
    tfidf = tfidf_v.fit_transform(documents)

    # build the LDA model with chosen components then fit the model to the tfidf matrix of the original documents
    LDA = LatentDirichletAllocation(n_components=n_components, random_state=0, learning_method='batch')
    LDA_fit = LDA.fit_transform(tfidf)

    features = tfidf_v.get_feature_names_out()

    return LDA, tfidf, features


def print_top_words(model, feature_names, n_top_words:int):
    '''
    Given an LDA model and the resulting features, print out a pre-determined number of top words.
    '''
    # loop through each of the components in the model
    for i, topic in enumerate(model.components_):
        # format topic heading Topic 0:
        topic_output = "Topic #%d: " % i
        # for each of the n_top_words, append the text to the message with a space in between
        topic_output += ", ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        # line break between text
        print(topic_output, '\n')
    print()


def process_text(documents:pd.Series, min_df, n_components:int, n_top_words:int):
    """
    Top-level function that takes a series of documents, pre-determined model parameters, the desired
    number of top words to return, and the intended form of output, either 'print' or 'plot'.

    Output displays either a list or plot of the input number of top words for the chosen number of components.
    """

    # call helper functions to lemmatize documents and build the LDA model and extract its features
    lem_docs = lemmatize_documents(documents)

    model, _, features = get_model_features(lem_docs, min_df, n_components)

    print_top_words(model, features, n_top_words)
