from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
from stop_words import get_stop_words
#from scipy import sparse
import gensim
import string
import numpy as np
from utils import random_idx
from utils import utils
from utils import lang_vectors_utils as lvu

tokenizer = RegexpTokenizer(r'\w+')

# create English stop words list
en_stop = get_stop_words('en')

# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()

num_topics = 50
passes = 20
#topn = 10
k = 5000
N = 10000
# cluster_sizes is mapping to n-gram size
# cluster_sz in random_idx referring to specific element (int) in cluster_sizes, array
cluster_sizes = [1, 2, 3, 4, 5, 6, 7, 8]
ordered = 1
#assuming this is the alphabet bc of precedent in generate_text.py
#alph = 'abc' 
alphabet = string.lowercase + ' '
RI_letters = random_idx.generate_letter_id_vectors(N, k, alphabet)
# words should be unique? (such as permutations)
# number of words added in a syntax vector
syntax_granularities = [100,1000,10000]
# number of stacked syntax vectors per meaning matrix
meaning_granularities = [10,100,1000,10000]
def create_doc_set(path, files):
    doc_set = []
    for filename in files:
        f = open(path + filename, "r")
        doc_set.append(f.read())
        f.close()
    return doc_set


def tokenize(doc_set):
    # list for tokenized documents in loop
    texts = []
    # loop through document list
    for i in doc_set:
        # clean and tokenize document string
        raw = i.lower()
        tokens = tokenizer.tokenize(raw)
        # remove stop words from tokens
        stopped_tokens = [i for i in tokens if not i in en_stop]
        # stem tokens
        stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
        # add tokens to list
        texts.append(stemmed_tokens)
    return texts

def meaning_matrix(ldamodel, topicid, topn, dictionary):
    " NO "
    # token 2 id dictionary
    # print dictionary.token2id
    matrix = np.zeros((N,N))
    id2token = dictionary.id2token
    topic_terms = []

    for tup in ldamodel.get_topic_terms(topicid, topn):
        topic_terms.append(str(id2token[tup[0]]))

    for i in range(0,topn):
        term_vector = random_idx.id_vector(N, topic_terms[i], alphabet, RI_letters, ordered)
        matrix[i] = term_vector
    return matrix


def meaning_matrices(ldamodel, num_topics, topn, dictionary):
    " NO "
    matrices = np.zeros((num_topics,N,N))
    for topicid in range(0,num_topics):
        matrices[topicid] = create_meaning_matrix(ldamodel, topicid, topn, dictionary)
    return matrices


def vectorize_dictionary(dictionary):
    """
    returns an array indexed by token_id, token_id->vector representation of the token
    """
    vectors = []
    for i in range(0,len(dictionary.keys())):
        vectors.append(random_idx.id_vector(N, dictionary[i], alphabet, RI_letters))
    return vectors


def similarity_matrix(vectorized_dictionary):
    """
    calculate cosine similarity of every word. 
    """
    num_tokens = len(vectorized_dictionary)
    sm = np.zeros((num_tokens,num_tokens))

    for row in range(num_tokens):
        for col in range(num_tokens):
            sm[row][col] = np.dot(np.transpose(vectorized_dictionary[row][0]), vectorized_dictionary[col][0])
    return sm

def syntax_space(similarity_matrix, vectorized_dictionary):
    """
    number of granularities of words encoded per syntax vector x number of tokens x N
    index in numpy array on axis "number of tokens" equivalent to token id
    http://stackoverflow.com/questions/6910641/how-to-get-indices-of-n-maximum-values-in-a-numpy-array
    """
    num_tokens = similarity_matrix.shape[0]
    matrices = np.zeros((len(syntax_granularities), num_tokens, N))
    for gran_i in range(len(syntax_granularities)):
        for row in range(num_tokens):
            granularity = min(syntax_granularities[gran_i],num_tokens)
            most_similar_indices = np.argpartition(similarity_matrix[row],-granularity)[-granularity:]
            for key in most_similar_indices:
                matrices[gran_i][row] += vectorized_dictionary[key][0]
    return matrices

def meaning_space(ldamodel, topn, dictionary, vectorized_dictionary, syntaxed_space):
    """
    number of granularities meaning space x number of tokens x granularity x N
    index in numpy array on axis "number of tokens" equivalent to token id
    """
    num_tokens = similarity_matrix.shape[0]
    matrices = []
    for gran_i in range(len(meaning_granularities)):
        matrix = np.zeros((num_tokens, meaning_granularities[gran_i], N))
        for token_id in range(num_tokens):
            # rank topics by how much the specific token contributes to it. start stacking words
            # either choose the max topic or a certain number from each topic.
            # let's choose max or else lose the less popular tokens
            token_topic = ldamodel.get_term_topics(token_id, minimum_probability=None)[0]
            print token_topic

            #for topic_id in token_topics:
            #    for tup in ldamodel.get_topic_terms(topicid, topn):
            #print token_topics
            
        matrices.append(matrix)
    
    return matrices

def run():
    # create sample documents
    raw_path = "raw_texts/texts_english/"
    preprocessed_path = "preprocessed_texts/english/"
    training_preprocessed_path = "preprocessed_texts/english/with_spaces/"

    training_files = ["a_christmas_carol.txt", "alice_in_wonderland.txt"]
    # this is for testing accuracy against the 
    # actual stream that will be the test input
    test_files = ["hamlet_english.txt", "percy_addleshaw.txt"]

    training_doc_set = create_doc_set(training_preprocessed_path, training_files)
    test_doc_set = create_doc_set(preprocessed_path, test_files)

    tokenized_training_documents = tokenize(training_doc_set)
    tokenized_test_documents = tokenize(test_doc_set)

    # turn our tokenized documents into a id <-> term dictionary
    dictionary = corpora.Dictionary(tokenized_training_documents)
    # convert tokenized documents into a document-term matrix
    corpus = [dictionary.doc2bow(text) for text in tokenized_training_documents]
    # generate LDA model
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=passes)

    vectorized_dictionary = vectorize_dictionary(dictionary)

    # similarity matrix
    sm = similarity_matrix(vectorized_dictionary)
    syntaxed_space = syntax_space(sm, vectorized_dictionary) 
    # meaning matrix
    # how to cope with drop out and encoding meaning even tho heavily syntactic bc permutations...
    # tokenizing actually gets rid of the permutations...

run()
