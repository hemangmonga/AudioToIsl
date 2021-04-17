from nltk.parse.stanford import StanfordParser
from nltk.stem import WordNetLemmatizer
from nltk.tree import *
import os
import json
from six.moves import urllib
import zipfile
import pickle as cPickle
import sys
import time
import ssl
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import wordnet

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')

ssl._create_default_https_context = ssl._create_unverified_context

# __file__ = "D:\\Final Year Project\\Final year project demo\\util"
# BASE_DIR = os.path.dirname(os.path.realpath(__file__))
# print(BASE_DIR)
# # Download zip file from https://nlp.stanford.edu/software/stanford-parser-full-2015-04-20.zip and extract in
# # stanford-parser-full-2015-04-20 folder in higher directory
# os.environ['CLASSPATH'] = os.path.join(BASE_DIR, 'stanford-parser-full-2018-10-17')
# os.environ['STANFORD_MODELS'] = os.path.join(BASE_DIR,
#                                              'stanford-parser-full-2018-10-17/edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz')
# os.environ['NLTK_DATA'] = '/usr/local/share/nltk_data/'
# os.environ['JAVAHOME'] = 'C:\\Program Files\\Java\\jdk1.8.0_221\\bin\\java.exe'
# print(os.environ.get('CLASSPATH'))

with open("data_dict.pickle", "rb") as input_file:
    data_dict = cPickle.load(input_file)

with open("parser.pickle", "rb") as input_file:
    parser = cPickle.load(input_file)


# def is_parser_jar_file_present():
#     stanford_parser_zip_file_path = os.environ.get('CLASSPATH') + ".jar"
#     return os.path.exists(stanford_parser_zip_file_path)
#
#
# def extract_models_jar_file():
#     stanford_models_zip_file_path = os.path.join(os.environ.get('CLASSPATH'), 'stanford-parser-3.9.2-models.jar')
#     stanford_models_dir = os.environ.get('CLASSPATH')
#     with zipfile.ZipFile(stanford_models_zip_file_path) as z:
#         z.extractall(path=stanford_models_dir)
#
#
# def reporthook(count, block_size, total_size):
#     global start_time
#     if count == 0:
#         start_time = time.time()
#         return
#     duration = time.time() - start_time
#     progress_size = int(count * block_size)
#     speed = int(progress_size / (1024 * duration))
#     percent = min(int(count * block_size * 100 / total_size), 100)
#     sys.stdout.write("\r...%d%%, %d MB, %d KB/s, %d seconds passed" %
#                      (percent, progress_size / (1024 * 1024), speed, duration))
#     sys.stdout.flush()
#
#
# def download_parser_jar_file():
#     stanford_parser_zip_file_path = os.environ.get('CLASSPATH') + ".jar"
#     url = "https://nlp.stanford.edu/software/stanford-parser-full-2018-10-17.zip"
#     urllib.request.urlretrieve(url, stanford_parser_zip_file_path, reporthook)
#
#
# def extract_parser_jar_file():
#     stanford_parser_zip_file_path = os.environ.get('CLASSPATH') + ".jar"
#     try:
#         with zipfile.ZipFile(stanford_parser_zip_file_path) as z:
#             z.extractall(path=BASE_DIR)
#     except Exception:
#         os.remove(stanford_parser_zip_file_path)
#         download_parser_jar_file()
#         extract_parser_jar_file()
#
#
# def download_required_packages():
#     if not os.path.exists(os.environ.get('CLASSPATH')):
#         if is_parser_jar_file_present():
#             pass
#         else:
#             download_parser_jar_file()
#         extract_parser_jar_file()
#
#     if not os.path.exists(os.environ.get('STANFORD_MODELS')):
#         extract_models_jar_file()


lemmatizer = WordNetLemmatizer()


def pos_tagger(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def label_parse_subtrees(parent_tree):
    tree_traversal_flag = {}

    for sub_tree in parent_tree.subtrees():
        tree_traversal_flag[sub_tree.treeposition()] = 0
    return tree_traversal_flag


def handle_noun_clause(i, tree_traversal_flag, modified_parse_tree, sub_tree):
    # if clause is Noun clause and not traversed then insert them in new tree first
    if tree_traversal_flag[sub_tree.treeposition()] == 0 and tree_traversal_flag[sub_tree.parent().treeposition()] == 0:
        tree_traversal_flag[sub_tree.treeposition()] = 1
        modified_parse_tree.insert(i, sub_tree)
        i = i + 1
    return i, modified_parse_tree


def handle_verb_prop_clause(i, tree_traversal_flag, modified_parse_tree, sub_tree):
    # if clause is Verb clause or Proportion clause recursively check for Noun clause
    for child_sub_tree in sub_tree.subtrees():
        if child_sub_tree.label() == "NP" or child_sub_tree.label() == 'PRP':
            if tree_traversal_flag[child_sub_tree.treeposition()] == 0 and tree_traversal_flag[
                child_sub_tree.parent().treeposition()] == 0:
                tree_traversal_flag[child_sub_tree.treeposition()] = 1
                modified_parse_tree.insert(i, child_sub_tree)
                i = i + 1
    return i, modified_parse_tree


def modify_tree_structure(parent_tree):
    # Mark all subtrees position as 0
    tree_traversal_flag = label_parse_subtrees(parent_tree)
    # Initialize new parse tree
    modified_parse_tree = Tree('ROOT', [])
    i = 0
    for sub_tree in parent_tree.subtrees():
        if sub_tree.label() == "NP":
            i, modified_parse_tree = handle_noun_clause(i, tree_traversal_flag, modified_parse_tree, sub_tree)
        if sub_tree.label() == "VP" or sub_tree.label() == "PRP":
            i, modified_parse_tree = handle_verb_prop_clause(i, tree_traversal_flag, modified_parse_tree, sub_tree)

    # recursively check for omitted clauses to be inserted in tree
    for sub_tree in parent_tree.subtrees():
        for child_sub_tree in sub_tree.subtrees():
            if len(child_sub_tree.leaves()) == 1:  # check if subtree leads to some word
                if tree_traversal_flag[child_sub_tree.treeposition()] == 0 and tree_traversal_flag[
                    child_sub_tree.parent().treeposition()] == 0:
                    tree_traversal_flag[child_sub_tree.treeposition()] = 1
                    modified_parse_tree.insert(i, child_sub_tree)
                    i = i + 1

    return modified_parse_tree


def remove_stop_words(sentence):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(sentence)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    filtered_sentence = ' '.join(filtered_sentence)
    return filtered_sentence


def convert_eng_to_isl(input_string):
    # get all required packages
    # download_required_packages()

    if len(list(input_string.split(' '))) is 1:
        return list(input_string.split(' '))

    # Initializing stanford parser
    # parser = StanfordParser()

    # Generates all possible parse trees sort by probability for the sentence
    possible_parse_tree_list = [tree for tree in parser.parse(input_string.split())]

    # Get most probable parse tree
    parse_tree = possible_parse_tree_list[0]
    print(parse_tree.pretty_print())
    # output = '(ROOT
    #               (S
    #                   (PP (IN As) (NP (DT an) (NN accountant)))
    #                   (NP (PRP I))
    #                   (VP (VBP want) (S (VP (TO to) (VP (VB make) (NP (DT a) (NN payment))))))
    #                )
    #             )'

    # Convert into tree data structure
    parent_tree = ParentedTree.convert(parse_tree)
    print(parent_tree.pretty_print())

    modified_parse_tree = modify_tree_structure(parent_tree)

    parsed_sent = modified_parse_tree.leaves()
    print(modified_parse_tree.pretty_print())
    return parsed_sent


# l = convert_eng_to_isl('As an accountant i want to make a payment')

def check_word(word):
    temp_list = []
    for i in data_dict.keys():
        if word in i:
            temp_list.append(i)
    return temp_list


# sentence = 'As an accountant, i want to make a payment'
def getISL(sentence):
    # tokenize the sentence and find the POS tag for each token
    pos_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))
    
    print(pos_tagged)
    
    # we use our own pos_tagger function to make things simpler to understand.
    wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged))
    print(wordnet_tagged)
    
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:
        if tag is None:
            # if there is no available tag, append the token as is
            lemmatized_sentence.append(word)
        else:
            # else use the tag to lemmatize the token
            lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
    lemmatized_sentence = " ".join(lemmatized_sentence)
    
    print(lemmatized_sentence)

    filtered_sentence = remove_stop_words(lemmatized_sentence)

    l = convert_eng_to_isl(filtered_sentence)
    
    links = []
    for i in l:
        if i in data_dict.keys():
            print(i, data_dict[i])
            links.append(data_dict[i].split('/')[-1])
        else:
            print(check_word(i))

    return l, links

# if __name__ == '__main__':
#     convert_eng_to_isl('As an accountant i want to make a payment')
