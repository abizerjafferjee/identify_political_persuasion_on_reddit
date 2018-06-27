import sys
import argparse
import os
import json
import html
import spacy
import re
from nltk.corpus import stopwords

indir = '../data/';
stopwords = '../lists/StopWords';
abbrevs = '../lists/abbrev.english';

def punctuation(text):
    text = text.replace("^", "")
    words = text.split()
    for index, word in enumerate(words):
        if len(word) != 1:

            if "." in word[-1] and "." not in word[0:-1]:
                words.insert(index + 1, word[-1])
                words[index] = word[0:-1]
            elif "," in word[-1] and "," not in word[0:-1]:
                words.insert(index + 1, word[-1])
                words[index] = word[0:-1]
            elif "?" in word[-1] and "?" not in word[0:-1]:
                words.insert(index + 1, word[-1])
                words[index] = word[0:-1]
            elif "!" in word[-1] and "!" not in word[0:-1]:
                words.insert(index + 1, word[-1])
                words[index] = word[0:-1]
            elif ":" in word[-1] and ":" not in word[0:-1]:
                words.insert(index + 1, word[-1])
                words[index] = word[0:-1]
            elif ";" in word[-1] and ";" not in word[0:-1]:
                words.insert(index + 1, word[-1])
                words[index] = word[0:-1]

    new_text = ""
    for idx, word in enumerate(words):
        if idx != len(words)-1:
                new_text += word + " "
        else:
                new_text += word

    return new_text

def clitics(text):

    words = text.split()

    for index, word in enumerate(words):
        if len(word) != 1:

            if "'" in word:
                idx = word.find("'")
                words[index] = word[:idx] + " " + word[idx:]

    new_text = ""
    for idx, word in enumerate(words):
        if idx != len(words)-1:
            new_text += word + " "
        else:
            new_text += word

    return new_text

def tags(text):
    nlp = spacy.load('en', disable=['parser', 'ner'])
    utt = nlp(text)
    new_text = ""
    for word in utt:
        new_text += word.text + "/" + word.tag_ + " "

    return new_text

def remove_stopwords(text):

    stopwords_file = open(stopwords)
    stop_words = stopwords_file.read().split("\n")
    word_list = text.split()
    words = []
    for word in word_list:
        word_part = word.split("/")[0]
        if word_part not in stop_words:
            words.append(word)

    new_text = ""
    for idx, word in enumerate(words):
        if idx != len(words)-1:
                new_text += word + " "
        else:
                new_text += word

    stopwords_file.close()
    return new_text

def lemmatize(text):
    nlp = spacy.load('en', disable=['parser', 'ner'])

    tokens = text.split()
    words = []

    for token in tokens:
        words.append(token.split("/")[0])

    word_text = ""
    for idx, word in enumerate(words):
        if idx != len(words)-1:
                word_text += word + " "
        else:
                word_text += word

    utt = nlp(word_text)
    new_text = ""
    for word in utt:
        if word.lemma_.startswith("-"):
            new_text += word.text + "/" + word.tag_ + " "
        else:
            new_text += word.lemma_ + "/" + word.tag_ + " "

    return new_text

def add_newline(text):

    abbrev_file = open(abbrevs)
    abbreviations = abbrev_file.read().split("\n")
    words = text.split()
    for idx, word in enumerate(words):
        token = word.split("/")[0]
        if token == ".":
            if words[idx-1] not in abbreviations:
                words.insert(idx+1, "\n")

    new_text = ""
    for idx, word in enumerate(words):
        if idx != len(words)-1:
                new_text += word + " "
        else:
                new_text += word

    abbrev_file.close()
    return new_text

def preproc1( comment , steps=range(1,11)):
    ''' This function pre-processes a single comment

    Parameters:
        comment : string, the body of a comment
        steps   : list of ints, each entry in this list corresponds to a preprocessing step

    Returns:
        modComm : string, the modified comment
    '''

    modComm = ''
    if 1 in steps:
        modComm = comment.replace("\n", " ")
    if 2 in steps:
        modComm = html.unescape(modComm)
    if 3 in steps:
        modComm = re.sub(r'http\S+', '', modComm)
        modComm = re.sub(r'www\S+', '', modComm)
    if 4 in steps:
        modComm = punctuation(modComm)
    if 5 in steps:
        modComm = clitics(modComm)
    if 6 in steps:
        modComm = tags(modComm)
    if 7 in steps:
        modComm = remove_stopwords(modComm)
    if 8 in steps:
        modComm = lemmatize(modComm)
    if 9 in steps:
        modComm = add_newline(modComm)
    if 10 in steps:
        modComm = modComm.lower()

    return modComm

def main( args ):

    allOutput = []
    for subdir, dirs, files in os.walk(indir):
        for file in files:

            fullFile = os.path.join(subdir, file)
            print ("Processing " + fullFile)
            data = json.load(open(fullFile))
            lines_completed = 0
            index = args.ID[0] % len(data)

            while lines_completed < int(args.max):
                print(lines_completed)

                if index == len(data):
                        index = 0

                line_dict = json.loads(data[index])
                line_id = line_dict["id"]
                line_body = preproc1(line_dict["body"])
                line_cat = file
                line = {"id":line_id, "body":line_body, "cat":line_cat}
                allOutput.append(line)

                lines_completed += 1
                index += 1

    fout = open(args.output, 'w')
    fout.write(json.dumps(allOutput))
    fout.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument('ID', metavar='N', type=int, nargs=1,
                        help='your student ID')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("--max", help="The maximum number of comments to read from each file", default=10000)
    args = parser.parse_args()

    if (int(args.max) > 200272):
        print ("Error: If you want to read more than 200,272 comments per file, you have to read them all.")
        sys.exit(1)

    main(args)
