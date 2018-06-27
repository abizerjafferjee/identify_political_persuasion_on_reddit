import numpy as np
import sys
import argparse
import os
import json
import csv

BRISTOL = "../lists/BristolNorms+GilhoolyLogie.csv";
WARRINGER = "../lists/Ratings_Warriner_et_al.csv";
slang_path = "../lists/Slang";
feats_dir = "../feats";

def extract1( comment ):
    ''' This function extracts features from a single comment

    Parameters:
        comment : string, the body of a comment (after preprocessing)

    Returns:
        feats : numpy Array, a 173-length vector of floating point features (only the first 29 are expected to be filled, here)
    '''
    feats = [0]*173

    first_person = ["i", "me", "my", "mine", "we", "us", "our", "ours"]
    second_person = ["you", "your", "yours", "u", "ur", "urs"]
    third_person = ["he", "him", "his", "she", "her", "hers", "it", "its", "they", "them", "their", "theirs"]
    future_tense = ["'ll", "will", "gonna", "going"]
    words = comment.split()

    # features 1-14
    for word in words:

        token_tag = word.split("/")
        token= token_tag[0]
        tag = token_tag[1]
        # first person
        if token in first_person:
            feats[0] += 1

        # second person
        elif token in second_person:
            feats[1] += 1

        # third person
        elif token in third_person:
            feats[2] += 1

        # coordinating conjunction
        if tag == "cc":
            feats[3] += 1

        # past tense
        if token in "vbd":
            feats[4] += 1

        # future tense
        if token in future_tense:
            feats[5] += 1

        # commas
        if token == ",":
            feats[6] += 1

        # multi-character
        if len(token) > 1 and tag in ["!", ".", "?"]:
            feats[7] += 1

        # common nouns
        if tag in ["nn", "nns"]:
            feats[8] += 1

        # proper nouns
        if tag in ["nnp", "nnps"]:
            feats[9] += 1

        # adverbs
        if tag in ["rb", "rbr", "rbs"]:
            feats[10] += 1

        # wh words
        if tag in ["wdt", "wp", "wrb", "wp$"]:
            feats[11] += 1

        # slang
        if token in slang:
            feats[12] += 1

        # uppercase
        if len(token) >= 3 and token.isupper():
            feats[13] += 1

    # features 15-17
    # average length of sentence
    sentences = comment.split("\n")
    num_sentences = len(sentences)
    num_words = 0
    for sentence in sentences:
        num_tokens = len(sentence.split())
        num_words += num_tokens

    average_words = num_words / num_sentences
    feats[14] += average_words

    # average number of characters
    num_words = 1
    num_characters = 0

    for i in words:
        token = i.split("/")[0]
        if token not in ["?","!",",",".",";",":"]:
            num_words += 1
            num_characters += len(token)

    average_characters = num_characters/num_words
    feats[15] = average_characters

    # number of sentences
    feats[16] = num_sentences

    # features 18-23
    bristol = open(BRISTOL, "r")
    breader = csv.reader(bristol, delimiter=",")

    """
    Average of AoA
    Average of IMG
    Average of FAM
    """
    sum_aoa = 0
    sum_img = 0
    sum_fam = 0
    word_count = 1
    for word in words:
        token_tag = word.split("/")
        token = token_tag[0]
        for row in breader:
            if token != "" and token == row[1]:
                if row[3] != " ":
                    sum_aoa += float(row[3])
                if row[4] != " ":
                    sum_img += float(row[4])
                if row[5] != " ":
                    sum_fam += float(row[5])
                word_count += 1
    average_aoa = sum_aoa / word_count
    average_img = sum_img / word_count
    average_fam = sum_fam / word_count
    feats[17] = average_aoa
    feats[18] = average_img
    feats[19] = average_fam

    """
    SD of AoA
    SD of IMG
    SD of FAM
    """
    sum_sd_aoa = 0
    sum_sd_img = 0
    sum_sd_fam = 0
    word_count = 1
    for word in words:
        token_tag = word.split("/")
        token= token_tag[0]
        for row in breader:
            if token != "" and token == row[1]:
                if row[3] != " ":
                    sum_sd_aoa += ((float(row[3])-average_aoa)**2)
                if row[4] != " ":
                    sum_sd_img += ((float(row[4])-average_img)**2)
                if row[5] != " ":
                    sum_sd_fam += ((float(row[5])-average_fam)**2)
                word_count += 1
    sd_aoa = ((sum_sd_aoa / word_count)**0.5)
    sd_img = ((sum_sd_img / word_count)**0.5)
    sd_fam = ((sum_sd_fam / word_count)**0.5)
    feats[20] = sd_aoa
    feats[21] = sd_img
    feats[22] = sd_fam

    # features 24-29
    warringer = open(WARRINGER, "r")
    wreader = csv.reader(warringer,delimiter=",")

    """
    Average of V.Mean.Sum
    Average of A.Mean.Sum
    Average of D.Mean.Sum
    """
    sum_v = 0
    sum_a = 0
    sum_d = 0
    word_count = 1
    for word in words:
        token_tag = word.split("/")
        token= token_tag[0]
        for row in wreader:
            if token != "" and token == row[1]:
                if row[2] != " ":
                    sum_v += float(row[2])
                if row[5] != " ":
                    sum_a += float(row[5])
                if row[8] != " ":
                    sum_d += float(row[8])
                word_count += 1
    average_v = sum_v / word_count
    average_a = sum_a / word_count
    average_d = sum_d / word_count
    feats[23] = average_v
    feats[24] = average_a
    feats[25] = average_d

    """
    SD of V.Mean.Sum
    SD of A.Mean.Sum
    SD of D.Mean.Sum
    """
    sum_sd_v = 0
    sum_sd_a = 0
    sum_sd_d = 0
    word_count = 1
    for word in words:
        token_tag = word.split("/")
        token= token_tag[0]
        for row in wreader:
            if token != "" and token == row[1]:
                if row[2] != " ":
                    sum_sd_v += ((float(row[2])-average_v)**2)
                if row[5] != " ":
                    sum_sd_a += ((float(row[5])-average_a)**2)
                if row[8] != " ":
                    sum_sd_d += ((float(row[8])-average_d)**2)
                word_count += 1
    sd_v = ((sum_sd_v / word_count)**0.5)
    sd_a = ((sum_sd_a / word_count)**0.5)
    sd_d = ((sum_sd_d / word_count)**0.5)
    feats[26] = sd_v
    feats[27] = sd_a
    feats[28] = sd_d

    feats = np.array(feats)
    return feats

def main( args ):

    data = json.load(open(args.input))
    feats = np.zeros( (len(data), 173+1))

    line_number = 0

    for line in data:

        body = line["body"]
        features = extract1(body)
        line_id = line["id"]
        cat = line["cat"]
        liwc = np.load("{}/{}_feats.dat.npy".format(feats_dir, cat))
        ids_txt = open("{}/{}_IDs.txt".format(feats_dir, cat)).read()
        ids = ids_txt.split("\n")
        idx = ids.index(line_id)
        liwc_feats = liwc[idx]

        i = 0
        while i < 29:
            feats[line_number][i] = features[i]
            i += 1

        j = 29
        k = 0
        while j < 173:
            feats[line_number][j] = liwc_feats[k]
            j += 1
            k += 1

        if cat == "Left":
            feats[line_number][173] = 0
        if cat == "Center":
            feats[line_number][173] = 1
        if cat == "Right":
            feats[line_number][173] = 2
        if cat == "Alt":
            feats[line_number][173] = 3

        line_number += 1

    np.savez_compressed( args.output, feats)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("-i", "--input", help="The input JSON file, preprocessed as in Task 1", required=True)
    args = parser.parse_args()

    main(args)
