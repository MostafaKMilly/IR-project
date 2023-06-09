from flask import Flask, jsonify, request
import ir_datasets
from collections import defaultdict
import math
import requests
import json

app = Flask(__name__)
dataset = ir_datasets.load("beir/quora")


def calculate_tf(terms):
    tf = {}
    term_count = len(terms)
    for term in terms:
        tf[term] = terms.count(term)/term_count
    return tf

def calculate_idf(corpus, inverted_index):
    idf = {}
    n_docs = len(corpus)
    for term, doc_ids in inverted_index.items():
        idf[term] = math.log(n_docs/len(doc_ids))
    return idf

def calculate_tfidf(document, corpus, inverted_index):
    tfidf = {}
    tf = calculate_tf(document)
    idf = calculate_idf(corpus, inverted_index)
    for term in tf:
        tfidf[term] = tf[term] * idf[term]
    return tfidf


# Splitting sentences and words from the body of text.
@app.route('/get-tf-idf/', methods=['POST'])
def get_tf_idf(): 
    query = request.args.get('query')

    print(query)

    if query:
        tokenization_response = requests.post('http://127.0.0.1:106/tokenize?query=' + query)
        proceed_query =  json.loads(tokenization_response.text)["data"]
        print("Finish tokenize")
        # tokenized_store_list_response = requests.post('http://127.0.0.1:106/tokenize')
        # tokenized_store_list =  json.loads(tokenized_store_list_response.text)["data"]

        tokenize_file = open("./data/tokens.json")
        tokenized_store_list =  json.load(tokenize_file)
        invertad_index_file = open("./data/inverted-index.json")
        invertad_index_list =  json.load(invertad_index_file)
        # inverted_index_response = requests.post('http://127.0.0.1:107/get-inverted-index')
        # invertad_index_list =  json.loads(inverted_index_response.text)["data"]
        query_tfi_df = calculate_tfidf(proceed_query, tokenized_store_list, invertad_index_list)
        print("Finish tf idf")
        return jsonify(
            proceed_query=proceed_query,
            query_tfi_df=query_tfi_df
        )
    else:
        tokenized_store_list = {}
        tf_list = {}
        tf_idf_list = {}

        tokenization_response = requests.post('http://127.0.0.1:106/tokenize/')
        tokenized_store_list =  json.loads(tokenization_response.text)["data"]

        inverted_index_response = requests.post('http://127.0.0.1:107/get-inverted-index/')
        invertad_index_list =  json.loads(inverted_index_response.text)["data"]

        # tokenization_file = open('./data/mini-tokens.json')
        # tokenized_store_list =  json.load(tokenization_file)

        # invertad_index_file = open("./data/mini-inverted-index.json")
        # invertad_index_list =  json.load(invertad_index_file)

        for key, value in tokenized_store_list.items():
            tf_list[key] = calculate_tf(value)

        for key, value in tokenized_store_list.items():
            tf_idf_list[key] = calculate_tfidf(value, tokenized_store_list, invertad_index_list)

        # with open("./data/tf-idf.json", "w") as outfile:
        #     json.dump(tf_idf_list, outfile)

        return jsonify(
            data=tf_idf_list
        )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=105)