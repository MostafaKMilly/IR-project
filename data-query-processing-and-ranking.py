from flask import Flask, render_template, request, jsonify
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import ir_datasets
import math
import requests
import json
from flask import Flask
from flask_cors import CORS


app = Flask(__name__)
CORS(app, origins=['*'])
dataset = ir_datasets.load("beir/quora")
lemmatizer = WordNetLemmatizer()


def cosine_similarity(query_tfidf, doc_tfidf):
    common_terms = set(query_tfidf.keys()) & set(doc_tfidf.keys())
    dot_product = sum(query_tfidf[term] * doc_tfidf[term] for term in common_terms)
    query_norm = math.sqrt(sum(query_tfidf[term] ** 2 for term in query_tfidf))
    doc_norm = math.sqrt(sum(doc_tfidf[term] ** 2 for term in doc_tfidf))
    return dot_product / (query_norm * doc_norm)

@app.route('/')
def index():
    return render_template('test.html')

# @app.route('/index/', methods=['POST'])
# def build_index():
#     tokenized_store_list = {}

#     tokenization_response = requests.post('http://127.0.0.1:106/tokenize/')
#     tokenized_store_list = json.loads(tokenization_response.text)["data"]

#     inverted_index = create_inverted_index(tokenized_store_list)

#     return jsonify(data=inverted_index)

@app.route('/query/', methods=['POST'])
def process_query():
    query = request.form['query']
    tokenized_query = nltk.word_tokenize(query.lower().strip())
    stop_words = set(stopwords.words('english'))
    preprocessed_query = [*set([lemmatizer.lemmatize(w) for w in tokenized_query if not w in stop_words])]

    # inverted_index_response = requests.post('http://127.0.0.1:107/get-inverted-index/')
    # inverted_index = json.loads(inverted_index_response.text)["data"]

    invertad_index_file = open("./data/inverted-index.json")
    inverted_index =  json.load(invertad_index_file)

    # tf_idf_store_response = requests.post('http://127.0.0.1:105/get-tf-idf/')
    # tf_idf_store = json.loads(tf_idf_store_response.text)["data"]

    tf_idf_file = open("./data/tf-idf.json")
    tf_idf_store =  json.load(tf_idf_file)

    tf_idf_query_response = requests.post('http://127.0.0.1:105/get-tf-idf?query=' + query)
    tf_idf_query = json.loads(tf_idf_query_response.text)["query_tfi_df"]

    relevant_docs = set()
    for term in preprocessed_query:
        if term in inverted_index:
            relevant_docs.update(inverted_index[term])


    ranked_docs = []
    for doc_id in relevant_docs:
        similarity = cosine_similarity(tf_idf_query, tf_idf_store[doc_id])
        ranked_docs.append((doc_id, similarity))

    ranked_docs = sorted(ranked_docs, key=lambda x: x[1], reverse=True)
    ranked_doc_ids = [doc_id for doc_id, _ in ranked_docs]

    return jsonify(data=ranked_doc_ids)


@app.route('/qquery/', methods=['POST'])
def process_qquery():
    query = request.args.get('query')
    tokenized_query = nltk.word_tokenize(query.lower().strip())
    stop_words = set(stopwords.words('english'))
    preprocessed_query = [*set([lemmatizer.lemmatize(w) for w in tokenized_query if not w in stop_words])]

    # inverted_index_response = requests.post('http://127.0.0.1:107/get-inverted-index/')
    # inverted_index = json.loads(inverted_index_response.text)["data"]

    invertad_index_file = open("./data/inverted-index.json")
    inverted_index =  json.load(invertad_index_file)

    # tf_idf_store_response = requests.post('http://127.0.0.1:105/get-tf-idf/')
    # tf_idf_store = json.loads(tf_idf_store_response.text)["data"]

    tf_idf_file = open("./data/tf-idf.json")
    tf_idf_store =  json.load(tf_idf_file)

    tf_idf_query_response = requests.post('http://127.0.0.1:105/get-tf-idf?query=' + query)
    tf_idf_query = json.loads(tf_idf_query_response.text)["query_tfi_df"]

    print(tf_idf_query)
    relevant_docs = set()
    for term in preprocessed_query:
        if term in inverted_index:
            relevant_docs.update(inverted_index[term])


    ranked_docs = []
    for doc_id in relevant_docs:
        similarity = cosine_similarity(tf_idf_query, tf_idf_store[doc_id])
        ranked_docs.append((doc_id, similarity))

    ranked_docs = sorted(ranked_docs, key=lambda x: x[1], reverse=True)
    ranked_doc_ids = [doc_id for doc_id, _ in ranked_docs]

    return jsonify(data=ranked_doc_ids)



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=109)
