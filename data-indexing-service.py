from flask import Flask, jsonify
from collections import defaultdict
import requests
import ir_datasets
import json

app = Flask(__name__)
dataset = ir_datasets.load("beir/quora")


def create_inverted_index(corpus):
    inverted_index = defaultdict(list)
    for doc_id, doc_content in corpus.items():
        for term in doc_content:
            inverted_index[term].append(doc_id)

    return dict(inverted_index)




@app.route('/get-inverted-index/', methods=['POST'])
def build_index():
    tokenized_store_list = {}
    inverted_index = {}
    tokenization_response = requests.post('http://127.0.0.1:106/tokenize/')
    tokenized_store_list = json.loads(tokenization_response.text)["data"]
    inverted_index = create_inverted_index(tokenized_store_list)


    # with open("./data/mini-inverted-index.json", "w") as outfile:
    #     json.dump(inverted_index, outfile)

    return jsonify(data=inverted_index)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=107)
