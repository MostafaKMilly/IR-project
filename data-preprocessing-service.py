from flask import Flask, jsonify, request
import nltk
import ir_datasets
from nltk.corpus import stopwords, wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import string
import numpy as np

app = Flask(__name__)
dataset = ir_datasets.load("beir/quora")
test_dataset = ir_datasets.load("beir/quora/test")
lemmatizer = WordNetLemmatizer()


def _get_wordnet_pos(tag: str) -> str:
        if tag.startswith('JJ'):
            return wordnet.ADJ
        elif tag.startswith('VB'):
            return wordnet.VERB
        elif tag.startswith('NN'):
            return wordnet.NOUN
        elif tag.startswith('RB'):
            return wordnet.ADV
        else:
            return wordnet.NOUN

@app.route('/get-question-by-id/', methods=['GET'])
def get_guestion_by_id(): 
    # store = dataset.docs_store().get_many(['{}'.format(x) for x in range(1,100000)])
    # store = dataset.docs_store().get_many(['{}'.format(x) for x in range(1,100000)])
    id = request.args.get('id')
    doc = dataset.docs_store().get(id)

    return jsonify(
        data=doc
    )

@app.route('/get-questions/', methods=['GET'])
def get_guestions(): 
    # store = dataset.docs_store().get_many(['{}'.format(x) for x in range(1,100000)])
    # store = dataset.docs_store().get_many(['{}'.format(x) for x in range(1,100000)])
    store = test_dataset.docs_store().get_many(['{}'.format(x) for x in range(1,100000)])

    return jsonify(
        data=store
    )

# Splitting sentences and words from the body of text.
@app.route('/tokenize/', methods=['POST'])
def tokenizing(): 
    query = request.args.get('query')

    if query:
        tokenized_query = nltk.word_tokenize(query.lower().strip())
        stop_words = set(stopwords.words('english'))
        punctuation = string.punctuation
        tagged_list =  nltk.pos_tag(tokenized_query)
        proceed_query = [*set([lemmatizer.lemmatize(str(np.char.replace(w[0], "'", " ")), pos = _get_wordnet_pos(w[1])) for w in tagged_list if not w[0] in stop_words and not w[0] in punctuation])]
        return jsonify(
            data=proceed_query
        )
    else: 
        store = dataset.docs_store().get_many(['{}'.format(x) for x in range(1,522931)])
        store_list = {}
        tokenized_store_list = {}

        for key, value in store.items():
            store_list[key] = value.text

        for key, value in store_list.items():
            terms = nltk.word_tokenize(value.lower().strip())
            stop_words = set(stopwords.words('english'))
            punctuation = string.punctuation

            abbreviations = []
            lemmatized_list = []
            for term in terms:
                synsets = wn.synsets(term)
                for synset in synsets:
                    for lemma in synset.lemmas():
                        if lemma.name().isupper():
                            abbreviations.append(lemma.name().lower())

            tagged_list =  nltk.pos_tag(terms)
            for w in tagged_list:
                # Check if word is not a stop word or punctuation and is not already in the lemmatized list
                if not w[0] in stop_words and not w[0] in punctuation and lemmatizer.lemmatize(w[0], pos=_get_wordnet_pos(w[1])) not in lemmatized_list:
                    # Lemmatize the word and add it to the lemmatized list
                    lemmatized_list.append(lemmatizer.lemmatize(w[0], pos=_get_wordnet_pos(w[1]))) 

                    lemmatized_list.extend(abbreviations)

            tokenized_store_list[key] = [*set(lemmatized_list)]

        # with open("./data/mini-tokens.json", "w") as outfile:
        #     json.dump(tokenized_store_list, outfile)
        return jsonify(
            data=tokenized_store_list
        )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=106)