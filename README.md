# Information Retrieval Project

This is an Information Retrieval project that aims to implement various services for text processing, indexing, and querying. The project utilizes IR Datasets, Flask, NLTK, and other tools to process textual data, build inverted indexes, calculate TF-IDF scores, and perform cosine similarity-based ranking.

## App Folder Structure

The project follows a specific folder structure to organize the code and resources. Here is an overview of the app folder structure:

- `data/`: This directory containes the data which getting from dataset.
- `data_preprocessing_service.py`: A service that performs tokenization and stop word removal on the input text data.
- `data_representation_service.py`: A service that calculates TF-IDF scores for the tokenized data.
- `data_indexing_service.py`: A service that builds an inverted index based on the tokenized data.
- `data_query_processing_and_ranking.py`: A service that processes user queries, calculates query TF-IDF scores, and performs cosine similarity-based ranking.
- `templates/`: This directory contains HTML templates used for rendering the web interface.

## Getting Started

To run the project locally, follow these steps:

1. Install the required dependencies listed in the `requirements.txt` file.
2. Run the anyof service file to start the Flask web server.
3. Access the project through the provided URL, usually `http://localhost:${SERVICE_PORT}`.

Make sure to have Python and the required packages installed on your machine.

## Usage

Once the project is up and running, you can access the different services through the provided routes:

- Tokenization and Stop Word Removal: `/tokenize/`
- TF-IDF Calculation: `/get-tf-idf/`
- Query Processing and Ranking: `/query/`

You can test the services using tools like Postman or by making HTTP requests to the appropriate endpoints.
