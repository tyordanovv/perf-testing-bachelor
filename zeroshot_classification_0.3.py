from flask import Flask, request, jsonify
from transformers import pipeline
import time

app = Flask(__name__)
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
candidate_labels = ['Delivery Inquiry', 'Service Complainment', 'Product Inquiry', "Return Policy", "Feedback"]

@app.route('/classify', methods=['POST'])
def classify_text():  
    start_time = time.time()

    data = request.json

    app.logger.info("Received request data: %s", data)

    text = data.get('text', '')
    id = data.get('id', '')

    result = classifier(text, candidate_labels, batch_size=32)

    end_time = time.time()

    execution_time = end_time - start_time
    app.logger.info("Execution time of " + str(id) + ": {:.2f} seconds".format(execution_time))

    return jsonify(result)


@app.route('/classify/batch', methods=['POST'])
def classify_text_batch():  
    data = request.json
    app.logger.info("Batch processing started!")
    # should fetch data from local repo
    text = []

    start_time = time.time()
    result = classifier(text, candidate_labels, batch_size=32)
    end_time = time.time()
    
    execution_time = end_time - start_time

    app.logger.info("Execution time of " + str(id) + ": {:.2f} seconds".format(execution_time))

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)