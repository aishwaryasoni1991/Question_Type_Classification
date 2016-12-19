from logging import DEBUG

from flask import Flask, render_template, request, jsonify

from model import model

app = Flask(__name__)
app.logger.setLevel(DEBUG)


@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/qtype')
def classify():
    if 'q' in request.args:
        
        predictions = model.classify_question_type(request.args['q'])
        data = dict()
        for rec in predictions:
            data[request.args['q']] = predictions[0]
        resp = jsonify(data)
        resp.status_code = 200
        return resp
    return jsonify(dict())

if __name__ == '__main__':
    app.run(debug=True)
