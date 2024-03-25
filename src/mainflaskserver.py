from flask import Flask, jsonify, request

app = Flask(__name__)

# Sample data
earnings = [
    {'locationId': 1, 'time': '01', 'earning': 100},
    {'locationId': 2, 'time': '02', 'earning': 200},
    {'locationId': 3, 'time': '03', 'earning': 300}
]

# Route to get all getEarning
@app.route('/getEarning', methods=['GET'])
def getEarning():
    return jsonify(earnings)

# Route to get a specific getEarning by locationId
@app.route('/getEarning/<int:locationId>', methods=['GET'])
def getEarning(locationId):
    earning = next((earning for earning in earnings if earning['locationId'] == locationId), None)
    if earning:
        return jsonify(earnings)
    else:
        return jsonify({'error': 'Earning not found'}), 404


if __name__ == '__main__':
    # app.run(debug=True)
    app.run(host="localhost", port=8000, debug=True)



# export FLASK_RUN_PORT=8000
# export FLASK_RUN_HOST="127.0.0.1"
# flask run
#  * Running on http://127.0.0.1:8000/