from flask import Flask, request, jsonify
from to_B.report_2B import construct_structured_data as report_2b
from to_C.report_2C import construct_structured_data as report_2c

app = Flask(__name__)

@app.route('/api/report2b', methods=['POST'])
def api_report_2b():
    data = request.get_json(force=True)
    dir_path = data.get('dir_path')
    category = data.get('category', '水果')
    if not dir_path:
        return jsonify({'error': 'dir_path required'}), 400
    result = report_2b(dir_path, category)
    return jsonify({'result': result})

@app.route('/api/report2c', methods=['POST'])
def api_report_2c():
    data = request.get_json(force=True)
    image1 = data.get('image1')
    image2 = data.get('image2')
    role = data.get('user_role', 'consumer')
    if not image1 or not image2:
        return jsonify({'error': 'image1 and image2 required'}), 400
    result = report_2c(image1, image2, role)
    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000)

