from server import app
from model import Pytorchmodel

if __name__ == '__main__':
    print("Starting Breast Detection Server, please wait ...")
    print("Please wait until server has fully started")

    app.run(host='0.0.0.0', port=5000, debug=False)
