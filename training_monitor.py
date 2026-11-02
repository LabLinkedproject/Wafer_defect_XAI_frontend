"""
training_monitor.py
카메라·모터를 건드리지 않는 학습 상태 전용 서버. 메인 서버가 꺼져 있어도 상태를 볼 수 있다.

실행: ./venv/bin/python training_monitor.py   → http://<보드 IP>:5001/training
"""
from flask import Flask, redirect

from training_routes import create_training_blueprint

PORT = 5001

app = Flask(__name__)
app.register_blueprint(create_training_blueprint())


@app.route("/")
def index():
    return redirect("/training")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, threaded=True)
