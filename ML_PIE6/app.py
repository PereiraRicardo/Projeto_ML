from flask import Flask, render_template
from routes.ml_routes import ml_bp

app = Flask(__name__)
app.register_blueprint(ml_bp)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=False)