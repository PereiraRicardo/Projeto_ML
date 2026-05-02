from waitress import serve
from app import app

print("Iniciando servidor em http://127.0.0.1:5000")
serve(app, host="0.0.0.0", port=5000)