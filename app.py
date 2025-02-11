from flask import Flask
import equationSolver 

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

if __name__ == "__main__":
    app.run(debug=True)


equation = equationSolver.get_user_input()