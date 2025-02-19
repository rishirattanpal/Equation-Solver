from flask import Flask, render_template, request
from equationSolver import classify_equation, solve_quadratic, solve_systems_and_linear, solve_equation


app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        equations = request.form["equations"].split("\n")
        equations = [eq.replace('\r', '') for eq in equations]
        print(equations)

        answer = solve_equation(equations)
        print(f"answer = {answer}")

        return render_template("index.html", answer=answer)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)


