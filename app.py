from flask import Flask, render_template, request
from equationSolver import solve_equation

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get equations from the form
        equations = request.form["equations"].split("\n")
        equations = [eq.replace('\r', '') for eq in equations]
        print(f"Equations: {equations}")

        # Solve the equations
        answer = solve_equation(equations)
        print(f"Answer: {answer}")

        # format arithmetic solution
        if "result" in answer:
            formattedEquations = answer["expression"]
            formattedSolutions = str(answer["result"])
        else:
            # format equation solutions
            inputEquations = answer.get("equation", [])  
            solutions = answer.get("solutions", {})       

            formattedEquations = "\n".join(inputEquations)
            formattedSolutions = "\n".join([f"{var} = {value}" for var, value in solutions.items()])

        # render with answewr
        return render_template(
            "index.html",
            equations=formattedEquations,
            solutions=formattedSolutions
        )

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)