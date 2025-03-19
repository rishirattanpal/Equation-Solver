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

        # Extract the equations and solutions
        input_equations = answer.get("equation", [])  # List of input equations
        solutions = answer.get("solutions", {})       # Dictionary of solutions

        # Format the equations into a clean string
        formatted_equations = "\n".join(input_equations)

        # Format the solutions into a clean string
        formatted_solutions = "\n".join([f"{var} = {value}" for var, value in solutions.items()])

        # Pass the formatted equations and solutions to the template
        return render_template(
            "index.html",
            equations=formatted_equations,
            solutions=formatted_solutions
        )

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)