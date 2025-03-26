from flask import Flask, render_template, request, flash, session
from equationSolver import solve_equation, show_graph
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def format_answer(answer):
    print(f"answer: {answer}")
    if "result" in answer:
        formattedEquations = answer["expression"]
        formattedSolutions = str(answer["result"])
        return formattedEquations, formattedSolutions
    
    inputEquations = answer.get("equation", [])
    solutions = answer.get("solutions", {})

    # format equation
    if isinstance(inputEquations, list):
        formattedEquations = "\n".join(inputEquations)
    else:
        formattedEquations = str(inputEquations)
    

    # format solutiin
    if isinstance(solutions, dict):
        formattedSolutions = "\n".join([f"{var} = {value}" for var, value in solutions.items()])
    elif isinstance(solutions, list):
        formattedSolutions = "\n".join([f"x = {value}" for value in solutions])
    elif isinstance(solutions, str):
        formattedSolutions = solutions
    else:
        formattedSolutions = "no solution found"

    return formattedEquations, formattedSolutions




app = Flask(__name__)
app.secret_key="dababy"
#app.config["SESSION_TYPE"] = "filesystem"

@app.route('/get_last_answer')
def get_last_answer():
    return {'lastAnswer': session.get('lastAnswer', '')}


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get equations from the form
        rawInput = request.form.get("equations", "").strip()

        if not rawInput:
            flash("Please enter an equation")
            return render_template("index.html", graphAvailable = False)

        equations = [eq.strip() for eq in rawInput.split("\n") if eq.strip()]
        print(f"Equations: {equations}")


        equations = [eq.replace('\r', '') for eq in equations]
        print(f"Equations: {equations}")

        try:

            # Solve the equations
            answer = solve_equation(equations)
            print(f"Answer: {answer}")


            formattedEquations, formattedSolutions = format_answer(answer)
            print(f"formatted equations: {formattedEquations}")

            if "note" in answer:
                note = answer['note']
                print(f"note {note}")
            else:
                note = ""

            try:
                show_graph(equations)
                graphAvailable = True
            except Exception as e:
                print(f"Failed to generate graph: {e}")
                graphAvailable = False

            
            return render_template(
                "index.html",
                equations=formattedEquations,
                solutions=formattedSolutions, graphAvailable=graphAvailable, note=note
            )
        except ValueError as e:
            if "No equation entered" in str(e):
                flash("Invalid equation entered")
            else:
                flash(f"Error: {str(e)}")
            return render_template("index.html", grSaphAvailable=False)

    return render_template("index.html", graphAvailable = False)

if __name__ == "__main__":
    app.run(debug=True)



# -- -- - - - - - - - 
# for today, add graph implementation here and then send to front

