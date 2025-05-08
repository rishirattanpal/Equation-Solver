from flask import Flask, render_template, request, flash, session
from equationSolver import solve_equation, show_graph, OPERATORS
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import re

def format_answer(answer):
    print(f"answer: {answer}")
    if "result" in answer:
        return answer["expression"], str(answer["result"])
    
    inputEquations = answer.get("equation", [])
    solutions = answer.get("solutions", {})
    operators = answer.get("operators", [])
    
    # replace '=' with og operator
    if isinstance(inputEquations, list):
        formattedEquations = []
        for i, eq in enumerate(inputEquations):
            op = operators[i] 

            formattedEq = eq.replace('=', f'{op}') 
            formattedEquations.append(formattedEq)
        
        formattedEquations = "\n".join(formattedEquations)  
    else:
        op = operators[0] if operators else "="
        formattedEquations = inputEquations.replace('=', f' {op} ')
    
    # Format solutions with operators
    if isinstance(solutions, dict):
        formattedSolutions = []
        for i, (var, value) in enumerate(solutions.items()):
            op = operators[i] if i < len(operators) else "="
            formattedSolutions.append(f"{var} {op} {value}")
        formattedSolutions = "\n".join(formattedSolutions)
    elif isinstance(solutions, list):
        op = operators[0] if operators else "="
        formattedSolutions = "\n".join([f"x {op} {value}" for value in solutions])
    elif isinstance(solutions, str):
        formattedSolutions = solutions
    else:
        formattedSolutions = "no solution found"

    return formattedEquations, formattedSolutions




app = Flask(__name__)
app.secret_key="dababy"

@app.route('/get_last_answer')
def get_last_answer():
    return {'lastAnswer': session.get('lastAnswer', '')}


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get equations from the form
        rawInput = request.form.get("equations", "").strip()
        print(f"rawinput: {rawInput}")

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


            if "result" in answer: # store for answer key
                session['lastAnswer'] = str(answer["result"])


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
        
        except TypeError as te:
            flash("Unsupported equation type")
            return render_template("index.html", graphAvailable = False)
        except ValueError as e:
            flash("Invalid equation entered")
            return render_template("index.html", grSaphAvailable=False)
        except Exception as e:
            flash(f"Unexpected error: {str(e)}")
            return render_template("index.html", graphAvailable = False)

    return render_template("index.html", graphAvailable = False)

if __name__ == "__main__":
    app.run(debug=True)




