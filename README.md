# Equation Solver Web Application  


A full-stack web application that solves mathematical equations, developed as a university dissertation project that earned first-class honors.  

## Features  

### Equation Solving  
- **Linear equations** (single and systems)  
- **Quadratic equations** using quadratic formula  
- **Polynomials** (degree n) using Newton-Raphson method  
- **Inequalities** with sign flip handling  
- **Basic arithmetic** evaluation  

### Technical Implementation  
**Backend (Python)**  
- Numerical methods implemented from scratch  
- `NumPy` for efficient computation  
- `SymPy` for symbolic mathematics  
- `Matplotlib` for equation visualization  
- Automatic equation classification system  

**Frontend**  
- Clean `Bootstrap` interface  
- Interactive equation calculator  
- Dynamic solution display  
- Graphical equation plotting  

**Flask Integration**  
- Handles input processing  
- Coordinates frontend-backend communication  
- Formats solutions for display  

## Installation  

```bash
git clone https://github.com/yourusername/equation-solver.git
cd equation-solver
pip install -r requirements.txt
python app.py
