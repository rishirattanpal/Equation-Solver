// Equation Solver Calculator Functions



function appendToEquation(input) {
    const equationsArea = document.getElementById("equations");
    
    // Handle special cases
    if (input === '\\n') {
        equationsArea.value += '\n';
    } else if (input === '\\sqrt{}') {
        // Insert sqrt and position cursor between braces
        const start = equationsArea.selectionStart;
        const end = equationsArea.selectionEnd;
        equationsArea.value = equationsArea.value.substring(0, start) + 
                            'sqrt()' + 
                            equationsArea.value.substring(end);
        // Position the cursor between the parentheses
        equationsArea.selectionStart = equationsArea.selectionEnd = start + 5;
    } else if (input === '\\frac{}{}') {
        // Insert fraction and position cursor for numerator
        const start = equationsArea.selectionStart;
        const end = equationsArea.selectionEnd;
        equationsArea.value = equationsArea.value.substring(0, start) + 
                            '()/()' + 
                            equationsArea.value.substring(end);
        equationsArea.selectionStart = equationsArea.selectionEnd = start + 1;
    } else {
        equationsArea.value += input;
    }
    
    // Focus on the textarea to allow continued typing
    equationsArea.focus();
}



function clearEquation() {
    document.getElementById("equations").value = "";
    document.getElementById("equations").focus();
}


function backspace() {
    const equationsArea = document.getElementById("equations");
    equationsArea.value = equationsArea.value.slice(0, -1);
    equationsArea.focus();
}



function solveEquation() {
    document.getElementById("equationForm").submit();
}


document.addEventListener('DOMContentLoaded', function() {
    const equationsArea = document.getElementById("equations");
    
    // Add keyboard event listener for keyboard shortcuts
    equationsArea.addEventListener('keydown', function(event) {
        // Allow special key combinations (like Ctrl+C, Ctrl+V)
        if (event.ctrlKey || event.metaKey) {
            return;
        }
        
        // Handle Enter key to add a new line instead of submitting the form
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            appendToEquation('\\n');
        }
    });
    
    // Initialize focus on the textarea when page loads
    equationsArea.focus();
});