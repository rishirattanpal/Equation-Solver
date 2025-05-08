// Equation Solver Calculator Functions
let lastAnswer = null;


function append_to_equation(input) {
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



function clear_equation() {
    document.getElementById("equations").value = "";
    document.getElementById("equations").focus();
}


function backspace() {
    const equationsArea = document.getElementById("equations");
    equationsArea.value = equationsArea.value.slice(0, -1);
    equationsArea.focus();
}



function solve_equation() {
    document.getElementById("equationForm").submit();
}


// document.addEventListener('DOMContentLoaded', function() {
//     const equationsArea = document.getElementById("equations");
    
//     // Add keyboard event listener for keyboard shortcuts
//     equationsArea.addEventListener('keydown', function(event) {
//         // Allow special key combinations (like Ctrl+C, Ctrl+V)
//         if (event.ctrlKey || event.metaKey) {
//             return;
//         }
        
//         // Handle Enter key to add a new line instead of submitting the form
//         if (event.key === 'Enter' && !event.shiftKey) {
//             event.preventDefault();
//             append_to_equation('\\n');
//         }
//     });
    
//     // Initialize focus on the textarea when page loads
//     equationsArea.focus();
// });


async function get_prev_ans() {  
    const equationsArea = document.getElementById("equations");
    
    try {
        const response = await fetch('/get_last_answer');
        const data = await response.json();
        
        if (data.lastAnswer) {
            equationsArea.value += data.lastAnswer;
            equationsArea.focus();
        } else {
            alert("No previous answer available");
        }
    } catch (error) {
        console.error("Error fetching last answer:", error);
    }
}