// Start the evaluation process when the page loads
document.addEventListener('DOMContentLoaded', function() {
    // Start the evaluation
    fetch('/run-evaluation/')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // Start checking the evaluation status
                checkEvaluationStatus();
            } else {
                console.error('Failed to start evaluation:', data.error);
                window.location.href = '/uploadimage/';
            }
        })
        .catch(error => {
            console.error('Error starting evaluation:', error);
            window.location.href = '/uploadimage/';
        });
});

// Function to check if the evaluation is complete
function checkEvaluationStatus() {
    fetch('/check-evaluation-status/')
        .then(response => response.json())
        .then(data => {
            if (data.complete) {
                // Evaluation is complete, redirect to results page
                window.location.href = '/result/';
            } else {
                // Check again after 2 seconds
                setTimeout(checkEvaluationStatus, 2000);
            }
        })
        .catch(error => {
            console.error('Error checking evaluation status:', error);
            // If there's an error, wait and try again
            setTimeout(checkEvaluationStatus, 2000);
        });
}
