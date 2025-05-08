// Function to check if the process is complete
function checkProcessStatus() {
    fetch('/check-process-status/')
        .then(response => response.json())
        .then(data => {
            if (data.complete) {
                // Get the trigger path
                fetch('/get-trigger-path/')
                    .then(response => response.json())
                    .then(data => {
                        if (data.trigger_path) {
                            console.log('Trigger path:', data.trigger_path);
                            // Store the trigger path in localStorage for later use
                            localStorage.setItem('lastTriggerPath', data.trigger_path);
                        }
                        // Redirect to upload image page
                        window.location.href = uploadImageUrl;
                    })
                    .catch(error => {
                        console.error('Error getting trigger path:', error);
                        window.location.href = uploadImageUrl;
                    });
            } else {
                // Check again after 5 seconds
                setTimeout(checkProcessStatus, 5000);
            }
        })
        .catch(error => {
            console.error('Error checking process status:', error);
            // If there's an error, wait and try again
            setTimeout(checkProcessStatus, 5000);
        });
}

// Start checking the process status
checkProcessStatus();