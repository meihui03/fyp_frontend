document.addEventListener('DOMContentLoaded', function() {
    const searchInput = document.getElementById('searchInput');
    const showAllBtn = document.getElementById('showAllBtn');
    const showLessBtn = document.getElementById('showLessBtn');
    const userRows = document.querySelectorAll('.user-row');
    let isShowingAll = false;

    // Function to show only first 10 users
    function showDefaultView() {
        userRows.forEach((row, index) => {
            if (index >= 10) {
                row.style.display = 'none';
            } else {
                row.style.display = '';
            }
        });
        showAllBtn.style.display = '';
        showLessBtn.style.display = 'none';
        isShowingAll = false;
    }

    // Search functionality
    searchInput.addEventListener('keyup', function() {
        const filter = this.value.toUpperCase();
        
        if (filter === '') {
            showDefaultView();
            return;
        }

        userRows.forEach(row => {
            const td = row.getElementsByTagName('td')[1]; // User ID is in the 2nd column
            if (td) {
                const txtValue = td.textContent || td.innerText;
                if (txtValue.toUpperCase().indexOf(filter) > -1) {
                    row.style.display = '';
                } else {
                    row.style.display = 'none';
                }
            }
        });
    });

    // Show All button functionality
    showAllBtn.addEventListener('click', function() {
        userRows.forEach(row => {
            row.style.display = '';
        });
        showAllBtn.style.display = 'none';
        showLessBtn.style.display = '';
        isShowingAll = true;
    });

    // Show Less button functionality
    showLessBtn.addEventListener('click', function() {
        showDefaultView();
    });

    // Initialize with default view
    showDefaultView();
});