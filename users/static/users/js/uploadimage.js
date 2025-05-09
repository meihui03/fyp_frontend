function previewImage(event) {
    const image = document.getElementById('preview');
    image.src = URL.createObjectURL(event.target.files[0]);
    image.style.display = 'block';
}

function uploadImage() {
    const formData = new FormData(document.getElementById('uploadForm'));
    
    fetch('/post-upload-images/', {
        method: 'POST',
        body: formData,
        headers: {
            'X-CSRFToken': '{{ csrf_token }}'  // Include CSRF token
        }
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            alert('Image uploaded successfully!');
            window.location.href = data.redirect_url;  // redirect on success
        } else {
            alert('Error: ' + data.error);
        }
    })
    .catch(error => {
        console.error('Error:', error);
    });
}

