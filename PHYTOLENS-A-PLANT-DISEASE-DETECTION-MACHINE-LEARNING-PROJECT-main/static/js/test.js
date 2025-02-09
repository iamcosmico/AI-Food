function previewImage(event) {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function (e) {
            const previewBox = document.getElementById("image-preview");
            previewBox.innerHTML = `<img src="${e.target.result}" alt="Uploaded Leaf Image">`;
        };
        reader.readAsDataURL(file);
    }
}

function resetImage() {
    document.getElementById("leaf-image").value = "";
    document.getElementById("image-preview").innerHTML = "<p>No image uploaded yet.</p>";
    document.getElementById("result-preview").innerHTML = "<p>Results will be displayed here after prediction.</p>";
}

function toggleMenu() {
    const navLinks = document.querySelector('.nav-links');
    if (navLinks) {
        navLinks.classList.toggle('show');
    } else {
        console.error('Error: Navigation links element not found.');
    }
}