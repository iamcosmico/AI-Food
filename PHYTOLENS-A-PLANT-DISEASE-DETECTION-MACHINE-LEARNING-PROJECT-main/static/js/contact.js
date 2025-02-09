let currentIndex = 0;
const sliderItems = document.querySelectorAll('.slider-item');
const totalSlides = sliderItems.length;

function showNextSlide() {
    // Remove the 'active' class from the current slide
    sliderItems[currentIndex].classList.remove('active');

    // Calculate the next slide index
    currentIndex = (currentIndex + 1) % totalSlides;

    // Add the 'active' class to the next slide
    sliderItems[currentIndex].classList.add('active');
}

// Initialize the first slide
sliderItems[currentIndex].classList.add('active');

// Change slide every 2 seconds
setInterval(showNextSlide, 2000);
