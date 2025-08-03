// Initialize carousel
document.addEventListener('DOMContentLoaded', function() {
    // Slider functionality
    const myCarousel = new bootstrap.Carousel(document.getElementById('carouselExample'), {
        interval: 3000,
        wrap: true
    });
    
    // Any other custom JS can go here
    console.log('Script loaded successfully');
    
    // Example: Add active class to current page in navbar
    const currentUrl = window.location.pathname;
    document.querySelectorAll('.navbar-nav a').forEach(link => {
        if (link.getAttribute('href') === currentUrl) {
            link.classList.add('active');
        }
    });
});