document.addEventListener('DOMContentLoaded', function() {
    const sky = document.querySelector('.starry-sky');
    const starCount = 200; // Number of stars

    for (let i = 0; i < starCount; i++) {
        const star = document.createElement('div');
        star.classList.add('star');
        const x = Math.random() * window.innerWidth;
        const y = Math.random() * window.innerHeight;
        const delay = Math.random() * 2; // Random delay for twinkle effect

        star.style.left = `${x}px`;
        star.style.top = `${y}px`;
        star.style.animationDelay = `${delay}s`;

        sky.appendChild(star);
    }
});
