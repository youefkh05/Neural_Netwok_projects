// Enhanced visuals and animations
document.addEventListener('DOMContentLoaded', () => {
  // Counter animation for statistics
  const stats = document.querySelectorAll('.stat strong');
  stats.forEach((stat) => {
    const text = stat.textContent;
    const isNumber = /^\d+$/.test(text);
    if (isNumber) {
      const target = parseInt(text);
      let current = 0;
      const increment = target / 40;
      
      const interval = setInterval(() => {
        current += increment;
        if (current >= target) {
          stat.textContent = target;
          clearInterval(interval);
        } else {
          stat.textContent = Math.floor(current);
        }
      }, 50);
    }
  });

  // Floating animation for cards on scroll
  const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
  };

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('visible');
        observer.unobserve(entry.target);
      }
    });
  }, observerOptions);

  document.querySelectorAll('.reveal').forEach(el => {
    observer.observe(el);
  });

  // Add subtle mouse tracking to hero visual
  const heroVisual = document.querySelector('.hero-visual img');
  if (heroVisual) {
    document.addEventListener('mousemove', (e) => {
      const rect = heroVisual.getBoundingClientRect();
      const centerX = rect.left + rect.width / 2;
      const centerY = rect.top + rect.height / 2;
      const angleX = (e.clientY - centerY) * 0.01;
      const angleY = (e.clientX - centerX) * 0.01;
      heroVisual.style.transform = `rotateX(${angleX}deg) rotateY(${angleY}deg) scale(1.02)`;
    });

    document.addEventListener('mouseleave', () => {
      heroVisual.style.transform = 'rotateX(0) rotateY(0) scale(1)';
    });
  }

  // Add glow effect to cards on hover
  const cards = document.querySelectorAll('.feature-card, .glass-panel');
  cards.forEach(card => {
    card.addEventListener('mouseenter', function() {
      this.style.setProperty('--glow-x', event.pageX + 'px');
      this.style.setProperty('--glow-y', event.pageY + 'px');
    });
  });

  // Smooth scroll for anchor links
  document.querySelectorAll('a[href^="#"]').forEach(link => {
    link.addEventListener('click', function(e) {
      const href = this.getAttribute('href');
      if (href && href !== '#') {
        const target = document.querySelector(href);
        if (target) {
          e.preventDefault();
          target.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
      }
    });
  });
});
