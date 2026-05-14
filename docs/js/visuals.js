// Small visual enhancements: parallax and reveal init
document.addEventListener('DOMContentLoaded', function(){
  // Initialize existing reveal observer from site.js
  if (typeof initReveal === 'function') initReveal();

  // Parallax hero image
  const hero = document.querySelector('.hero-visual img');
  if (hero){
    const wrap = hero.parentElement;
    wrap.addEventListener('mousemove', e => {
      const r = wrap.getBoundingClientRect();
      const x = (e.clientX - r.left) / r.width - 0.5;
      const y = (e.clientY - r.top) / r.height - 0.5;
      hero.style.transform = `translate3d(${x*10}px, ${y*-8}px, 0) scale(1.01)`;
    });
    wrap.addEventListener('mouseleave', ()=>{ hero.style.transform = 'translateY(0)'; });
  }

  // Soft entrance for badges
  const badges = document.querySelectorAll('.badge-row img');
  badges.forEach((b,i)=>{
    b.style.opacity = 0; b.style.transform = 'translateY(8px)';
    setTimeout(()=>{ b.style.transition = 'all .6s cubic-bezier(.2,.9,.2,1)'; b.style.opacity = 1; b.style.transform='translateY(0)'; }, 350 + i*120);
  });
});