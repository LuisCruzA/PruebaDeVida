// Función para abrir/cerrar el menú lateral
function toggleMenu() {
    var menu = document.getElementById("sideMenu");
    if (menu.style.width === "250px") {
        menu.style.width = "0";  // Cierra el menú
    } else {
        menu.style.width = "250px";  // Abre el menú
    }
}

const animados = document.querySelectorAll('.animado');

const observer = new IntersectionObserver(entries => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      entry.target.style.opacity = 1;
      entry.target.style.animationPlayState = 'running';
      observer.unobserve(entry.target);
    }
  });
}, {
  threshold: 0.1
});

animados.forEach(el => {
  el.style.animationPlayState = 'paused';
  observer.observe(el);
});
