const links = ['link-data-science', 'link-web-projects', 'link-ml-nerd'];
links.forEach(id => {
  const el = document.getElementById(id);
  if (el) {
    el.addEventListener('click', e => {
      e.preventDefault();
      window.location.href = 'comingsoon.html';
    });
  }
});
