// AI Ethics Auditor — global JS

// Persist login state in navbar
(function() {
  const user = localStorage.getItem('username');
  const link = document.querySelector('.nav-link[href="/login"]');
  if (user && link) {
    link.innerHTML = `<i class="fas fa-user-check me-1 text-info"></i>${user}`;
    link.href = '#';
    link.onclick = () => { localStorage.clear(); location.reload(); };
  }
})();

// Global fetch helper with auth header
async function apiFetch(url, options = {}) {
  const token = localStorage.getItem('token');
  const headers = { 'Content-Type': 'application/json', ...(options.headers || {}) };
  if (token) headers['Authorization'] = `Bearer ${token}`;
  return fetch(url, { ...options, headers });
}
