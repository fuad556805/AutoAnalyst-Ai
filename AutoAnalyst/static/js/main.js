// ── Mobile menu ────────────────────────────────────
const hamburger     = document.querySelector('#hamburger');
const navMobile     = document.querySelector('#nav-mobile');
const navBackdrop   = document.querySelector('#nav-mobile-backdrop');
const navCloseBtn   = document.querySelector('#nav-mobile-close');

function openMobileMenu() {
  navMobile.classList.add('open');
  navBackdrop.classList.add('open');
  hamburger.classList.add('open');
  hamburger.setAttribute('aria-expanded', 'true');
  document.body.style.overflow = 'hidden';
}

function closeMobileMenu() {
  navMobile.classList.remove('open');
  navBackdrop.classList.remove('open');
  hamburger.classList.remove('open');
  hamburger.setAttribute('aria-expanded', 'false');
  document.body.style.overflow = '';
}

if (hamburger && navMobile) {
  hamburger.addEventListener('click', (e) => {
    e.stopPropagation();
    if (navMobile.classList.contains('open')) {
      closeMobileMenu();
    } else {
      openMobileMenu();
    }
  });

  if (navCloseBtn) {
    navCloseBtn.addEventListener('click', closeMobileMenu);
  }

  if (navBackdrop) {
    navBackdrop.addEventListener('click', closeMobileMenu);
  }

  // Close on nav link click
  navMobile.querySelectorAll('a').forEach(link => {
    link.addEventListener('click', closeMobileMenu);
  });

  // Close on Escape key
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') closeMobileMenu();
  });
}

// ── Upload drag & drop ─────────────────────────────
const uploadZone = document.querySelector('#upload-zone');
const fileInput  = document.querySelector('#dataset-file');
const uploadForm = document.querySelector('#upload-form');

if (uploadZone && fileInput && uploadForm) {
  uploadZone.addEventListener('click', (e) => {
    if (e.target !== fileInput) fileInput.click();
  });

  uploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadZone.classList.add('drag-over');
  });
  uploadZone.addEventListener('dragleave', (e) => {
    if (!uploadZone.contains(e.relatedTarget)) {
      uploadZone.classList.remove('drag-over');
    }
  });
  uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadZone.classList.remove('drag-over');
    const dt = e.dataTransfer;
    if (dt.files.length > 0) {
      setFile(dt.files[0]);
    }
  });

  fileInput.addEventListener('change', () => {
    if (fileInput.files.length > 0) {
      setFile(fileInput.files[0]);
    }
  });

  function setFile(file) {
    const allowed = ['csv', 'xlsx', 'xls'];
    const ext = file.name.split('.').pop().toLowerCase();
    if (!allowed.includes(ext)) {
      showZoneError('Only CSV and XLSX files are supported.');
      return;
    }
    const label = uploadZone.querySelector('.upload-filename');
    if (label) {
      label.textContent = '📎 ' + file.name;
    }
    setTimeout(() => uploadForm.submit(), 300);
  }

  function showZoneError(msg) {
    const label = uploadZone.querySelector('.upload-filename');
    if (label) {
      label.textContent = '⚠ ' + msg;
      label.style.color = 'var(--error)';
    }
  }
}

// ── Loading overlay for train form ────────────────
const trainForm      = document.querySelector('#train-form');
const loadingOverlay = document.querySelector('#loading-overlay');
if (trainForm && loadingOverlay) {
  trainForm.addEventListener('submit', () => {
    loadingOverlay.classList.remove('hidden');
  });
}

// ── Animate score bars ─────────────────────────────
document.querySelectorAll('.score-bar-fill').forEach(bar => {
  const w = bar.getAttribute('data-width');
  requestAnimationFrame(() => {
    setTimeout(() => { bar.style.width = w + '%'; }, 150);
  });
});

// ── Animate feature importance bars ───────────────
document.querySelectorAll('.feat-bar-fill').forEach(bar => {
  const w = bar.getAttribute('data-width');
  requestAnimationFrame(() => {
    setTimeout(() => { bar.style.width = w + '%'; }, 250);
  });
});

// ── Auto-dismiss alerts ────────────────────────────
setTimeout(() => {
  document.querySelectorAll('.messages li').forEach(el => {
    el.style.transition = 'opacity 0.5s ease';
    el.style.opacity = '0';
    setTimeout(() => el.remove(), 550);
  });
}, 4500);

// ── Password toggle (show / hide) ─────────────────
document.querySelectorAll('.pwd-toggle').forEach(btn => {
  btn.addEventListener('click', () => {
    const wrap  = btn.closest('.pwd-wrap');
    const input = wrap ? wrap.querySelector('input') : null;
    if (!input) return;

    if (input.type === 'password') {
      input.type = 'text';
      btn.innerHTML = iconEyeOff();
      btn.setAttribute('aria-label', 'Hide password');
    } else {
      input.type = 'password';
      btn.innerHTML = iconEye();
      btn.setAttribute('aria-label', 'Show password');
    }
  });
});

function iconEye() {
  return `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
    <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/>
    <circle cx="12" cy="12" r="3"/>
  </svg>`;
}

function iconEyeOff() {
  return `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
    <path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94"/>
    <path d="M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19"/>
    <line x1="1" y1="1" x2="23" y2="23"/>
  </svg>`;
}
