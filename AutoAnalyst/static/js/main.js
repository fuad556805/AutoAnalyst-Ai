// ── Mobile menu ────────────────────────────────────
const hamburger = document.querySelector('#hamburger');
const navMobile = document.querySelector('#nav-mobile');
if (hamburger && navMobile) {
  hamburger.addEventListener('click', () => {
    navMobile.classList.toggle('open');
  });
  document.addEventListener('click', (e) => {
    if (!hamburger.contains(e.target) && !navMobile.contains(e.target)) {
      navMobile.classList.remove('open');
    }
  });
}

// ── Upload drag & drop ─────────────────────────────
const uploadZone = document.querySelector('#upload-zone');
const fileInput  = document.querySelector('#dataset-file');
const uploadForm = document.querySelector('#upload-form');

if (uploadZone && fileInput && uploadForm) {
  // Click to open file dialog
  uploadZone.addEventListener('click', (e) => {
    if (e.target !== fileInput) fileInput.click();
  });

  // Drag events
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

  // File input change
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
    // Small delay so user sees the filename, then submit
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
const trainForm       = document.querySelector('#train-form');
const loadingOverlay  = document.querySelector('#loading-overlay');
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
