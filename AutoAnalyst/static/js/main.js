// Mobile menu
const hamburger = document.querySelector('.hamburger');
const navMobile = document.querySelector('.nav-mobile');
if (hamburger && navMobile) {
  hamburger.addEventListener('click', () => {
    navMobile.classList.toggle('open');
  });
}

// Upload drag & drop
const uploadZone = document.querySelector('.upload-zone');
const fileInput = document.querySelector('#dataset-file');
if (uploadZone && fileInput) {
  uploadZone.addEventListener('click', () => fileInput.click());
  uploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadZone.classList.add('drag-over');
  });
  uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('drag-over'));
  uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadZone.classList.remove('drag-over');
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      fileInput.files = files;
      updateUploadLabel(files[0].name);
      document.querySelector('#upload-form').submit();
    }
  });
  fileInput.addEventListener('change', () => {
    if (fileInput.files.length > 0) {
      updateUploadLabel(fileInput.files[0].name);
      document.querySelector('#upload-form').submit();
    }
  });
}

function updateUploadLabel(name) {
  const label = document.querySelector('.upload-filename');
  if (label) label.textContent = name;
}

// Loading overlay for train
const trainForm = document.querySelector('#train-form');
const loadingOverlay = document.querySelector('.loading-overlay');
if (trainForm && loadingOverlay) {
  trainForm.addEventListener('submit', (e) => {
    loadingOverlay.classList.remove('hidden');
  });
}

// Score bar animation
document.querySelectorAll('.score-bar-fill').forEach(bar => {
  const w = bar.getAttribute('data-width');
  setTimeout(() => { bar.style.width = w + '%'; }, 200);
});

// Feature importance bars
document.querySelectorAll('.feat-bar-fill').forEach(bar => {
  const w = bar.getAttribute('data-width');
  setTimeout(() => { bar.style.width = w + '%'; }, 300);
});

// Auto dismiss messages
setTimeout(() => {
  document.querySelectorAll('.messages li').forEach(el => {
    el.style.transition = 'opacity 0.5s';
    el.style.opacity = '0';
    setTimeout(() => el.remove(), 600);
  });
}, 4000);
