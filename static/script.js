/* ===== PlantCare AI — script.js ===== */

document.addEventListener('DOMContentLoaded', () => {

  /* ── Navbar scroll effect ── */
  const navbar = document.querySelector('.navbar');
  if (navbar) {
    window.addEventListener('scroll', () => {
      navbar.classList.toggle('scrolled', window.scrollY > 24);
    }, { passive: true });
  }

  /* ── Active nav link ── */
  const currentPath = window.location.pathname;
  document.querySelectorAll('.nav-links a').forEach(link => {
    const href = link.getAttribute('href');
    if (href === currentPath || (href !== '/' && currentPath.startsWith(href))) {
      link.classList.add('active');
    }
  });

  /* ── Mobile hamburger ── */
  const hamburger = document.querySelector('.nav-hamburger');
  const navLinks  = document.querySelector('.nav-links');
  if (hamburger && navLinks) {
    hamburger.addEventListener('click', () => {
      navLinks.classList.toggle('open');
      hamburger.setAttribute('aria-expanded', navLinks.classList.contains('open'));
    });
    // close on outside click
    document.addEventListener('click', (e) => {
      if (!navbar.contains(e.target)) navLinks.classList.remove('open');
    });
  }

  /* ── Scroll reveal ── */
  const revealEls = document.querySelectorAll('.reveal');
  if (revealEls.length) {
    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry, i) => {
        if (entry.isIntersecting) {
          setTimeout(() => entry.target.classList.add('visible'), i * 80);
          observer.unobserve(entry.target);
        }
      });
    }, { threshold: 0.1, rootMargin: '0px 0px -40px 0px' });
    revealEls.forEach(el => observer.observe(el));
  }

  /* ── Counter animation ── */
  const counters = document.querySelectorAll('[data-count]');
  if (counters.length) {
    const countObserver = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          const el     = entry.target;
          const target = parseFloat(el.dataset.count);
          const suffix = el.dataset.suffix || '';
          const dec    = el.dataset.dec || 0;
          const dur    = 1400;
          const start  = performance.now();
          const update = (now) => {
            const p = Math.min((now - start) / dur, 1);
            const v = p < .5 ? 2 * p * p : 1 - Math.pow(-2 * p + 2, 2) / 2;
            el.textContent = (target * v).toFixed(dec) + suffix;
            if (p < 1) requestAnimationFrame(update);
            else el.textContent = target.toFixed(dec) + suffix;
          };
          requestAnimationFrame(update);
          countObserver.unobserve(el);
        }
      });
    }, { threshold: 0.5 });
    counters.forEach(c => countObserver.observe(c));
  }

  /* ═══════════════════════════════
     UPLOAD PAGE
  ═══════════════════════════════ */
  const dropZone   = document.getElementById('dropZone');
  const fileInput  = document.getElementById('fileInput');
  const previewWrap= document.getElementById('previewWrap');
  const previewImg = document.getElementById('previewImg');
  const previewFileName = document.getElementById('previewFileName');
  const removeBtn  = document.getElementById('removeBtn');
  const submitBtn  = document.getElementById('submitBtn');
  const uploadForm = document.getElementById('uploadForm');
  const spinner    = document.getElementById('spinnerOverlay');

  if (dropZone && fileInput) {

    /* click to open file picker */
    dropZone.addEventListener('click', (e) => {
      if (e.target === removeBtn) return;
      fileInput.click();
    });

    /* drag events */
    ['dragenter','dragover'].forEach(ev => {
      dropZone.addEventListener(ev, (e) => { e.preventDefault(); dropZone.classList.add('drag-over'); });
    });
    ['dragleave','drop'].forEach(ev => {
      dropZone.addEventListener(ev, (e) => { e.preventDefault(); dropZone.classList.remove('drag-over'); });
    });

    dropZone.addEventListener('drop', (e) => {
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    });

    fileInput.addEventListener('change', () => {
      if (fileInput.files[0]) handleFile(fileInput.files[0]);
    });

    if (removeBtn) {
      removeBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        clearPreview();
      });
    }
  }

  function handleFile(file) {
    const allowed = ['image/jpeg','image/jpg','image/png','image/webp'];
    if (!allowed.includes(file.type)) {
      showToast('Please upload a JPG, PNG or WEBP image.', 'error');
      return;
    }
    if (file.size > 10 * 1024 * 1024) {
      showToast('File size must be under 10 MB.', 'error');
      return;
    }
    const reader = new FileReader();
    reader.onload = (e) => {
      if (previewImg)      previewImg.src = e.target.result;
      if (previewWrap)     previewWrap.classList.add('show');
      if (previewFileName) previewFileName.textContent = `📄 ${file.name} (${(file.size/1024).toFixed(0)} KB)`;
      if (submitBtn)       submitBtn.disabled = false;
    };
    reader.readAsDataURL(file);
    // transfer to real input if needed
    const dt = new DataTransfer();
    dt.items.add(file);
    if (fileInput) fileInput.files = dt.files;
  }

  function clearPreview() {
    if (previewImg)      previewImg.src = '';
    if (previewWrap)     previewWrap.classList.remove('show');
    if (fileInput)       fileInput.value = '';
    if (submitBtn)       submitBtn.disabled = true;
  }

  /* form submission with spinner */
  if (uploadForm && spinner) {
    uploadForm.addEventListener('submit', (e) => {
      if (!fileInput || !fileInput.files[0]) {
        e.preventDefault();
        showToast('Please select an image first.', 'error');
        return;
      }
      showSpinner();
    });
  }

  /* ── Spinner ── */
  function showSpinner() {
    if (!spinner) return;
    spinner.classList.add('active');
    const steps   = spinner.querySelectorAll('.spinner-step');
    const progress= spinner.querySelector('.progress-fill');
    const delays  = [800, 1800, 3000, 4200];
    const pcts    = ['20%','45%','72%','90%'];

    steps.forEach((s, i) => {
      setTimeout(() => {
        steps.forEach(x => x.classList.remove('active'));
        s.classList.add('active');
        if (i > 0) steps[i-1].classList.add('done');
        if (progress) progress.style.width = pcts[i];
      }, delays[i]);
    });
  }

  /* ═══════════════════════════════
     RESULT PAGE
  ═══════════════════════════════ */
  const confFill = document.querySelector('.conf-fill');
  const confPct  = document.querySelector('.conf-pct');

  if (confFill) {
    const targetW = confFill.dataset.confidence || '94';
    setTimeout(() => {
      confFill.style.width = targetW + '%';
    }, 300);
  }

  /* download result */
  const downloadBtn = document.getElementById('downloadBtn');
  if (downloadBtn) {
    downloadBtn.addEventListener('click', () => {
      const title   = document.querySelector('.result-title')?.textContent || 'Result';
      const pct     = document.querySelector('.conf-pct')?.textContent || '';
      const plant   = document.querySelector('[data-plant]')?.dataset.plant || '';
      const content = `PlantCare AI — Detection Report\n${'='.repeat(40)}\n\nDisease: ${title}\nPlant: ${plant}\nConfidence: ${pct}\nDate: ${new Date().toLocaleDateString()}\n\nGenerated by PlantCare AI\nhttps://plantcare.ai`;
      const blob = new Blob([content], { type: 'text/plain' });
      const url  = URL.createObjectURL(blob);
      const a    = document.createElement('a');
      a.href = url; a.download = 'plantcare-result.txt'; a.click();
      URL.revokeObjectURL(url);
      showToast('Report downloaded!', 'success');
    });
  }

  /* share */
  const shareBtn = document.getElementById('shareBtn');
  if (shareBtn) {
    shareBtn.addEventListener('click', async () => {
      const title = document.querySelector('.result-title')?.textContent;
      if (navigator.share) {
        try { await navigator.share({ title: 'PlantCare AI Result', text: `Detected: ${title}`, url: window.location.href }); }
        catch {}
      } else {
        navigator.clipboard.writeText(window.location.href).then(() => showToast('Link copied to clipboard!', 'success'));
      }
    });
  }

  /* ── Toast notification ── */
  function showToast(message, type = 'info') {
    const existing = document.querySelector('.toast');
    if (existing) existing.remove();
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.innerHTML = `<span>${type === 'success' ? '✅' : type === 'error' ? '❌' : 'ℹ️'}</span> ${message}`;
    Object.assign(toast.style, {
      position: 'fixed', bottom: '24px', right: '24px', zIndex: '999',
      background: type === 'success' ? '#1a4a22' : type === 'error' ? '#7f1d1d' : '#1a4a22',
      color: '#fff', padding: '14px 20px', borderRadius: '10px',
      fontFamily: 'var(--font-body)', fontSize: '.9rem', fontWeight: '500',
      boxShadow: '0 8px 32px rgba(0,0,0,.3)', display: 'flex', alignItems: 'center',
      gap: '10px', transform: 'translateY(80px)', opacity: '0',
      transition: 'transform .3s ease, opacity .3s ease', maxWidth: '340px'
    });
    document.body.appendChild(toast);
    requestAnimationFrame(() => { toast.style.transform = 'translateY(0)'; toast.style.opacity = '1'; });
    setTimeout(() => { toast.style.transform = 'translateY(80px)'; toast.style.opacity = '0'; setTimeout(() => toast.remove(), 350); }, 3500);
  }

  /* ── Smooth page transitions ── */
  document.querySelectorAll('a[href]').forEach(link => {
    const href = link.getAttribute('href');
    if (href.startsWith('/') && !href.startsWith('//')) {
      link.addEventListener('click', (e) => {
        e.preventDefault();
        document.body.style.opacity = '0';
        document.body.style.transition = 'opacity .2s ease';
        setTimeout(() => { window.location.href = href; }, 180);
      });
    }
  });

  document.body.style.opacity = '0';
  document.body.style.transition = 'opacity .3s ease';
  requestAnimationFrame(() => { document.body.style.opacity = '1'; });

});
