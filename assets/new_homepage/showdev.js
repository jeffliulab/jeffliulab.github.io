document.addEventListener('DOMContentLoaded', () => {

	/* ──────────────────────────────
	   DARK MODE
	   ────────────────────────────── */
	const body = document.body;
	const darkBtn = document.getElementById('darkModeBtn');
	const starsContainer = document.getElementById('starsContainer');
	const stored = localStorage.getItem('darkMode');

	if (stored === 'true') {
		body.classList.add('dark-mode');
		updateDarkIcon(true);
		createStars();
	}

	if (darkBtn) {
		darkBtn.addEventListener('click', () => {
			body.classList.toggle('dark-mode');
			const isDark = body.classList.contains('dark-mode');
			localStorage.setItem('darkMode', isDark);
			updateDarkIcon(isDark);
			if (isDark) createStars(); else clearStars();
		});
	}

	function updateDarkIcon(isDark) {
		if (!darkBtn) return;
		darkBtn.innerHTML = isDark
			? '<i class="bi bi-sun-fill"></i>'
			: '<i class="bi bi-moon-fill"></i>';
	}

	function createStars() {
		if (!starsContainer) return;
		clearStars();
		for (let i = 0; i < 80; i++) {
			const star = document.createElement('div');
			const sizes = ['small','medium','large'];
			star.className = 'star ' + sizes[Math.floor(Math.random()*3)];
			star.style.left = Math.random() * 100 + '%';
			star.style.top  = Math.random() * 100 + '%';
			star.style.animationDelay = (Math.random() * 3) + 's';
			starsContainer.appendChild(star);
		}
	}

	function clearStars() {
		if (starsContainer) starsContainer.innerHTML = '';
	}

	/* ──────────────────────────────
	   TYPING ANIMATION
	   ────────────────────────────── */
	const dynamicEl = document.getElementById('dynamicText');
	const phrases = [
		'Agentic AI Solutions',
		'Robotics & Embodied AI Solutions',
		'0-1 POC/Demo Builder',
		'Bridging Business & Tech'
	];
	let phraseIdx = 0, charIdx = 0, deleting = false;

	function typeLoop() {
		if (!dynamicEl) return;
		const current = phrases[phraseIdx];
		if (!deleting) {
			dynamicEl.textContent = current.substring(0, charIdx + 1);
			charIdx++;
			if (charIdx === current.length) {
				deleting = true;
				setTimeout(typeLoop, 1800);
				return;
			}
			setTimeout(typeLoop, 80);
		} else {
			dynamicEl.textContent = current.substring(0, charIdx - 1);
			charIdx--;
			if (charIdx === 0) {
				deleting = false;
				phraseIdx = (phraseIdx + 1) % phrases.length;
			}
			setTimeout(typeLoop, 40);
		}
	}
	typeLoop();

	/* ──────────────────────────────
	   NAVBAR & SIDEBAR (always visible)
	   ────────────────────────────── */

	/* ──────────────────────────────
	   PROJECT META ROW (status + tech tags)
	   ────────────────────────────── */
	document.querySelectorAll('.text-section').forEach(section => {
		const title = section.querySelector('.project-title');
		const status = section.querySelector('.project-status');
		const techStack = section.querySelector('.tech-stack');
		if (!title || !techStack) return;

		const metaRow = document.createElement('div');
		metaRow.className = 'project-meta-row';
		title.after(metaRow);

		if (status) metaRow.appendChild(status);
		metaRow.appendChild(techStack);
	});

	/* ──────────────────────────────
	   PROJECT DESCRIPTION CLAMP
	   ────────────────────────────── */
	document.querySelectorAll('.project-description').forEach(desc => {
		const wrap = document.createElement('div');
		wrap.className = 'project-desc-wrap';
		desc.parentNode.insertBefore(wrap, desc);
		wrap.appendChild(desc);

		const btn = document.createElement('button');
		btn.className = 'desc-toggle';
		btn.innerHTML = '<span class="lang-en">Show more</span><span class="lang-cn">展开</span>';
		wrap.appendChild(btn);

		wrap.classList.add('clamped');

		requestAnimationFrame(() => {
			if (desc.scrollHeight <= desc.clientHeight + 2) {
				wrap.classList.remove('clamped');
			}
		});

		btn.addEventListener('click', () => {
			if (wrap.classList.contains('expanded')) {
				wrap.classList.remove('expanded');
				wrap.classList.add('clamped');
				btn.innerHTML = '<span class="lang-en">Show more</span><span class="lang-cn">展开</span>';
			} else {
				wrap.classList.remove('clamped');
				wrap.classList.add('expanded');
				btn.innerHTML = '<span class="lang-en">Show less</span><span class="lang-cn">收起</span>';
			}
		});
	});

	/* ──────────────────────────────
	   ACTION BUTTON ICONS
	   ────────────────────────────── */
	const iconMap = {
		'github': 'bi-github',
		'live demo': 'bi-play-circle-fill',
		'pypi': 'bi-box-seam-fill',
		'report': 'bi-file-earmark-text-fill',
		'details': 'bi-info-circle-fill',
		'view': 'bi-eye-fill'
	};
	document.querySelectorAll('.case-study-btn').forEach(btn => {
		const text = btn.textContent.trim().toLowerCase();
		const icon = iconMap[text];
		if (icon) {
			btn.innerHTML = '<i class="bi ' + icon + '"></i> ' + btn.textContent.trim();
		}
	});

	/* ──────────────────────────────
	   TAB FILTERING
	   ────────────────────────────── */
	const tabBtns = document.querySelectorAll('.tab-btn');
	const groups  = document.querySelectorAll('.project-group');

	tabBtns.forEach(btn => {
		btn.addEventListener('click', () => {
			tabBtns.forEach(b => b.classList.remove('active'));
			btn.classList.add('active');
			const target = btn.dataset.tab;
			groups.forEach(g => {
				if (target === 'all' || g.dataset.group === target) {
					g.style.display = '';
				} else {
					g.style.display = 'none';
				}
			});
		});
	});

	/* ──────────────────────────────
	   LANGUAGE TOGGLE
	   ────────────────────────────── */
	const langBtn   = document.getElementById('langToggle');
	const langBtnEn = document.getElementById('langBtnEn');
	const langBtnCn = document.getElementById('langBtnCn');

	if (langBtn) {
		langBtn.addEventListener('click', () => {
			const html = document.documentElement;
			if (html.classList.contains('lang-en')) {
				html.classList.replace('lang-en', 'lang-cn');
				langBtnEn.classList.remove('active');
				langBtnCn.classList.add('active');
			} else {
				html.classList.replace('lang-cn', 'lang-en');
				langBtnCn.classList.remove('active');
				langBtnEn.classList.add('active');
			}
		});
	}

	/* ──────────────────────────────
	   PDF MODAL
	   ────────────────────────────── */
	window.openPdfModal = function(url) {
		const modal = document.getElementById('pdfModal');
		const frame = document.getElementById('pdfFrame');
		if (!modal || !frame) return;
		frame.src = url + '#page=1';
		modal.style.display = 'flex';
		document.body.style.overflow = 'hidden';
	};

	window.closePdfModal = function() {
		const modal = document.getElementById('pdfModal');
		const frame = document.getElementById('pdfFrame');
		if (!modal || !frame) return;
		frame.src = '';
		modal.style.display = 'none';
		document.body.style.overflow = '';
		document.getElementById('pdfModalContent').classList.remove('fullscreen');
	};

	window.togglePdfFullscreen = function() {
		document.getElementById('pdfModalContent').classList.toggle('fullscreen');
	};

	/* ──────────────────────────────
	   VIDEO MODAL
	   ────────────────────────────── */
	window.openVideoModal = function(videoId) {
		const modal  = document.getElementById('videoModal');
		const iframe = document.getElementById('ytIframe');
		if (!modal || !iframe) return;
		iframe.src = 'https://www.youtube.com/embed/' + videoId + '?autoplay=1';
		modal.style.display = 'block';
	};

	window.closeVideoModal = function() {
		const modal  = document.getElementById('videoModal');
		const iframe = document.getElementById('ytIframe');
		if (!modal || !iframe) return;
		iframe.src = '';
		modal.style.display = 'none';
	};

	document.addEventListener('keydown', e => {
		if (e.key === 'Escape') {
			closeVideoModal();
			closePdfModal();
		}
	});

	document.addEventListener('click', e => {
		const videoModal = document.getElementById('videoModal');
		const pdfModal   = document.getElementById('pdfModal');
		if (videoModal && e.target === videoModal) closeVideoModal();
		if (pdfModal   && e.target === pdfModal)   closePdfModal();
	});

	/* ──────────────────────────────
	   AOS INIT
	   ────────────────────────────── */
	if (typeof AOS !== 'undefined') {
		AOS.init({ duration: 700, once: true, offset: 60 });
	}

});
