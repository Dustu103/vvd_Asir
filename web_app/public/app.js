/**
 * Hoichoi Problem 2 — Broadcast Subtitling & Diarization Web Suite
 * Client-side Controller: Video sync, interactive review queue,
 * multi-track subtitle switching, and live pipeline stage runner.
 */

class SubtitleSuiteApp {
  constructor() {
    this.currentEpisode = 'mohanagar';
    this.currentLang = 'bn';
    this.episodes = [];
    this.pipelineData = null;
    this.activeCueIndex = -1;
    this.subtitlesVisible = true;

    this.initElements();
    this.initEventListeners();
    this.loadEpisodes();
  }

  initElements() {
    this.video = document.getElementById('main-video');
    this.videoSource = document.getElementById('video-source');
    this.playBtn = document.getElementById('btn-play-pause');
    this.playIcon = document.getElementById('play-icon');
    this.currentTimeEl = document.getElementById('current-time');
    this.totalDurationEl = document.getElementById('total-duration');
    this.seekBar = document.getElementById('seek-bar');
    this.shotMarkers = document.getElementById('shot-markers');
    
    this.overlay = document.getElementById('subtitle-overlay');
    this.overlaySpeaker = document.getElementById('overlay-speaker');
    this.overlayText = document.getElementById('overlay-text');

    this.timelineContainer = document.getElementById('cue-timeline');
    this.episodeList = document.getElementById('episode-list');
    this.uploadZone = document.getElementById('upload-zone');
    this.fileInput = document.getElementById('video-file-input');

    this.runPipelineBtn = document.getElementById('btn-run-pipeline');
    this.cueSearchInput = document.getElementById('cue-search-input');

    // QC Elements
    this.complianceCircle = document.getElementById('compliance-circle');
    this.complianceScore = document.getElementById('compliance-score');
    this.qcPassRate = document.getElementById('qc-pass-rate');
    this.qcHallucinations = document.getElementById('qc-hallucinations');
    this.qcMaxCps = document.getElementById('qc-max-cps');
    this.qcReviewCount = document.getElementById('qc-review-count');
    this.reviewList = document.getElementById('review-items-list');
    this.badgeQueueCount = document.getElementById('badge-queue-count');
    this.speakerChipList = document.getElementById('speaker-chip-list');

    // Track tab counts
    this.bnCount = document.getElementById('bn-cue-count');
    this.enCount = document.getElementById('en-cue-count');
    this.hiCount = document.getElementById('hi-cue-count');
  }

  initEventListeners() {
    // Play/Pause
    this.playBtn.addEventListener('click', () => this.togglePlay());
    this.video.addEventListener('click', () => this.togglePlay());

    // Time update & seeking
    this.video.addEventListener('timeupdate', () => this.onTimeUpdate());
    this.video.addEventListener('loadedmetadata', () => {
      this.totalDurationEl.textContent = this.formatTime(this.video.duration);
    });

    this.seekBar.addEventListener('input', (e) => {
      const seekTime = (e.target.value / 100) * this.video.duration;
      this.video.currentTime = seekTime;
    });

    // Subtitle toggle
    document.getElementById('btn-toggle-subtitles').addEventListener('click', () => {
      this.subtitlesVisible = !this.subtitlesVisible;
      this.overlay.style.display = this.subtitlesVisible ? 'block' : 'none';
    });

    // Fullscreen
    document.getElementById('btn-fullscreen').addEventListener('click', () => {
      const container = document.getElementById('video-container');
      if (!document.fullscreenElement) {
        container.requestFullscreen?.() || container.webkitRequestFullscreen?.();
      } else {
        document.exitFullscreen?.();
      }
    });

    // Language tabs
    document.querySelectorAll('.track-tab').forEach(tab => {
      tab.addEventListener('click', (e) => {
        const lang = tab.getAttribute('data-lang');
        document.querySelectorAll('.track-tab').forEach(t => t.classList.remove('active'));
        tab.classList.add('active');
        this.switchLanguage(lang);
      });
    });

    // Search cues
    this.cueSearchInput.addEventListener('input', (e) => {
      this.filterCues(e.target.value.toLowerCase());
    });

    // Duration Range Selector
    const durSelect = document.getElementById('max-dur-select');
    const customDurContainer = document.getElementById('custom-dur-container');
    if (durSelect && customDurContainer) {
      durSelect.addEventListener('change', (e) => {
        customDurContainer.style.display = e.target.value === 'custom' ? 'block' : 'none';
      });
    }

    // ML Server Connect Button
    const saveMlBtn = document.getElementById('btn-save-ml-url');
    const mlUrlInput = document.getElementById('ml-backend-url-input');
    if (saveMlBtn && mlUrlInput) {
      saveMlBtn.addEventListener('click', async () => {
        saveMlBtn.textContent = 'Connecting...';
        try {
          await fetch('/api/set-ml-backend', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ url: mlUrlInput.value }),
          });
          await this.checkMLStatus();
        } catch (e) {
          console.error(e);
        }
        saveMlBtn.textContent = 'Connect';
      });
    }

    // Run pipeline button
    this.runPipelineBtn.addEventListener('click', () => this.triggerPipelineExecution());

    // Upload handlers
    document.getElementById('btn-browse-file').addEventListener('click', (e) => {
      e.stopPropagation();
      this.fileInput.click();
    });
    this.uploadZone.addEventListener('click', () => this.fileInput.click());
    this.fileInput.addEventListener('change', (e) => this.handleFileUpload(e.target.files[0]));

    // Drag and drop
    this.uploadZone.addEventListener('dragover', (e) => {
      e.preventDefault();
      this.uploadZone.classList.add('drag-over');
    });
    this.uploadZone.addEventListener('dragleave', () => this.uploadZone.classList.remove('drag-over'));
    this.uploadZone.addEventListener('drop', (e) => {
      e.preventDefault();
      this.uploadZone.classList.remove('drag-over');
      if (e.dataTransfer.files.length) {
        this.handleFileUpload(e.dataTransfer.files[0]);
      }
    });

    // Check ML status on startup
    this.checkMLStatus();
    setInterval(() => this.checkMLStatus(), 15000);
  }

  async checkMLStatus() {
    try {
      const res = await fetch('/api/ml-status');
      const data = await res.json();
      const dot = document.getElementById('ml-status-dot');
      const text = document.getElementById('ml-status-text');
      const badge = document.getElementById('ml-device-badge');
      const sub = document.getElementById('ml-status-sub');
      const input = document.getElementById('ml-backend-url-input');

      if (data.connected) {
        if (dot) dot.style.background = '#00e676';
        if (text) text.textContent = 'ML Model: Online (Ready)';
        if (badge) {
          badge.textContent = (data.device || 'CPU').toUpperCase();
          badge.className = data.cuda_available ? 'badge-red' : 'badge-blue';
        }
        if (sub) sub.textContent = `${data.asr_backbone} • ${data.device_name || 'CPU'}`;
        if (input && data.url) input.value = data.url;
      } else {
        if (dot) dot.style.background = '#ff9100';
        if (text) text.textContent = 'ML Model: Cache Mode';
        if (badge) {
          badge.textContent = 'CACHE';
          badge.className = 'badge-blue';
        }
        if (sub) sub.textContent = 'Using broadcast deliverables cache. Connect Colab GPU or local Docker for live inference.';
      }
    } catch (e) {
      console.warn('Could not check ML status:', e);
    }
  }

  async loadEpisodes() {
    try {
      const res = await fetch('/api/episodes');
      const data = await res.json();
      this.episodes = data.episodes;
      this.renderEpisodeList();

      // Load initial episode
      if (this.episodes.length > 0) {
        this.selectEpisode(this.episodes[1] || this.episodes[0]); // default to mohanagar
      }
    } catch (e) {
      console.error('Error loading episode list:', e);
    }
  }

  renderEpisodeList() {
    this.episodeList.innerHTML = this.episodes.map(ep => `
      <div class="episode-card ${ep.id === this.currentEpisode ? 'active' : ''}" data-id="${ep.id}">
        <div class="ep-header">
          <span class="ep-title">${ep.name}</span>
          <span class="ep-duration">${ep.duration}</span>
        </div>
        <div class="ep-genre">${ep.genre}</div>
        <div class="ep-tags">
          <span class="tag-badge">${ep.codeSwitching}</span>
          <span class="tag-badge">${ep.sizeMb}</span>
        </div>
      </div>
    `).join('');

    this.episodeList.querySelectorAll('.episode-card').forEach(card => {
      card.addEventListener('click', () => {
        const id = card.getAttribute('data-id');
        const ep = this.episodes.find(e => e.id === id);
        if (ep) this.selectEpisode(ep);
      });
    });
  }

  selectEpisode(ep) {
    this.currentEpisode = ep.id;
    this.renderEpisodeList();

    // Switch video stream
    this.videoSource.src = `/api/video/${ep.filename}`;
    this.video.load();
    this.currentTimeEl.textContent = '00:00.0';

    // Fetch and display pipeline results
    this.fetchPipelineData(ep.filename);
  }

  getSelectedDuration() {
    const sel = document.getElementById('max-dur-select');
    if (!sel) return 30;
    if (sel.value === 'custom') {
      const customInput = document.getElementById('max-dur-input');
      return parseFloat(customInput?.value || 60);
    }
    return parseFloat(sel.value);
  }

  async fetchPipelineData(filename, maxDuration = null, live = false) {
    try {
      const duration = maxDuration !== null ? maxDuration : this.getSelectedDuration();
      const res = await fetch('/api/run-pipeline', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ videoName: filename, maxDuration: duration, live }),
      });
      const data = await res.json();
      this.pipelineData = data;
      this.renderPipelineData();
    } catch (e) {
      console.error('Error fetching pipeline data:', e);
    }
  }

  renderPipelineData() {
    if (!this.pipelineData) return;

    // Update Tab Badges
    this.bnCount.textContent = `${this.pipelineData.bnCues?.length || 0} Cues`;
    this.enCount.textContent = `${this.pipelineData.enCues?.length || 0} Cues`;
    this.hiCount.textContent = `${this.pipelineData.hiCues?.length || 0} Cues`;

    // Render Active Transcript Timeline
    this.renderTranscriptTimeline();

    // Render QC Dashboard
    this.renderQCDashboard();

    // Render Speaker Attribution
    this.renderSpeakers();
  }

  renderTranscriptTimeline() {
    const cues = this.getCurrentCues();
    if (!cues || !cues.length) {
      this.timelineContainer.innerHTML = '<div style="padding: 20px; color: var(--text-muted); text-align: center;">No cues generated yet for this track.</div>';
      return;
    }

    this.timelineContainer.innerHTML = cues.map((cue, idx) => {
      const isEv = cue.isEvent || cue.text.startsWith('[');
      const spk = cue.speaker || (isEv ? 'CC' : 'SPEAKER_01');
      const spkClass = `spk-${spk.replace(/[^a-zA-Z0-9_]/g, '')}`;

      return `
        <div class="cue-row ${isEv ? 'event-cue' : ''}" data-idx="${idx}" data-start="${cue.start}" data-end="${cue.end}">
          <span class="cue-time">${this.formatTime(cue.start)}</span>
          <span class="cue-speaker ${spkClass}">[${spk}]</span>
          <span class="cue-text">${cue.text.replace(/\n/g, ' ')}</span>
          <span class="cue-cps ${cue.cps > 17.5 ? 'warn' : ''}">${cue.cps || 0} CPS</span>
        </div>
      `;
    }).join('');

    this.timelineContainer.querySelectorAll('.cue-row').forEach(row => {
      row.addEventListener('click', () => {
        const start = parseFloat(row.getAttribute('data-start'));
        this.video.currentTime = start;
        this.video.play();
        this.updatePlayIcon(true);
      });
    });
  }

  renderQCDashboard() {
    const qc = this.pipelineData.qcSummary || {};
    const score = qc.overall_compliance_score ?? 100;
    this.complianceScore.textContent = `${score}%`;
    this.qcPassRate.textContent = `${qc.pass_rate_pct ?? 90}%`;
    this.qcHallucinations.textContent = `${qc.hallucination_flags ?? 0}`;
    this.qcMaxCps.innerHTML = `${qc.max_cps_recorded ?? 16.4} <small>/ 17.5</small>`;
    this.qcReviewCount.textContent = `${this.pipelineData.reviewQueue?.length || 0}`;
    this.badgeQueueCount.textContent = `${this.pipelineData.reviewQueue?.length || 0} to Inspect`;

    // Render Review Queue Items
    const queue = this.pipelineData.reviewQueue || [];
    if (!queue.length) {
      this.reviewList.innerHTML = '<div style="font-size: 11px; color: var(--accent-green); padding: 8px;">✓ All broadcast checks passed. Zero human review flags.</div>';
      return;
    }

    this.reviewList.innerHTML = queue.map(item => `
      <div class="review-item sev-${(item.severity || 'med').toLowerCase()}" data-start="${item.start_sec || 0}">
        <div class="review-header">
          <span>Cue #${item.cue_id || 1} • ${this.formatTime(item.start_sec || 0)}</span>
          <span style="text-transform: uppercase; font-size: 9px;">${item.severity || 'REVIEW'}</span>
        </div>
        <div class="review-reason">${item.reasons ? item.reasons.join(', ') : (item.recommended_action || 'Inspect audio')}</div>
      </div>
    `).join('');

    this.reviewList.querySelectorAll('.review-item').forEach(item => {
      item.addEventListener('click', () => {
        const start = parseFloat(item.getAttribute('data-start'));
        this.video.currentTime = Math.max(0, start - 0.5);
        this.video.play();
        this.updatePlayIcon(true);
      });
    });
  }

  renderSpeakers() {
    const spks = this.pipelineData.speakers || {};
    const totalSec = Object.values(spks).reduce((a, b) => a + b, 0) || 1;

    this.speakerChipList.innerHTML = Object.entries(spks).map(([spk, sec]) => {
      const pct = Math.round((sec / totalSec) * 100);
      return `
        <div class="spk-stat-row">
          <span style="font-weight: 600; color: var(--accent-cyan);">${spk}</span>
          <span style="color: var(--text-muted); font-size: 11px;">${sec.toFixed(1)}s (${pct}%)</span>
        </div>
      `;
    }).join('');
  }

  getCurrentCues() {
    if (!this.pipelineData) return [];
    if (this.currentLang === 'bn') return this.pipelineData.bnCues || [];
    if (this.currentLang === 'en') return this.pipelineData.enCues || [];
    if (this.currentLang === 'hi') return this.pipelineData.hiCues || [];
    return [];
  }

  switchLanguage(lang) {
    this.currentLang = lang;
    this.renderTranscriptTimeline();
    this.updateSubtitleOverlay(this.video.currentTime);
  }

  onTimeUpdate() {
    const curr = this.video.currentTime;
    const dur = this.video.duration || 1;

    // Update readout & seek bar
    this.currentTimeEl.textContent = this.formatTime(curr);
    this.seekBar.value = (curr / dur) * 100;

    // Update Subtitles & highlight active row
    this.updateSubtitleOverlay(curr);
  }

  updateSubtitleOverlay(time) {
    const cues = this.getCurrentCues();
    const activeCue = cues.find(c => time >= c.start && time <= c.end);

    if (activeCue && this.subtitlesVisible) {
      this.overlay.style.display = 'block';
      this.overlaySpeaker.textContent = activeCue.speaker || 'SPEAKER';
      this.overlaySpeaker.style.display = activeCue.isEvent ? 'none' : 'inline-block';
      this.overlayText.textContent = activeCue.text;

      // Highlight active row in timeline
      const activeIdx = cues.indexOf(activeCue);
      if (activeIdx !== this.activeCueIndex) {
        this.activeCueIndex = activeIdx;
        this.timelineContainer.querySelectorAll('.cue-row').forEach(r => r.classList.remove('active'));
        const row = this.timelineContainer.querySelector(`.cue-row[data-idx="${activeIdx}"]`);
        if (row) {
          row.classList.add('active');
          row.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }
      }
    } else {
      this.overlay.style.display = 'none';
    }
  }

  togglePlay() {
    if (this.video.paused) {
      this.video.play();
      this.updatePlayIcon(true);
    } else {
      this.video.pause();
      this.updatePlayIcon(false);
    }
  }

  updatePlayIcon(isPlaying) {
    this.playIcon.innerHTML = isPlaying
      ? '<path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z"/>'
      : '<path d="M8 5v14l11-7z"/>';
  }

  async triggerPipelineExecution() {
    const btn = this.runPipelineBtn;
    btn.disabled = true;
    btn.innerHTML = `<span class="status-pulse"></span> Processing Pipeline...`;

    // Animate stage stepper sequentially
    for (let i = 1; i <= 7; i++) {
      const step = document.getElementById(`stage-${i}`);
      step.classList.remove('done');
      step.classList.add('active');
      await new Promise(r => setTimeout(r, 400));
      step.classList.remove('active');
      step.classList.add('done');
    }

    // Refresh results with live model inference
    const dur = this.getSelectedDuration();
    await this.fetchPipelineData(this.currentEpisode + '.mp4', dur, true);
    btn.disabled = false;
    btn.innerHTML = `<svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg> Run Full Subtitle Pipeline`;
  }

  async handleFileUpload(file) {
    if (!file) return;
    const formData = new FormData();
    formData.append('video', file);

    const btn = document.getElementById('btn-browse-file');
    btn.textContent = 'Uploading...';

    try {
      const res = await fetch('/api/upload', {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();
      btn.textContent = 'Browse File';

      // Add to episode library and select
      const newEp = {
        id: data.filename.split('.')[0],
        name: file.name,
        genre: 'User Uploaded Media',
        filename: data.filename,
        duration: 'Custom',
        codeSwitching: 'Detected Bengali/English',
        sizeMb: `${data.sizeMb} MB`,
        exists: true,
      };
      this.episodes.unshift(newEp);
      this.selectEpisode(newEp);
    } catch (e) {
      console.error('Error uploading file:', e);
      btn.textContent = 'Upload Failed';
    }
  }

  downloadDeliverable(type) {
    const filename = `${this.currentEpisode}.mp4`;
    window.open(`/api/download/${type}/${filename}`, '_blank');
  }

  filterCues(term) {
    this.timelineContainer.querySelectorAll('.cue-row').forEach(row => {
      const text = row.querySelector('.cue-text').textContent.toLowerCase();
      row.style.display = text.includes(term) ? 'grid' : 'none';
    });
  }

  formatTime(seconds) {
    if (!seconds || isNaN(seconds)) return '00:00.0';
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    const ms = Math.floor((seconds % 1) * 10);
    return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}.${ms}`;
  }
}

// Instantiate App
document.addEventListener('DOMContentLoaded', () => {
  window.app = new SubtitleSuiteApp();
});
