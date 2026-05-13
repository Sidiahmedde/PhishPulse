const form = document.getElementById('composeForm');
const inbox = document.getElementById('inbox');
const sendBtn = document.getElementById('sendBtn');
const btnText = document.getElementById('btnText');
const btnSpinner = document.getElementById('btnSpinner');

let totalAnalyzed = 0;
let totalThreats = 0;
let totalSafe = 0;

const STORAGE_KEY = 'phishguard_history_v1';

ensureDashboard();
loadHistory();

form.addEventListener('submit', async (e) => {
  e.preventDefault();

  const sender = document.getElementById('sender').value.trim();
  const receiver = document.getElementById('receiver').value.trim();
  const subject = document.getElementById('subject').value.trim();
  const body = document.getElementById('body').value.trim();

  sendBtn.disabled = true;
  btnText.textContent = 'Analyzing...';
  btnSpinner.classList.remove('hidden');

  showAnalysisOverlay();

  try {
    await simulateAnalysisSteps();

    const res = await fetch('/send', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sender, receiver, subject, body })
    });

    if (!res.ok) throw new Error(`Server error: ${res.status}`);

    const result = await res.json();
    const record = {
      email: { sender, receiver, subject, body },
      result,
      timestamp: new Date().toISOString()
    };

    addEmailCard(record.email, record.result, true);
    saveRecord(record);
    form.reset();
  } catch (err) {
    alert('Something went wrong: ' + err.message);
  } finally {
    hideAnalysisOverlay();
    sendBtn.disabled = false;
    btnText.textContent = 'Analyze Email';
    btnSpinner.classList.add('hidden');
  }
});

function ensureDashboard() {
  const header = document.querySelector('header');

  if (!document.querySelector('.shield-status')) {
    const shield = document.createElement('div');
    shield.className = 'shield-status';
    shield.innerHTML = `<span class="shield-dot"></span> Shield Active`;
    header.appendChild(shield);
  }

  if (!document.querySelector('.dashboard-stats')) {
    const stats = document.createElement('div');
    stats.className = 'dashboard-stats';
    stats.innerHTML = `
      <div class="stat-card">
        <span id="totalAnalyzed">0</span>
        <small>Analyzed</small>
      </div>
      <div class="stat-card threat">
        <span id="totalThreats">0</span>
        <small>Threats</small>
      </div>
      <div class="stat-card safe">
        <span id="totalSafe">0</span>
        <small>Safe</small>
      </div>
    `;
    header.appendChild(stats);
  }

  if (!document.querySelector('.sample-actions')) {
    const sampleBox = document.createElement('div');
    sampleBox.className = 'sample-actions';
    sampleBox.innerHTML = `
      <button type="button" data-sample="paypal">PayPal Scam</button>
      <button type="button" data-sample="bank">Bank Alert</button>
      <button type="button" data-sample="safe">Safe Email</button>
      <button type="button" data-sample="mfa">MFA Reset</button>
    `;

    const composePanel = document.querySelector('.compose-panel');
    const h2 = composePanel.querySelector('h2');
    h2.insertAdjacentElement('afterend', sampleBox);

    sampleBox.addEventListener('click', (e) => {
      if (e.target.tagName !== 'BUTTON') return;
      fillSample(e.target.dataset.sample);
    });
  }

  if (!document.querySelector('.clear-history-btn')) {
    const clearBtn = document.createElement('button');
    clearBtn.type = 'button';
    clearBtn.className = 'clear-history-btn';
    clearBtn.textContent = 'Clear History';

    const inboxPanel = document.querySelector('.inbox-panel');
    const inboxTitle = inboxPanel.querySelector('h2');
    inboxTitle.appendChild(clearBtn);
    const exportBtn = document.createElement('button');
    exportBtn.type = 'button';
    exportBtn.className = 'export-btn';
    exportBtn.textContent = 'Export Report';
    inboxTitle.appendChild(exportBtn);

exportBtn.addEventListener('click', exportReport);

    clearBtn.addEventListener('click', () => {
      localStorage.removeItem(STORAGE_KEY);
      totalAnalyzed = 0;
      totalThreats = 0;
      totalSafe = 0;
      inbox.innerHTML = `<p class="empty-state">No emails yet. Send one to see the prediction.</p>`;
      updateStats();
    });
  }

  if (!document.querySelector('.analysis-overlay')) {
    const overlay = document.createElement('div');
    overlay.className = 'analysis-overlay hidden';
    overlay.innerHTML = `
      <div class="analysis-box">
        <div class="analysis-ring"></div>
        <h3>Analyzing Email Threat Signals</h3>
        <p id="analysisStep">Preparing detection pipeline...</p>
      </div>
    `;
    document.body.appendChild(overlay);
  }
}

function fillSample(type) {
  const samples = {
    paypal: {
      sender: 'security@paypal-alert.com',
      receiver: 'you@gmail.com',
      subject: 'URGENT: Verify your account immediately',
      body: 'Your account has been compromised. Click the link below immediately to verify your login credentials:\n\nhttp://paypal-security-login-verification.com'
    },
    bank: {
      sender: 'support@secure-bank-alerts.com',
      receiver: 'you@gmail.com',
      subject: 'Account locked: action required now',
      body: 'We detected suspicious activity on your account. Confirm your identity immediately to avoid account closure:\n\nhttp://secure-bank-login-check.com'
    },
    safe: {
      sender: 'alex.smith@gmail.com',
      receiver: 'you@gmail.com',
      subject: 'Project meeting notes',
      body: 'Hi, I attached the meeting notes from today. Let me know if you want to review the next steps tomorrow.'
    },
    mfa: {
      sender: 'it-support@company-security-reset.com',
      receiver: 'you@gmail.com',
      subject: 'MFA reset required within 15 minutes',
      body: 'Your multi-factor authentication token expired. Reset your password and verify your access now:\n\nhttp://company-mfa-reset-login.com'
    }
  };

  const sample = samples[type];
  document.getElementById('sender').value = sample.sender;
  document.getElementById('receiver').value = sample.receiver;
  document.getElementById('subject').value = sample.subject;
  document.getElementById('body').value = sample.body;
}

function addEmailCard(email, result, animate = false) {
  const empty = inbox.querySelector('.empty-state');
  if (empty) empty.remove();

  const isPhishing = result.prediction === 'Phishing';
  const badgeClass = isPhishing ? 'phishing' : 'legitimate';
  const scorePercent = Number(result.final_score * 100);
  const severity = getSeverity(result.final_score);
  const moduleScores = result.module_scores || {};
  const indicators = detectIndicators(email, moduleScores);
  const timeLabel = formatTimestamp(new Date());

  totalAnalyzed += 1;
  if (isPhishing) totalThreats += 1;
  else totalSafe += 1;
  updateStats();

  const barsHtml = Object.entries(moduleScores).map(([key, val]) => {
    const pct = Number(val * 100).toFixed(1);
    const riskClass = val >= 0.70 ? 'high' : val >= 0.40 ? 'medium' : 'low';

    return `
      <div class="score-row">
        <div class="score-label">
          <span>${formatLabel(key)}</span>
          <strong>${pct}%</strong>
        </div>
        <div class="score-track">
          <div class="score-fill ${riskClass}" data-width="${pct}" style="width: ${animate ? '0%' : pct + '%'}"></div>
        </div>
      </div>
    `;
  }).join('');

  const explanation = buildExplanation(moduleScores, result.final_score, isPhishing, indicators);

  const indicatorHtml = indicators.length
    ? indicators.map(item => `<span class="indicator-pill">${escHtml(item)}</span>`).join('')
    : `<span class="indicator-pill neutral">No obvious phishing indicators extracted</span>`;

  const card = document.createElement('div');
  card.className = `email-card ${badgeClass}`;
  card.innerHTML = `
    <div class="card-header">
      <div>
        <div class="card-subject">${escHtml(email.subject || '(No subject)')}</div>
        <div class="card-meta">From: ${escHtml(email.sender)} &rarr; To: ${escHtml(email.receiver)}</div>
<div class="card-time">Analyzed ${timeLabel}</div>
      </div>
      <span class="badge ${badgeClass}">${result.prediction}</span>
    </div>

    <div class="risk-summary">
      <div class="risk-meter ${severity.className}">
        <div class="risk-value" data-score="${scorePercent.toFixed(1)}">${animate ? '0.0%' : scorePercent.toFixed(1) + '%'}</div>
        <div class="risk-label">${severity.label}</div>
      </div>

      <div class="risk-details">
        <h3>Detection Breakdown</h3>
        ${barsHtml}
      </div>
    </div>

    <div class="indicator-section">
      <h3>Indicators Detected</h3>
      <div class="indicator-list">${indicatorHtml}</div>
    </div>

    <div class="explanation-box ${badgeClass}">
      <strong>Assessment:</strong> ${explanation}
    </div>

    <details class="message-details">
      <summary>View message body</summary>
      <div class="card-body-text ${email.body ? '' : 'empty'}">
        ${email.body ? escHtml(email.body) : 'No message body.'}
      </div>
    </details>
  `;

  inbox.prepend(card);

  if (animate) {
    requestAnimationFrame(() => animateCard(card));
  }
}

function animateCard(card) {
  card.querySelectorAll('.score-fill').forEach(fill => {
    const width = fill.dataset.width;
    setTimeout(() => {
      fill.style.width = `${width}%`;
    }, 120);
  });

  const riskValue = card.querySelector('.risk-value');
  const target = parseFloat(riskValue.dataset.score);
  animateNumber(riskValue, target, 700);
}

function animateNumber(element, target, duration) {
  const startTime = performance.now();

  function update(now) {
    const progress = Math.min((now - startTime) / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3);
    const value = target * eased;
    element.textContent = `${value.toFixed(1)}%`;

    if (progress < 1) {
      requestAnimationFrame(update);
    }
  }

  requestAnimationFrame(update);
}

function updateStats() {
  document.getElementById('totalAnalyzed').textContent = totalAnalyzed;
  document.getElementById('totalThreats').textContent = totalThreats;
  document.getElementById('totalSafe').textContent = totalSafe;
}

function getSeverity(score) {
  if (score >= 0.75) return { label: 'High Risk', className: 'high' };
  if (score >= 0.45) return { label: 'Suspicious', className: 'medium' };
  return { label: 'Low Risk', className: 'low' };
}

function buildExplanation(scores, finalScore, isPhishing, indicators) {
  const sorted = Object.entries(scores || {}).sort((a, b) => b[1] - a[1]);
  const topSignals = sorted
    .filter(([, value]) => value >= 0.40)
    .slice(0, 2)
    .map(([key]) => formatLabel(key).toLowerCase());

  if (isPhishing && indicators.length) {
    return `The message was classified as phishing because the model detected ${indicators.slice(0, 3).join(', ')}. The strongest model signals came from ${topSignals.join(' and ') || 'combined module behavior'}. Final fusion score: ${(finalScore * 100).toFixed(1)}%.`;
  }

  if (isPhishing) {
    return `The fusion model classified this message as phishing based on combined risk signals across the sender, subject, body, and URL modules. Final fusion score: ${(finalScore * 100).toFixed(1)}%.`;
  }

  return `The message appears lower risk because the combined module scores did not strongly match phishing behavior. Final fusion score: ${(finalScore * 100).toFixed(1)}%.`;
}

function detectIndicators(email, scores) {
  const indicators = [];
  const text = `${email.sender} ${email.subject} ${email.body}`.toLowerCase();

  const urgencyWords = ['urgent', 'immediately', 'now', 'within 15 minutes', 'action required', 'locked', 'suspended', 'expired'];
  const credentialWords = ['verify', 'login', 'password', 'credentials', 'reset', 'confirm your identity', 'account closure'];
  const moneyWords = ['invoice', 'payment', 'refund', 'wire', 'bank', 'billing'];

  if (urgencyWords.some(word => text.includes(word))) indicators.push('Urgency language');
  if (credentialWords.some(word => text.includes(word))) indicators.push('Credential request');
  if (moneyWords.some(word => text.includes(word))) indicators.push('Financial/account theme');
  if (/https?:\/\/[^\s]+/i.test(email.body || '')) indicators.push('Embedded URL');
  if ((email.sender || '').includes('-') || (email.sender || '').includes('alert') || (email.sender || '').includes('security')) {
    indicators.push('Suspicious sender pattern');
  }
  if ((scores.subject || 0) >= 0.70) indicators.push('High-risk subject signal');
  if ((scores.sender || 0) >= 0.70) indicators.push('High-risk sender signal');
  if ((scores.body || 0) >= 0.70) indicators.push('High-risk body signal');

  return [...new Set(indicators)];
}

function showAnalysisOverlay() {
  const overlay = document.querySelector('.analysis-overlay');
  overlay.classList.remove('hidden');
}

function hideAnalysisOverlay() {
  const overlay = document.querySelector('.analysis-overlay');
  overlay.classList.add('hidden');
}

async function simulateAnalysisSteps() {
  const steps = [
    'Extracting sender reputation features...',
    'Scanning subject-line urgency patterns...',
    'Analyzing body text for credential language...',
    'Checking URL-based phishing indicators...',
    'Running fusion model for final classification...'
  ];

  const stepEl = document.getElementById('analysisStep');

  for (const step of steps) {
    stepEl.textContent = step;
    await wait(220);
  }
}

function wait(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function saveRecord(record) {
  const history = getHistory();
  history.unshift(record);
  localStorage.setItem(STORAGE_KEY, JSON.stringify(history.slice(0, 20)));
}

function loadHistory() {
  const history = getHistory();

  if (!history.length) {
    updateStats();
    return;
  }

  history.reverse().forEach(record => {
    addEmailCard(record.email, record.result, false);
  });
}

function getHistory() {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY)) || [];
  } catch {
    return [];
  }
}

function formatLabel(key) {
  return key
    .replace(/_/g, ' ')
    .replace(/\b\w/g, char => char.toUpperCase());
}

function formatTimestamp(date) {
  return date.toLocaleString([], {
    month: 'short',
    day: 'numeric',
    hour: 'numeric',
    minute: '2-digit'
  });
}
function escHtml(str) {
  return String(str || '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function exportReport() {
  const history = getHistory();

  if (!history.length) {
    alert('No analysis history to export yet.');
    return;
  }

  const report = history.map((item, index) => {
    const email = item.email;
    const result = item.result;
    const scores = result.module_scores || {};

    return `
==============================
PhishPulse Analysis #${index + 1}
==============================
Timestamp: ${new Date(item.timestamp).toLocaleString()}
Prediction: ${result.prediction}
Final Score: ${(result.final_score * 100).toFixed(1)}%

From: ${email.sender}
To: ${email.receiver}
Subject: ${email.subject}

Module Scores:
${Object.entries(scores).map(([key, value]) => `- ${formatLabel(key)}: ${(value * 100).toFixed(1)}%`).join('\n')}

Message:
${email.body}
`;
  }).join('\n-----------------------------\n');

  const blob = new Blob([report], { type: 'text/plain' });
  const url = URL.createObjectURL(blob);

  const link = document.createElement('a');
  link.href = url;
  link.download = 'phishpulse-analysis-report.txt';
  link.click();

  URL.revokeObjectURL(url);
}
