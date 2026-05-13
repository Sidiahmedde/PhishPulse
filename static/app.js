const form = document.getElementById('composeForm');
const inbox = document.getElementById('inbox');
const sendBtn = document.getElementById('sendBtn');
const btnText = document.getElementById('btnText');
const btnSpinner = document.getElementById('btnSpinner');

form.addEventListener('submit', async (e) => {
  e.preventDefault();

  const sender  = document.getElementById('sender').value.trim();
  const receiver = document.getElementById('receiver').value.trim();
  const subject = document.getElementById('subject').value.trim();
  const body    = document.getElementById('body').value.trim();

  // loading state
  sendBtn.disabled = true;
  btnText.textContent = 'Sending...';
  btnSpinner.classList.remove('hidden');

  try {
    const res = await fetch('/send', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sender, receiver, subject, body })
    });

    if (!res.ok) throw new Error(`Server error: ${res.status}`);
    const data = await res.json();

    addEmailCard({ sender, receiver, subject, body }, data);
    form.reset();
  } catch (err) {
    alert('Something went wrong: ' + err.message);
  } finally {
    sendBtn.disabled = false;
    btnText.textContent = 'Send';
    btnSpinner.classList.add('hidden');
  }
});

function addEmailCard(email, result) {
  // remove empty state
  const empty = inbox.querySelector('.empty-state');
  if (empty) empty.remove();

  const isPhishing = result.prediction === 'Phishing';
  const badgeClass = isPhishing ? 'phishing' : 'legitimate';
  const scorePercent = (result.final_score * 100).toFixed(1);

  // build module score chips
  const moduleScores = result.module_scores || {};
  const chipsHtml = Object.entries(moduleScores).map(([key, val]) =>
    `<div class="score-chip">${key}: <span>${(val * 100).toFixed(1)}%</span></div>`
  ).join('');

  const card = document.createElement('div');
  card.className = 'email-card';
  card.innerHTML = `
    <div class="card-header">
      <div>
        <div class="card-subject">${escHtml(email.subject)}</div>
        <div class="card-meta">From: ${escHtml(email.sender)} &rarr; To: ${escHtml(email.receiver)}</div>
      </div>
      <span class="badge ${badgeClass}">${result.prediction}</span>
    </div>
    <div class="card-body-text ${email.body ? '' : 'empty'}">
      ${email.body ? escHtml(email.body) : 'No message body.'}
    </div>
    <div class="card-scores">
      ${chipsHtml}
      <div class="final-score ${badgeClass}">Final: ${scorePercent}%</div>
    </div>
  `;

  inbox.prepend(card);
}

function escHtml(str) {
  return str.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}
