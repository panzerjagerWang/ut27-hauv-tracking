'use strict';
const tabs = [...document.querySelectorAll('[role="tab"]')];
const panel = document.querySelector('#weight-panel');
let models = [];
function selectTab(tab, focus = false) {
  for (const button of tabs) {
    const active = button === tab;
    button.setAttribute('aria-selected', String(active));
    button.tabIndex = active ? 0 : -1;
  }
  if (focus) tab.focus();
  panel.setAttribute('aria-labelledby', tab.id);
  panel.replaceChildren();
  for (const model of models.filter(model => model.representation === tab.dataset.representation)) {
    const card = document.createElement('article'); card.className = 'weight-card';
    const seed = document.createElement('strong'); seed.className = 'seed'; seed.textContent = `Seed ${model.seed}`;
    const architecture = document.createElement('p'); architecture.className = 'architecture';
    architecture.textContent = `${model.input_dim} → ${model.hidden_widths.join(' → ')} → 4 · ${model.actor_parameters.toLocaleString('en-US')} parameters`;
    const checkpoint = document.createElement('p'); checkpoint.className = 'checkpoint';
    checkpoint.textContent = model.is_bc_initialization ? 'Selected checkpoint: BC initialization' : `Selected at ${model.selected_online_transitions.toLocaleString('en-US')} online transitions`;
    const link = document.createElement('a'); link.href = `downloads/${model.file}`; link.download = model.file.split('/').pop(); link.textContent = 'Download .pt ↓';
    card.append(seed, architecture, checkpoint, link); panel.append(card);
  }
}
fetch('downloads/weights/manifest.json').then(response => { if (!response.ok) throw Error('Manifest unavailable'); return response.json(); }).then(data => { models = data.models; selectTab(tabs[0]); }).catch(() => { panel.innerHTML = '<p>Individual checkpoint details could not be loaded. The complete release ZIP above contains all nine policies.</p>'; });
for (const tab of tabs) {
  tab.addEventListener('click', () => selectTab(tab));
  tab.addEventListener('keydown', event => {
    if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
    event.preventDefault();
    const index = event.key === 'Home' ? 0 : event.key === 'End' ? tabs.length - 1 : (tabs.indexOf(tab) + (event.key === 'ArrowRight' ? 1 : -1) + tabs.length) % tabs.length;
    selectTab(tabs[index], true);
  });
}
const copy = document.querySelector('#copy-command');
copy.addEventListener('click', async () => {
  try { await navigator.clipboard.writeText(document.querySelector('#run-command').textContent); copy.textContent = 'Copied'; setTimeout(() => { copy.textContent = 'Copy'; }, 1800); }
  catch { copy.textContent = 'Select text to copy'; }
});
