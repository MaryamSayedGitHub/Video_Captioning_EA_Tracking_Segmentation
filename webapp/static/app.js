(() => {
const form = document.getElementById('form');
const kind = form.dataset.kind;
const $ = id => document.getElementById(id);
const PALETTE = ['#e74c3c','#3498db','#2ecc71','#f1c40f','#9b59b6','#1abc9c','#e67e22','#8e44ad','#34495e','#27ae60','#c0392b','#16a085'];
const col = i => PALETTE[i % PALETTE.length];
const charts = [];
const esc = s => String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const fmt = n => (n === undefined || n === null) ? '–' : n;

form.querySelector('input[type=file]').addEventListener('change', e => {
  const f = e.target.files[0];
  $('dropText').textContent = f ? f.name : 'Choose a video';
  $('drop').classList.toggle('has', !!f);
});

form.addEventListener('submit', async e => {
  e.preventDefault();
  const btn = $('go'); btn.disabled = true;
  setStatus(0, 'uploading…');
  try {
    const r = await fetch('/api/start/' + kind, {method: 'POST', body: new FormData(form)});
    const j = await r.json();
    if (!r.ok) throw new Error(j.error || 'Upload failed');
    poll(j.job_id);
  } catch (err) { fail(err.message); }
});

function setStatus(p, msg, err) {
  $('status').hidden = false;
  $('status').classList.toggle('err', !!err);
  $('barFill').style.width = Math.round(p * 100) + '%';
  $('statusText').textContent = msg;
}
function fail(msg) { setStatus(0, msg, true); $('go').disabled = false; }

async function poll(id) {
  const j = await (await fetch('/api/job/' + id)).json();
  if (j.status === 'error') return fail(j.message);
  if (j.status === 'done') {
    setStatus(1, 'Done'); $('go').disabled = false;
    return render(id, j.result);
  }
  setStatus(j.progress || 0, j.message || j.status);
  setTimeout(() => poll(id), 1000);
}

/* ---------- rendering ---------- */
function render(id, r) {
  charts.splice(0).forEach(c => c.destroy());
  const R = $('results'); R.innerHTML = '';
  const url = n => `/files/${id}/${n}`;
  if (kind === 'captioning') return renderCaptioning(R, r, url);
  R.insertAdjacentHTML('beforeend', `<div><h2>Annotated video</h2><video controls src="${url(r.video_file)}"></video></div>`);
  const v = r.video || {};
  if (r.totals) R.insertAdjacentHTML('beforeend', facts([
    ['Unique objects', r.totals.unique_objects], ['Most at once', r.totals.max_simultaneous],
    ['Video length', v.duration_s + ' s'], ['Processing speed', r.runtime.fps_processing + ' fps']]));
  else R.insertAdjacentHTML('beforeend', facts([
    ['Classes found', Object.keys(r.segmentation.classes).length], ['Video length', v.duration_s + ' s'],
    ['Processing speed', r.runtime.fps_processing + ' fps'], ['Model', r.weights]]));
  if (r.totals) renderTracking(R, r, url);
  if (r.segmentation) renderSegmentation(R, r);
  if (r.keyframe_captions) R.insertAdjacentHTML('beforeend', captionList(r.keyframe_captions, 'Captions in the video'));
}

function facts(items) {
  return '<dl class="facts">' + items.map(([k, v]) => `<div><dt>${esc(k)}</dt><dd>${esc(fmt(v))}</dd></div>`).join('') + '</dl>';
}
function captionList(list, title) {
  return `<div class="box"><h3>${title}</h3><ol class="caps">` +
    list.map(c => `<li><time>${c.t}s</time>${esc(c.caption)}</li>`).join('') + '</ol></div>';
}
function addChart(canvas, cfg) {
  if (typeof Chart === 'undefined') { canvas.replaceWith(Object.assign(document.createElement('p'), {className: 'hint', textContent: 'Chart library could not load (check your internet connection).'})); return null; }
  const c = new Chart(canvas, cfg); charts.push(c); return c;
}
function chartBox(title, hint) {
  const box = document.createElement('div'); box.className = 'box';
  box.innerHTML = `<h3>${title}</h3><canvas></canvas>` + (hint ? `<p class="hint">${hint}</p>` : '');
  return box;
}

function renderTracking(R, r, url) {
  const g = document.createElement('div'); g.className = 'grid2'; R.appendChild(g);

  const b1 = chartBox('Objects by class', 'Each tracked ID is counted once.'); g.appendChild(b1);
  const cls = Object.keys(r.totals.by_class);
  addChart(b1.querySelector('canvas'), {type: 'bar',
    data: {labels: cls, datasets: [{data: cls.map(c => r.totals.by_class[c]), backgroundColor: cls.map((_, i) => col(i))}]},
    options: {plugins: {legend: {display: false}}, scales: {y: {beginAtZero: true, ticks: {precision: 0}}}}});

  const b2 = chartBox('Objects on screen over time', 'Average number of tracked objects per second.'); g.appendChild(b2);
  addChart(b2.querySelector('canvas'), {type: 'line',
    data: {labels: r.timeline_per_second.map((_, i) => i + 's'), datasets: [{data: r.timeline_per_second, borderColor: '#3498db', backgroundColor: '#3498db33', fill: true, tension: .25, pointRadius: 0}]},
    options: {plugins: {legend: {display: false}}, scales: {y: {beginAtZero: true, ticks: {precision: 0}}}}});

  const lc = r.line_crossing;
  const rows = cls.map(c => `<tr><td>${esc(c)}</td><td class="num">${lc.down[c] || 0}</td><td class="num">${lc.up[c] || 0}</td></tr>`).join('');
  g.insertAdjacentHTML('beforeend', `<div class="box"><h3>Crossing the centre line</h3>
    <table><tr><th>Class</th><th class="num">Downward</th><th class="num">Upward</th></tr>${rows}</table>
    <p class="hint">An object counts when its centre moves from one side of the horizontal middle line to the other.</p></div>`);

  const q = r.quality_proxies;
  g.insertAdjacentHTML('beforeend', `<div class="box"><h3>Tracker health (no ground truth needed)</h3>
    <table><tr><td>ID churn</td><td class="num">${q.id_churn}</td></tr>
    <tr><td>Tracks shorter than 1 s</td><td class="num">${q.short_track_pct}%</td></tr>
    <tr><td>Average track length</td><td class="num">${q.avg_track_s} s</td></tr></table>
    <p class="hint">ID churn is unique IDs divided by the most objects seen at once. Near 1 is healthy; high values mean the tracker keeps re-issuing IDs. These are hints only. Real accuracy (IDF1, MOTA, HOTA) needs ground truth.</p></div>`);

  const v = r.video;
  R.insertAdjacentHTML('beforeend', `<div class="grid2">
    <div class="box"><h3>Where objects spent time</h3><img src="${url(r.heatmap_file)}" alt="Heatmap of object positions"></div>
    <div class="box"><h3>Trajectories</h3><div class="traj" id="traj" style="aspect-ratio:${v.width}/${v.height}"><canvas></canvas></div></div></div>`);
  drawTrajectories($('traj').querySelector('canvas'), r);

  const t = r.tracks.map(t => `<tr><td><i class="chip" style="background:${col(t.id)}"></i>#${t.id}</td><td>${esc(t.class)}</td>
    <td class="num">${t.first_s}</td><td class="num">${t.dwell_s}</td><td class="num">${t.path_px}</td><td class="num">${t.avg_conf}</td></tr>`).join('');
  R.insertAdjacentHTML('beforeend', `<div class="box"><h3>Every tracked object</h3><div class="scroll"><table>
    <tr><th>ID</th><th>Class</th><th class="num">First seen (s)</th><th class="num">Time on screen (s)</th><th class="num">Distance (px)</th><th class="num">Avg. confidence</th></tr>${t}</table></div>
    <p class="hint"><a class="dl" href="${url(r.mot_file)}" download>Download tracks (MOT format)</a>: use it with motmetrics or TrackEval to compute IDF1, MOTA and HOTA against your own ground truth.</p></div>`);
}

function drawTrajectories(canvas, r) {
  const v = r.video, dpr = window.devicePixelRatio || 1;
  const w = canvas.parentElement.clientWidth, h = w * v.height / v.width;
  canvas.width = w * dpr; canvas.height = h * dpr;
  const ctx = canvas.getContext('2d'); ctx.scale(dpr, dpr);
  ctx.fillStyle = getComputedStyle(document.body).getPropertyValue('--card'); ctx.fillRect(0, 0, w, h);
  ctx.strokeStyle = getComputedStyle(document.body).getPropertyValue('--line'); ctx.strokeRect(0, 0, w, h);
  const ly = h * r.line_crossing.line_y_ratio;
  ctx.setLineDash([6, 6]); ctx.beginPath(); ctx.moveTo(0, ly); ctx.lineTo(w, ly); ctx.stroke(); ctx.setLineDash([]);
  Object.entries(r.trajectories).forEach(([id, t]) => {
    ctx.strokeStyle = col(+id); ctx.lineWidth = 2; ctx.beginPath();
    t.pts.forEach(([x, y], i) => { const X = x / v.width * w, Y = y / v.height * h; i ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y); });
    ctx.stroke();
    const [x, y] = t.pts[t.pts.length - 1]; ctx.fillStyle = col(+id);
    ctx.beginPath(); ctx.arc(x / v.width * w, y / v.height * h, 3.5, 0, 7); ctx.fill();
  });
}

function renderSegmentation(R, r) {
  const s = r.segmentation, cls = Object.keys(s.classes);
  const rows = cls.map((c, i) => { const k = s.classes[c];
    return `<tr><td><i class="chip" style="background:${col(i)}"></i>${esc(c)}</td><td class="num">${k.instances}</td>
    <td class="num">${k.mean_area_pct}%</td><td class="num">${k.max_area_pct}%</td><td class="num">${k.avg_conf}</td></tr>`; }).join('');
  R.insertAdjacentHTML('beforeend', `<div class="box"><h3>Segmented classes</h3><table>
    <tr><th>Class</th><th class="num">Instances</th><th class="num">Mean area</th><th class="num">Max area</th><th class="num">Avg. confidence</th></tr>${rows}</table>
    <p class="hint">Area is the share of the frame covered by that class's masks.</p></div>`);
  const b = chartBox('Frame area covered by each class over time');
  R.appendChild(b);
  const n = Math.max(...Object.values(s.area_timeline).map(a => a.length));
  addChart(b.querySelector('canvas'), {type: 'line',
    data: {labels: Array.from({length: n}, (_, i) => (i * s.timeline_step_s).toFixed(1) + 's'),
      datasets: cls.map((c, i) => ({label: c, data: s.area_timeline[c], borderColor: col(i), pointRadius: 0, tension: .2}))},
    options: {scales: {y: {beginAtZero: true, title: {display: true, text: '% of frame'}}}}});
}

function renderCaptioning(R, r, url) {
  const langs = Object.keys(r.videos);
  if (!langs.length) R.insertAdjacentHTML('beforeend', '<p class="empty">No video was produced.</p>');
  R.insertAdjacentHTML('beforeend', '<div class="vids">' + langs.map(l =>
    `<div><div class="vcap">${l === 'ar' ? 'Arabic' : 'English'}</div><video controls src="${url(r.videos[l])}"></video>
     <p class="hint">${esc(r.narratives[l])}</p></div>`).join('') + '</div>');
  (r.notes || []).forEach(n => R.insertAdjacentHTML('beforeend', `<p class="note">${esc(n)}</p>`));
  const m = r.metrics;
  R.insertAdjacentHTML('beforeend', facts([
    ['Frames sampled', m.frames_sampled], ['Keyframes kept', m.keyframes],
    ['Compression', m.compression_ratio + '×'], ['CLIPScore', m.clip_score_mean]]));
  R.insertAdjacentHTML('beforeend', `<div class="box"><h3>How to read these numbers</h3><table>
    <tr><td>Compression</td><td>Frames sampled ÷ keyframes. How much redundant video CLIP removed before captioning.</td></tr>
    <tr><td>CLIPScore</td><td>2.5 × cosine(image, caption) in CLIP space, averaged over keyframes. No reference captions needed.</td></tr>
    <tr><td>Redundancy</td><td class="num">${m.redundancy}</td></tr>
    <tr><td>Unique captions</td><td class="num">${m.unique_caption_pct}%</td></tr>
    <tr><td>Stage times</td><td>${Object.entries(m.timings).map(([k, v]) => k.replace(/_s$/, '').replace(/_/g, ' ') + ' ' + v + ' s').join(', ')}</td></tr></table></div>`);
  R.insertAdjacentHTML('beforeend', captionList(r.keyframe_captions, 'BLIP caption for each keyframe'));
}
})();
