"use strict";

const $ = (id) => document.getElementById(id);
const HEADERS = { "X-CharlieCam": "1" };
const POLL_MS = 600;
const AVATAR_COLOURS = ["#6366f1", "#0ea5e9", "#10b981", "#f59e0b", "#ec4899", "#a855f7", "#14b8a6", "#f97316"];

let lastActivityId = 0;
let activity = [];
let filter = "all";
let peopleCount = -1;
let wasEnrolling = false;
let lastResultKey = "";
let ready = false;
let enrolling = false;
let stopped = false;
let failures = 0;
let confirmTimer = null;

// ------------------------------------------------------------------ helpers
async function api(path, options = {}) {
  const res = await fetch(path, { ...options, headers: { ...HEADERS, ...(options.headers || {}) } });
  let body = null;
  try { body = await res.json(); } catch { /* empty body */ }
  if (!res.ok) throw new Error((body && body.detail) || `Request failed (${res.status})`);
  return body;
}

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else if (v !== false && v != null) node.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat()) if (c != null) node.append(c);
  return node;
}

function svg(paths) {
  const ns = "http://www.w3.org/2000/svg";
  const s = document.createElementNS(ns, "svg");
  s.setAttribute("viewBox", "0 0 24 24");
  s.setAttribute("aria-hidden", "true");
  for (const d of paths) {
    const p = document.createElementNS(ns, "path");
    p.setAttribute("d", d);
    s.append(p);
  }
  return s;
}

const ICON_TRASH = ["M4 7h16", "M10 11v6", "M14 11v6", "M6 7l1 12a2 2 0 0 0 2 2h6a2 2 0 0 0 2-2l1-12", "M9 7V4h6v3"];
const ICON_USER = ["M12 12a4 4 0 1 0 0-8 4 4 0 0 0 0 8Z", "M4 20a8 8 0 0 1 16 0"];
const ICON_QUESTION = ["M9.5 9a2.5 2.5 0 1 1 3.5 2.3c-.7.3-1 1-1 1.7v.5", "M12 17h.01"];

function toast(text, ok = true) {
  const t = el("div", { class: "toast" + (ok ? "" : " error") }, text);
  $("toasts").append(t);
  setTimeout(() => { t.classList.add("out"); setTimeout(() => t.remove(), 300); }, 3500);
}

function colourFor(name) {
  let h = 0;
  for (const ch of name) h = (h * 31 + ch.codePointAt(0)) >>> 0;
  return AVATAR_COLOURS[h % AVATAR_COLOURS.length];
}

function initials(name) {
  const parts = name.trim().split(/\s+/);
  return ((parts[0] || "")[0] + (parts.length > 1 ? parts[parts.length - 1][0] : "")).toUpperCase();
}

function timeAgo(iso) {
  const d = new Date(iso);
  const secs = Math.round((Date.now() - d.getTime()) / 1000);
  if (secs < 10) return "just now";
  if (secs < 60) return `${secs}s ago`;
  if (secs < 3600) return `${Math.floor(secs / 60)} min ago`;
  const sameDay = d.toDateString() === new Date().toDateString();
  return sameDay ? d.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" })
                 : d.toLocaleString([], { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" });
}

// ------------------------------------------------------------------ video
function startStream() {
  const img = $("stream");
  img.onload = () => { img.hidden = false; $("placeholder").hidden = true; };
  img.onerror = () => {
    img.hidden = true;
    $("placeholder").hidden = false;
    if (!stopped) setTimeout(startStream, 2000);
  };
  img.src = "/api/stream?t=" + Date.now();
}

// ------------------------------------------------------------------ state
function renderStatus(s) {
  const box = $("status");
  box.classList.toggle("ok", s.status_ok);
  box.classList.toggle("error", !s.status_ok && /not|could|disconnect|ended/i.test(s.status));
  $("status-text").textContent = s.status;
  $("fps").textContent = s.status_ok && s.fps ? `${s.fps.toFixed(0)} fps` : "";
  if (!s.ready) $("placeholder-text").textContent = s.status;
}

function renderEnrollment(s) {
  const e = s.enrolling;
  enrolling = !!e;
  $("enroll-overlay").hidden = !e;
  if (e) {
    $("enroll-name").textContent = e.name;
    $("enroll-msg").textContent = e.message;
    $("enroll-bar").style.width = `${(100 * e.captured) / e.total}%`;
    $("enroll-count").textContent = `${e.captured} of ${e.total} photos`;
  }
  if (s.last_result) {
    const key = JSON.stringify(s.last_result);
    if (key !== lastResultKey && wasEnrolling) {
      toast(s.last_result.message, s.last_result.ok);
      if (s.last_result.ok) $("name").value = "";
      loadPeople();
    }
    lastResultKey = key;
  }
  wasEnrolling = enrolling;
  updateButtons();
}

function updateButtons() {
  const hasName = $("name").value.trim().length > 0;
  $("capture").disabled = !ready || enrolling || !hasName;
  $("upload").disabled = !ready || enrolling || !hasName;
  $("name").disabled = enrolling;
}

async function poll() {
  if (stopped) return;
  try {
    const s = await api(`/api/state?since=${lastActivityId}`);
    failures = 0;
    ready = s.ready;
    renderStatus(s);
    renderEnrollment(s);
    if (s.people_count !== peopleCount) { peopleCount = s.people_count; loadPeople(); scheduleAttendance(); }
    if (s.activity.length) {
      if (s.activity.some((a) => a.known)) scheduleAttendance();
      addActivity(s.activity, lastActivityId > 0);
    }
    syncSettings(s.settings);
  } catch {
    failures += 1;
    if (failures >= 3) {
      $("status").className = "status error";
      $("status-text").textContent = "Can't reach CharlieCam. Is it still running?";
      $("fps").textContent = "";
    }
  }
  setTimeout(poll, POLL_MS);
}

// ------------------------------------------------------------------ people
async function loadPeople() {
  let people;
  try { people = await api("/api/people"); } catch { return; }
  const list = $("people");
  list.replaceChildren(...people.map(personRow));
  $("people-count").textContent = people.length;
  $("people-empty").hidden = people.length > 0;
}

function personRow(p) {
  const avatar = el("div", { class: "avatar", style: `background:${colourFor(p.name)}` });
  if (p.has_photo) {
    const img = el("img", { alt: "", loading: "lazy", src: `/api/people/${encodeURIComponent(p.name)}/photo?v=${p.samples}` });
    img.onerror = () => { img.remove(); avatar.textContent = initials(p.name); };
    avatar.append(img);
  } else {
    avatar.textContent = initials(p.name);
  }
  const actions = el("div", {});
  const removeBtn = el("button", {
    class: "icon-btn", type: "button", title: `Remove ${p.name}`, "aria-label": `Remove ${p.name}`,
    onclick: () => askRemove(p.name, actions, removeBtn),
  }, svg(ICON_TRASH));
  actions.append(removeBtn);
  return el("li", { class: "person" },
    avatar,
    el("div", { class: "person-info" },
      el("div", { class: "person-name" }, p.name),
      el("div", { class: "person-meta" }, `${p.samples} photo${p.samples === 1 ? "" : "s"}`)),
    actions);
}

function askRemove(name, actions, removeBtn) {
  clearTimeout(confirmTimer);
  const restore = () => actions.replaceChildren(removeBtn);
  actions.replaceChildren(el("div", { class: "confirm" },
    el("button", { class: "btn small danger", type: "button", onclick: () => removePerson(name) }, "Remove"),
    el("button", { class: "btn small ghost", type: "button", onclick: restore }, "Keep")));
  confirmTimer = setTimeout(restore, 5000);
}

async function removePerson(name) {
  try {
    await api(`/api/people/${encodeURIComponent(name)}`, { method: "DELETE" });
    toast(`Removed ${name}`);
  } catch (err) {
    toast(err.message, false);
  }
  loadPeople();
}

// ------------------------------------------------------------------ add person
function showNameError(msg) {
  $("name-error").textContent = msg || "";
  $("name").classList.toggle("invalid", !!msg);
}

async function startCapture(ev) {
  ev.preventDefault();
  const name = $("name").value.trim();
  if (!name) { showNameError("Please enter a name"); return; }
  try {
    await api("/api/enroll", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name }) });
    showNameError("");
    wasEnrolling = true;
    enrolling = true;
    updateButtons();
  } catch (err) {
    showNameError(err.message);
  }
}

async function uploadFiles(files) {
  const name = $("name").value.trim();
  if (!name || !files.length) return;
  $("upload").disabled = true;
  let added = 0;
  const problems = [];
  for (const file of files) {
    try {
      await api(`/api/people/${encodeURIComponent(name)}/photos`, {
        method: "POST", headers: { "Content-Type": "application/octet-stream" }, body: file,
      });
      added += 1;
    } catch (err) {
      problems.push(`${file.name}: ${err.message}`);
    }
  }
  if (added) {
    toast(`Added ${added} photo${added === 1 ? "" : "s"} of ${name}`);
    $("name").value = "";
    showNameError("");
  }
  if (problems.length) {
    showNameError(problems.length === 1 ? problems[0] : `${problems.length} photos skipped. ${problems[0]}`);
    if (!added) toast("No photos were added", false);
  }
  updateButtons();
  loadPeople();
}

// ------------------------------------------------------------------ activity
function addActivity(items, fresh) {
  const known = new Set(activity.map((a) => a.id));
  const incoming = items.filter((a) => !known.has(a.id)).map((a) => ({ ...a, fresh }));
  activity = [...incoming, ...activity].sort((a, b) => b.id - a.id).slice(0, 50);
  lastActivityId = Math.max(lastActivityId, ...items.map((a) => a.id));
  renderActivity();
}

function renderActivity() {
  const shown = activity.filter((a) => filter === "all" || !a.known);
  $("activity").replaceChildren(...shown.map(activityRow));
  $("activity-empty").hidden = shown.length > 0;
  $("activity-empty").textContent = filter === "all" ? "Nobody seen yet." : "No unknown faces seen.";
  for (const a of activity) a.fresh = false;
}

function activityRow(a) {
  let icon;
  if (a.snapshot) {
    const url = `/api/snapshots/${encodeURIComponent(a.snapshot)}`;
    icon = el("button", { class: "event-icon", type: "button", title: "View snapshot", "aria-label": `View snapshot of ${a.label}`,
      onclick: () => openViewer(url, `${a.label}, ${new Date(a.timestamp).toLocaleString()}`) },
      el("img", { alt: "", loading: "lazy", src: url }));
  } else {
    icon = el("div", { class: "event-icon" }, svg(a.known ? ICON_USER : ICON_QUESTION));
  }
  const pct = a.similarity != null && a.known ? ` · ${Math.round(a.similarity * 100)}% match` : "";
  return el("li", { class: `event ${a.known ? "known" : "unknown"}${a.fresh ? " fresh" : ""}` },
    icon,
    el("div", { class: "event-info" },
      el("div", { class: "event-label" }, a.label),
      el("div", { class: "event-meta", "data-ts": a.timestamp, "data-suffix": pct }, timeAgo(a.timestamp) + pct)),
    el("span", { class: `chip ${a.known ? "known" : "unknown"}` }, a.known ? "Known" : "Unknown"));
}

function refreshTimes() {
  for (const m of document.querySelectorAll(".event-meta[data-ts]")) {
    m.textContent = timeAgo(m.dataset.ts) + (m.dataset.suffix || "");
  }
}

function openViewer(src, caption) {
  $("viewer-img").src = src;
  $("viewer-caption").textContent = caption;
  $("viewer").showModal();
}

// ------------------------------------------------------------------ attendance
const ICON_CHECK = ["M5 12.5l4.5 4.5L19 7.5"];
let attendanceDay = "";
let attendanceTimer = null;

function scheduleAttendance() {
  clearTimeout(attendanceTimer);
  attendanceTimer = setTimeout(loadAttendance, 400);
}

async function loadAttendance() {
  const query = attendanceDay ? `?date=${attendanceDay}` : "";
  let rep;
  try { rep = await api(`/api/attendance${query}`); } catch { return; }
  const today = new Date();
  const todayIso = `${today.getFullYear()}-${String(today.getMonth() + 1).padStart(2, "0")}-${String(today.getDate()).padStart(2, "0")}`;
  const days = [...new Set([todayIso, rep.date, ...rep.days])].sort().reverse();
  const select = $("attendance-day");
  select.replaceChildren(...days.map((d) => el("option", { value: d, selected: d === rep.date },
    d === todayIso ? "Today" : new Date(`${d}T12:00:00`).toLocaleDateString([], { weekday: "short", month: "short", day: "numeric" }))));
  const total = rep.present.length + rep.absent.length;
  $("tally-present").textContent = rep.present.length;
  $("tally-text").textContent = total ? `of ${total} present` : "present";
  $("roll").replaceChildren(
    ...rep.present.map((a) => el("li", { class: "present" },
      el("span", { class: "mark" }, svg(ICON_CHECK)),
      el("span", { class: "who" }, a.name),
      el("span", { class: "when" }, new Date(a.first_seen).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" })))),
    ...rep.absent.map((name) => el("li", { class: "absent" },
      el("span", { class: "mark" }),
      el("span", { class: "who" }, name),
      el("span", { class: "when" }, "Not seen"))));
  $("attendance-csv").href = `/api/attendance.csv?date=${rep.date}`;
}

// ------------------------------------------------------------------ settings
let settingsTimer = null;
let editingSettings = false;

function syncSettings(s) {
  if (editingSettings) return;
  $("threshold").value = s.threshold;
  $("cooldown").value = Math.min(120, s.cooldown);
  showSettingValues();
}

function showSettingValues() {
  $("threshold-out").textContent = Number($("threshold").value).toFixed(3);
  const c = Number($("cooldown").value);
  $("cooldown-out").textContent = c === 0 ? "Every sighting" : `${c} s`;
}

function saveSettings() {
  showSettingValues();
  editingSettings = true;
  clearTimeout(settingsTimer);
  settingsTimer = setTimeout(async () => {
    try {
      await api("/api/settings", {
        method: "PUT", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ threshold: Number($("threshold").value), cooldown: Number($("cooldown").value) }),
      });
      $("settings-error").textContent = "";
    } catch (err) {
      $("settings-error").textContent = err.message;
    }
    editingSettings = false;
  }, 250);
}

// ------------------------------------------------------------------ quit
async function quit() {
  if (!confirm("Stop CharlieCam? The camera will turn off.")) return;
  try { await api("/api/quit", { method: "POST" }); } catch { /* server may already be gone */ }
  stopped = true;
  $("stream").src = "";
  $("stopped").hidden = false;
}

// ------------------------------------------------------------------ wire up
$("add-form").addEventListener("submit", startCapture);
$("attendance-day").addEventListener("change", (e) => { attendanceDay = e.target.value; loadAttendance(); });
$("name").addEventListener("input", () => { showNameError(""); updateButtons(); });
$("upload").addEventListener("click", () => $("file").click());
$("file").addEventListener("change", (e) => { uploadFiles([...e.target.files]); e.target.value = ""; });
$("enroll-cancel").addEventListener("click", () => api("/api/enroll/cancel", { method: "POST" }).catch(() => {}));
$("open-settings").addEventListener("click", () => $("settings").showModal());
$("threshold").addEventListener("input", saveSettings);
$("cooldown").addEventListener("input", saveSettings);
$("quit").addEventListener("click", quit);
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && enrolling) api("/api/enroll/cancel", { method: "POST" }).catch(() => {});
});
for (const btn of document.querySelectorAll(".seg-btn")) {
  btn.addEventListener("click", () => {
    filter = btn.dataset.filter;
    for (const b of document.querySelectorAll(".seg-btn")) {
      b.classList.toggle("active", b === btn);
      b.setAttribute("aria-selected", b === btn);
    }
    renderActivity();
  });
}

updateButtons();
loadAttendance();
startStream();
poll();
setInterval(refreshTimes, 15000);
