"use strict";

// The play app's page: long-polls /api/state and renders what a player at the
// table may see. All game logic and validation live on the server.

const SUITS = { s: "♠", h: "♥", d: "♦", c: "♣" };
const STREET_TITLES = { preflop: "Preflop", flop: "Flop", turn: "Turn", river: "River" };
const SECOND_PERSON = { folds: "fold", checks: "check", calls: "call", bets: "bet", raises: "raise", posts: "post" };
const MIN_HANDS_FOR_CI = 30;

// "raises to 250" -> "raise to 250", for lines about the person
function forYou(text) {
  const [verb, ...rest] = text.split(" ");
  return [SECOND_PERSON[verb] || verb, ...rest].join(" ");
}
const $ = (id) => document.getElementById(id);

let view = null;
let version = 0;
let busy = false;
let autoTimer = null;
let presets = [];

const prefs = loadPrefs();

function loadPrefs() {
  try {
    return JSON.parse(localStorage.getItem("pokerbot-play") || "{}") || {};
  } catch (e) {
    return {};
  }
}

function savePrefs() {
  try {
    localStorage.setItem("pokerbot-play", JSON.stringify(prefs));
  } catch (e) {
    // private window or blocked storage: preferences last for this page only
  }
}

const fmt = (n) => Number(n).toLocaleString("en-US");

function signed(n, digits = 0) {
  const s = Math.abs(n).toLocaleString("en-US", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
  return (n > 0 ? "+" : n < 0 ? "−" : "") + s;
}

// -- cards -------------------------------------------------------------------

function cardEl(card) {
  const el = document.createElement("div");
  if (!card) {
    el.className = "card back";
    el.setAttribute("aria-label", "hidden card");
    return el;
  }
  const suit = card[1];
  el.className = `card ${suit}`;
  el.setAttribute("aria-label", card);
  const rank = document.createElement("span");
  rank.className = "rank";
  rank.textContent = card[0] === "T" ? "10" : card[0];
  const pip = document.createElement("span");
  pip.className = "suit";
  pip.textContent = SUITS[suit];
  el.append(rank, pip);
  return el;
}

function renderCards(container, cards) {
  container.replaceChildren(...(cards ? cards.map(cardEl) : [cardEl(null), cardEl(null)]));
}

// -- rendering -----------------------------------------------------------------

function seatOf(hand, human) {
  return hand.players.find((p) => p.human === human);
}

function renderSeat(prefix, p, hand) {
  $(`${prefix}-name`).textContent = p.name;
  $(`${prefix}-stack`).textContent = fmt(p.stack);
  $(`${prefix}-dealer`).classList.toggle("on", p.button);
  $(`${prefix}-bet`).textContent = p.bet > 0 ? fmt(p.bet) : "";
  const seat = $(prefix === "bot" ? "seat-bot" : "seat-human");
  seat.classList.toggle("to-act", hand.to_act === p.seat);
  seat.classList.toggle("folded", p.folded);
  renderCards($(`${prefix}-cards`), p.cards);
}

function renderTable(v) {
  const hand = v.hand;
  const names = v.session.names;
  if (!hand) {
    $("bot-name").textContent = names.bot;
    $("human-name").textContent = names.human;
    $("board").replaceChildren(...Array.from({ length: 5 }, slot));
    $("pot").textContent = "";
    $("result").hidden = true;
    return;
  }
  renderSeat("bot", seatOf(hand, false), hand);
  renderSeat("human", seatOf(hand, true), hand);
  const board = hand.board.map(cardEl);
  while (board.length < 5) board.push(slot());
  $("board").replaceChildren(...board);
  $("pot").textContent = hand.pot ? `Pot ${fmt(hand.pot)}` : "";
  $("bot-thinking").classList.toggle("on", v.status === "bot");
  const res = $("result");
  if (hand.result) {
    const net = hand.result.human_net;
    res.textContent = `${hand.result.text} · ${names.human} ${signed(net)}`;
    res.className = `result ${net > 0 ? "win" : net < 0 ? "lose" : ""}`;
    res.hidden = false;
  } else {
    res.hidden = true;
  }
}

function slot() {
  const el = document.createElement("div");
  el.className = "slot";
  return el;
}

function statusText(v) {
  const s = v.session;
  switch (v.status) {
    case "warmup":
      return "Loading the bot. Its first decisions also compile GPU kernels, so this can take a minute.";
    case "ready":
      return s.hands_played ? "Ready. Deal the next hand." : "Ready. Deal the first hand.";
    case "human":
      return "Your turn";
    case "bot":
      return `${s.names.bot} is thinking`;
    case "hand_over":
      return "Hand over";
    case "finished":
      if (s.hands_target && s.hands_played >= s.hands_target) {
        return `Session complete: all ${fmt(s.hands_target)} hands played. Thank you!`;
      }
      return `Session ended after ${fmt(s.hands_played)} hands.`;
    case "error":
      return "The session stopped with an error (details in errors.log in the session directory).";
    default:
      return "";
  }
}

function renderStatus(v) {
  const el = $("status");
  el.textContent = statusText(v);
  if (v.message) {
    const m = document.createElement("span");
    m.className = "msg";
    m.textContent = v.message;
    el.append(m);
  }
}

function renderLog(v) {
  const log = $("log");
  const hand = v.hand;
  $("hand-no").textContent = hand ? `#${fmt(hand.number + 1)}` : "";
  if (!hand) {
    log.replaceChildren();
    return;
  }
  const items = [];
  let street = null;
  for (const a of hand.actions) {
    if (a.street !== street) {
      street = a.street;
      if (street !== "preflop") {
        const h = document.createElement("li");
        h.className = "street";
        h.textContent = STREET_TITLES[street];
        items.push(h);
      }
    }
    const li = document.createElement("li");
    const human = hand.players[a.seat].human;
    li.className = human ? "human" : "bot";
    const who = document.createElement("span");
    who.className = "who";
    who.textContent = hand.players[a.seat].name;
    li.append(who, ` ${human ? forYou(a.text) : a.text}`);
    items.push(li);
  }
  if (hand.result && hand.result.showdown) {
    for (const [name, desc] of Object.entries(hand.result.hands)) {
      const li = document.createElement("li");
      li.className = "street";
      li.textContent = `${name}: ${desc}`;
      items.push(li);
    }
  }
  log.replaceChildren(...items);
  log.scrollTop = log.scrollHeight;
  renderLastHand(v);
}

// the previous hand's result stays readable after auto-deal starts the next one
function renderLastHand(v) {
  const el = $("last-hand");
  const last = v.last_hand;
  if (!last || !v.hand || last.number === v.hand.number) {
    el.hidden = true;
    return;
  }
  const head = document.createElement("b");
  head.textContent = `Last hand #${fmt(last.number + 1)}: `;
  let text = `${last.text}.`;
  if (last.bot_cards) {
    const shown = last.bot_cards.map((c) => `${c[0] === "T" ? "10" : c[0]}${SUITS[c[1]]}`);
    text += ` ${v.session.names.bot} showed ${shown.join(" ")}.`;
  }
  text += ` ${v.session.names.human} ${signed(last.human_net)}.`;
  el.replaceChildren(head, text);
  el.hidden = false;
}

function stat(label, value, ci) {
  const row = document.createElement("div");
  row.className = "stat";
  const a = document.createElement("span");
  a.textContent = label;
  const b = document.createElement("span");
  b.textContent = value;
  if (ci) {
    const c = document.createElement("span");
    c.className = "ci";
    c.textContent = ci;
    b.append(c);
  }
  row.append(a, b);
  return row;
}

function rate(r, hands) {
  // mbb/hand to bb/100 (divide by 10), with the CI once there are enough hands for one
  if (!r || r.mbb === null || r.mbb === undefined) return ["–", ""];
  if (hands < MIN_HANDS_FOR_CI) {
    return [`${signed(r.mbb / 10, 1)} bb/100`, `CI after ${MIN_HANDS_FOR_CI} hands`];
  }
  const pct = Math.round(100 * (view.stats.confidence || 0.95));
  const lo = r.lo === null ? "?" : signed(r.lo / 10, 1);
  const hi = r.hi === null ? "?" : signed(r.hi / 10, 1);
  return [`${signed(r.mbb / 10, 1)} bb/100`, `${pct}% CI ${lo} to ${hi}`];
}

function renderStats(v) {
  const el = $("stats");
  $("toggle-stats").textContent = prefs.hideStats ? "show" : "hide";
  const s = v.stats || { hands: 0 };
  const ses = v.session;
  if (prefs.hideStats) {
    const p = document.createElement("p");
    p.className = "hint";
    p.textContent = "Hidden. Every hand is still logged.";
    el.replaceChildren(p);
    return;
  }
  const rows = [
    stat("Hands", ses.hands_target ? `${fmt(s.hands)} of ${fmt(ses.hands_target)}` : fmt(s.hands)),
  ];
  if (ses.abandoned) rows.push(stat("Abandoned", fmt(ses.abandoned)));
  if (s.hands) {
    const bb = s.net_bb || 0;
    rows.push(stat(`${ses.names.human}, net`, `${signed(s.net)} chips`, `${signed(bb, 1)} big blinds`));
    rows.push(stat("Win rate", ...rate(s.raw, s.hands)));
    rows.push(stat("Luck-adjusted", ...rate(s.adjusted, s.hands)));
  }
  el.replaceChildren(...rows);
}

function renderControls(v) {
  const hand = v.hand;
  const L = v.status === "human" && hand ? hand.legal : null;
  $("controls").hidden = !(v.status === "human" || v.status === "bot");
  $("between").hidden = !(v.status === "ready" || v.status === "hand_over");
  $("btn-next").disabled = busy;
  $("btn-stop").disabled = v.status === "human" || v.status === "bot" || v.status === "finished";
  const me = hand ? seatOf(hand, true) : null;
  const off = !L || busy;
  $("btn-fold").disabled = off || !L.can_fold;
  $("btn-call").disabled = off;
  if (L) {
    const call = L.can_check ? "Check" : `Call ${fmt(L.call)}${me && L.call >= me.stack ? " (all-in)" : ""}`;
    $("btn-call").firstChild.textContent = `${call} `;
  }
  const canRaise = !!L && L.min_raise_to > 0;
  const slider = $("raise-slider");
  const input = $("raise-input");
  slider.disabled = input.disabled = off || !canRaise;
  $("btn-raise").disabled = off || !canRaise;
  presets = canRaise ? L.presets : [];
  const row = $("presets");
  row.replaceChildren(
    ...presets.map((p, i) => {
      const b = document.createElement("button");
      b.type = "button";
      b.disabled = off;
      b.textContent = `${p.label} · ${fmt(p.to)}`;
      if (i < 9) {
        const k = document.createElement("kbd");
        k.textContent = String(i + 1);
        b.append(k);
      }
      b.addEventListener("click", () => setRaise(p.to));
      return b;
    })
  );
  if (canRaise) {
    const lo = Math.min(L.min_raise_to, L.max_raise_to);
    slider.min = lo;
    slider.max = L.max_raise_to;
    slider.step = 1;
    input.min = lo;
    input.max = L.max_raise_to;
    // a new decision starts at the second preset; within one, keep the person's size
    const decision = `${hand.number}:${hand.actions.length}`;
    if (input.dataset.decision !== decision) {
      input.dataset.decision = decision;
      setRaise(presets.length > 1 ? presets[Math.min(1, presets.length - 2)].to : lo);
    }
    updateRaiseLabel();
  } else {
    $("btn-raise").firstChild.textContent = "Raise ";
  }
}

function setRaise(amount) {
  const input = $("raise-input");
  input.value = String(amount);
  $("raise-slider").value = String(amount);
  updateRaiseLabel();
}

function updateRaiseLabel() {
  if (!view || !view.hand || !view.hand.legal) return;
  const L = view.hand.legal;
  const to = Number($("raise-input").value);
  const unopened = view.hand.street !== "preflop" && view.hand.players.every((p) => p.bet === 0);
  let label = unopened ? `Bet ${fmt(to)}` : `Raise to ${fmt(to)}`;
  if (to >= L.max_raise_to) label = `All-in ${fmt(L.max_raise_to)}`;
  $("btn-raise").firstChild.textContent = `${label} `;
}

function renderTopbar(v) {
  const s = v.session;
  $("mode").textContent = s.mode;
  const parts = [`session ${s.id}`];
  if (v.hand) {
    parts.push(
      s.hands_target
        ? `hand ${fmt(v.hand.number + 1)} of ${fmt(s.hands_target)}`
        : `hand ${fmt(v.hand.number + 1)}`
    );
  }
  $("session-info").textContent = parts.join(" · ");
}

function render(v) {
  view = v;
  busy = false;
  renderTopbar(v);
  renderTable(v);
  renderStatus(v);
  renderControls(v);
  renderLog(v);
  renderStats(v);
  scheduleAutoDeal(v);
}

// -- actions -------------------------------------------------------------------

async function post(path, body) {
  try {
    const r = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body || {}),
    });
    const j = await r.json();
    if (!j.ok) showMessage(j.error || `request failed (${r.status})`);
    return !!j.ok;
  } catch (e) {
    showMessage("Cannot reach the server");
    return false;
  }
}

function showMessage(text) {
  const el = $("status");
  let m = el.querySelector(".msg");
  if (!m) {
    m = document.createElement("span");
    m.className = "msg";
    el.append(m);
  }
  m.textContent = text;
}

async function act(kind, amount) {
  if (busy) return;
  busy = true;
  renderControls(view);
  const ok = await post("/api/action", { kind, amount });
  if (!ok) {
    busy = false;
    renderControls(view);
  }
}

function raise() {
  if ($("btn-raise").disabled) return;
  const L = view.hand.legal;
  const to = Math.round(Number($("raise-input").value));
  if (to >= L.max_raise_to) act("allin");
  else act("raise", to);
}

async function next() {
  if (busy || $("btn-next").disabled || $("between").hidden) return;
  busy = true;
  $("btn-next").disabled = true;
  const ok = await post("/api/next");
  if (!ok) {
    busy = false;
    $("btn-next").disabled = false;
  }
}

function scheduleAutoDeal(v) {
  clearTimeout(autoTimer);
  autoTimer = null;
  if (v.status === "hand_over" && prefs.autoDeal) {
    const showdown = v.hand && v.hand.result && v.hand.result.showdown;
    autoTimer = setTimeout(next, showdown ? 5000 : 2500);
  }
}

// -- wiring --------------------------------------------------------------------

$("btn-fold").addEventListener("click", () => act("fold"));
$("btn-call").addEventListener("click", () => {
  const L = view && view.hand && view.hand.legal;
  if (L) act(L.can_check ? "check" : "call");
});
$("btn-raise").addEventListener("click", raise);
$("btn-next").addEventListener("click", next);
$("raise-slider").addEventListener("input", (e) => setRaise(e.target.value));
$("raise-input").addEventListener("input", () => {
  $("raise-slider").value = $("raise-input").value;
  updateRaiseLabel();
});
$("auto-deal").checked = !!prefs.autoDeal;
$("auto-deal").addEventListener("change", (e) => {
  prefs.autoDeal = e.target.checked;
  savePrefs();
  if (view) scheduleAutoDeal(view);
});
$("toggle-stats").addEventListener("click", () => {
  prefs.hideStats = !prefs.hideStats;
  savePrefs();
  if (view) renderStats(view);
});
$("btn-stop").addEventListener("click", async () => {
  if (confirm("End this session? You can resume it later with --resume.")) {
    await post("/api/stop");
  }
});

document.addEventListener("keydown", (e) => {
  if (e.ctrlKey || e.metaKey || e.altKey) return;
  if (e.target instanceof HTMLInputElement && e.target.type === "number") {
    if (e.key === "Enter") {
      e.preventDefault();
      raise();
    }
    return;
  }
  const k = e.key.toLowerCase();
  if (k === "f" && !$("btn-fold").disabled) act("fold");
  else if ((k === "c" || k === "k") && !$("btn-call").disabled) $("btn-call").click();
  else if (k === "r" && !$("raise-input").disabled) {
    e.preventDefault();
    $("raise-input").focus();
    $("raise-input").select();
  } else if (k === "n") next();
  else if (/^[1-9]$/.test(k) && presets[Number(k) - 1] && !$("raise-input").disabled) {
    setRaise(presets[Number(k) - 1].to);
  }
});

async function poll() {
  for (;;) {
    try {
      const r = await fetch(`/api/state?since=${version}`, { cache: "no-store" });
      const v = await r.json();
      if (v.version > version) {
        version = v.version;
        render(v);
      }
    } catch (e) {
      $("status").textContent = "Connection lost. Retrying…";
      await new Promise((ok) => setTimeout(ok, 1000));
    }
  }
}

poll();
