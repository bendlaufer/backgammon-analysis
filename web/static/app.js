let state = null;
let dice = null;
let legalActions = [];
let busy = false;

const boardEl = document.querySelector("#board");
const statusEl = document.querySelector("#status");
const diceEl = document.querySelector("#dice");
const movesEl = document.querySelector("#moves");
const engineLogEl = document.querySelector("#engineLog");
const rollBtn = document.querySelector("#roll");

document.querySelector("#newGame").addEventListener("click", newGame);
rollBtn.addEventListener("click", roll);

async function api(path, payload = null) {
  const response = await fetch(path, payload ? {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(payload)
  } : undefined);
  const data = await response.json();
  if (!response.ok || data.error) throw new Error(data.error || response.statusText);
  return data;
}

async function newGame() {
  const data = await api("/api/new");
  state = data.state;
  dice = null;
  legalActions = [];
  engineLogEl.innerHTML = "";
  log(`Engine: ${data.engine}`);
  render();
}

async function roll() {
  if (busy || state.turn !== 0) return;
  busy = true;
  try {
    dice = [randDie(), randDie()];
    const data = await api("/api/legal", {state, dice});
    legalActions = data.legal_actions;
    if (legalActions.length === 1 && legalActions[0].action.length === 0) {
      log(`You rolled ${dice.join("-")} and cannot move.`);
      const pass = await api("/api/play", {state, dice, action: []});
      state = pass.state;
      await engineTurn();
    }
    render();
  } catch (err) {
    log(err.message);
  } finally {
    busy = false;
  }
}

async function play(action) {
  if (busy) return;
  busy = true;
  try {
    const data = await api("/api/play", {state, dice, action});
    log(`You played ${formatAction(action)}.`);
    state = data.state;
    dice = null;
    legalActions = [];
    if (data.winner !== null) {
      log(`Player ${data.winner} wins.`);
      render();
      return;
    }
    await engineTurn();
  } catch (err) {
    log(err.message);
  } finally {
    busy = false;
    render();
  }
}

async function engineTurn() {
  const data = await api("/api/engine", {state});
  state = data.state;
  log(`Engine rolled ${data.dice.join("-")} and played ${formatAction(data.action)}.`);
  if (data.winner !== null) log(`Player ${data.winner} wins.`);
}

function render() {
  renderBoard();
  renderStacks();
  renderDice();
  renderMoves();
  rollBtn.disabled = busy || !state || state.turn !== 0 || dice !== null || state.off[0] === 15 || state.off[1] === 15;
  const turn = state?.turn === 0 ? "Your turn" : "Engine turn";
  const score = state ? `Off: you ${state.off[0]}, engine ${state.off[1]}` : "";
  statusEl.textContent = state ? `${turn}. ${score}` : "Loading...";
}

function renderBoard() {
  boardEl.innerHTML = "";
  const order = [23,22,21,20,19,18,17,16,15,14,13,12,0,1,2,3,4,5,6,7,8,9,10,11];
  order.forEach((point, idx) => {
    const div = document.createElement("div");
    const isTop = idx < 12;
    div.className = `point ${isTop ? "top" : "bottom"} ${(idx + Math.floor(idx / 6)) % 2 ? "dark" : ""}`;
    const label = document.createElement("div");
    label.className = "point-label";
    label.textContent = String(point);
    div.appendChild(label);

    const stack = document.createElement("div");
    stack.className = "checkers";
    const count = state.points[point];
    const playerClass = count > 0 ? "p0" : "p1";
    const visible = Math.min(Math.abs(count), 5);
    for (let i = 0; i < visible; i++) {
      const checker = document.createElement("div");
      checker.className = `checker ${playerClass}`;
      stack.appendChild(checker);
    }
    if (Math.abs(count) > 5) {
      const badge = document.createElement("div");
      badge.className = "count-badge";
      badge.textContent = Math.abs(count);
      stack.appendChild(badge);
    }
    div.appendChild(stack);
    boardEl.appendChild(div);
  });
}

function renderStacks() {
  renderCountStack("#bar0", state.bar[0], "p0");
  renderCountStack("#bar1", state.bar[1], "p1");
  renderCountStack("#off0", state.off[0], "p0");
  renderCountStack("#off1", state.off[1], "p1");
}

function renderCountStack(selector, count, klass) {
  const el = document.querySelector(selector);
  el.innerHTML = "";
  const visible = Math.min(count, 5);
  for (let i = 0; i < visible; i++) {
    const checker = document.createElement("div");
    checker.className = `checker ${klass}`;
    el.appendChild(checker);
  }
  if (count > 5) {
    const badge = document.createElement("div");
    badge.className = "count-badge";
    badge.textContent = count;
    el.appendChild(badge);
  }
}

function renderDice() {
  diceEl.innerHTML = "";
  if (!dice) return;
  dice.forEach(value => {
    const die = document.createElement("div");
    die.className = "die";
    die.textContent = value;
    diceEl.appendChild(die);
  });
}

function renderMoves() {
  movesEl.innerHTML = "";
  if (!dice) {
    movesEl.textContent = "Roll to see legal moves.";
    return;
  }
  legalActions.forEach(item => {
    const button = document.createElement("button");
    button.className = "move-button";
    button.textContent = item.action_key || "pass";
    button.addEventListener("click", () => play(item.action));
    movesEl.appendChild(button);
  });
}

function log(message) {
  const entry = document.createElement("div");
  entry.className = "log-entry";
  entry.textContent = message;
  engineLogEl.prepend(entry);
}

function formatAction(action) {
  return action && action.length ? action.join(" ") : "pass";
}

function randDie() {
  return Math.floor(Math.random() * 6) + 1;
}

newGame();
