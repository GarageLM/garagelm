/* Toy decoder-only transformer — one block.
   Weights start hand-set but are MUTABLE: trainStep() runs real gradient descent
   (central finite differences over all 172 trainable weights). forward(tokens, w?)
   uses the live weights by default; pass LLM.ORIG for the untrained baseline.
   Exposes window.LLM with a pure forward() so every rendered number is live+consistent. */
(function () {
  const VOCAB = ["the", "cat", "sat", "on", "mat", "dog", "ran", "fast"];
  const D = 4, H = 2, DH = 2, DFF = 8;
  const DIMS = ["d0", "d1", "d2", "d3"];

  // Token embeddings (8 x 4)
  const E = [
    [ 0.9,  0.1, -0.2,  0.3], // the
    [ 0.2,  0.8,  0.4, -0.5], // cat
    [-0.3,  0.5,  0.7,  0.2], // sat
    [ 0.6, -0.4,  0.1,  0.5], // on
    [-0.1,  0.7,  0.9, -0.2], // mat
    [ 0.3,  0.6,  0.2, -0.6], // dog
    [-0.5,  0.2,  0.8,  0.4], // ran
    [ 0.7, -0.3, -0.1,  0.6]  // fast
  ];
  const WQ = [[0.4,-0.2,0.1,0.3],[0.1,0.5,-0.3,0.2],[-0.2,0.3,0.4,-0.1],[0.3,0.1,0.2,0.4]];
  const WK = [[0.3,0.2,-0.1,0.4],[-0.2,0.4,0.3,0.1],[0.5,-0.1,0.2,0.3],[0.1,0.3,-0.2,0.2]];
  const WV = [[0.2,0.4,0.1,-0.3],[0.3,-0.2,0.5,0.2],[-0.1,0.3,0.2,0.4],[0.4,0.1,-0.3,0.2]];
  const WO = [[0.3,0.1,0.4,-0.2],[0.2,0.4,-0.1,0.3],[-0.2,0.3,0.2,0.4],[0.4,-0.1,0.3,0.1]];
  const W1 = [[0.3,-0.2,0.4,0.1,-0.3,0.2,0.1,0.4],[0.2,0.3,-0.1,0.4,0.2,-0.3,0.3,0.1],[-0.1,0.4,0.2,-0.2,0.3,0.1,-0.2,0.3],[0.4,0.1,-0.3,0.2,0.1,0.3,0.4,-0.1]];
  const b1 = [0.1,-0.1,0.0,0.1,-0.1,0.0,0.1,-0.1];
  const W2 = [[0.2,-0.1,0.3,0.1],[0.1,0.3,-0.2,0.2],[-0.2,0.2,0.1,0.3],[0.3,0.1,0.2,-0.1],[0.1,-0.2,0.3,0.2],[0.2,0.3,-0.1,0.1],[-0.1,0.2,0.2,0.3],[0.3,-0.1,0.1,0.2]];
  const b2 = [0.0,0.1,-0.1,0.0];
  // Unembedding = E^T (weight tying) — for compat; forward() recomputes from live E
  const U = [0,1,2,3].map(k => E.map(row => row[k])); // 4 x 8

  const clone = o => JSON.parse(JSON.stringify(o));
  const ORIG = { E, WQ, WK, WV, WO, W1, b1, W2, b2 };
  let W = clone(ORIG);   // live, trainable weights
  let version = 0;       // bumped on every weight change

  const matmul = (A, B) => {
    const m = A.length, n = B.length, p = B[0].length, C = [];
    for (let i = 0; i < m; i++) { C[i] = []; for (let j = 0; j < p; j++) { let s = 0; for (let k = 0; k < n; k++) s += A[i][k] * B[k][j]; C[i][j] = s; } }
    return C;
  };
  const T = A => A[0].map((_, j) => A.map(r => r[j]));
  const addM = (A, B) => A.map((r, i) => r.map((v, j) => v + B[i][j]));
  const gelu = x => 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x * x * x)));
  function softmax(a) { const m = Math.max(...a.filter(v => isFinite(v))); const e = a.map(v => isFinite(v) ? Math.exp(v - m) : 0); const s = e.reduce((x, y) => x + y, 0); return e.map(v => v / s); }
  function layernorm(row) { const m = row.reduce((a, b) => a + b, 0) / row.length; const v = row.reduce((a, b) => a + (b - m) ** 2, 0) / row.length; return row.map(x => (x - m) / Math.sqrt(v + 1e-5)); }
  function PE(pos) { const r = []; for (let i = 0; i < D / 2; i++) { const f = 1 / Math.pow(10000, (2 * i) / D); r.push(Math.sin(pos * f)); r.push(Math.cos(pos * f)); } return r; }

  function forward(tokens, wArg) {
    const w = wArg || W;
    const Uw = [0, 1, 2, 3].map(k => w.E.map(row => row[k])); // tied unembedding, from live E
    const seq = tokens.length;
    const words = tokens.map(t => VOCAB[t]);
    const emb = tokens.map(t => w.E[t].slice());
    const pos = tokens.map((_, i) => PE(i));
    const x0 = addM(emb, pos);
    const Q = matmul(x0, w.WQ), K = matmul(x0, w.WK), V = matmul(x0, w.WV);
    const heads = [];
    const concat = x0.map(() => new Array(D).fill(0));
    for (let h = 0; h < H; h++) {
      const c0 = h * DH;
      const Qh = Q.map(r => r.slice(c0, c0 + DH));
      const Kh = K.map(r => r.slice(c0, c0 + DH));
      const Vh = V.map(r => r.slice(c0, c0 + DH));
      const raw = matmul(Qh, T(Kh)).map(r => r.map(v => v / Math.sqrt(DH)));
      const masked = raw.map((r, i) => r.map((v, j) => (j > i ? -Infinity : v)));
      const weights = masked.map(r => softmax(r));
      const out = matmul(weights, Vh);
      heads.push({ Qh, Kh, Vh, raw, masked, weights, out });
      for (let i = 0; i < seq; i++) for (let j = 0; j < DH; j++) concat[i][c0 + j] = out[i][j];
    }
    const attnOut = matmul(concat, w.WO);
    const res1 = addM(x0, attnOut);
    const ln1 = res1.map(layernorm);
    const preAct = addM(matmul(ln1, w.W1), ln1.map(() => w.b1));
    const act = preAct.map(r => r.map(gelu));
    const ff2 = addM(matmul(act, w.W2), ln1.map(() => w.b2));
    const res2 = addM(ln1, ff2);
    const ln2 = res2.map(layernorm);
    const last = ln2[seq - 1];
    // logits + probs at EVERY position (needed for teacher-forcing loss)
    const allLogits = ln2.map(h => Uw[0].map((_, j) => h.reduce((s, _v, k) => s + h[k] * Uw[k][j], 0)));
    const allProbs = allLogits.map(softmax);
    const logits = allLogits[seq - 1];
    const probs = allProbs[seq - 1];
    const ranked = probs.map((p, i) => ({ i, word: VOCAB[i], p })).sort((a, b) => b.p - a.p);
    // teacher forcing: position i predicts token i+1
    const perTokenLoss = [];
    for (let i = 0; i < seq - 1; i++) {
      const target = tokens[i + 1];
      perTokenLoss.push({ pos: i, target, targetWord: VOCAB[target], p: allProbs[i][target], loss: -Math.log(allProbs[i][target] + 1e-12) });
    }
    const avgLoss = perTokenLoss.length ? perTokenLoss.reduce((s, x) => s + x.loss, 0) / perTokenLoss.length : 0;
    const perplexity = Math.exp(avgLoss);
    return { tokens, words, seq, emb, pos, x0, Q, K, V, heads, concat, attnOut, res1, ln1, preAct, act, ff2, res2, ln2, last, logits, probs, ranked, allLogits, allProbs, perTokenLoss, avgLoss, perplexity };
  }

  // ---- training: real gradient descent via central finite differences ----
  function paramRefs(w) {
    const refs = [];
    ['E', 'WQ', 'WK', 'WV', 'WO', 'W1', 'W2'].forEach(k => w[k].forEach((row, i) => row.forEach((_, j) => refs.push({ k, i, j }))));
    ['b1', 'b2'].forEach(k => w[k].forEach((_, i) => refs.push({ k, i })));
    return refs; // 172 trainable weights (LN γ/β fixed at 1/0 in this toy)
  }
  const getP = (w, r) => (r.j == null ? w[r.k][r.i] : w[r.k][r.i][r.j]);
  const setP = (w, r, v) => { if (r.j == null) w[r.k][r.i] = v; else w[r.k][r.i][r.j] = v; };

  function trainStep(tokens, lr) {
    const eps = 1e-3, refs = paramRefs(W), g = new Array(refs.length);
    for (let n = 0; n < refs.length; n++) {
      const r = refs[n], v = getP(W, r);
      setP(W, r, v + eps); const lp = forward(tokens).avgLoss;
      setP(W, r, v - eps); const lm = forward(tokens).avgLoss;
      setP(W, r, v);
      g[n] = (lp - lm) / (2 * eps);
    }
    for (let n = 0; n < refs.length; n++) setP(W, refs[n], getP(W, refs[n]) - lr * g[n]);
    version++;
    return forward(tokens).avgLoss;
  }

  function resetWeights() { W = clone(ORIG); version++; }
  function setWeights(w2) { W = clone(w2); version++; }
  const getWeights = () => W;
  const getVersion = () => version;
  const bump = () => { version++; };

  window.LLM = { VOCAB, DIMS, D, H, DH, DFF, E, WQ, WK, WV, WO, W1, b1, W2, b2, U, forward, matmul, PE,
                 ORIG, getWeights, getVersion, bump, trainStep, resetWeights, setWeights };
})();
