import { useEffect, useMemo, useRef, useState } from 'react';

type P = { temperature: number; topK: number; topP: number; minP: number; repeat: number; seed: number; ctx: number; maxTok: number; kLo: number; pLo: number; freq: number; pres: number; stream: boolean; mirostat: boolean };
type Key = 'temperature' | 'topK' | 'topP' | 'minP' | 'repeat' | 'seed';
const DEF: P = { temperature: 0.8, topK: 40, topP: 0.95, minP: 0.05, repeat: 1.1, seed: 42, kLo: 1, pLo: 0, ctx: 4096, maxTok: 256, freq: 0, pres: 0, stream: true, mirostat: false };
const RANGE: Record<string, [number, number, number]> = { temperature: [0, 2, 0.01], topK: [1, 100, 1], topP: [0, 1, 0.01], minP: [0, 0.5, 0.01], repeat: [1, 2, 0.01], seed: [0, 99999, 1], kLo: [1, 100, 1], pLo: [0, 1, 0.01], ctx: [512, 32768, 512], maxTok: [16, 4096, 16], freq: [-2, 2, 0.1], pres: [-2, 2, 0.1] };
const INFO: Record<string, string> = { temperature: 'Flattens or sharpens the distribution. Low = focused, high = wild.', topK: 'Keep only the K most likely tokens.', topP: 'Keep the smallest set whose cumulative probability reaches P.', minP: 'Drop tokens below P x the top token probability.', repeat: 'Penalizes tokens already used. >1 discourages loops.', seed: 'Fixes the random path so results repeat.', kLo: 'Min keep: candidates always retained, even after other filters (hollow blue node).', pLo: 'Mass floor: probability mass always retained regardless of Min-P (hollow red node).', ctx: 'Context window size in tokens.', maxTok: 'Maximum tokens to generate.', freq: 'Penalty scaled by how often a token appeared.', pres: 'Penalty for any token that appeared at all.' };

// Adapters: map common params to backend-specific names and support
type Adapter = { name: string; url: string; supports: string[]; map: (p: P) => Record<string, unknown> };
const ADAPTERS: Record<string, Adapter> = {
  demo: { name: 'Demo (simulated)', url: '', supports: Object.keys(RANGE).concat(['stream', 'mirostat']), map: p => ({ ...p }) },
  lmstudio: { name: 'LM Studio', url: 'http://localhost:1234/v1', supports: ['temperature', 'topK', 'topP', 'minP', 'repeat', 'seed', 'maxTok', 'freq', 'pres', 'stream'], map: p => ({ min_keep: p.kLo, temperature: p.temperature, top_k: p.topK, top_p: p.topP, min_p: p.minP, repeat_penalty: p.repeat, seed: p.seed, max_tokens: p.maxTok, frequency_penalty: p.freq, presence_penalty: p.pres, stream: false }) },
  ollama: { name: 'Ollama', url: 'http://localhost:11434', supports: ['temperature', 'topK', 'topP', 'minP', 'repeat', 'seed', 'ctx', 'maxTok', 'freq', 'pres', 'mirostat', 'kLo'], map: p => ({ stream: false, options: { temperature: p.temperature, top_k: p.topK, top_p: p.topP, min_p: p.minP, repeat_penalty: p.repeat, seed: p.seed, num_ctx: p.ctx, num_predict: p.maxTok, frequency_penalty: p.freq, presence_penalty: p.pres, mirostat: p.mirostat ? 2 : 0 } }) },
  openai: { name: 'llama.cpp / OpenAI', url: 'http://localhost:8080/v1', supports: ['temperature', 'topP', 'seed', 'maxTok', 'freq', 'pres', 'topK', 'minP', 'repeat', 'kLo'], map: p => ({ min_keep: p.kLo, temperature: p.temperature, top_p: p.topP, top_k: p.topK, min_p: p.minP, repeat_penalty: p.repeat, seed: p.seed, max_tokens: p.maxTok, frequency_penalty: p.freq, presence_penalty: p.pres, stream: false }) },
};

const TOKENS = ['the', 'a', 'little', 'quiet', 'strange', 'mind', 'river', 'clock', 'moth', 'glass', 'orbit', 'yes', 'perhaps', 'echo', 'blue', 'never'];
const USED = [0, 1, 5]; // tokens already present in context (simulated)
const LOGITS = TOKENS.map((_, i) => 3.2 - i * 0.42 - (i % 3) * 0.08);
function dist(p: P) {
  const t = Math.max(p.temperature, 0.02);
  const ex = LOGITS.map((l, i) => Math.exp((l - (USED.includes(i) ? (p.repeat - 1) * 1.6 : 0)) / t));
  const s = ex.reduce((a, b) => a + b, 0);
  const pr = ex.map(e => e / s);
  const order = pr.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]);
  const mx = order[0][0]; let cum = 0;
  const ok = new Array(pr.length).fill(false);
  order.forEach(([v, i], r) => { const keep = r < Math.min(p.kLo, p.topK) || cum < p.pLo || (r < p.topK && (cum < p.topP || r === 0) && v >= p.minP * mx); cum += v; ok[i] = keep; });
  return pr.map((v, i) => ({ tok: TOKENS[i], v, ok: ok[i] }));
}
const rng = (s: number) => () => { s = (s * 16807) % 2147483647 || 1; return s / 2147483647; };
const fmt = (k: string, v: number) => (RANGE[k][2] < 1 ? v.toFixed(2) : String(Math.round(v)));

function useDrag(k: string, val: number, set: (v: number) => void, onStart: () => void, onEnd: () => void) {
  const st = useRef<{ y: number; v: number } | null>(null);
  const [mn, mx] = RANGE[k];
  return {
    onPointerDown: (e: React.PointerEvent) => { (e.target as Element).setPointerCapture(e.pointerId); st.current = { y: e.clientY - e.clientX, v: val }; onStart(); },
    onPointerMove: (e: React.PointerEvent) => { if (!st.current) return; const d = (st.current.y - (e.clientY - e.clientX)) / 220; set(Math.min(mx, Math.max(mn, st.current.v + d * (mx - mn)))); },
    onPointerUp: () => { st.current = null; onEnd(); },
    onPointerCancel: () => { st.current = null; onEnd(); },
  };
}

type Expr = { eyeW: number; eyeH: number; lx: number; ly: number; browL: number; browR: number; browY: number; mouth: number; mouthW: number; wob: number };
const NEUTRAL: Expr = { eyeW: 1, eyeH: 1, lx: 0, ly: 0, browL: 0, browR: 0, browY: 0, mouth: 2, mouthW: 1, wob: 0 };
function expr(k: Key | null, p: P): Expr {
  const e = { ...NEUTRAL };
  if (k === 'temperature') { const t = p.temperature / 2; e.eyeW = .8 + t * .7; e.eyeH = .45 + t * 1.1; e.browY = 4 - t * 10; e.browL = (.5 - t) * 10; e.browR = -(.5 - t) * 10; e.mouth = t > .55 ? -4 - t * 4 : 1; e.mouthW = .7 + t * .4; e.wob = Math.max(0, t - .35) * 4; }
  if (k === 'topK') { const t = p.topK / 100; e.eyeH = .35 + t * .9; e.eyeW = .9 + t * .3; e.lx = -3 + t * 6; e.browR = -t * 8; e.browY = -t * 3; e.mouth = 1 + t * 2; }
  if (k === 'topP') { const t = p.topP; e.ly = 3; e.lx = -2 + t * 4; e.eyeH = .6 + t * .5; e.browY = -t * 4; e.browL = -t * 4; e.browR = -t * 4; e.mouth = 3 * t; }
  if (k === 'minP') { const t = p.minP / .5; e.eyeH = 1 - t * .65; e.browY = 2 + t * 4; e.browL = 6 + t * 8; e.browR = -6 - t * 8; e.ly = 3; e.lx = 3; e.mouth = 0; e.mouthW = .8 - t * .3; }
  if (k === 'repeat') { const t = p.repeat - 1; e.eyeH = .8 - t * .4; e.browL = -t * 14; e.browR = t * 14; e.browY = 2; e.mouth = -t * 3; e.mouthW = 1 - t * .4; }
  if (k === 'seed') { e.lx = 3; e.ly = -3; e.browR = -10; e.browY = -2; e.eyeH = 1.15; e.mouth = 1; e.mouthW = .6; }
  return e;
}

function Face({ p, active }: { p: P; active: Key | null }) {
  const [blink, setBlink] = useState(false);
  const [look, setLook] = useState([0, 0]);
  const [tick, setTick] = useState(0);
  useEffect(() => { let id: number; const go = () => { id = window.setTimeout(() => { setBlink(true); setTimeout(() => setBlink(false), 120); if (Math.random() < .5) setLook([(Math.random() - 0.5) * 3, (Math.random() - 0.5) * 1.6]); go(); }, 2400 + Math.random() * 3600); }; go(); return () => clearTimeout(id); }, []);
  const e = expr(active, p);
  useEffect(() => { if (!e.wob) return; let r = 0; const f = () => { setTick(t => t + 1); r = requestAnimationFrame(f); }; r = requestAnimationFrame(f); return () => cancelAnimationFrame(r); }, [e.wob > 0]);
  const w = (i: number) => e.wob ? Math.sin(tick * 0.12 + i * 1.7) * e.wob : 0;
  const head = `M${100 + w(1)} ${40 + w(2)} C ${140 + w(3)} 38, 158 ${70 + w(4)}, ${156 + w(5)} 110 C 155 ${150 + w(6)}, ${135 + w(7)} 178, 104 180 C ${72 + w(8)} 182, 48 ${158 + w(9)}, ${46 + w(1)} 118 C 44 ${78 + w(2)}, ${64 + w(3)} 42, ${100 + w(1)} ${40 + w(2)} Z`;
  const lx = active ? e.lx : look[0], ly = active ? e.ly : look[1];
  const eh = blink && !active ? .08 : e.eyeH;
  const sp = 'transition-all duration-500 [transition-timing-function:cubic-bezier(.34,1.45,.64,1)]';
  const m = e.mouth, mw = 13 * e.mouthW;
  return (
    <svg viewBox="20 20 160 180" className="w-full select-none overflow-visible">
      {active === 'repeat' && [1, 2, 3].map(i => <path key={i} d={head} fill="none" stroke="#141414" strokeWidth=".6" opacity={.35 / i} className={sp} transform={`translate(${(p.repeat - 1) * 9 * i} 0)`} />)}
      <g stroke="#141414" strokeWidth="1.5" fill="none" strokeLinecap="round" strokeLinejoin="round">
        <path d={head} fill="#f4f1ea" />
        <path d="M70 64 C 80 56, 92 55, 100 59" strokeWidth=".8" opacity=".6" />
        <g className={sp} style={{ transform: `translateY(${e.browY}px)` }}>
          <line x1="84" y1="82" x2="97" y2="82" className={sp} style={{ transformOrigin: '90px 82px', transform: `rotate(${e.browL}deg)` }} />
          <line x1="112" y1="81" x2="125" y2="81" className={sp} style={{ transformOrigin: '118px 81px', transform: `rotate(${e.browR}deg)` }} />
        </g>
        {[[91, 96], [119, 95]].map(([x, y], i) => <ellipse key={i} cx={x} cy={y} rx="3.2" ry="3.2" fill="#141414" stroke="none" className="transition-all duration-200 ease-out" style={{ transformOrigin: `${x}px ${y}px`, transform: `translate(${lx}px,${ly}px) scale(${e.eyeW},${eh})` }} />)}
        <path d={`M104 100 C 108 112, ${114 + w(3)} 124, 118 130 C 114 134, 108 134, 103 132`} />
        <path d={`M${105 - mw} 152 Q 105 ${152 + m} ${105 + mw} 151`} className={sp} />
        <path d="M48 112 C 40 108, 38 124, 48 128" />
      </g>
    </svg>
  );
}

function Num({ k, v, set, reset }: { k: string; v: number; set: (n: number) => void; reset: () => void }) {
  const [mn, mx, st] = RANGE[k];
  return <div className="flex items-center gap-2 text-[11px]">
    <input type="number" min={mn} max={mx} step={st} value={fmt(k, v)} onChange={e => { const n = parseFloat(e.target.value); if (!isNaN(n)) set(Math.min(mx, Math.max(mn, n))); }} aria-label={k} />
    <button onClick={reset} className="opacity-50 hover:opacity-100 hover:text-[#ff3b1f]" title="reset">↺</button>
  </div>;
}

function Tag({ n, k, label, v, set, reset, off }: { n: string; k: string; label: string; v: number; set: (n: number) => void; reset: () => void; off?: boolean }) {
  return <div className="tip flex items-center gap-1 text-[9px] uppercase tracking-[.15em]" tabIndex={0}>
    <span className="opacity-40">{n}</span><span>{label}</span>{off && <span className="text-[#ff3b1f]">n/a</span>}
    <Num k={k} v={v} set={set} reset={reset} />
    <span className="tt normal-case tracking-normal">{INFO[k]}</span>
  </div>;
}

function Label({ n, k, label, disabled }: { n: string; k: string; label: string; disabled?: boolean }) {
  return <div className="tip flex items-baseline gap-2 text-[10px] uppercase tracking-[.18em]" tabIndex={0}>
    <span className="opacity-40">{n}</span><span>{label}</span>{disabled && <span className="text-[#ff3b1f]">n/a</span>}
    <span className="tt normal-case tracking-normal">{INFO[k]}</span>
  </div>;
}

export default function App() {
  const [p, setP] = useState<P>(() => { try { return { ...DEF, ...JSON.parse(localStorage.getItem('alm-cur') || '{}') }; } catch { return DEF; } });
  const [active, setActive] = useState<Key | null>(null);
  const [backend, setBackend] = useState('demo');
  const [url, setUrl] = useState(ADAPTERS.demo.url);
  const [model, setModel] = useState('');
  const [models, setModels] = useState<string[]>([]);
  const [status, setStatus] = useState<'idle' | 'ok' | 'err' | 'wait'>('ok');
  const [presets, setPresets] = useState<Record<string, P>>(() => JSON.parse(localStorage.getItem('alm-presets') || '{"Precise":{"temperature":0.2,"topK":10,"topP":0.8,"minP":0.1,"repeat":1.05,"seed":1,"ctx":4096,"maxTok":256,"freq":0,"pres":0,"stream":true,"mirostat":false}}'));
  const [prompt, setPrompt] = useState('Describe a small machine that dreams.');
  const [out, setOut] = useState('');
  const ad = ADAPTERS[backend];
  const sup = (k: string) => ad.supports.includes(k);
  useEffect(() => { localStorage.setItem('alm-cur', JSON.stringify(p)); }, [p]);
  useEffect(() => { localStorage.setItem('alm-presets', JSON.stringify(presets)); }, [presets]);
  const set = (k: keyof P) => (v: number | boolean) => setP(o => ({ ...o, [k]: typeof v === 'number' && RANGE[k as string][2] >= 1 ? Math.round(v) : v }));
  const reset = (k: keyof P) => () => setP(o => ({ ...o, [k]: DEF[k] }));
  const D = useMemo(() => dist(p), [p]);
  const dragProps = (k: Key) => [p[k] as number, set(k) as (v: number) => void, () => setActive(k), () => setActive(null)] as const;
  const dT = useDrag('temperature', ...dragProps('temperature'));
  const dS = useDrag('seed', ...dragProps('seed'));

  const connect = async (b = backend, u = url) => {
    if (b === 'demo') { setStatus('ok'); setModels(['little-mind-sim']); setModel('little-mind-sim'); return; }
    setStatus('wait');
    try {
      const r = await fetch(b === 'ollama' ? `${u}/api/tags` : `${u}/models`);
      const j = await r.json();
      const ms: string[] = b === 'ollama' ? j.models.map((m: { name: string }) => m.name) : j.data.map((m: { id: string }) => m.id);
      setModels(ms); setModel(ms[0] || ''); setStatus('ok');
    } catch { setStatus('err'); setModels([]); }
  };
  useEffect(() => { connect('demo', ''); }, []);

  const generate = async () => {
    setOut('');
    if (backend === 'demo') {
      const r = rng(p.seed + Math.round(p.temperature * 100)); let s = ''; const n = Math.min(40, Math.round(p.maxTok / 8));
      for (let i = 0; i < n; i++) { const el = dist(p).filter(d => d.ok); const tot = el.reduce((a, b) => a + b.v, 0); let x = r() * tot; const pick = el.find(d => (x -= d.v) <= 0) || el[0]; s += (i ? ' ' : '') + pick.tok; setOut('[simulated] ' + s); await new Promise(res => setTimeout(res, 40)); }
      return;
    }
    try {
      const body = ad.map(p);
      const r = backend === 'ollama'
        ? await fetch(`${url}/api/generate`, { method: 'POST', body: JSON.stringify({ model, prompt, ...body }) })
        : await fetch(`${url}/chat/completions`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ model, messages: [{ role: 'user', content: prompt }], ...body }) });
      const j = await r.json();
      setOut(backend === 'ollama' ? j.response : j.choices?.[0]?.message?.content ?? JSON.stringify(j));
    } catch (e) { setOut('Connection failed: ' + (e as Error).message); setStatus('err'); }
  };

  const hit = 'cursor-ns-resize touch-none';
  const pulse = (k: Key) => active === k ? 'scale-105' : 'hover:scale-[1.03]';
  const box = (k: string) => `flex flex-col gap-2 transition-opacity ${sup(k) ? '' : 'opacity-30 pointer-events-none'}`;

  return (
    <div className="min-h-screen overflow-x-auto px-6 md:px-12 py-6 max-w-[1400px] mx-auto">
      <header className="flex flex-wrap items-center justify-between gap-4 text-[10px] uppercase tracking-[.2em]">
        <div className="flex items-baseline gap-3"><span className="disp italic text-3xl normal-case tracking-normal">a little mind</span><span className="opacity-40">model no. ALM-01 / sampling instrument</span></div>
        <div className="flex flex-wrap items-center gap-3">
          <select value={backend} onChange={e => { const b = e.target.value; setBackend(b); setUrl(ADAPTERS[b].url); setStatus('idle'); if (b === 'demo') connect('demo', ''); }} className="bg-transparent border-b border-black outline-none">
            {Object.entries(ADAPTERS).map(([k, a]) => <option key={k} value={k}>{a.name}</option>)}
          </select>
          {backend !== 'demo' && <input value={url} onChange={e => setUrl(e.target.value)} className="bg-transparent border-b border-black w-48 normal-case outline-none" />}
          {backend !== 'demo' && <button onClick={() => connect()} className="border border-black px-2 py-1 hover:bg-black hover:text-[#efece4]">connect</button>}
          {models.length > 0 && <select value={model} onChange={e => setModel(e.target.value)} className="bg-transparent border-b border-black normal-case outline-none">{models.map(m => <option key={m}>{m}</option>)}</select>}
          <span className="flex items-center gap-1"><span className={`w-2 h-2 rounded-full ${status === 'ok' ? 'bg-[#22e04a] shadow-[0_0_8px_#22e04a]' : status === 'err' ? 'bg-[#ff3b1f]' : status === 'wait' ? 'bg-[#3b5bff] animate-pulse' : 'bg-black/30'}`} />{backend === 'demo' ? 'simulated' : status}</span>
        </div>
      </header>

      <main className="relative mx-auto mt-6 w-full max-w-[1120px] min-w-[920px] aspect-[1120/640]">
        {/* silhouette: face chamber fused into a capsule */}
        {/* sculpted housing */}
        <svg viewBox="0 0 1120 640" className="absolute inset-0 w-full h-full pointer-events-none">
          <path fill="#e2dfd7" d="M250 70 L 407 70 A 172 172 0 1 0 633 70 L 870 70 C 990 70 1080 170 1080 300 C 1080 440 990 530 870 530 L 250 530 C 130 530 40 440 40 300 C 40 170 130 70 250 70 Z" />
          <path d="M407 70 A 172 172 0 1 0 633 70" fill="none" stroke="#141414" strokeOpacity=".18" />
        </svg>
        {/* gradient flares */}
        <div className="absolute left-[11%] top-[17%] w-[13%] aspect-square rounded-full blur-2xl opacity-45 pointer-events-none" style={{ background: 'conic-gradient(from 200deg,#22e04a,#2a4bff,#ff6fb5,#ff2a1f,#22e04a)' }} />
        <div className="absolute left-[66%] top-[57%] w-[27%] h-[12%] rounded-full bg-[#ff2a1f] blur-3xl opacity-30 pointer-events-none" />
        <div className="absolute left-[76%] top-[14%] w-[8%] h-[6%] rounded-full bg-[#22e04a] blur-2xl opacity-60 pointer-events-none" />
        <div className="absolute left-[33.04%] w-[26.8%] top-[7.8%] aspect-square rounded-full border border-black/15 pointer-events-none" />
        <svg viewBox="0 0 1120 640" className="absolute inset-0 pointer-events-none" fill="none" stroke="#141414" strokeWidth=".7">
          {/* connectors: everything speaks to the face (cx 520, cy 200) */}
          <path d="M330 170 C 370 120, 410 110, 440 130" strokeDasharray="2 4" opacity={active === 'temperature' ? 1 : .35} />
          <path d="M870 150 C 790 90, 720 110, 660 160" strokeDasharray="2 4" opacity={active === 'seed' ? 1 : .35} />
          <path d="M740 290 C 800 330, 840 350, 900 360" strokeDasharray="2 4" opacity={active === 'minP' ? 1 : .35} />
          <path d="M520 30 C 640 20, 760 40, 830 70" opacity=".25" />
          {Array.from({ length: 60 }, (_, i) => { const a = i / 60 * Math.PI * 2; const r = 158; return <line key={i} x1={520 + Math.cos(a) * r} y1={200 + Math.sin(a) * r} x2={520 + Math.cos(a) * (r + (i % 5 ? 3 : 8))} y2={200 + Math.sin(a) * (r + (i % 5 ? 3 : 8))} opacity=".4" />; })}
          {[[400, 40, 2], [985, 250, 3], [60, 360, 2], [1050, 540, 2]].map(([x, y, r], i) => <circle key={i} cx={x} cy={y} r={r} />)}
          <text x="70" y="300" fontSize="8" fill="#141414" stroke="none" opacity=".4" letterSpacing="2">ALM—01</text>
          <text x="595" y="40" fontSize="8" fill="#141414" stroke="none" opacity=".4" letterSpacing="2">fig. subject / readout</text>
        </svg>

        {/* face */}
        <div className="absolute left-[37%] w-[18.8%] top-[10%]"><Face p={p} active={active} /></div>
        <div className="absolute left-[44%] top-[2%] text-[8px] uppercase tracking-[.25em] opacity-40">readout · {active ?? 'idle'}</div>

        {/* seed constellation, perched on the chamber */}
        <div className={`absolute left-[74%] top-[16%] flex flex-col items-center gap-1 ${box('seed')}`}>
          <svg viewBox="-25 -25 50 50" className={`w-12 ${hit} transition-transform ${pulse('seed')}`} {...dS}>
            <circle r="23" fill="none" stroke="#141414" strokeWidth=".4" strokeDasharray="1 2" />
            {(() => { const r = rng(p.seed + 7); const s = Array.from({ length: 6 }, () => [r() * 34 - 17, r() * 34 - 17]); return s.map(([x, y], i) => <g key={i}><circle cx={x} cy={y} r={i === 0 ? 2.6 : 1.5} fill={i === 0 ? '#ff6fb5' : '#141414'} className="transition-all duration-500" />{i > 0 && <line x1={s[i - 1][0]} y1={s[i - 1][1]} x2={x} y2={y} stroke="#141414" strokeWidth=".5" />}</g>); })()}
          </svg>
          <div className="flex flex-col gap-1">
            <Tag n="06" k="seed" label="seed" v={p.seed} set={set('seed')} reset={reset('seed')} />
            <button onClick={() => { setActive('seed'); setP(o => ({ ...o, seed: Math.floor(Math.random() * 99999) })); setTimeout(() => setActive(null), 800); }} className="self-start text-[8px] uppercase tracking-[.2em] border-b border-black hover:text-[#ff6fb5]">randomize</button>
          </div>
        </div>

        {/* temperature: energy dial */}
        <div className={`absolute left-[9%] top-[13%] flex flex-col items-start gap-2 ${box('temperature')}`}>
          <div className={`relative w-44 h-44 ${hit} transition-transform duration-300 ${pulse('temperature')}`} {...dT}>
            <div className="absolute inset-2 rounded-full blur-xl transition-opacity duration-500" style={{ background: 'radial-gradient(circle,#ff6fb5,transparent 70%)', opacity: active === 'temperature' ? .2 + p.temperature * .25 : .1 }} />
            <svg viewBox="0 0 100 100" className="relative overflow-visible">
              {Array.from({ length: 41 }, (_, i) => { const a = (135 + i * 6.75) * Math.PI / 180; const on = i / 40 <= p.temperature / 2; return <line key={i} x1={50 + Math.cos(a) * 44} y1={50 + Math.sin(a) * 44} x2={50 + Math.cos(a) * (i % 10 ? 47 : 50)} y2={50 + Math.sin(a) * (i % 10 ? 47 : 50)} stroke={on ? '#141414' : '#14141433'} strokeWidth=".8" />; })}
              <circle cx="50" cy="50" r="36" fill="#141414" />
              <clipPath id="tc"><circle cx="50" cy="50" r="34" /></clipPath>
              <g clipPath="url(#tc)" fill="none" strokeWidth=".7">
                {[0, 1, 2, 3, 4, 5].map(j => { const r = rng(j * 31 + Math.round(p.temperature * 40)); let d = ''; for (let i = 0; i <= 30; i++) { const x = 14 + i * 2.4; const y = 30 + j * 8 + Math.sin(i * (.4 + p.temperature * .5) + j) * p.temperature * 3 + (r() - .5) * p.temperature * 5; d += `${i ? 'L' : 'M'}${x} ${y}`; } return <path key={j} d={d} stroke={j === 2 ? '#22e04a' : '#efece4'} opacity={j === 2 ? 1 : .55} className="transition-all duration-300" />; })}
              </g>
              {(() => { const a = (135 + p.temperature / 2 * 270) * Math.PI / 180; return <circle cx={50 + Math.cos(a) * 40} cy={50 + Math.sin(a) * 40} r="3.2" fill="#ff6fb5" stroke="#141414" strokeWidth=".8" />; })()}
            </svg>
          </div>
          <div className="flex flex-col gap-1">
            <span className="text-[8px] uppercase tracking-[.25em] opacity-50">{p.temperature < .4 ? 'ordered' : p.temperature < 1.1 ? 'lively' : 'turbulent'}</span>
            <span className="disp text-3xl tabular-nums leading-none">{p.temperature.toFixed(2)}</span>
            <Tag n="01" k="temperature" label="temp" v={p.temperature} set={set('temperature')} reset={reset('temperature')} />
          </div>
        </div>

        {/* 04 min-p: ranked tokens with a draggable floor */}
        <Constraint p={p} D={D} set={set} reset={reset} sup={sup} setActive={setActive} />

        <button onClick={generate} className="absolute left-[63%] top-[70%] w-12 h-12 rounded-full bg-[#141414] text-[#efece4] text-[8px] uppercase tracking-widest hover:scale-110 transition-transform shadow-[0_0_0_5px_#ff6fb5,0_0_0_8px_#22e04a,0_0_28px_8px_#ff6fb588]" title="Generate">run</button>
        <ShareTrack p={p} setActive={setActive} set={set} sup={sup} reset={reset} active={active} />
      </main>

      <section className="grid grid-cols-1 lg:grid-cols-[1fr_1.3fr] gap-10 mt-10 pt-4 max-w-[1180px] mx-auto text-[11px]">
        <div className="flex flex-col gap-4">
          <div className="text-[10px] uppercase tracking-[.2em] opacity-60">secondary · {ad.name}</div>
          <div className="grid grid-cols-2 gap-x-6 gap-y-4">
            {(['ctx', 'maxTok', 'freq', 'pres'] as const).map((k, i) => <div key={k} className={box(k)}>
              <Label n={`0${7 + i}`} k={k} label={{ ctx: 'Context', maxTok: 'Max tokens', freq: 'Freq. pen.', pres: 'Pres. pen.' }[k]} disabled={!sup(k)} />
              <input type="range" min={RANGE[k][0]} max={RANGE[k][1]} step={RANGE[k][2]} value={p[k]} onChange={e => set(k)(parseFloat(e.target.value))} className="accent-[#ff3b1f]" />
              <Num k={k} v={p[k]} set={set(k)} reset={reset(k)} />
            </div>)}
          </div>
          <div className="flex gap-6 text-[10px] uppercase tracking-[.15em]">
            {(['stream', 'mirostat'] as const).map(k => <label key={k} className={`flex items-center gap-2 ${sup(k) ? '' : 'opacity-30'}`}><input type="checkbox" disabled={!sup(k)} checked={p[k]} onChange={e => set(k)(e.target.checked)} className="accent-[#22e04a]" />{k}</label>)}
          </div>
          <div className="flex flex-wrap items-center gap-2 text-[10px] uppercase tracking-[.15em]">
            <span className="opacity-60">presets</span>
            {Object.keys(presets).map(n => <span key={n} className="flex border border-black"><button onClick={() => setP({ ...DEF, ...presets[n] })} className="px-2 py-1 hover:bg-[#3b5bff] hover:text-white">{n}</button><button onClick={() => setPresets(o => { const c = { ...o }; delete c[n]; return c; })} className="px-1 opacity-40 hover:opacity-100">×</button></span>)}
            <button onClick={() => { const n = prompt_('Preset name'); if (n) setPresets(o => ({ ...o, [n]: p })); }} className="px-2 py-1 border border-dashed border-black">+ save</button>
            <button onClick={() => setP(DEF)} className="px-2 py-1 opacity-60 hover:opacity-100">reset all</button>
          </div>
        </div>
        <div className="flex flex-col gap-3">
          <div className="text-[10px] uppercase tracking-[.2em] opacity-60">test bench {backend === 'demo' && <span className="text-[#ff3b1f]">· simulated sampling over sample tokens</span>}</div>
          <div className="flex gap-3"><textarea value={prompt} onChange={e => setPrompt(e.target.value)} rows={2} className="flex-1 bg-transparent border-b border-black outline-none text-sm resize-none" />
            </div>
          <pre className="whitespace-pre-wrap text-sm min-h-[80px] disp text-lg leading-snug">{out || <span className="opacity-30">output appears here.</span>}</pre>
        </div>
      </section>
    </div>
  );
}
function Constraint({ p, D, set, reset, sup, setActive }: { p: P; D: { tok: string; v: number; ok: boolean }[]; set: (k: keyof P) => (v: number | boolean) => void; reset: (k: keyof P) => () => void; sup: (k: string) => boolean; setActive: (k: Key | null) => void }) {
  const ref = useRef<HTMLDivElement>(null);
  const ghost = Object.fromEntries(dist({ ...p, repeat: 1 }).map(d => [d.tok, d.v]));
  const ranked = [...D].sort((a, b) => b.v - a.v);
  const top = Math.max(ranked[0].v, ...Object.values(ghost));
  const at = (y: number) => { const r = ref.current!.getBoundingClientRect(); set('minP')(Math.round(Math.min(.5, Math.max(0, 1 - (y - r.top) / r.height)) * 100) / 100); };
  const down = (e: React.PointerEvent) => { if (!sup('minP')) return; e.preventDefault(); setActive('minP'); at(e.clientY); const mv = (ev: PointerEvent) => at(ev.clientY); const up = () => { setActive(null); window.removeEventListener('pointermove', mv); window.removeEventListener('pointerup', up); }; window.addEventListener('pointermove', mv); window.addEventListener('pointerup', up); };
  const floor = p.minP;
  const isUsed = (t: string) => USED.includes(TOKENS.indexOf(t));
  const isCut = (v: number) => v < p.minP * ranked[0].v;
  const cut = ranked.filter(d => isCut(d.v)).length;
  return <div className="absolute right-[7%] w-[27%] top-[50%] flex flex-col gap-1">
    <div className="flex justify-between text-[8px] uppercase tracking-[.2em]"><span className="opacity-50">constraint · filter + suppress</span><span className="text-[#ff6fb5]">simulated</span></div>
    <div ref={ref} onPointerDown={down} className={`relative h-24 touch-none ${sup('minP') ? 'cursor-ns-resize' : ''}`}>
      <div className="absolute inset-x-0 bottom-0 h-px bg-[#141414]" />
      <div className="absolute inset-x-0 bottom-0 bg-[#ff6fb5]/15 transition-all duration-150" style={{ height: `${floor * 100}%` }} />
      <div className="absolute inset-0 flex items-end">
        {ranked.map(d => { const rel = d.v / ranked[0].v, g = Math.min(1, ghost[d.tok] / ranked[0].v), used = isUsed(d.tok), below = isCut(d.v); return <div key={d.tok} className="relative flex-1 h-full flex flex-col justify-end items-center">
          {used && <div className="absolute bottom-0 w-[5px] border border-dashed border-[#2a4bff]/60 transition-all duration-500" style={{ height: `${g * 100}%` }} />}
          <span className={`w-[6px] h-[6px] -mb-[3px] rounded-full z-10 transition-colors ${below ? 'border border-[#141414]/40 bg-[#e2dfd7]' : used ? 'bg-[#2a4bff]' : d.ok ? 'bg-[#141414]' : 'bg-[#141414]/30'}`} />
          <div className="w-px transition-all duration-500" style={{ height: `${rel * 100}%`, background: below ? '#14141433' : used ? '#2a4bff' : '#141414' }} />
        </div>; })}
      </div>
      <div className="absolute inset-x-0 transition-[bottom] duration-150 pointer-events-none" style={{ bottom: `${floor * 100}%` }}>
        <div className="border-t border-[#ff2a1f]" />
        <span className="absolute -left-2 -top-[7px] w-3.5 h-3.5 rounded-full bg-[#ff6fb5] border border-[#141414]" />
        <span className="absolute right-0 -top-3.5 text-[7px] uppercase tracking-[.15em] text-[#ff2a1f]">floor</span>
      </div>
    </div>
    <div className="flex gap-[1px]">{ranked.map(d => <span key={d.tok} className={`flex-1 text-[6.5px] text-center truncate ${isCut(d.v) ? 'opacity-30 line-through' : ''} ${isUsed(d.tok) ? 'text-[#2a4bff]' : ''}`}>{d.tok}</span>)}</div>
    <div className="grid grid-cols-2 gap-4 mt-1">
      <div className={`flex flex-col gap-1 ${sup('minP') ? '' : 'opacity-30 pointer-events-none'}`}>
        <div className="text-[7px] uppercase tracking-[.15em] opacity-60"><span className="inline-block w-2 h-2 rounded-full bg-[#ff6fb5] border border-black mr-1 align-middle" />drag plot · cut weak <span className="text-[#ff2a1f]">{cut}</span></div>
        <Tag n="04" k="minP" label="min-p" v={p.minP} set={set('minP') as (n: number) => void} reset={reset('minP')} off={!sup('minP')} />
      </div>
      <div className={`flex flex-col gap-1 ${sup('repeat') ? '' : 'opacity-30 pointer-events-none'}`}>
        <div className="text-[7px] uppercase tracking-[.15em] opacity-60"><span className="inline-block w-2 h-2 border border-dashed border-[#2a4bff] mr-1 align-middle" />already said · damped</div>
        <HSlider k="repeat" v={p.repeat} set={set('repeat') as (n: number) => void} onStart={() => setActive('repeat')} onEnd={() => setActive(null)} />
        <Tag n="05" k="repeat" label="repeat" v={p.repeat} set={set('repeat')} reset={reset('repeat')} off={!sup('repeat')} />
      </div>
    </div>
  </div>;
}

function HSlider({ k, v, set, onStart, onEnd }: { k: string; v: number; set: (n: number) => void; onStart: () => void; onEnd: () => void }) {
  const ref = useRef<HTMLDivElement>(null);
  const [mn, mx] = RANGE[k]; const t = (v - mn) / (mx - mn);
  const at = (x: number) => { const r = ref.current!.getBoundingClientRect(); set(mn + Math.min(1, Math.max(0, (x - r.left) / r.width)) * (mx - mn)); };
  const down = (e: React.PointerEvent) => { e.preventDefault(); onStart(); at(e.clientX); const mv = (ev: PointerEvent) => at(ev.clientX); const up = () => { onEnd(); window.removeEventListener('pointermove', mv); window.removeEventListener('pointerup', up); }; window.addEventListener('pointermove', mv); window.addEventListener('pointerup', up); };
  return <div ref={ref} onPointerDown={down} className="relative h-5 cursor-ew-resize touch-none">
    <div className="absolute inset-x-0 top-1/2 h-px bg-[#141414]" />
    <div className="absolute left-0 top-1/2 -mt-[1px] h-[3px] bg-[#141414]" style={{ width: `${t * 100}%` }} />
    <span className="absolute top-1/2 w-3.5 h-3.5 -mt-[7px] -ml-[7px] rounded-full bg-[#141414] ring-2 ring-[#e2dfd7] hover:scale-125 transition-transform" style={{ left: `${t * 100}%` }} />
  </div>;
}
const CX = 520, CY = 200, RK = 196, RP = 222, A0 = 165, A1 = 15;
const ang = (t: number) => (A0 - t * (A0 - A1)) * Math.PI / 180;
const pt = (t: number, r: number) => [CX + Math.cos(ang(t)) * r, CY + Math.sin(ang(t)) * r];
const arc = (t0: number, t1: number, r: number) => { const [x0, y0] = pt(t0, r), [x1, y1] = pt(t1, r); return `M${x0} ${y0} A ${r} ${r} 0 ${(t1 - t0) * (A0 - A1) > 180 ? 1 : 0} 0 ${x1} ${y1}`; };
function ShareTrack({ p, set, setActive, sup, reset, active }: { p: P; set: (k: keyof P) => (v: number | boolean) => void; setActive: (k: Key | null) => void; sup: (k: string) => boolean; reset: (k: keyof P) => () => void; active: Key | null }) {
  const ref = useRef<SVGSVGElement>(null);
  const kt = (p.topK - 1) / 99, pp = p.topP;
  const loc = (cx: number, cy: number) => { const r = ref.current!.getBoundingClientRect(); const sc = 1120 / r.width; const x = (cx - r.left) * sc - CX, y = (cy - r.top) * sc - CY; let a = Math.atan2(y, x) * 180 / Math.PI; if (a < -90) a += 360; return { t: Math.min(1, Math.max(0, (A0 - a) / (A0 - A1))), d: Math.hypot(x, y) }; };
  const setRef = useRef(set); setRef.current = set;
  const pRef = useRef(p); pRef.current = p;
  type H = 'kHi' | 'kLo' | 'pHi' | 'pLo';
  const handles: Record<H, { t: number; r: number; key: 'topK' | 'topP' }> = { kHi: { t: kt, r: RK, key: 'topK' }, kLo: { t: (p.kLo - 1) / 99, r: RK, key: 'topK' }, pHi: { t: pp, r: RP, key: 'topP' }, pLo: { t: p.pLo, r: RP, key: 'topP' } };
  const grab = useRef<H | null>(null);
  const applyH = (h: H, t: number) => {
    const S = setRef.current, cur = pRef.current;
    if (h === 'kHi') S('topK')(Math.max(cur.kLo, 1 + t * 99));
    if (h === 'kLo') S('kLo')(Math.min(cur.topK, 1 + t * 99));
    if (h === 'pHi') S('topP')(Math.max(cur.pLo, Math.round(t * 100) / 100));
    if (h === 'pLo') S('pLo')(Math.min(cur.topP, Math.round(t * 100) / 100));
  };
  const down = (e: React.PointerEvent) => {
    e.preventDefault();
    const r = ref.current!.getBoundingClientRect(), sc = 1120 / r.width;
    const mx = (e.clientX - r.left) * sc, my = (e.clientY - r.top) * sc;
    const { t, d } = loc(e.clientX, e.clientY);
    // nearest handle within 30 units wins; otherwise nearest handle in the clicked lane
    let best: H | null = null, bd = 1e9;
    (Object.keys(handles) as H[]).forEach(h => { const H0 = handles[h]; if (!sup(H0.key)) return; const [x, y] = pt(H0.t, H0.r); const dd = Math.hypot(x - mx, y - my) - (h.endsWith('Hi') ? .5 : 0); if (dd < bd) { bd = dd; best = h; } });
    if (!best || bd > 30) { const lane = d < (RK + RP) / 2 ? 'k' : 'p'; if (!sup(lane === 'k' ? 'topK' : 'topP')) return; const lo = handles[`${lane}Lo` as H].t, hi = handles[`${lane}Hi` as H].t; best = (Math.abs(t - lo) < Math.abs(t - hi) ? `${lane}Lo` : `${lane}Hi`) as H; }
    const h = best as H; grab.current = h; setActive(handles[h].key); applyH(h, t);
    const mv = (ev: PointerEvent) => applyH(h, loc(ev.clientX, ev.clientY).t);
    const up = () => { grab.current = null; setActive(null); window.removeEventListener('pointermove', mv); window.removeEventListener('pointerup', up); window.removeEventListener('pointercancel', up); };
    window.addEventListener('pointermove', mv); window.addEventListener('pointerup', up); window.addEventListener('pointercancel', up);
  };
  const node = (t: number, r: number, c: string, filled: boolean, on: boolean) => { const [x, y] = pt(t, r); return <g pointerEvents="none">{<circle cx={x} cy={y} r={on ? 13 : 10} fill="none" stroke={c} strokeWidth=".7" strokeDasharray="2 2" opacity={on ? 1 : .55} />}<circle cx={x} cy={y} r={filled ? (on ? 8 : 6.5) : (on ? 6.5 : 5)} fill={filled ? c : '#e2dfd7'} stroke={c} strokeWidth="1.6" style={{ filter: filled && on ? `drop-shadow(0 0 6px ${c})` : 'none' }} /></g>; };
  const [lk, lp] = [pt(1, RK - 26), pt(1, RP + 22)];
  return (
    <>
      <svg ref={ref} viewBox="0 0 1120 640" className="absolute inset-0 w-full h-full pointer-events-none">
        <g className="cursor-grab active:cursor-grabbing touch-none" onPointerDown={down}>
          <path d={arc(-.03, 1.03, (RK + RP) / 2)} stroke="#000" strokeOpacity="0" strokeWidth="64" fill="none" pointerEvents="stroke" />
          <path d={arc(0, 1, (RK + RP) / 2)} stroke="#141414" strokeWidth=".8" fill="none" />
          {Array.from({ length: 21 }, (_, i) => { const [x0, y0] = pt(i / 20, (RK + RP) / 2 - (i % 5 ? 3 : 7)); const [x1, y1] = pt(i / 20, (RK + RP) / 2 + (i % 5 ? 3 : 7)); return <line key={i} x1={x0} y1={y0} x2={x1} y2={y1} stroke="#141414" strokeWidth=".6" />; })}
          <g opacity={sup('topK') ? 1 : .2}><path d={arc(0, Math.max(handles.kLo.t, .001), RK)} stroke="#2a4bff" strokeWidth="1" strokeDasharray="1 3" fill="none" /><path d={arc(handles.kLo.t, Math.max(kt, handles.kLo.t + .001), RK)} stroke="#2a4bff" strokeWidth="3" fill="none" />{node(handles.kLo.t, RK, '#2a4bff', false, grab.current === 'kLo')}{node(kt, RK, '#2a4bff', true, grab.current === 'kHi')}</g>
          <g opacity={sup('topP') ? 1 : .2}><path d={arc(0, Math.max(p.pLo, .001), RP)} stroke="#ff2a1f" strokeWidth="1" strokeDasharray="1 3" fill="none" /><path d={arc(p.pLo, Math.max(pp, p.pLo + .001), RP)} stroke="#ff2a1f" strokeWidth="3" fill="none" />{node(p.pLo, RP, '#ff2a1f', false, grab.current === 'pLo')}{node(pp, RP, '#ff2a1f', true, grab.current === 'pHi')}</g>
          <text x={lk[0] - 4} y={lk[1]} fontSize="8" fill="#2a4bff" letterSpacing="2" textAnchor="end">K · CANDIDATES</text>
          <text x={lp[0] + 4} y={lp[1] + 10} fontSize="8" fill="#ff2a1f" letterSpacing="2">P · MASS</text>
        </g>
      </svg>
      <div className="absolute left-[37%] top-[50%] flex flex-col items-start gap-1 text-[#2a4bff]">
        <span className="disp text-2xl tabular-nums leading-none">{p.topK}</span>
        <Tag n="02" k="topK" label="top-k" v={p.topK} set={set('topK') as (n: number) => void} reset={reset('topK')} off={!sup('topK')} />
        <Tag n="○" k="kLo" label="keep" v={p.kLo} set={set('kLo') as (n: number) => void} reset={reset('kLo')} />
      </div>
      <div className="absolute left-[47%] top-[74%] flex flex-col items-start gap-1 text-[#ff2a1f]">
        <span className="disp text-2xl tabular-nums leading-none">{p.topP.toFixed(2)}</span>
        <Tag n="03" k="topP" label="top-p" v={p.topP} set={set('topP') as (n: number) => void} reset={reset('topP')} off={!sup('topP')} />
        <Tag n="○" k="pLo" label="floor" v={p.pLo} set={set('pLo') as (n: number) => void} reset={reset('pLo')} />
      </div>
    </>
  );
}
const prompt_ = (m: string) => window.prompt(m);
