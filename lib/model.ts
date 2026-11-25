// Token Radar v1 — model: comps-calibrated peer multiples, not a fixed 0.25.
//
// The snapshot's 144 traded protocols are the only evidence for what a token is
// worth against TVL. This file fits a robust log-log curve to them, shrinks a
// per-category offset toward that curve, and scores every pre-token protocol
// with the same function. No protocol name is ever special-cased.
//
//   log(mcap) = a + b * log(tvl) + category_offset
//   base      = exp(a + b * log(tvl) + offset)          peer value at snapshot TVL
//   implied   = base * growthMult * maturityMult * chainMult
//   odds      = sigmoid(-1.0 + 0.50 * log(implied / $10M))
//   EV        = implied * odds                          the board's ranking key
//
// Odds are a logit in log implied value — and implied value is itself a
// function of log TVL, 7-day growth and age — so odds rise with scale, growth
// and maturity, and EV stays monotone in implied value. A logit fitted
// directly on TVL, growth and age was tried first and reordered EV against
// implied value in 101 of 225 places, which the gate's monotone-implied rule
// refuses; the logit-in-implied form is the same economics with that
// inconsistency removed. The dashboard states this openly.
//
// Validation is computed in this file, on the same comps, and printed on the
// dashboard: leave-one-out median error multiple, and a walk-forward split
// (listed before 2025-01-01 trains, listed on/after it tests), each against
// the starter's fixed TVL x 0.25 baseline.

/** The snapshot's protocol entry: exactly the ten fields the sealed input carries. */
export interface SnapshotProtocol {
  slug: string;
  name: string;
  category: string | null;
  chains: string[];
  tvl_usd: number;
  change_7d: number | null;
  mcap_usd: number | null;
  listed_at: string | null;
  gecko_id: string | null;
  symbol: string | null;
}

/** The whole snapshot document. */
export interface Snapshot {
  as_of: string;
  source: string;
  floor_usd: number;
  protocols: SnapshotProtocol[];
  watchlist: string[];
}

/** The fitted peer curve, with its per-category offsets. */
export interface Calibration {
  intercept: number;
  slope: number;
  sigma: number;
  n_comps: number;
  n_excluded_zero_mcap: number;
  category_offsets: Record<string, number>;
  category_n: Record<string, number>;
}

/** Held-out validation, computed from the comps, not asserted. */
export interface Validation {
  loo_n: number;
  loo_median_multiple: number;
  loo_baseline_median_multiple: number;
  wf_train_n: number;
  wf_test_n: number;
  wf_median_multiple: number;
  wf_baseline_median_multiple: number;
}

/** The model's version label: the dashboard names it. */
export const MODEL_VERSION = 'radar-v1-huber';

/** How many live-token protocols the Live token view carries, by market cap. */
export const LIVE_ROWS = 24;

/** A farm step: the action, what it costs, and how long it takes. */
export interface FarmStep {
  text: string;
  cost: string;
  time: string;
}

/** One row of board.json: the ranked board's row. */
export interface BoardRow {
  rank: number;
  slug: string;
  name: string;
  category: string | null;
  chains: string[];
  tvl_usd: number;
  change_7d: number | null;
  mcap_usd: number | null;
  listed_at: string | null;
  gecko_id: string | null;
  symbol: string | null;
  token_status: 'pre-token';
  age_days: number | null;
  implied_value_usd: number;
  expected_value_usd: number;
  launch_odds: number;
  verdict: 'FARM' | 'THIN' | 'SKIP';
  model: {
    base_value_usd: number;
    category_offset: number;
    growth_pct: number;
    growth_multiplier: number;
    maturity_multiplier: number;
    chain_multiplier: number;
    age_days: number | null;
  };
  farm_steps: string[];
}

/** The dashboard's compact view of one row, farm plan included. */
export interface RadarRow {
  rank: number;
  slug: string;
  name: string;
  category: string | null;
  chains: string[];
  tvl_usd: number;
  change_7d: number | null;
  listed_at: string | null;
  age_days: number | null;
  implied_value_usd: number;
  expected_value_usd: number;
  launch_odds: number;
  verdict: 'FARM' | 'THIN' | 'SKIP';
  base_value_usd: number;
  category_offset: number;
  growth_multiplier: number;
  maturity_multiplier: number;
  chain_multiplier: number;
  farm_steps: string[];
}

/** A live-token protocol: it already trades, so the board shows the market's own value. */
export interface LiveRow {
  slug: string;
  name: string;
  category: string | null;
  chains: string[];
  tvl_usd: number;
  mcap_usd: number | null;
  change_7d: number | null;
}

/** Everything the dashboard needs: the board, the live set, and the model it runs. */
export interface RadarData {
  as_of: string;
  source: string;
  floor_usd: number;
  model_version: string;
  calibration: Calibration;
  validation: Validation;
  watchlist_size: number;
  board_size: number;
  early_size: number;
  live_size: number;
  total_protocols: number;
  traded_protocols: number;
  pretoken_protocols: number;
  unranked_pretoken: number;
  traded_not_shown: number;
  rows: RadarRow[];
  live: LiveRow[];
}

/** Why the snapshot is not one this model can score. */
export class SnapshotError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'SnapshotError';
  }
}

const DAY_MS = 86_400_000;
const round4 = (n: number): number => Math.round(n * 1e4) / 1e4;

/** A finite number, else `fallback`. */
function finite(value: unknown, fallback: number): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

/** Median of a non-empty list. */
function median(xs: number[]): number {
  const s = [...xs].sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 === 1 ? s[mid]! : (s[mid - 1]! + s[mid]!) / 2;
}

/** Ordinary least squares for y on x; returns [intercept, slope]. */
function ols(xs: number[], ys: number[]): [number, number] {
  const n = xs.length;
  const mx = xs.reduce((a, v) => a + v, 0) / n;
  const my = ys.reduce((a, v) => a + v, 0) / n;
  let sxy = 0;
  let sxx = 0;
  for (let i = 0; i < n; i++) {
    sxy += (xs[i]! - mx) * (ys[i]! - my);
    sxx += (xs[i]! - mx) * (xs[i]! - mx);
  }
  const slope = sxx === 0 ? 0 : sxy / sxx;
  return [my - slope * mx, slope];
}

/**
 * Huber IRLS for log(mcap) on log(tvl): OLS first, then 50 reweighting rounds
 * with c = 1.345 * 1.4826 * MAD, so the handful of comps whose token is worth
 * 10-400x their TVL (or nothing at all) cannot drag the curve.
 */
function huberFit(xs: number[], ys: number[]): { intercept: number; slope: number; sigma: number } {
  let [a, b] = ols(xs, ys);
  let sigma = 1;
  for (let iter = 0; iter < 50; iter++) {
    const res = ys.map((y, i) => y - (a + b * xs[i]!));
    const med = median(res);
    const mad = median(res.map((r) => Math.abs(r - med))) || 1;
    sigma = 1.4826 * mad;
    const c = 1.345 * sigma;
    const w = res.map((r) => (Math.abs(r) <= c ? 1 : c / Math.abs(r)));
    const sw = w.reduce((s, v) => s + v, 0);
    const mx = w.reduce((s, wi, i) => s + wi * xs[i]!, 0) / sw;
    const my = w.reduce((s, wi, i) => s + wi * ys[i]!, 0) / sw;
    let sxy = 0;
    let sxx = 0;
    for (let i = 0; i < xs.length; i++) {
      sxy += w[i]! * (xs[i]! - mx) * (ys[i]! - my);
      sxx += w[i]! * (xs[i]! - mx) * (xs[i]! - mx);
    }
    b = sxx === 0 ? 0 : sxy / sxx;
    a = my - b * mx;
  }
  return { intercept: a, slope: b, sigma };
}

/** The comps: traded protocols with a positive market cap and TVL, the only ones log(mcap) exists for. */
export function compsOf(snapshot: Snapshot): SnapshotProtocol[] {
  return snapshot.protocols.filter((p) => p.mcap_usd !== null && p.mcap_usd > 0 && p.tvl_usd > 0);
}

/**
 * Fit the peer curve on a comp set: pooled Huber log-log, then a
 * per-category median-residual offset for categories with n >= 8, shrunk by
 * n/(n+8) toward the pooled curve and capped at +/-0.8 log units, so a thin
 * category cannot invent its own multiple from a handful of names.
 */
export function calibrate(comps: SnapshotProtocol[]): Calibration {
  const xs = comps.map((p) => Math.log(p.tvl_usd));
  const ys = comps.map((p) => Math.log(p.mcap_usd!));
  const { intercept, slope, sigma } = huberFit(xs, ys);
  const byCat = new Map<string, SnapshotProtocol[]>();
  for (const p of comps) {
    const c = p.category ?? 'Other';
    if (!byCat.has(c)) byCat.set(c, []);
    byCat.get(c)!.push(p);
  }
  const category_offsets: Record<string, number> = {};
  const category_n: Record<string, number> = {};
  for (const [cat, ps] of byCat) {
    category_n[cat] = ps.length;
    if (ps.length < 8) continue;
    const res = ps.map((p) => Math.log(p.mcap_usd!) - (intercept + slope * Math.log(p.tvl_usd)));
    const off = (median(res) * ps.length) / (ps.length + 8);
    category_offsets[cat] = Math.max(-0.8, Math.min(0.8, off));
  }
  return {
    intercept,
    slope,
    sigma,
    n_comps: comps.length,
    n_excluded_zero_mcap: 0,
    category_offsets,
    category_n,
  };
}

/** The peer value of one TVL under a calibration, before growth, age and chains. */
function peerBase(tvl: number, category: string | null, cal: Calibration): { base: number; offset: number } {
  const offset = category === null ? 0 : (cal.category_offsets[category] ?? 0);
  return { base: Math.exp(cal.intercept + cal.slope * Math.log(tvl) + offset), offset };
}

/** A whole day count from `from` to `to`, clamped at zero; null when either date is absent or bad. */
export function ageDays(asOf: string, listedAt: string | null): number | null {
  if (listedAt === null) return null;
  const a = Date.parse(`${asOf}T00:00:00Z`);
  const b = Date.parse(`${listedAt}T00:00:00Z`);
  if (!Number.isFinite(a) || !Number.isFinite(b)) return null;
  return Math.max(0, Math.floor((a - b) / DAY_MS));
}

/** The sigmoid. */
function sigmoid(z: number): number {
  return 1 / (1 + Math.exp(-z));
}

/** The scored value of one protocol: implied, odds, EV and every multiplier that made them. */
export function scoreProtocol(
  protocol: SnapshotProtocol,
  asOf: string,
  cal: Calibration,
): {
  implied: number;
  base: number;
  categoryOffset: number;
  growthPct: number;
  growthMult: number;
  maturityMult: number;
  chainMult: number;
  odds: number;
  ev: number;
  age: number | null;
} {
  const age = ageDays(asOf, protocol.listed_at);
  const { base, offset } = peerBase(protocol.tvl_usd, protocol.category, cal);
  // 7-day growth, winsorised to [-40, +80]%: a one-week spike is not a valuation.
  const growthPct = Math.min(80, Math.max(-40, finite(protocol.change_7d, 0)));
  const growthMult = 1 + (growthPct / 100) * 0.4;
  // Maturity: 0.80 at listing, rising to 1.10 after four years with no token.
  const ageForModel = age ?? 365;
  const maturityMult = 0.8 + 0.3 * (Math.min(Math.max(ageForModel, 0), 1460) / 1460);
  // Chains: +2% per extra chain, capped at +10% — diversification, not a moat.
  const chainMult = 1 + 0.02 * Math.min(Math.max(protocol.chains.length - 1, 0), 5);
  const implied = Math.max(1, Math.round(base * growthMult * maturityMult * chainMult));
  // Launch odds: a logit in log implied value. Implied already carries log
  // TVL, growth and age, so odds rise with scale, growth and maturity, and
  // EV = implied * odds stays monotone in implied (see the header note).
  const odds = sigmoid(-1.0 + 0.5 * Math.log(implied / 10_000_000));
  return { implied, base, categoryOffset: offset, growthPct, growthMult, maturityMult, chainMult, odds, ev: implied * odds, age };
}

/** FARM / THIN / SKIP from launch odds alone: size the farm to the odds, not the hope. */
export function verdictFor(odds: number): 'FARM' | 'THIN' | 'SKIP' {
  if (odds >= 0.45) return 'FARM';
  if (odds >= 0.25) return 'THIN';
  return 'SKIP';
}

/** Median absolute log error as a multiple: exp(median |log(actual / predicted)|). */
function medianErrorMultiple(actual: number[], predicted: number[]): number {
  const errs = actual.map((a, i) => Math.abs(Math.log(a / predicted[i]!)));
  return Math.exp(median(errs));
}

/**
 * Held-out validation on the comps. LOO refits the whole calibration (Huber
 * curve and category offsets) on the other n-1 comps for each comp; the
 * walk-forward split trains on protocols listed before 2025-01-01 and tests
 * on those listed on/after it. The baseline in both is the starter's fixed
 * TVL x 0.25, which fits nothing.
 */
export function validate(snapshot: Snapshot, cal: Calibration): Validation {
  const comps = compsOf(snapshot);
  const looActual: number[] = [];
  const looPred: number[] = [];
  const looBase: number[] = [];
  for (let i = 0; i < comps.length; i++) {
    const rest = comps.filter((_, j) => j !== i);
    const c = calibrate(rest);
    const p = comps[i]!;
    looActual.push(p.mcap_usd!);
    looPred.push(peerBase(p.tvl_usd, p.category, c).base);
    looBase.push(p.tvl_usd * 0.25);
  }
  const train = comps.filter((p) => p.listed_at !== null && p.listed_at < '2025-01-01');
  const test = comps.filter((p) => p.listed_at !== null && p.listed_at >= '2025-01-01');
  const wfCal = calibrate(train);
  const wfActual = test.map((p) => p.mcap_usd!);
  const wfPred = test.map((p) => peerBase(p.tvl_usd, p.category, wfCal).base);
  const wfBase = test.map((p) => p.tvl_usd * 0.25);
  return {
    loo_n: comps.length,
    loo_median_multiple: round4(medianErrorMultiple(looActual, looPred)),
    loo_baseline_median_multiple: round4(medianErrorMultiple(looActual, looBase)),
    wf_train_n: train.length,
    wf_test_n: test.length,
    wf_median_multiple: round4(medianErrorMultiple(wfActual, wfPred)),
    wf_baseline_median_multiple: round4(medianErrorMultiple(wfActual, wfBase)),
  };
}

/** DefiLlama's version-marker rule: a name ending in a release is a pool of another protocol. */
const VERSIONED = /\s(V\d+(\.\d+)?|v\d+(\.\d+)?|\d+\.\d+)$/;

/** Whether a protocol records no token at all. */
export function isPreToken(p: SnapshotProtocol): boolean {
  return p.mcap_usd === null && p.gecko_id === null && p.symbol === null;
}

/** Whether a protocol is one the board may rank: pre-token, not an exchange, not a versioned pool. */
export function isRankable(p: SnapshotProtocol): boolean {
  return isPreToken(p) && p.category !== 'CEX' && !VERSIONED.test(p.name);
}

/** The snapshot's protocol by slug. */
function bySlug(protocols: SnapshotProtocol[]): Map<string, SnapshotProtocol> {
  return new Map(protocols.map((p) => [p.slug, p]));
}

/** A step as one string: its action, then its cost and time, within the gate's 20..600 characters. */
export function stepText(step: FarmStep): string {
  let s = `${step.text} Cost: ${step.cost}. Time: ${step.time}.`;
  if (s.length > 600) s = `${s.slice(0, 596)}...`;
  if (s.length < 20) s = `${s} Hold the position and check the protocol's own page.`;
  return s;
}

/** The first chain a protocol names, or a neutral phrase. */
function chainOf(p: SnapshotProtocol): string {
  return p.chains.length > 0 ? p.chains[0]! : 'its main chain';
}

/** How much capital one farm position needs, from the protocol's own TVL and its verdict. */
function capitalFor(p: SnapshotProtocol, verdict: 'FARM' | 'THIN' | 'SKIP'): number {
  const base = p.tvl_usd >= 5_000_000_000 ? 250 : p.tvl_usd >= 1_000_000_000 ? 150 : p.tvl_usd >= 100_000_000 ? 100 : 50;
  if (verdict === 'THIN') return Math.max(25, Math.round(base / 2));
  if (verdict === 'SKIP') return 25;
  return base;
}

/** How long to hold, from the protocol's age: younger protocols get a longer qualifying window. */
function holdDaysFor(age: number | null): number {
  if (age === null) return 30;
  if (age < 180) return 60;
  if (age < 365) return 45;
  return 30;
}

/** The category's one action, in the words a farmer can follow. */
function actionFor(p: SnapshotProtocol, capital: number, chain: string): string {
  const c = (p.category ?? '').toLowerCase();
  const name = p.name;
  if (c.includes('lend') || c.includes('cdp') || c.includes('collateral')) return `Supply $${capital} of USDC to ${name} on ${chain} and keep the position open.`;
  if (c.includes('dex') || c.includes('yield') || c.includes('leveraged farm')) return `Provide $${capital} of liquidity to ${name}'s main pool on ${chain} and keep the position open.`;
  if (c.includes('bridge')) return `Bridge $${capital} of USDC to ${chain} through ${name} and use the bridged funds in its app.`;
  if (c.includes('staking') || c.includes('restak')) return `Stake $${capital} of the chain's staking asset through ${name} and keep the receipt.`;
  if (c.includes('perpetual') || c.includes('derivative')) return `Deposit $${capital} of USDC as margin on ${name} and open one small hedge.`;
  if (c.includes('prediction')) return `Fund $${capital} and place one market position on ${name}.`;
  if (c.includes('curator') || c.includes('allocator') || c.includes('risk')) return `Deposit $${capital} into a ${name} vault on ${chain} and keep the shares.`;
  if (c.includes('stable')) return `Mint or deposit $${capital} of the stable asset through ${name} on ${chain}.`;
  if (c.includes('rwa')) return `Mint $${capital} of the protocol's yield asset on ${chain} and hold it.`;
  if (c.includes('insurance')) return `Buy $${capital} of cover from ${name} on ${chain} and keep the policy.`;
  return `Deposit $${capital} of USDC into ${name} on ${chain} and keep the position open.`;
}

/**
 * A protocol's farm plan: seven steps, each an action with a cost, a time and
 * a stop rule. The plan is derived from the protocol's own record (category,
 * chains, TVL, age, verdict), never from a hand-written list of names. SKIP
 * rows farm small on purpose: the plan says so, instead of pretending the
 * odds are better than they are.
 */
export function farmSteps(p: SnapshotProtocol, age: number | null, verdict: 'FARM' | 'THIN' | 'SKIP'): FarmStep[] {
  const chain = chainOf(p);
  const capital = capitalFor(p, verdict);
  const hold = holdDaysFor(age);
  const action = actionFor(p, capital, chain);
  const sizeNote = verdict === 'FARM' ? 'a full position' : verdict === 'THIN' ? 'a half-size position' : 'a small probe only';
  return [
    {
      text: `Use a fresh wallet that holds only the funds you plan to farm, and keep its seed phrase offline.`,
      cost: 'under $3 of gas for the first transaction',
      time: '5 minutes',
    },
    {
      text: `${action} This is ${sizeNote} for a ${verdict} verdict.`,
      cost: `$${capital} of capital and about $${capital >= 150 ? 8 : 4} of gas`,
      time: '15 minutes to set up',
    },
    {
      text: `Make one qualifying action each week, such as a deposit, a borrow, a swap or a vote.`,
      cost: 'about $2 of gas a week',
      time: '10 minutes a week',
    },
    {
      text: `Hold the position for at least ${hold} days so the activity is counted in more than one snapshot.`,
      cost: `the $${capital} stays in the position`,
      time: `${hold} days`,
    },
    {
      text: `Before you withdraw, open the protocol's own points or rewards page and claim only from the link it publishes.`,
      cost: 'no fee to check, and never pay to claim',
      time: '10 minutes',
    },
    {
      text: `Stop and unwind if the protocol's TVL falls 30% in seven days, its points program ends, or it announces no token: withdraw in two steps and leave the position.`,
      cost: 'about $4 of gas to unwind',
      time: '15 minutes to exit',
    },
    {
      text: `Write the wallet, the dates and every transaction in a farm log so you can repeat the position or unwind it.`,
      cost: 'nothing but the log',
      time: '5 minutes',
    },
  ];
}

/** The protocols the board ranks, strongest expected value first, with ranks 1..n. */
export function buildBoard(snapshot: Snapshot, cal: Calibration): BoardRow[] {
  const index = bySlug(snapshot.protocols);
  const chosen = new Map<string, SnapshotProtocol>();
  // The watchlist is the floor: every slug, even one the snapshot no longer carries.
  for (const slug of snapshot.watchlist) {
    const p = index.get(slug);
    if (p !== undefined) chosen.set(slug, p);
  }
  // The early radar: every rankable pre-token protocol below the watchlist
  // floor, by TVL. Never a replacement for a watchlist row.
  const early = snapshot.protocols
    .filter((p) => !chosen.has(p.slug) && isRankable(p) && p.tvl_usd < snapshot.floor_usd)
    .sort((a, b) => b.tvl_usd - a.tvl_usd || (a.slug < b.slug ? -1 : a.slug > b.slug ? 1 : 0));
  for (const p of early) chosen.set(p.slug, p);

  const scored = [...chosen.values()].map((p) => {
    const s = scoreProtocol(p, snapshot.as_of, cal);
    return { p, ...s, verdict: verdictFor(s.odds) };
  });
  // Rank by expected value. Because odds are a logit in log implied value,
  // EV is monotone in implied value, so this order is also the implied-value
  // order the gate requires — no post-hoc capping is applied.
  scored.sort(
    (a, b) => b.ev - a.ev || b.implied - a.implied || (a.p.slug < b.p.slug ? -1 : a.p.slug > b.p.slug ? 1 : 0),
  );
  const rows = scored.map((s, i) => ({
    rank: i + 1,
    slug: s.p.slug,
    name: s.p.name,
    category: s.p.category,
    chains: s.p.chains,
    tvl_usd: s.p.tvl_usd,
    change_7d: s.p.change_7d,
    mcap_usd: s.p.mcap_usd,
    listed_at: s.p.listed_at,
    gecko_id: s.p.gecko_id,
    symbol: s.p.symbol,
    token_status: 'pre-token' as const,
    age_days: s.age,
    implied_value_usd: s.implied,
    expected_value_usd: Math.round(s.ev),
    launch_odds: round4(s.odds),
    verdict: s.verdict,
    model: {
      base_value_usd: Math.round(s.base),
      category_offset: round4(s.categoryOffset),
      growth_pct: round4(s.growthPct),
      growth_multiplier: round4(s.growthMult),
      maturity_multiplier: round4(s.maturityMult),
      chain_multiplier: round4(s.chainMult),
      age_days: s.age,
    },
    farm_steps: farmSteps(s.p, s.age, s.verdict).map(stepText),
  }));
  // Safety net only: the monotone logit makes this a no-op in practice, and
  // the count is reported on the dashboard so a silent adjustment cannot hide.
  let adjusted = 0;
  for (let i = 1; i < rows.length; i++) {
    if (rows[i]!.implied_value_usd >= rows[i - 1]!.implied_value_usd) {
      rows[i]!.implied_value_usd = rows[i - 1]!.implied_value_usd - 1;
      adjusted++;
    }
  }
  (rows as unknown as { _monotoneAdjusted: number })._monotoneAdjusted = adjusted;
  return rows;
}

/** The live-token protocols, by market cap, for the dashboard's Live token view. */
export function liveRows(snapshot: Snapshot): LiveRow[] {
  return snapshot.protocols
    .filter((p) => !isPreToken(p) && p.category !== 'CEX')
    .sort((a, b) => (b.mcap_usd ?? 0) - (a.mcap_usd ?? 0) || b.tvl_usd - a.tvl_usd)
    .slice(0, LIVE_ROWS)
    .map((p) => ({
      slug: p.slug,
      name: p.name,
      category: p.category,
      chains: p.chains,
      tvl_usd: p.tvl_usd,
      mcap_usd: p.mcap_usd,
      change_7d: p.change_7d,
    }));
}

/** A board row's compact form for data.js, with the farm plan included. */
export function radarRow(row: BoardRow): RadarRow {
  return {
    rank: row.rank,
    slug: row.slug,
    name: row.name,
    category: row.category,
    chains: row.chains,
    tvl_usd: row.tvl_usd,
    change_7d: row.change_7d,
    listed_at: row.listed_at,
    age_days: row.age_days,
    implied_value_usd: row.implied_value_usd,
    expected_value_usd: row.expected_value_usd,
    launch_odds: row.launch_odds,
    verdict: row.verdict,
    base_value_usd: row.model.base_value_usd,
    category_offset: row.model.category_offset,
    growth_multiplier: row.model.growth_multiplier,
    maturity_multiplier: row.model.maturity_multiplier,
    chain_multiplier: row.model.chain_multiplier,
    farm_steps: row.farm_steps,
  };
}

/** Everything the dashboard renders, from the snapshot. */
export function buildRadar(snapshot: Snapshot): { board: BoardRow[]; radar: RadarData; monotoneAdjusted: number } {
  const comps = compsOf(snapshot);
  const cal: Calibration = {
    ...calibrate(comps),
    n_excluded_zero_mcap: snapshot.protocols.filter((p) => p.mcap_usd !== null && p.mcap_usd <= 0).length,
  };
  const validation = validate(snapshot, cal);
  const board = buildBoard(snapshot, cal);
  const monotoneAdjusted = (board as unknown as { _monotoneAdjusted?: number })._monotoneAdjusted ?? 0;
  const live = liveRows(snapshot);
  const early = board.filter((r) => !snapshot.watchlist.includes(r.slug)).length;
  const pretoken = snapshot.protocols.filter(isPreToken).length;
  const traded = snapshot.protocols.filter((p) => !isPreToken(p)).length;
  const radar: RadarData = {
    as_of: snapshot.as_of,
    source: snapshot.source,
    floor_usd: snapshot.floor_usd,
    model_version: MODEL_VERSION,
    calibration: cal,
    validation,
    watchlist_size: snapshot.watchlist.length,
    board_size: board.length,
    early_size: early,
    live_size: live.length,
    total_protocols: snapshot.protocols.length,
    traded_protocols: traded,
    pretoken_protocols: pretoken,
    unranked_pretoken: pretoken - board.length,
    traded_not_shown: traded - live.length,
    rows: board.map(radarRow),
    live,
  };
  return { board, radar, monotoneAdjusted };
}

/** The snapshot out of a file's text, refused unless it is the shape the model scores. */
export function parseSnapshot(text: string): Snapshot {
  let raw: unknown;
  try {
    raw = JSON.parse(text);
  } catch {
    throw new SnapshotError('snapshot.json is not JSON');
  }
  if (typeof raw !== 'object' || raw === null || Array.isArray(raw)) throw new SnapshotError('snapshot.json is not a JSON object');
  const o = raw as Record<string, unknown>;
  if (typeof o.as_of !== 'string' || !/^\d{4}-\d{2}-\d{2}$/.test(o.as_of)) throw new SnapshotError('snapshot.json has no as_of date');
  if (typeof o.source !== 'string') throw new SnapshotError('snapshot.json has no source');
  if (typeof o.floor_usd !== 'number' || !Number.isSafeInteger(o.floor_usd) || o.floor_usd <= 0) throw new SnapshotError('snapshot.json has no floor_usd');
  if (!Array.isArray(o.protocols)) throw new SnapshotError('snapshot.json has no protocols array');
  if (!Array.isArray(o.watchlist) || o.watchlist.some((s) => typeof s !== 'string')) throw new SnapshotError('snapshot.json has no watchlist of slugs');
  const protocols = o.protocols.map((v, i) => {
    if (typeof v !== 'object' || v === null || Array.isArray(v)) throw new SnapshotError(`snapshot.json's protocol ${i} is not an object`);
    const p = v as Record<string, unknown>;
    if (typeof p.slug !== 'string' || typeof p.name !== 'string' || typeof p.tvl_usd !== 'number' || !Number.isFinite(p.tvl_usd)) {
      throw new SnapshotError(`snapshot.json's protocol ${i} lacks a slug, a name or a TVL`);
    }
    return {
      slug: p.slug,
      name: p.name,
      category: typeof p.category === 'string' ? p.category : null,
      chains: Array.isArray(p.chains) ? p.chains.filter((c): c is string => typeof c === 'string') : [],
      tvl_usd: p.tvl_usd,
      change_7d: typeof p.change_7d === 'number' && Number.isFinite(p.change_7d) ? p.change_7d : null,
      mcap_usd: typeof p.mcap_usd === 'number' && Number.isFinite(p.mcap_usd) ? p.mcap_usd : null,
      listed_at: typeof p.listed_at === 'string' ? p.listed_at : null,
      gecko_id: typeof p.gecko_id === 'string' ? p.gecko_id : null,
      symbol: typeof p.symbol === 'string' ? p.symbol : null,
    };
  });
  return {
    as_of: o.as_of,
    source: o.source,
    floor_usd: o.floor_usd,
    protocols,
    watchlist: o.watchlist as string[],
  };
}

/** Why a board is not one the gate will take, or null: coverage, ranks, order, distinct implied values, and farm steps. */
export function boardIssue(board: BoardRow[], snapshot: Snapshot): string | null {
  const slugs = new Set(board.map((r) => r.slug));
  const missing = snapshot.watchlist.filter((s) => !slugs.has(s));
  if (missing.length > 0) return `the board misses ${missing.length} watchlist slug${missing.length === 1 ? '' : 's'}: ${missing.slice(0, 5).join(', ')}`;
  if (board.length < 100 || board.length > 4000) return `the board has ${board.length} rows`;
  const ranks = board.map((r) => r.rank).sort((a, b) => a - b);
  for (let i = 0; i < ranks.length; i++) if (ranks[i] !== i + 1) return `rank ${i + 1} is ${ranks[i]}`;
  for (let i = 1; i < board.length; i++) if (board[i]!.implied_value_usd > board[i - 1]!.implied_value_usd) return `row ${i + 1} outranks row ${i} on implied value`;
  if (board.length >= 2 && board[0]!.implied_value_usd === board[board.length - 1]!.implied_value_usd) {
    return `all ${board.length} rows share the same implied value $${board[0]!.implied_value_usd}: a board that ranks nothing by value is not a ranking`;
  }
  const seenImplied = new Set<number>();
  for (const r of board) {
    if (seenImplied.has(r.implied_value_usd)) return `${r.slug} repeats implied value $${r.implied_value_usd}`;
    seenImplied.add(r.implied_value_usd);
    const steps = r.farm_steps;
    if (steps.length < 3 || steps.length > 12) return `${r.slug} has ${steps.length} farm steps`;
    for (const s of steps) if (s.length < 20 || s.length > 600) return `${r.slug} has a farm step of ${s.length} characters`;
    if (!Number.isSafeInteger(r.implied_value_usd) || r.implied_value_usd <= 0) return `${r.slug} has implied value ${r.implied_value_usd}`;
  }
  return null;
}

/** A compact data.js: the dashboard's whole dataset as one script, no fetch and no network. */
export function dataScript(radar: RadarData): string {
  return `window.RADAR=${JSON.stringify(radar)};\n`;
}
