//! PyO3 bindings for the abstractions, the canonical indexer and the MCCFR
//! solver: `ActionAbstraction`, `CardAbstraction`, `Trainer`,
//! `BlueprintStrategy` and helper functions.

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::{PyKeyboardInterrupt, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use crate::abstraction::{
    pseudo_harmonic as ph, raise_fraction, AbstractAction, ActionAbstraction, ActionList, Bucketer, CardAbstraction,
    CardAbstractionSpec,
};
use crate::cards::{validate_cards, Card};
use crate::game::{Action, GameConfig};
use crate::isomorphism::{self, board_len, CANONICAL_SIZES};
use crate::mccfr::{decode_key, make_key, BlueprintStrategy, Seq, SolverConfig, Trainer};
use crate::python::{to_cards, PyAction, PyGameConfig, PyGameState};
use crate::sim::Rng;

fn verr(e: impl ToString) -> PyErr {
    PyValueError::new_err(e.to_string())
}

fn check_street(street: usize) -> PyResult<()> {
    if street > 3 {
        return Err(verr(format!("street must be in 0..4, got {street}")));
    }
    Ok(())
}

/// Validate `hole` + `board` for `street`.
fn hand_cards(street: usize, hole: &[i64], board: &[i64]) -> PyResult<([Card; 2], Vec<Card>)> {
    check_street(street)?;
    if hole.len() != 2 {
        return Err(verr(format!("need 2 hole cards, got {}", hole.len())));
    }
    if board.len() != board_len(street) {
        return Err(verr(format!("street {street} needs {} board cards, got {}", board_len(street), board.len())));
    }
    let mut all = hole.to_vec();
    all.extend_from_slice(board);
    let c = to_cards(&all)?;
    Ok(([c[0], c[1]], c[2..].to_vec()))
}

type ActionProbs<'py> = (Vec<u32>, Vec<Bound<'py, PyAny>>, Vec<f32>, bool);

fn list_to_py<'py>(py: Python<'py>, l: &ActionList) -> PyResult<Vec<Bound<'py, PyTuple>>> {
    (0..l.len)
        .map(|k| {
            let a = Bound::new(py, PyAction { inner: l.action[k] })?;
            (l.index[k], a).into_pyobject(py)
        })
        .collect()
}

// ----------------------------------------------------------------------
// Config parsing
// ----------------------------------------------------------------------

fn parse_action(obj: &Bound<'_, PyAny>) -> PyResult<AbstractAction> {
    let (name, arg): (String, Option<f64>) = if let Ok(s) = obj.extract::<String>() {
        (s, None)
    } else {
        let items: Vec<Bound<'_, PyAny>> = obj.try_iter()?.collect::<PyResult<_>>()?;
        if items.is_empty() {
            return Err(verr("empty abstract action"));
        }
        let name: String = items[0].extract()?;
        let arg = if items.len() > 1 { Some(items[1].extract::<f64>()?) } else { None };
        (name, arg)
    };
    match (name.to_ascii_lowercase().as_str(), arg) {
        ("fold" | "f", None) => Ok(AbstractAction::Fold),
        ("check_call" | "call" | "check" | "c", None) => Ok(AbstractAction::CheckCall),
        ("allin" | "all_in" | "all-in" | "a", None) => Ok(AbstractAction::AllIn),
        ("raise" | "raise_pot" | "r", Some(f)) => Ok(AbstractAction::RaisePot(f)),
        _ => Err(verr(format!(
            "bad abstract action {obj}: use 'fold', 'check_call', 'allin' or ('raise', pot_fraction)"
        ))),
    }
}

fn action_to_py<'py>(py: Python<'py>, a: &AbstractAction) -> PyResult<Bound<'py, PyTuple>> {
    match a {
        AbstractAction::Fold => ("fold",).into_pyobject(py),
        AbstractAction::CheckCall => ("check_call",).into_pyobject(py),
        AbstractAction::RaisePot(f) => ("raise", *f).into_pyobject(py),
        AbstractAction::AllIn => ("allin",).into_pyobject(py),
    }
}

const STREET_NAMES: [&str; 4] = ["preflop", "flop", "turn", "river"];

fn parse_action_abstraction(streets: Option<&Bound<'_, PyAny>>, max_raises: Option<u8>) -> PyResult<ActionAbstraction> {
    let mut a = ActionAbstraction::default();
    if let Some(m) = max_raises {
        a.max_raises = m;
    }
    if let Some(obj) = streets {
        if !obj.is_none() {
            let lists: Vec<Bound<'_, PyAny>> = if let Ok(d) = obj.cast::<PyDict>() {
                let mut v = Vec::new();
                for name in STREET_NAMES {
                    v.push(d.get_item(name)?.ok_or_else(|| verr(format!("actions: missing street {name}")))?);
                }
                v
            } else {
                obj.try_iter()?.collect::<PyResult<_>>()?
            };
            if lists.len() != 4 {
                return Err(verr("need one action list per street (4)"));
            }
            for (s, l) in lists.iter().enumerate() {
                a.streets[s] = l.try_iter()?.map(|x| parse_action(&x?)).collect::<PyResult<_>>()?;
            }
        }
    }
    a.validate().map_err(verr)?;
    Ok(a)
}

fn parse_game(obj: &Bound<'_, PyAny>) -> PyResult<GameConfig> {
    if let Ok(c) = obj.extract::<PyRef<'_, PyGameConfig>>() {
        return Ok(c.to_rust());
    }
    let d = obj.cast::<PyDict>().map_err(|_| verr("game must be a GameConfig or a dict"))?;
    let n: usize = d.get_item("num_players")?.map(|x| x.extract()).transpose()?.unwrap_or(2);
    let stacks: Vec<i64> = match d.get_item("stacks")? {
        None => vec![20_000; n],
        Some(x) => {
            if let Ok(v) = x.extract::<i64>() {
                vec![v; n]
            } else {
                x.extract()?
            }
        }
    };
    let get =
        |k: &str, def: i64| -> PyResult<i64> { Ok(d.get_item(k)?.map(|x| x.extract()).transpose()?.unwrap_or(def)) };
    let g = GameConfig {
        num_players: n,
        stacks,
        small_blind: get("small_blind", 50)?,
        big_blind: get("big_blind", 100)?,
        ante: get("ante", 0)?,
    };
    g.validate().map_err(verr)?;
    Ok(g)
}

fn parse_cards(obj: Option<&Bound<'_, PyAny>>) -> PyResult<CardAbstractionSpec> {
    let mut spec = CardAbstractionSpec::default();
    let Some(obj) = obj else { return Ok(spec) };
    if obj.is_none() {
        return Ok(spec);
    }
    let d = obj.cast::<PyDict>().map_err(|_| verr("cards must be a dict"))?;
    for (k, v) in d.iter() {
        let k: String = k.extract()?;
        match k.as_str() {
            "buckets" => {
                let b: Vec<u32> = v.extract()?;
                if b.len() != 4 {
                    return Err(verr("cards.buckets needs 4 entries"));
                }
                spec.buckets.copy_from_slice(&b);
            }
            "hs_samples" => spec.hs_samples = v.extract()?,
            "tables" => {
                let t: Vec<Option<String>> = v.extract()?;
                if t.len() != 4 {
                    return Err(verr("cards.tables needs 4 entries (None or a .npy path)"));
                }
                spec.tables.clone_from_slice(&t[..4]);
            }
            _ => return Err(verr(format!("unknown cards option {k:?}"))),
        }
    }
    Ok(spec)
}

fn parse_solver_config(d: &Bound<'_, PyDict>) -> PyResult<(SolverConfig, String)> {
    let mut c = SolverConfig::default();
    let mut meta = String::new();
    let mut actions_streets: Option<Bound<'_, PyAny>> = None;
    let mut max_raises = None;
    for (k, v) in d.iter() {
        let k: String = k.extract()?;
        match k.as_str() {
            "game" => c.game = parse_game(&v)?,
            "actions" => {
                if let Ok(ad) = v.cast::<PyDict>() {
                    for (ak, av) in ad.iter() {
                        let ak: String = ak.extract()?;
                        match ak.as_str() {
                            "streets" => actions_streets = Some(av),
                            "max_raises" | "max_raises_per_street" => max_raises = Some(av.extract()?),
                            _ => return Err(verr(format!("unknown actions option {ak:?}"))),
                        }
                    }
                } else {
                    actions_streets = Some(v);
                }
            }
            "cards" => c.cards = parse_cards(Some(&v))?,
            "seed" => c.seed = v.extract()?,
            "lcfr_discount_every" => c.lcfr_discount_every = v.extract()?,
            "lcfr_stop" => c.lcfr_stop = v.extract()?,
            "prune_start" => c.prune_start = if v.is_none() { u64::MAX } else { v.extract()? },
            "prune_prob" => c.prune_prob = v.extract()?,
            "prune_threshold" => c.prune_threshold = v.extract()?,
            "regret_floor" => c.regret_floor = v.extract()?,
            "checkpoint_path" => c.checkpoint_path = v.extract()?,
            "checkpoint_interval" => c.checkpoint_interval = if v.is_none() { 0.0 } else { v.extract()? },
            "shards" => c.shards = v.extract()?,
            "meta" => meta = v.extract()?,
            _ => return Err(verr(format!("unknown solver option {k:?}"))),
        }
    }
    c.actions = parse_action_abstraction(actions_streets.as_ref(), max_raises)?;
    c.validate().map_err(verr)?;
    Ok((c, meta))
}

// ----------------------------------------------------------------------
// Free functions
// ----------------------------------------------------------------------

/// Canonical suit-isomorphic index of hole + board on `street` (see the
/// module docs of `isomorphism.rs` for the exact definition).
#[pyfunction]
fn canonical_index(street: usize, hole: Vec<i64>, board: Vec<i64>) -> PyResult<u32> {
    let (h, b) = hand_cards(street, &hole, &board)?;
    Ok(isomorphism::canonical_index(street, h, &b))
}

/// Canonical indices of the rows of a uint8 array of shape (N, 2 + board_len).
#[pyfunction]
fn canonical_index_batch<'py>(
    py: Python<'py>,
    street: usize,
    cards: PyReadonlyArray2<'py, u8>,
) -> PyResult<Bound<'py, PyArray1<u32>>> {
    check_street(street)?;
    let w = 2 + board_len(street);
    if cards.shape()[1] != w {
        return Err(verr(format!("expected shape (N, {w})")));
    }
    let a = cards.as_array();
    let mut out = Vec::with_capacity(a.shape()[0]);
    let mut row = vec![0u8; w];
    for r in a.rows() {
        for (j, &c) in r.iter().enumerate() {
            row[j] = c;
        }
        validate_cards(&row).map_err(|e| verr(e.0))?;
        out.push(isomorphism::canonical_index(street, [row[0], row[1]], &row[2..]));
    }
    Ok(PyArray1::from_vec(py, out))
}

/// Canonical representative `(hole, board)` of an index.
#[pyfunction]
fn canonical_unindex(street: usize, index: u32) -> PyResult<(Vec<i64>, Vec<i64>)> {
    check_street(street)?;
    let (h, b) = isomorphism::canonical_unindex(street, index).ok_or_else(|| verr("index out of range"))?;
    Ok((h.iter().map(|&c| c as i64).collect(), b.iter().map(|&c| c as i64).collect()))
}

/// Number of canonical classes on `street` (169, 1286792, 13960050, 123156254).
#[pyfunction]
fn canonical_size(street: usize) -> PyResult<u64> {
    check_street(street)?;
    Ok(CANONICAL_SIZES[street])
}

/// Lossless preflop class in 0..169 (13x13 grid: pairs r*13+r, suited
/// hi*13+lo, offsuit lo*13+hi; ranks 0 = deuce .. 12 = ace).
#[pyfunction]
fn preflop_class(hole: Vec<i64>) -> PyResult<u32> {
    let (h, _) = hand_cards(0, &hole, &[])?;
    Ok(isomorphism::preflop_class(h))
}

/// Probability that the pseudo-harmonic mapping sends pot fraction `x` to
/// the smaller size `a` (vs `b`): (b - x)(1 + a) / ((b - a)(1 + x)).
#[pyfunction]
fn pseudo_harmonic(a: f64, b: f64, x: f64) -> f64 {
    ph(a, b, x)
}

/// Equity vs one random hand: exact on the river, `samples` Monte Carlo
/// draws (from `seed`) earlier.
#[pyfunction]
#[pyo3(signature = (hole, board, samples=256, seed=0))]
fn hand_strength(hole: Vec<i64>, board: Vec<i64>, samples: u32, seed: u64) -> PyResult<f64> {
    let street = match board.len() {
        0 => 0,
        3 => 1,
        4 => 2,
        5 => 3,
        n => return Err(verr(format!("board must have 0, 3, 4 or 5 cards, got {n}"))),
    };
    let (h, b) = hand_cards(street, &hole, &board)?;
    Ok(crate::abstraction::hand_strength(h, &b, samples, &mut Rng::new(seed)))
}

/// Infoset key from street, bucket and abstract betting sequence.
#[pyfunction]
fn make_infoset_key(street: usize, bucket: u32, sequence: Vec<u32>) -> PyResult<u128> {
    check_street(street)?;
    if sequence.iter().any(|&i| i >= 16) {
        return Err(verr("abstract indices must be < 16"));
    }
    let sequence: Vec<u8> = sequence.into_iter().map(|i| i as u8).collect();
    if bucket >= 1 << 29 {
        return Err(verr("bucket must be < 2**29"));
    }
    Ok(make_key(street, bucket, Seq::from_indices(&sequence).map_err(verr)?))
}

/// `(street, bucket, sequence)` of an infoset key.
#[pyfunction]
fn decode_infoset_key(key: u128) -> (usize, u32, Vec<u32>) {
    let (s, b, q) = decode_key(key);
    (s, b, q.indices().into_iter().map(u32::from).collect())
}

// ----------------------------------------------------------------------
// ActionAbstraction
// ----------------------------------------------------------------------

/// Per-street abstract actions with a raise cap. `streets` is a list of 4
/// lists (or a dict preflop/flop/turn/river) of `"fold"`, `"check_call"`,
/// `"allin"` or `("raise", pot_fraction)`; `None` is the DESIGN.md table.
#[pyclass(name = "ActionAbstraction", module = "poker_engine", frozen, skip_from_py_object)]
#[derive(Clone)]
pub(crate) struct PyActionAbstraction {
    inner: ActionAbstraction,
}

#[pymethods]
impl PyActionAbstraction {
    #[new]
    #[pyo3(signature = (streets=None, max_raises=4))]
    fn new(streets: Option<&Bound<'_, PyAny>>, max_raises: u8) -> PyResult<Self> {
        Ok(PyActionAbstraction { inner: parse_action_abstraction(streets, Some(max_raises))? })
    }

    /// The abstract action lists, as tuples.
    #[getter]
    fn streets<'py>(&self, py: Python<'py>) -> PyResult<Vec<Vec<Bound<'py, PyTuple>>>> {
        self.inner.streets.iter().map(|l| l.iter().map(|a| action_to_py(py, a)).collect()).collect()
    }

    #[getter]
    fn max_raises(&self) -> u8 {
        self.inner.max_raises
    }

    /// `[(abstract_index, Action), ...]` available in `state` (de-duplicated).
    fn legal<'py>(&self, py: Python<'py>, state: PyRef<'_, PyGameState>) -> PyResult<Vec<Bound<'py, PyTuple>>> {
        list_to_py(py, &self.inner.legal(&state.inner))
    }

    /// Concrete action of abstract index `index` in `state`, or None.
    fn to_concrete(&self, state: PyRef<'_, PyGameState>, index: u8) -> Option<PyAction> {
        self.inner.to_concrete(&state.inner, index).map(|a| PyAction { inner: a })
    }

    /// Map a concrete `action` taken in `real_state` onto the abstract
    /// actions of `abs_state` with the pseudo-harmonic mapping; `u` in [0, 1)
    /// randomizes between the two neighbouring sizes (0.5 = deterministic).
    #[pyo3(signature = (abs_state, real_state, action, u=0.5))]
    fn translate(
        &self,
        abs_state: PyRef<'_, PyGameState>,
        real_state: PyRef<'_, PyGameState>,
        action: PyRef<'_, PyAction>,
        u: f64,
    ) -> Option<u8> {
        self.inner.translate(&abs_state.inner, &real_state.inner, action.inner, u)
    }

    /// Abstract indices of `state`'s history (ValueError if off-tree).
    fn sequence(&self, state: PyRef<'_, PyGameState>) -> PyResult<Vec<u32>> {
        Ok(self.inner.sequence(&state.inner).map_err(verr)?.into_iter().map(u32::from).collect())
    }

    /// Decision nodes per street of the abstract betting tree for `config`.
    #[pyo3(signature = (config, limit=100_000_000))]
    fn count_tree(&self, config: &Bound<'_, PyAny>, limit: u64) -> PyResult<Option<Vec<u64>>> {
        let g = parse_game(config)?;
        Ok(self.inner.count_tree(&g, limit).map(|c| c.to_vec()))
    }

    /// Pot fraction of `raise_to` for the player to act in `state`.
    #[staticmethod]
    fn pot_fraction(state: PyRef<'_, PyGameState>, raise_to: i64) -> f64 {
        raise_fraction(&state.inner, raise_to)
    }

    fn __repr__(&self) -> String {
        format!("ActionAbstraction(streets={:?}, max_raises={})", self.inner.streets, self.inner.max_raises)
    }
}

// ----------------------------------------------------------------------
// CardAbstraction
// ----------------------------------------------------------------------

/// Card abstraction: `buckets` per street (preflop 169 = lossless), default
/// hand-strength buckets or per-street tables (`.npy` path or a 1-D integer
/// numpy array indexed by `canonical_index`).
#[pyclass(name = "CardAbstraction", module = "poker_engine", frozen, skip_from_py_object)]
pub(crate) struct PyCardAbstraction {
    inner: CardAbstraction,
}

#[pymethods]
impl PyCardAbstraction {
    #[new]
    #[pyo3(signature = (buckets=None, hs_samples=256, tables=None))]
    fn new(
        buckets: Option<Vec<u32>>,
        hs_samples: u32,
        tables: Option<Vec<Option<Bound<'_, PyAny>>>>,
    ) -> PyResult<Self> {
        let mut spec = CardAbstractionSpec { hs_samples, ..Default::default() };
        if let Some(b) = buckets {
            if b.len() != 4 {
                return Err(verr("buckets needs 4 entries"));
            }
            spec.buckets.copy_from_slice(&b);
        }
        let mut mem: [Option<Vec<u32>>; 4] = Default::default();
        if let Some(t) = tables {
            if t.len() != 4 {
                return Err(verr("tables needs 4 entries"));
            }
            for (s, x) in t.into_iter().enumerate() {
                let Some(x) = x else { continue };
                if x.is_none() {
                    continue;
                }
                if let Ok(p) = x.extract::<String>() {
                    mem[s] = Some(crate::npy::read_npy_u32(&p).map_err(verr)?);
                    spec.tables[s] = Some(p);
                } else {
                    let arr: PyReadonlyArray1<'_, i64> = x
                        .call_method1("astype", ("int64",))?
                        .extract()
                        .map_err(|_| verr("bucket table must be a 1-D integer array"))?;
                    let mut v = Vec::with_capacity(arr.len());
                    for &b in arr.as_array().iter() {
                        if !(0..=u32::MAX as i64).contains(&b) {
                            return Err(verr("negative bucket in table"));
                        }
                        v.push(b as u32);
                    }
                    mem[s] = Some(v);
                }
            }
        }
        Ok(PyCardAbstraction { inner: CardAbstraction::with_tables(spec, mem).map_err(verr)? })
    }

    fn bucket(&self, street: usize, hole: Vec<i64>, board: Vec<i64>) -> PyResult<u32> {
        let (h, b) = hand_cards(street, &hole, &board)?;
        Ok(self.inner.bucket(street, h, &b))
    }

    /// Buckets of the rows of a uint8 array of shape (N, 2 + board_len).
    fn buckets_batch<'py>(
        &self,
        py: Python<'py>,
        street: usize,
        cards: PyReadonlyArray2<'py, u8>,
    ) -> PyResult<Bound<'py, PyArray1<u32>>> {
        check_street(street)?;
        let w = 2 + board_len(street);
        if cards.shape()[1] != w {
            return Err(verr(format!("expected shape (N, {w})")));
        }
        let rows: Vec<Vec<u8>> = cards.as_array().rows().into_iter().map(|r| r.to_vec()).collect();
        for r in &rows {
            validate_cards(r).map_err(|e| verr(e.0))?;
        }
        let inner = &self.inner;
        let out: Vec<u32> = py.detach(|| rows.iter().map(|r| inner.bucket(street, [r[0], r[1]], &r[2..])).collect());
        Ok(PyArray1::from_vec(py, out))
    }

    fn num_buckets(&self, street: usize) -> PyResult<u32> {
        check_street(street)?;
        Ok(self.inner.num_buckets(street))
    }
}

// ----------------------------------------------------------------------
// Trainer
// ----------------------------------------------------------------------

/// External-sampling MCCFR trainer (see engine/src/mccfr.rs and
/// pokerbot/blueprint/mccfr/README.md). `config` is a dict with optional
/// keys: game, actions {streets, max_raises}, cards {buckets, hs_samples,
/// tables}, seed, lcfr_discount_every, lcfr_stop, prune_start (None
/// disables), prune_prob, prune_threshold, regret_floor, checkpoint_path,
/// checkpoint_interval (seconds), shards, meta (JSON string).
#[pyclass(name = "Trainer", module = "poker_engine", skip_from_py_object)]
pub(crate) struct PyTrainer {
    inner: Trainer,
}

fn run_stats<'py>(py: Python<'py>, t: &Trainer, detailed: bool) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("iterations", t.iterations())?;
    d.set_item("nodes", t.nodes())?;
    d.set_item("infosets", t.tables.len())?;
    d.set_item("table_bytes", t.tables.memory_bytes())?;
    d.set_item("rss_bytes", rss_bytes())?;
    d.set_item("seconds_total", t.total_seconds)?;
    let r = t.last_run;
    d.set_item("last_run_iterations", r.iterations)?;
    d.set_item("last_run_seconds", r.seconds)?;
    let rate = |x: u64| if r.seconds > 0.0 { x as f64 / r.seconds } else { 0.0 };
    d.set_item("nodes_per_second", rate(r.nodes))?;
    d.set_item("iterations_per_second", rate(r.iterations))?;
    if detailed {
        let mut per = [0u64; 4];
        t.tables.for_each(|k, _, _| per[decode_key(k).0] += 1);
        d.set_item("infosets_per_street", per.to_vec())?;
    }
    Ok(d)
}

/// Resident set size of this process (Linux), else 0.
fn rss_bytes() -> u64 {
    std::fs::read_to_string("/proc/self/statm")
        .ok()
        .and_then(|s| s.split_whitespace().nth(1).and_then(|x| x.parse::<u64>().ok()))
        .map(|pages| pages * 4096)
        .unwrap_or(0)
}

#[pymethods]
impl PyTrainer {
    #[new]
    #[pyo3(signature = (config=None))]
    fn new(config: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let (c, meta) = match config {
            Some(d) => parse_solver_config(d)?,
            None => (SolverConfig::default(), String::new()),
        };
        let mut t = Trainer::new(c).map_err(verr)?;
        t.meta = meta;
        Ok(PyTrainer { inner: t })
    }

    /// Run `iterations` iterations (one traversal per player each) on
    /// `threads` threads, releasing the GIL. Ctrl-C stops after the current
    /// batch and raises KeyboardInterrupt. Returns `stats()`.
    #[pyo3(signature = (iterations, threads=1))]
    fn run<'py>(&mut self, py: Python<'py>, iterations: u64, threads: usize) -> PyResult<Bound<'py, PyDict>> {
        let t = &mut self.inner;
        let mut interrupted = false;
        let res = py.detach(|| {
            t.run(iterations, threads, || {
                let ok = Python::attach(|py| py.check_signals().is_ok());
                if !ok {
                    interrupted = true;
                }
                ok
            })
        });
        res.map_err(verr)?;
        if interrupted {
            return Err(PyKeyboardInterrupt::new_err("training interrupted"));
        }
        run_stats(py, &self.inner, false)
    }

    /// Counters: iterations, nodes, infosets, table_bytes (estimated heap of
    /// the tables), rss_bytes, seconds_total, last-run rates; `detailed`
    /// adds infosets_per_street (walks the whole table).
    #[pyo3(signature = (detailed=false))]
    fn stats<'py>(&self, py: Python<'py>, detailed: bool) -> PyResult<Bound<'py, PyDict>> {
        run_stats(py, &self.inner, detailed)
    }

    #[getter]
    fn iterations(&self) -> u64 {
        self.inner.iterations()
    }

    /// Solver configuration as a JSON string.
    #[getter]
    fn config_json(&self) -> String {
        self.inner.config.to_json()
    }

    #[getter]
    fn meta(&self) -> String {
        self.inner.meta.clone()
    }

    #[setter]
    fn set_meta(&mut self, meta: String) {
        self.inner.meta = meta;
    }

    #[getter]
    fn game_config(&self) -> PyGameConfig {
        PyGameConfig::from_rust(&self.inner.config.game)
    }

    /// Change where and how often `run` writes checkpoints.
    #[pyo3(signature = (path, interval_seconds))]
    fn set_checkpoint(&mut self, path: Option<String>, interval_seconds: f64) {
        self.inner.config.checkpoint_path = path;
        self.inner.config.checkpoint_interval = interval_seconds;
    }

    fn action_abstraction(&self) -> PyActionAbstraction {
        PyActionAbstraction { inner: self.inner.config.actions.clone() }
    }

    fn bucket(&self, street: usize, hole: Vec<i64>, board: Vec<i64>) -> PyResult<u32> {
        let (h, b) = hand_cards(street, &hole, &board)?;
        Ok(self.inner.cards.bucket(street, h, &b))
    }

    /// Write a checkpoint (tables, config, counters).
    fn save(&self, py: Python<'_>, path: String) -> PyResult<()> {
        let t = &self.inner;
        py.detach(|| t.save(&path)).map_err(verr)
    }

    /// Load a checkpoint.
    #[staticmethod]
    fn load(py: Python<'_>, path: String) -> PyResult<Self> {
        let t = py.detach(|| Trainer::load(&path)).map_err(verr)?;
        Ok(PyTrainer { inner: t })
    }

    /// Write the averaged strategy file; returns the number of infosets.
    fn export_strategy(&self, py: Python<'_>, path: String) -> PyResult<u64> {
        let t = &self.inner;
        py.detach(|| t.export_strategy(&path)).map_err(verr)
    }

    /// Strategy of `player` (the player to act, hole cards visible) at an
    /// on-tree `state`, one probability per entry of
    /// `action_abstraction().legal(state)`. `average=False` gives the
    /// current regret-matching strategy.
    #[pyo3(signature = (state, player, average=true))]
    fn strategy(&self, state: PyRef<'_, PyGameState>, player: usize, average: bool) -> PyResult<Vec<f32>> {
        let (_, p) = self.inner.strategy(&state.inner, player, average).map_err(verr)?;
        Ok(p)
    }

    /// Infoset key of `player` (to act) at an on-tree `state`.
    fn infoset_key(&self, state: PyRef<'_, PyGameState>, player: usize) -> PyResult<u128> {
        Ok(self.inner.key_of(&state.inner, player).map_err(verr)?.0)
    }

    /// `(regrets, strategy_sums)` stored for `key`, or None.
    fn lookup(&self, key: u128) -> Option<(Vec<f32>, Vec<f32>)> {
        self.inner.tables.get(key)
    }

    /// All `(key, regrets, strategy_sums)` rows (for small games and tests).
    fn entries<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let out = PyList::empty(py);
        let mut err = None;
        self.inner.tables.for_each(|k, r, s| {
            if err.is_none() {
                if let Err(e) = out.append((k, r.to_vec(), s.to_vec())) {
                    err = Some(e);
                }
            }
        });
        match err {
            Some(e) => Err(e),
            None => Ok(out),
        }
    }
}

// ----------------------------------------------------------------------
// BlueprintStrategy
// ----------------------------------------------------------------------

/// An exported averaged strategy, loaded for play (see `Trainer.export_strategy`).
#[pyclass(name = "BlueprintStrategy", module = "poker_engine", frozen, skip_from_py_object)]
pub(crate) struct PyBlueprintStrategy {
    inner: BlueprintStrategy,
}

#[pymethods]
impl PyBlueprintStrategy {
    #[new]
    fn new(py: Python<'_>, path: String) -> PyResult<Self> {
        let s = py.detach(|| BlueprintStrategy::load(&path)).map_err(verr)?;
        Ok(PyBlueprintStrategy { inner: s })
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Solver configuration the strategy was trained with (JSON string).
    #[getter]
    fn config_json(&self) -> String {
        self.inner.config_json.clone()
    }

    /// Training metadata (JSON string, may be empty).
    #[getter]
    fn meta(&self) -> String {
        self.inner.meta.clone()
    }

    #[getter]
    fn game_config(&self) -> PyGameConfig {
        PyGameConfig::from_rust(&self.inner.config.game)
    }

    fn action_abstraction(&self) -> PyActionAbstraction {
        PyActionAbstraction { inner: self.inner.config.actions.clone() }
    }

    fn bucket(&self, street: usize, hole: Vec<i64>, board: Vec<i64>) -> PyResult<u32> {
        let (h, b) = hand_cards(street, &hole, &board)?;
        Ok(self.inner.cards.bucket(street, h, &b))
    }

    /// Normalized probabilities stored for an infoset key, or None.
    fn lookup(&self, key: u128) -> Option<Vec<f32>> {
        self.inner.lookup(key)
    }

    /// For the player to act at the on-tree `abs_state` holding `hole` with
    /// the real `board`: `(indices, actions, probs, found)` where `indices`
    /// are abstract indices, `actions` the concrete actions in `abs_state`,
    /// and `found` is False when the infoset was not in the file (uniform).
    fn action_probs<'py>(
        &self,
        py: Python<'py>,
        abs_state: PyRef<'_, PyGameState>,
        hole: Vec<i64>,
        board: Vec<i64>,
    ) -> PyResult<ActionProbs<'py>> {
        let street = abs_state.inner.street();
        let (h, b) = hand_cards(street, &hole, &board)?;
        let (list, probs, found) = self.inner.action_probs(&abs_state.inner, h, &b).map_err(verr)?;
        let acts = list
            .actions()
            .iter()
            .map(|&a| Ok(Bound::new(py, PyAction { inner: a })?.into_any()))
            .collect::<PyResult<Vec<_>>>()?;
        Ok((list.indices().iter().map(|&i| i as u32).collect(), acts, probs, found))
    }
}

/// A concrete action from kind/amount (helper for converting other engines' actions).
#[pyfunction]
fn action_from(kind: u8, amount: i64) -> PyResult<PyAction> {
    match kind {
        0 => Ok(PyAction { inner: Action::fold() }),
        1 => Ok(PyAction { inner: Action::check_call() }),
        2 => Ok(PyAction { inner: Action::raise_to(amount) }),
        _ => Err(verr(format!("bad action kind {kind}"))),
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyActionAbstraction>()?;
    m.add_class::<PyCardAbstraction>()?;
    m.add_class::<PyTrainer>()?;
    m.add_class::<PyBlueprintStrategy>()?;
    m.add_function(wrap_pyfunction!(canonical_index, m)?)?;
    m.add_function(wrap_pyfunction!(canonical_index_batch, m)?)?;
    m.add_function(wrap_pyfunction!(canonical_unindex, m)?)?;
    m.add_function(wrap_pyfunction!(canonical_size, m)?)?;
    m.add_function(wrap_pyfunction!(preflop_class, m)?)?;
    m.add_function(wrap_pyfunction!(pseudo_harmonic, m)?)?;
    m.add_function(wrap_pyfunction!(hand_strength, m)?)?;
    m.add_function(wrap_pyfunction!(make_infoset_key, m)?)?;
    m.add_function(wrap_pyfunction!(decode_infoset_key, m)?)?;
    m.add_function(wrap_pyfunction!(action_from, m)?)?;
    Ok(())
}
