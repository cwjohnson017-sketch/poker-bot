//! PyO3 bindings: the `poker_engine` Python module.

use numpy::{PyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyTuple, PyType};

use crate::cards::{self, validate_cards, Card};
use crate::eval;
use crate::game::{
    Action, ActionKind, Chips, GameConfig, GameError, GameState, LegalActions, CHECK_CALL, FOLD, MAX_PLAYERS, RAISE,
};

fn game_err(e: GameError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

fn to_cards(v: &[i64]) -> PyResult<Vec<Card>> {
    let mut out = Vec::with_capacity(v.len());
    for &c in v {
        if !(0..52).contains(&c) {
            return Err(PyValueError::new_err(format!("invalid card {c}: must be in 0..52")));
        }
        out.push(c as Card);
    }
    validate_cards(&out).map_err(|e| PyValueError::new_err(e.0))?;
    Ok(out)
}

fn eval_n(cards: Vec<i64>, n: usize) -> PyResult<i64> {
    if cards.len() != n {
        return Err(PyValueError::new_err(format!("expected exactly {n} cards, got {}", cards.len())));
    }
    let c = to_cards(&cards)?;
    Ok(eval::evaluate(&c) as i64)
}

/// Rank of exactly 5 cards; higher is better.
#[pyfunction]
fn evaluate5(cards: Vec<i64>) -> PyResult<i64> {
    eval_n(cards, 5)
}

/// Rank of the best 5-card hand out of exactly 6 cards.
#[pyfunction]
fn evaluate6(cards: Vec<i64>) -> PyResult<i64> {
    eval_n(cards, 6)
}

/// Rank of the best 5-card hand out of exactly 7 cards.
#[pyfunction]
fn evaluate7(cards: Vec<i64>) -> PyResult<i64> {
    eval_n(cards, 7)
}

/// Rank of the best 5-card hand out of 5, 6 or 7 cards.
#[pyfunction]
fn evaluate(cards: Vec<i64>) -> PyResult<i64> {
    let n = cards.len();
    if !(5..=7).contains(&n) {
        return Err(PyValueError::new_err(format!("expected 5 to 7 cards, got {n}")));
    }
    eval_n(cards, n)
}

/// Evaluate a uint8 array of shape (N, 7); returns int32 ranks of shape (N,).
#[pyfunction]
fn evaluate_batch<'py>(py: Python<'py>, cards: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyArray1<i32>>> {
    let arr: PyReadonlyArray2<'py, u8> = cards
        .extract()
        .map_err(|_| PyValueError::new_err("evaluate_batch expects a numpy uint8 array of shape (N, 7)"))?;
    let shape = arr.shape();
    if shape[1] != 7 {
        return Err(PyValueError::new_err(format!("expected shape (N, 7), got ({}, {})", shape[0], shape[1])));
    }
    let n = shape[0];
    let flat: Vec<u8> = match arr.as_slice() {
        Ok(s) => s.to_vec(),
        Err(_) => arr.as_array().iter().copied().collect(),
    };
    for (i, row) in flat.chunks_exact(7).enumerate() {
        validate_cards(row).map_err(|e| PyValueError::new_err(format!("row {i}: {}", e.0)))?;
    }
    let mut out = vec![0i32; n];
    eval::evaluate_batch(&flat, &mut out);
    Ok(PyArray1::from_vec(py, out))
}

/// Category of a rank: 0 high card, 1 pair, 2 two pair, 3 trips,
/// 4 straight, 5 flush, 6 full house, 7 quads, 8 straight flush.
#[pyfunction]
fn hand_category(rank: i64) -> PyResult<i64> {
    if !(0..(9 << eval::CATEGORY_SHIFT)).contains(&rank) {
        return Err(PyValueError::new_err(format!("invalid hand rank {rank}")));
    }
    Ok(eval::hand_category(rank as u32) as i64)
}

/// Parse a card string such as "As" or "Td".
#[pyfunction]
fn card_from_str(s: &str) -> PyResult<i64> {
    cards::card_from_str(s).map(|c| c as i64).map_err(|e| PyValueError::new_err(e.0))
}

/// Format a card int as a string such as "As".
#[pyfunction]
fn card_to_str(card: i64) -> PyResult<String> {
    if !(0..52).contains(&card) {
        return Err(PyValueError::new_err(format!("invalid card {card}: must be in 0..52")));
    }
    cards::card_to_str(card as Card).map_err(|e| PyValueError::new_err(e.0))
}

// ----------------------------------------------------------------------
// GameConfig
// ----------------------------------------------------------------------

/// Table configuration.
#[pyclass(name = "GameConfig", module = "poker_engine", get_all, set_all, eq, skip_from_py_object)]
#[derive(Clone, PartialEq)]
struct PyGameConfig {
    num_players: usize,
    stacks: Vec<Chips>,
    small_blind: Chips,
    big_blind: Chips,
    ante: Chips,
}

impl PyGameConfig {
    fn to_rust(&self) -> GameConfig {
        GameConfig {
            num_players: self.num_players,
            stacks: self.stacks.clone(),
            small_blind: self.small_blind,
            big_blind: self.big_blind,
            ante: self.ante,
        }
    }
}

#[pymethods]
impl PyGameConfig {
    /// `stacks` defaults to 20,000 chips per seat; `num_players` defaults
    /// to `len(stacks)` when stacks are given, else 2.
    #[new]
    #[pyo3(signature = (num_players=None, stacks=None, small_blind=50, big_blind=100, ante=0))]
    fn new(
        num_players: Option<usize>,
        stacks: Option<Vec<Chips>>,
        small_blind: Chips,
        big_blind: Chips,
        ante: Chips,
    ) -> PyResult<Self> {
        let n = num_players.unwrap_or_else(|| stacks.as_ref().map_or(2, |s| s.len()));
        let stacks = stacks.unwrap_or_else(|| vec![20_000; n]);
        let c = PyGameConfig { num_players: n, stacks, small_blind, big_blind, ante };
        c.to_rust().validate().map_err(game_err)?;
        Ok(c)
    }

    /// Raise ValueError if the configuration is invalid.
    fn validate(&self) -> PyResult<()> {
        self.to_rust().validate().map_err(game_err)
    }

    fn __repr__(&self) -> String {
        format!(
            "GameConfig(num_players={}, stacks={:?}, small_blind={}, big_blind={}, ante={})",
            self.num_players, self.stacks, self.small_blind, self.big_blind, self.ante
        )
    }

    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
        let py = slf.py();
        let c = slf.borrow();
        let args = (c.num_players, c.stacks.clone(), c.small_blind, c.big_blind, c.ante);
        (slf.get_type(), args).into_pyobject(py)
    }
}

// ----------------------------------------------------------------------
// Action / LegalActions
// ----------------------------------------------------------------------

/// A concrete action. `amount` is the raise-to total for RAISE, else 0.
#[pyclass(name = "Action", module = "poker_engine", frozen, eq, hash, skip_from_py_object)]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct PyAction {
    inner: Action,
}

#[pymethods]
impl PyAction {
    #[new]
    #[pyo3(signature = (kind, amount=0))]
    fn new(kind: i64, amount: Chips) -> PyResult<Self> {
        let k = u8::try_from(kind)
            .ok()
            .and_then(ActionKind::from_u8)
            .ok_or_else(|| PyValueError::new_err(format!("invalid action kind {kind}")))?;
        if k != ActionKind::Raise && amount != 0 {
            return Err(PyValueError::new_err("only RAISE actions take an amount"));
        }
        if k == ActionKind::Raise && amount <= 0 {
            return Err(PyValueError::new_err("raise amount must be positive"));
        }
        Ok(PyAction { inner: Action { kind: k, amount } })
    }

    #[staticmethod]
    fn fold() -> Self {
        PyAction { inner: Action::fold() }
    }

    #[staticmethod]
    fn check_call() -> Self {
        PyAction { inner: Action::check_call() }
    }

    /// Raise (or bet) so this player's total on the street becomes `amount`.
    #[staticmethod]
    fn raise_to(amount: Chips) -> PyResult<Self> {
        PyAction::new(RAISE as i64, amount)
    }

    #[getter]
    fn kind(&self) -> u8 {
        self.inner.kind as u8
    }

    #[getter]
    fn amount(&self) -> Chips {
        self.inner.amount
    }

    fn __repr__(&self) -> String {
        match self.inner.kind {
            ActionKind::Fold => "Action.fold()".into(),
            ActionKind::CheckCall => "Action.check_call()".into(),
            ActionKind::Raise => format!("Action.raise_to({})", self.inner.amount),
        }
    }

    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
        let py = slf.py();
        let a = slf.get().inner;
        (slf.get_type(), (a.kind as u8, a.amount)).into_pyobject(py)
    }
}

/// Legal actions of the player to act.
#[pyclass(name = "LegalActions", module = "poker_engine", frozen, eq, skip_from_py_object)]
#[derive(Clone, Copy, PartialEq, Eq)]
struct PyLegalActions {
    inner: LegalActions,
}

#[pymethods]
impl PyLegalActions {
    #[getter]
    fn can_fold(&self) -> bool {
        self.inner.can_fold
    }
    #[getter]
    fn can_check(&self) -> bool {
        self.inner.can_check
    }
    #[getter]
    fn call_amount(&self) -> Chips {
        self.inner.call_amount
    }
    #[getter]
    fn min_raise_to(&self) -> Chips {
        self.inner.min_raise_to
    }
    #[getter]
    fn max_raise_to(&self) -> Chips {
        self.inner.max_raise_to
    }
    /// True iff some raise is legal (`min_raise_to > 0`).
    #[getter]
    fn can_raise(&self) -> bool {
        self.inner.can_raise()
    }
    /// Whether `GameState.apply(action)` would accept `action`.
    fn is_legal(&self, action: PyRef<'_, PyAction>) -> bool {
        self.inner.is_legal(action.inner)
    }
    /// The legal raise-to amount closest to `amount` (0 if no raise is legal).
    fn clamp_raise_to(&self, amount: Chips) -> Chips {
        self.inner.clamp_raise_to(amount)
    }
    fn __repr__(&self) -> String {
        let l = &self.inner;
        format!(
            "LegalActions(can_fold={}, can_check={}, call_amount={}, min_raise_to={}, max_raise_to={})",
            py_bool(l.can_fold),
            py_bool(l.can_check),
            l.call_amount,
            l.min_raise_to,
            l.max_raise_to
        )
    }
}

fn py_bool(b: bool) -> &'static str {
    if b {
        "True"
    } else {
        "False"
    }
}

// ----------------------------------------------------------------------
// GameState
// ----------------------------------------------------------------------

/// State of one hand. Mutated by `apply`; `child` and `clone` copy.
#[pyclass(name = "GameState", module = "poker_engine", skip_from_py_object)]
#[derive(Clone)]
struct PyGameState {
    inner: GameState,
}

#[pymethods]
impl PyGameState {
    /// Post blinds (and antes) and deal from `deck` (a permutation of
    /// 0..52, or at least its first 2 * num_players + 5 cards).
    #[staticmethod]
    fn new_hand(config: PyRef<'_, PyGameConfig>, button: usize, deck: Vec<i64>) -> PyResult<Self> {
        let d = to_cards(&deck)?;
        GameState::new_hand(&config.to_rust(), button, &d).map(|inner| PyGameState { inner }).map_err(game_err)
    }

    #[getter]
    fn num_players(&self) -> usize {
        self.inner.num_players()
    }
    #[getter]
    fn button(&self) -> usize {
        self.inner.button()
    }
    #[getter]
    fn street(&self) -> usize {
        self.inner.street()
    }
    #[getter]
    fn board(&self) -> Vec<i64> {
        self.inner.board().iter().map(|&c| c as i64).collect()
    }
    /// The player's two hole cards in deal order; `[]` when masked.
    fn hole_cards(&self, player: usize) -> PyResult<Vec<i64>> {
        if player >= self.inner.num_players() {
            return Err(game_err(GameError::InvalidPlayer(player)));
        }
        Ok(self.inner.hole_cards(player).map(|h| h.iter().map(|&c| c as i64).collect()).unwrap_or_default())
    }
    #[getter]
    fn stacks(&self) -> Vec<Chips> {
        self.inner.stacks().to_vec()
    }
    #[getter]
    fn street_bets(&self) -> Vec<Chips> {
        self.inner.street_bets().to_vec()
    }
    /// Chips committed in the whole hand per seat (antes included).
    #[getter]
    fn contributed(&self) -> Vec<Chips> {
        self.inner.contributed().to_vec()
    }
    #[getter]
    fn pot(&self) -> Chips {
        self.inner.pot()
    }
    #[getter]
    fn current_player(&self) -> i64 {
        self.inner.current_player().map_or(-1, |p| p as i64)
    }
    #[getter]
    fn is_terminal(&self) -> bool {
        self.inner.is_terminal()
    }
    #[getter]
    fn folded(&self) -> Vec<bool> {
        self.inner.folded().to_vec()
    }
    #[getter]
    fn all_in(&self) -> Vec<bool> {
        self.inner.all_in().to_vec()
    }
    /// List of `(street, player, Action)`.
    #[getter]
    fn history<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyTuple>>> {
        self.inner
            .history()
            .iter()
            .map(|h| {
                let a = Bound::new(py, PyAction { inner: h.action })?;
                (h.street, h.player, a).into_pyobject(py)
            })
            .collect()
    }
    /// Street bet level every player must match.
    #[getter]
    fn current_bet(&self) -> Chips {
        self.inner.current_bet()
    }
    /// Minimum raise increment on the current street.
    #[getter]
    fn last_raise_size(&self) -> Chips {
        self.inner.last_raise_size()
    }
    #[getter]
    fn num_raises_this_street(&self) -> usize {
        self.inner.num_raises_this_street()
    }
    #[getter]
    fn is_masked(&self) -> bool {
        self.inner.is_masked()
    }
    #[getter]
    fn config(&self) -> PyGameConfig {
        // Starting stacks are recovered from the chips committed and behind.
        let n = self.inner.num_players();
        PyGameConfig {
            num_players: n,
            stacks: (0..n).map(|p| self.inner.stacks()[p] + self.inner.contributed()[p]).collect(),
            small_blind: self.inner.small_blind(),
            big_blind: self.inner.big_blind(),
            ante: self.inner.ante(),
        }
    }

    fn legal_actions(&self) -> PyLegalActions {
        PyLegalActions { inner: self.inner.legal_actions() }
    }

    /// Apply an action in place; raises ValueError if illegal.
    fn apply(&mut self, action: PyRef<'_, PyAction>) -> PyResult<()> {
        self.inner.apply(action.inner).map_err(game_err)
    }

    /// A new state with `action` applied.
    fn child(&self, action: PyRef<'_, PyAction>) -> PyResult<Self> {
        self.inner.child(action.inner).map(|inner| PyGameState { inner }).map_err(game_err)
    }

    fn clone(&self) -> Self {
        PyGameState { inner: self.inner.clone() }
    }

    fn __copy__(&self) -> Self {
        self.clone()
    }

    #[pyo3(signature = (_memo=None))]
    fn __deepcopy__(&self, _memo: Option<&Bound<'_, PyAny>>) -> Self {
        self.clone()
    }

    /// Net chip change per seat; terminal states only.
    fn payoffs(&self) -> PyResult<Vec<Chips>> {
        self.inner.payoffs().map_err(game_err)
    }

    /// Board and betting history as bytes (no hole cards).
    fn public_key<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.inner.public_key())
    }

    /// public_key + seat + the player's hole cards (ascending).
    fn infoset_key<'py>(&self, py: Python<'py>, player: usize) -> PyResult<Bound<'py, PyBytes>> {
        let k = self.inner.infoset_key(player).map_err(game_err)?;
        Ok(PyBytes::new(py, &k))
    }

    /// Copy as seen by `viewer`: other hole cards and undealt cards hidden.
    fn masked(&self, viewer: usize) -> PyResult<Self> {
        if viewer >= self.inner.num_players() {
            return Err(game_err(GameError::InvalidPlayer(viewer)));
        }
        Ok(PyGameState { inner: self.inner.masked(viewer) })
    }

    /// Order in which live players show at showdown (empty if none).
    fn showdown_order(&self) -> Vec<usize> {
        self.inner.showdown_order()
    }

    fn __repr__(&self) -> String {
        format!("GameState({})", self.inner)
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        match other.cast::<PyGameState>() {
            Ok(o) => o.borrow().inner == self.inner,
            Err(_) => false,
        }
    }

    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    #[classmethod]
    fn __class_getitem__(cls: &Bound<'_, PyType>, _item: &Bound<'_, PyAny>) -> Py<PyType> {
        cls.clone().unbind()
    }
}

#[pymodule]
fn poker_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGameConfig>()?;
    m.add_class::<PyAction>()?;
    m.add_class::<PyLegalActions>()?;
    m.add_class::<PyGameState>()?;
    m.add_function(wrap_pyfunction!(evaluate, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate5, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate6, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate7, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_batch, m)?)?;
    m.add_function(wrap_pyfunction!(hand_category, m)?)?;
    m.add_function(wrap_pyfunction!(card_from_str, m)?)?;
    m.add_function(wrap_pyfunction!(card_to_str, m)?)?;
    m.add("FOLD", FOLD)?;
    m.add("CHECK_CALL", CHECK_CALL)?;
    m.add("RAISE", RAISE)?;
    m.add("MAX_PLAYERS", MAX_PLAYERS)?;
    m.add("HAND_CATEGORY_NAMES", eval::CATEGORY_NAMES.to_vec())?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
