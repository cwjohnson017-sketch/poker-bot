//! Single-threaded throughput: hand evaluations and random hands per second.
//!
//! ```text
//! cargo run --release --example throughput [-- <seconds_per_benchmark>]
//! ```

use std::hint::black_box;
use std::time::Instant;

use poker_engine::eval::evaluate7;
use poker_engine::game::{GameConfig, GameState};
use poker_engine::sim::{fresh_deck, play_random_hand, random_action, shuffled_deck, Rng};

fn main() {
    let secs: f64 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(1.0);
    let mut rng = Rng::new(12345);

    // Pre-generate hands so RNG cost is excluded from the evaluator timing.
    let n_hands = 1 << 20;
    let mut hands = Vec::with_capacity(n_hands);
    let mut deck = fresh_deck();
    for _ in 0..n_hands {
        rng.partial_shuffle(&mut deck, 7);
        let mut h = [0u8; 7];
        h.copy_from_slice(&deck[..7]);
        hands.push(h);
    }
    let mut evals = 0u64;
    let mut acc = 0u64;
    let t = Instant::now();
    while t.elapsed().as_secs_f64() < secs {
        for h in &hands {
            acc = acc.wrapping_add(evaluate7(black_box(h)) as u64);
        }
        evals += n_hands as u64;
    }
    let dt = t.elapsed().as_secs_f64();
    black_box(acc);
    println!("evaluate7:          {:>12.0} evals/s", evals as f64 / dt);

    for players in [2usize, 6] {
        let config = GameConfig::new(players, 20_000, 50, 100, 0);
        let mut deck = fresh_deck();
        let mut count = 0u64;
        let mut sum = 0i64;
        let t = Instant::now();
        while t.elapsed().as_secs_f64() < secs {
            for i in 0..10_000 {
                let p = play_random_hand(&config, i % players, &mut rng, &mut deck).unwrap();
                sum = sum.wrapping_add(p[0]);
            }
            count += 10_000;
        }
        let dt = t.elapsed().as_secs_f64();
        black_box(sum);
        println!("random hands ({players}p):  {:>12.0} hands/s", count as f64 / dt);
    }

    // Clone-heavy workload typical of tree traversal: clone at every node.
    let config = GameConfig::default();
    let mut count = 0u64;
    let t = Instant::now();
    while t.elapsed().as_secs_f64() < secs {
        for i in 0..10_000 {
            let deck = shuffled_deck(&mut rng);
            let mut s = GameState::new_hand(&config, i % 2, &deck).unwrap();
            while !s.is_terminal() {
                let a = random_action(&s, &mut rng);
                s = s.child(a).unwrap();
                black_box(s.public_key());
            }
            black_box(s.payoffs().unwrap());
        }
        count += 10_000;
    }
    let dt = t.elapsed().as_secs_f64();
    println!("hands via child()+public_key (2p): {:>8.0} hands/s", count as f64 / dt);
}
