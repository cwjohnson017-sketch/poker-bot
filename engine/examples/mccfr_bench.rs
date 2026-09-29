//! MCCFR traversal throughput and abstract betting-tree sizes.
//!
//! ```text
//! cargo run --release --example mccfr_bench [-- <threads> <seconds> <stack_bb> <buckets>]
//! ```

use std::time::Instant;

use poker_engine::abstraction::{ActionAbstraction, CardAbstractionSpec};
use poker_engine::game::GameConfig;
use poker_engine::mccfr::{SolverConfig, Trainer};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let threads: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(4);
    let secs: f64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(10.0);
    let stack_bb: i64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(100);
    let buckets: u32 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(50);

    let game = GameConfig::new(2, stack_bb * 100, 50, 100, 0);
    let actions = ActionAbstraction::default();
    match actions.count_tree(&game, 200_000_000) {
        Some(c) => println!(
            "betting tree ({stack_bb}bb, default actions): decision nodes per street {:?}, total {}",
            c,
            c.iter().sum::<u64>()
        ),
        None => println!("betting tree: more than 200M decision nodes"),
    }
    let config = SolverConfig {
        game,
        actions,
        cards: CardAbstractionSpec {
            buckets: [169, buckets, buckets, buckets],
            hs_samples: 64,
            tables: Default::default(),
        },
        lcfr_discount_every: 50_000,
        lcfr_stop: 1_000_000,
        prune_start: 200_000,
        ..Default::default()
    };
    let mut t = Trainer::new(config).unwrap();
    let start = Instant::now();
    let mut last = Instant::now();
    while start.elapsed().as_secs_f64() < secs {
        let chunk = 20_000;
        let st = t.run(chunk, threads, || true).unwrap();
        if last.elapsed().as_secs_f64() >= 2.0 || start.elapsed().as_secs_f64() >= secs {
            last = Instant::now();
            println!(
                "{:>8.1}s it={:>9} nodes/s={:>10.0} it/s={:>8.0} infosets={:>10} table={:>7.1} MB",
                start.elapsed().as_secs_f64(),
                t.iterations(),
                st.nodes as f64 / st.seconds,
                st.iterations as f64 / st.seconds,
                t.tables.len(),
                t.tables.memory_bytes() as f64 / 1e6
            );
        }
    }
}
