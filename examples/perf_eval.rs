//! Performance evaluation harness for single-threaded optimisations.
//!
//! Measures three independent things against a large synthetic batch of words:
//!   1. Buffered vs unbuffered I/O when writing results.
//!   2. Reusing the output `String` (`clear()`) vs allocating a fresh one per word.
//!   3. Reusing one `IncrementalBuffer` (`clear()`) vs `transform_buffer` (fresh buffer per word).
//!
//! Each variant reports wall-clock ns/word and heap allocations/word (via a counting
//! allocator), so the cost of `Syllable`'s per-call allocations is visible directly.
//!
//! Run with: `cargo run --release --example perf_eval [word_count]`

use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::io::{self, Write};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use vi::methods::IncrementalBuffer;

// --- Counting allocator: tracks number of allocations and bytes requested. ---

static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);
static ALLOC_BYTES: AtomicUsize = AtomicUsize::new(0);

struct CountingAlloc;

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        ALLOC_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        System.alloc(layout)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
    }
}

#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;

fn alloc_snapshot() -> (usize, usize) {
    (
        ALLOC_COUNT.load(Ordering::Relaxed),
        ALLOC_BYTES.load(Ordering::Relaxed),
    )
}

/// Representative telex input sequences covering tones, modifications and finals.
const BASE_WORDS: &[&str] = &[
    "vieejt", "nam", "ddaay", "nghieng", "tuyeets", "hoaf", "ngoonf", "ddoongf", "thuongw", "chuw",
    "nguoiwf", "khoong", "ddaau", "buoonf", "quee", "gif", "trois", "xanh", "dduowngf", "muoiws",
    "anh", "em", "hocj", "sinh", "vieen", "truowngf", "ddaij", "hocj", "baachs", "khoa",
];

fn build_corpus(n: usize) -> Vec<String> {
    (0..n)
        .map(|i| BASE_WORDS[i % BASE_WORDS.len()].to_string())
        .collect()
}

struct Measured {
    name: &'static str,
    ns_per_word: f64,
    allocs_per_word: f64,
    bytes_per_word: f64,
}

fn measure<F: FnMut()>(name: &'static str, n: usize, mut f: F) -> Measured {
    // Warmup.
    f();
    let (a0, b0) = alloc_snapshot();
    let t0 = Instant::now();
    f();
    let elapsed = t0.elapsed();
    let (a1, b1) = alloc_snapshot();
    Measured {
        name,
        ns_per_word: elapsed.as_nanos() as f64 / n as f64,
        allocs_per_word: (a1 - a0) as f64 / n as f64,
        bytes_per_word: (b1 - b0) as f64 / n as f64,
    }
}

fn print_row(m: &Measured) {
    println!(
        "  {:<32} {:>10.2} ns/word {:>8.2} allocs/word {:>9.1} bytes/word",
        m.name, m.ns_per_word, m.allocs_per_word, m.bytes_per_word
    );
}

fn main() {
    let n: usize = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(200_000);

    let corpus = build_corpus(n);
    println!("Corpus: {n} words\n");

    // === 2 & 3: transform engine + output allocation ===
    println!("Transform (CPU + allocation):");

    // Baseline: fresh String per word (mirrors the doc examples / repl).
    let fresh_string = measure("transform_buffer + fresh String", n, || {
        for w in &corpus {
            let mut out = String::new();
            vi::transform_buffer(&vi::TELEX, w.chars(), &mut out);
            black_box(&out);
        }
    });

    // Reuse the output String across calls.
    let reuse_string = measure("transform_buffer + reused String", n, || {
        let mut out = String::new();
        for w in &corpus {
            out.clear();
            vi::transform_buffer(&vi::TELEX, w.chars(), &mut out);
            black_box(&out);
        }
    });

    // Reuse one IncrementalBuffer (clear() instead of allocating a new buffer per word).
    let reuse_buffer = measure("reused IncrementalBuffer", n, || {
        let mut buf = IncrementalBuffer::new(&vi::TELEX);
        for w in &corpus {
            buf.clear();
            for ch in w.chars() {
                buf.push(ch);
            }
            black_box(buf.view());
        }
    });

    print_row(&fresh_string);
    print_row(&reuse_string);
    print_row(&reuse_buffer);

    // === 1: Buffered vs unbuffered I/O ===
    println!("\nI/O (writing {n} result lines to /dev/null):");

    // Pre-transform once so I/O is the only thing measured.
    let outputs: Vec<String> = corpus
        .iter()
        .map(|w| {
            let mut out = String::new();
            vi::transform_buffer(&vi::TELEX, w.chars(), &mut out);
            out
        })
        .collect();

    let io_unbuffered = measure("unbuffered (write per line)", n, || {
        let sink = std::fs::OpenOptions::new()
            .write(true)
            .open("/dev/null")
            .unwrap();
        let mut sink = sink;
        for line in &outputs {
            sink.write_all(line.as_bytes()).unwrap();
            sink.write_all(b"\n").unwrap();
        }
    });

    let io_buffered = measure("BufWriter", n, || {
        let sink = std::fs::OpenOptions::new()
            .write(true)
            .open("/dev/null")
            .unwrap();
        let mut sink = io::BufWriter::new(sink);
        for line in &outputs {
            sink.write_all(line.as_bytes()).unwrap();
            sink.write_all(b"\n").unwrap();
        }
        sink.flush().unwrap();
    });

    print_row(&io_unbuffered);
    print_row(&io_buffered);

    // === Summary ===
    println!("\nSpeedups:");
    println!(
        "  reused String   vs fresh String : {:.2}x time, {:.2}x fewer allocs",
        fresh_string.ns_per_word / reuse_string.ns_per_word,
        fresh_string.allocs_per_word / reuse_string.allocs_per_word.max(0.001)
    );
    println!(
        "  reused buffer    vs fresh String : {:.2}x time, {:.2}x fewer allocs",
        fresh_string.ns_per_word / reuse_buffer.ns_per_word,
        fresh_string.allocs_per_word / reuse_buffer.allocs_per_word.max(0.001)
    );
    println!(
        "  BufWriter        vs unbuffered   : {:.2}x time",
        io_unbuffered.ns_per_word / io_buffered.ns_per_word
    );
}
