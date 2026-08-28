//! Reproducible heap measurements for the cache-aware approximate tree.
//!
//! `max_tree_size` counts *characters per tenant* (a tenant is one worker URL
//! inside one model's tree), not nodes and not bytes. These tests measure the
//! real heap cost behind that unit with a counting global allocator, so the
//! production memory bound can be derived from evidence instead of guesswork.
//! See `docs/load_balancing/cache_aware_memory.md` for the analysis these
//! numbers feed into.
//!
//! Facts encoded here (values are deterministic for the pinned dependency
//! versions; assertions keep wide margins so they act as structural guards,
//! not exact pins):
//! - Every node carries two 8-shard DashMaps, so a node costs a fixed couple of
//!   kilobytes regardless of how much text it holds.
//! - With realistic ~2 KB prompts that amortizes to roughly 2 bytes per char,
//!   so filling one tenant to the default `max_tree_size` of 2^26 chars costs
//!   on the order of a hundred+ MB of heap.
//! - Nothing bounds growth between eviction passes: `insert` never consults
//!   `max_tree_size`, so a burst can allocate far past the cap before the
//!   background eviction thread wakes up.
//! - Eviction brings the char counters down, but surviving DashMap shard
//!   tables keep their high-water capacity, so live heap does not return to
//!   the post-eviction text size.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::{atomic::AtomicUsize, atomic::Ordering, Mutex};

use vllm_router_rs::tree::Tree;

static ALLOCATED: AtomicUsize = AtomicUsize::new(0);
static DEALLOCATED: AtomicUsize = AtomicUsize::new(0);

struct Counting;

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            ALLOCATED.fetch_add(layout.size(), Ordering::Relaxed);
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
        DEALLOCATED.fetch_add(layout.size(), Ordering::Relaxed);
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new_ptr = unsafe { System.realloc(ptr, layout, new_size) };
        if !new_ptr.is_null() {
            DEALLOCATED.fetch_add(layout.size(), Ordering::Relaxed);
            ALLOCATED.fetch_add(new_size, Ordering::Relaxed);
        }
        new_ptr
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

fn heap_live() -> usize {
    ALLOCATED.load(Ordering::Relaxed) - DEALLOCATED.load(Ordering::Relaxed)
}

/// Serialize the measurement sections of every test in this binary: a delta
/// only means something if no sibling test allocates inside the window.
static MEASURE_LOCK: Mutex<()> = Mutex::new(());

/// Deterministic PRNG so repeated runs measure identical trees.
struct XorShift(u64);

impl XorShift {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    fn hex_string(&mut self, len: usize) -> String {
        const HEX: &[u8; 16] = b"0123456789abcdef";
        let mut s = String::with_capacity(len);
        while s.len() < len {
            let mut r = self.next_u64();
            for _ in 0..16 {
                if s.len() == len {
                    break;
                }
                s.push(HEX[(r & 0xF) as usize] as char);
                r >>= 4;
            }
        }
        s
    }
}

/// A leading char unique per string: children are keyed by first char, so
/// every insert lands as exactly one fresh leaf with no path splitting.
fn unique_lead(i: usize) -> char {
    // CJK block: plenty of distinct scalars, all below the surrogate range.
    char::from_u32(0x4E00 + i as u32).expect("lead char range is valid")
}

fn tenant_chars(tree: &Tree, tenant: &str) -> usize {
    tree.tenant_char_count.get(tenant).map(|v| *v).unwrap_or(0)
}

#[test]
fn measures_per_node_fixed_overhead() {
    let _guard = MEASURE_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    const NODES: usize = 20_000;
    const CHARS_PER_NODE: usize = 16;

    let mut rng = XorShift(0x9E37_79B9_7F4A_7C15);
    // Build the strings up front so their allocations cancel out of the delta.
    let texts: Vec<String> = (0..NODES)
        .map(|i| format!("{}{}", unique_lead(i), rng.hex_string(CHARS_PER_NODE - 1)))
        .collect();

    let tree = Tree::new();
    let before = heap_live();
    for text in &texts {
        tree.insert(text, "worker-a");
    }
    let delta = heap_live() - before;

    let per_node = delta / NODES;
    println!("per-node heap cost: {per_node} bytes (text: {CHARS_PER_NODE} chars/node)");
    println!(
        "per-char amortized at this node size: {:.1} bytes",
        per_node as f64 / CHARS_PER_NODE as f64
    );

    // Two 8-shard DashMaps dominate the fixed node cost. Keep wide margins so
    // the guard flags structural regressions without pinning exact layouts.
    assert!(
        (256..8 * 1024).contains(&per_node),
        "per-node heap cost {per_node} bytes is outside the plausible band"
    );

    assert_eq!(
        tenant_chars(&tree, "worker-a"),
        NODES * CHARS_PER_NODE,
        "char counter must match the inserted characters"
    );
}

#[test]
fn measures_per_char_cost_for_long_prompts() {
    let _guard = MEASURE_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    const PROMPTS: usize = 2_000;
    const CHARS_PER_PROMPT: usize = 5_000;

    let mut rng = XorShift(0x0DDB_1A5E_5EED_5EED);
    let texts: Vec<String> = (0..PROMPTS)
        .map(|i| format!("{}{}", unique_lead(i), rng.hex_string(CHARS_PER_PROMPT - 1)))
        .collect();

    let tree = Tree::new();
    let before = heap_live();
    for text in &texts {
        tree.insert(text, "worker-a");
    }
    let delta = heap_live() - before;

    let per_char = delta as f64 / (PROMPTS * CHARS_PER_PROMPT) as f64;
    println!("long-prompt per-char heap cost: {per_char:.2} bytes");

    // Long texts amortize the fixed node overhead; flag only gross inflation.
    assert!(
        per_char < 8.0,
        "per-char heap cost {per_char:.2} bytes is implausibly high for long prompts"
    );
}

#[test]
fn measures_full_tenant_at_default_max_tree_size() {
    let _guard = MEASURE_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    // The CLI default max_tree_size is 2^26 chars per tenant. Model one tenant
    // filled by ~2 KB prompts (4-char unique head + 1,996 hex chars): about
    // 33,554 requests reach the cap, with no eviction call in between.
    const REQUESTS: usize = 33_554;
    const CHARS_PER_REQUEST: usize = 2_000;
    const DEFAULT_MAX_TREE_SIZE: usize = 1 << 26;

    let mut rng = XorShift(0x5EED_5EED_5EED_5EED);
    let texts: Vec<String> = (0..REQUESTS)
        .map(|i| format!("P{:03x}{}", i, rng.hex_string(CHARS_PER_REQUEST - 5)))
        .collect();

    let tree = Tree::new();
    let before = heap_live();
    for text in &texts {
        tree.insert(text, "worker-a");
    }
    let delta = heap_live() - before;

    let chars = tenant_chars(&tree, "worker-a");
    println!(
        "tenant at max_tree_size: {chars} chars (cap {DEFAULT_MAX_TREE_SIZE}), heap {delta} bytes ({:.0} MiB)",
        delta as f64 / (1024.0 * 1024.0)
    );
    println!(
        "projected 8 workers at this fill: {:.0} MiB (a 1 GiB cgroup OOMs)",
        delta as f64 * 8.0 / (1024.0 * 1024.0)
    );

    // Every insert must be counted and land just under the cap. Random hex
    // content shares more prefixes than a back-of-the-envelope estimate, so
    // accept anything within ~6% of the cap.
    assert!(
        chars <= DEFAULT_MAX_TREE_SIZE
            && chars > DEFAULT_MAX_TREE_SIZE - DEFAULT_MAX_TREE_SIZE / 16,
        "inserted {chars} chars, expected just under the 2^26 cap"
    );

    // Wide evidence band: roughly a hundred MB per tenant at the cap.
    assert!(
        (32 * 1024 * 1024..512 * 1024 * 1024).contains(&delta),
        "full-tenant heap {delta} bytes is outside the expected order of magnitude"
    );
}

#[test]
fn eviction_caps_char_count_but_heap_stays_above_text_size() {
    let _guard = MEASURE_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    const INSERTS: usize = 30_000;
    const CHARS_PER_INSERT: usize = 128;
    const EVICT_TO: usize = 1_000_000;

    let mut rng = XorShift(0x1234_5678_9ABC_DEF0);
    let texts: Vec<String> = (0..INSERTS)
        .map(|i| format!("{}{}", unique_lead(i), rng.hex_string(CHARS_PER_INSERT - 1)))
        .collect();

    let tree = Tree::new();
    let before = heap_live();
    for text in &texts {
        tree.insert(text, "worker-a");
    }
    let filled = heap_live() - before;
    let filled_chars = tenant_chars(&tree, "worker-a");

    tree.evict_tenant_by_size(EVICT_TO);

    let after = heap_live() - before;
    let after_chars = tenant_chars(&tree, "worker-a");
    println!(
        "eviction: chars {filled_chars} -> {after_chars} (target {EVICT_TO}), heap {filled} -> {after} bytes"
    );

    assert!(
        after_chars <= EVICT_TO,
        "eviction must bring the tenant under the cap, got {after_chars}"
    );
    assert!(
        after < filled,
        "evicted nodes must release heap: {filled} -> {after}"
    );
    // Surviving shard tables keep their high-water capacity, so the heap does
    // not fall back to the size of the text that remains. Guard only against
    // the implausible extremes.
    assert!(
        after > after_chars / 2,
        "retained heap {after} bytes fell implausibly far below the remaining {after_chars} chars"
    );
}

#[test]
fn insert_capped_bounds_tenant_heap_between_evictions() {
    let _guard = MEASURE_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    // 200 requests of ~130 chars against a 4,000-char budget, with no
    // eviction call at all: the insert-time cap is the only bound, and it
    // must hold without waiting for the background eviction thread.
    const REQUESTS: usize = 200;
    const CHARS_PER_REQUEST: usize = 130;
    const BUDGET: usize = 4_000;

    let mut rng = XorShift(0x0F1E_2D3C_4B5A_6978);
    let texts: Vec<String> = (0..REQUESTS)
        .map(|i| {
            format!(
                "{}{}",
                unique_lead(i),
                rng.hex_string(CHARS_PER_REQUEST - 1)
            )
        })
        .collect();

    let tree = Tree::new();
    let before = heap_live();
    for text in &texts {
        tree.insert_capped(text, "worker-a", BUDGET);
    }
    let delta = heap_live() - before;

    let chars = tenant_chars(&tree, "worker-a");
    println!(
        "capped tenant: {chars} chars (budget {BUDGET}), heap {delta} bytes ({:.1} KiB)",
        delta as f64 / 1024.0
    );

    // The budget holds even though nothing ever evicted, with overshoot
    // bounded by one request's text.
    assert!(
        chars <= BUDGET + CHARS_PER_REQUEST,
        "capped tenant grew to {chars} chars, budget {BUDGET}"
    );
    assert!(
        chars >= BUDGET,
        "capped tenant should have filled its budget, got {chars}"
    );

    // And the heap reflects it: ~2.4 KB/node × ~31 nodes, not 200 nodes.
    assert!(
        delta < BUDGET * 64,
        "capped tenant heap {delta} bytes is out of proportion to its {BUDGET}-char budget"
    );
}

#[test]
fn remove_tenant_releases_removed_worker_memory() {
    let _guard = MEASURE_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    const INSERTS_PER_WORKER: usize = 10_000;
    const CHARS_PER_INSERT: usize = 64;

    let build = |lead_base: u32, salt: u64| {
        let mut rng = XorShift(salt);
        (0..INSERTS_PER_WORKER)
            .map(|i| {
                format!(
                    "{}{}",
                    char::from_u32(lead_base + i as u32).expect("lead char range is valid"),
                    rng.hex_string(CHARS_PER_INSERT - 1)
                )
            })
            .collect::<Vec<_>>()
    };
    // Disjoint leading-char ranges: the workers then own fully disjoint
    // subtrees, so no node text or tenant map is shared between them.
    let worker_a_texts = build(0x4E00, 0xAAAA_0000_1111_2222);
    let worker_b_texts = build(0x4E00 + INSERTS_PER_WORKER as u32, 0xBBBB_0000_3333_4444);

    let tree = Tree::new();
    let before = heap_live();
    for text in &worker_a_texts {
        tree.insert(text, "worker-a");
    }
    for text in &worker_b_texts {
        tree.insert(text, "worker-b");
    }
    let both = heap_live() - before;

    tree.remove_tenant("worker-a");
    let after = heap_live() - before;

    println!("both workers: {both} bytes, after removing worker-a: {after} bytes");
    assert!(
        !tree.tenant_char_count.contains_key("worker-a"),
        "removed tenant must disappear from the char counter"
    );
    // The two workers hold disjoint subtrees of equal size, so removal should
    // free close to half; keep a generous margin.
    assert!(
        after <= both * 6 / 10,
        "removing one of two equal workers released too little: {both} -> {after} bytes"
    );
    assert!(
        tenant_chars(&tree, "worker-b") == INSERTS_PER_WORKER * CHARS_PER_INSERT,
        "surviving tenant must keep its char count"
    );
}
