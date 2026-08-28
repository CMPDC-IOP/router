# Cache-aware tree memory: measured bounds

Production ran this router in a 1 GiB cgroup and observed RSS climbing to
~986 MiB before being OOM-killed. This document pins down what the
cache-aware approximate tree actually costs, where `max_tree_size` does and
does not bound it, and what this fork changed as a result. Every number
below is produced by `tests/cache_memory_test.rs` (a counting global
allocator measures live heap deltas); re-run with:

```sh
cargo test --test cache_memory_test -- --test-threads=1 --nocapture
```

## What `max_tree_size` actually measures

`max_tree_size` is a **character budget per tenant**. A tenant is one
worker URL inside one model's tree (`Tree::tenant_char_count`), and the
count is *deduplicated*: when tenants share prefix nodes, each character is
billed once per tenant that owns it, not once per insert. It is not a node
count, not a byte count, and not a per-tree or global limit.

A tree exists **per model** (`CacheAwarePolicy.trees`), so the total across
the router is:

```
total chars ≈ Σ over models Σ over workers (per-tenant budget)
total heap  ≈ chars × bytes/char + nodes × bytes/node
```

## Measured costs

| Quantity | Measured value | Source test |
|---|---|---|
| Fixed heap cost per node | **2,409 bytes** | `measures_per_node_fixed_overhead` |
| Per-char cost, ~5 KB prompts | **1.48 bytes/char** | `measures_per_char_cost_for_long_prompts` |
| One tenant filled to the 2^26 default | **150 MiB** | `measures_full_tenant_at_default_max_tree_size` |
| 8 tenants at the default (production shape) | **1,200 MiB** | same (projection printed) |
| Heap after evicting 3.84 M → 1.0 M chars | 75.9 → 20.6 MB | `eviction_caps_char_count_but_heap_stays_above_text_size` |
| Heap released by removing one of two disjoint workers | 49.2 → 24.9 MB (50%) | `remove_tenant_releases_removed_worker_memory` |

The fixed node cost comes from the node's two 8-shard DashMaps (children
and per-tenant access times) plus the text/parent/last-tenant locks: a node
costs ~2.4 KB even if it stores one character. That is why per-char cost is
dominated by *node count*: prompts that share no prefixes pay ~2.4 KB each,
while long prompts amortize it toward the raw text size (~1 byte/char for
ASCII).

## Where the bound was not a bound

Before the fix in this fork, `Tree::insert` never consulted
`max_tree_size`. The cap was enforced only by the background eviction
thread, once per `eviction_interval_secs` (120 s by default in the CLI and
`RouterArgs`). Between passes, growth was unbounded: a burst of inserts
could allocate far past the cap before the next eviction pass. That is the
mechanism behind the production OOM — 8 workers × ~2 KB prompts × steady
traffic reaches ~1 GiB of tree heap well within one eviction interval once
tenants approach the 2^26-char default budget (measured projection:
**1,200 MiB**).

The per-tenant budget also multiplies silently: it is per worker **and**
per model, so 10 models × 8 workers each get their own 2^26 chars.

## The fix in this fork

`Tree::insert_capped(text, tenant, max_chars)` is now the insert path used
by the cache-aware policy's routing hot path. It performs an O(1) check of
the tenant's current char count and skips the insert once the tenant is at
its budget:

- The cap now holds **continuously**, not just at eviction time. Overshoot
  past the budget is bounded by one request's text.
- Skipping an insert loses one cache-tree update, never a request; routing
  decisions for that request were already made from the tree state before
  the insert.
- The tenant resumes inserting as soon as the eviction pass frees space
  (≤ one eviction interval later).
- Worker registration (`tree.insert("", url)`) stays uncapped — it adds no
  characters and must always register the tenant.

With the cap in place, `max_tree_size` becomes a real per-tenant memory
budget: `max_tree_size × ~2.4 bytes/char` is the practical planning figure
for ~2 KB prompts (measured 2.35 bytes/char at the cap), with a worst case
of ~2.4 KB per *node* for adversarial one-char-branching workloads.

`tests/cache_memory_test.rs::insert_capped_bounds_tenant_heap_between_evictions`
asserts the budget holds with the eviction thread disabled, and
`Tree::insert_capped` unit tests cover the budget/resume/zero-budget
semantics.

## Eviction and the high-water mark

Eviction (`evict_tenant_by_size`) removes LRU leaf nodes until a tenant is
under its budget. It does real work — measured: chars 3.84 M → 1.0 M, heap
75.9 → 20.6 MB — but two properties are worth knowing:

1. **Heap does not return to the size of the remaining text.** After the
   eviction above, 1.0 M chars of text remained (~1 MB) but ~20 MB of heap
   was still live: DashMap shard tables keep their high-water capacity, and
   the allocator does not hand freed pages straight back to the OS. RSS
   should be read as a high-water mark, not a live-content gauge.
2. **Eviction is O(nodes in the tree)** — it walks the whole tree and builds
   an LRU heap of leaves. That cost lands on the eviction thread, not the
   request path, but very large trees make each pass expensive; the
   insert-time cap keeps trees at or near their budget so passes stay small.

## Worker and model deletion

`remove_tenant` (used by `remove_worker`) walks the tree, removes the
tenant from every node, deletes nodes that become empty, and drops the
tenant's char-count entry. With disjoint prompt streams the released heap
matches the tenant's share exactly (measured 50% for one of two equal
workers). Shared prefix nodes that other tenants still use correctly
survive.

Two defects made deletion incomplete in practice; both are fixed in this
fork:

1. **DP-expanded workers were never removed from the tree.** The DP branch
   of `Router::remove_worker` re-fetched each worker from the registry
   *after* removing it, so the lookup always failed and the
   `url@rank` tenants of a removed host stayed behind (kept alive by the
   default-policy Arc) until a cache-hit-to-dead-worker self-heal. Fixed by
   capturing each worker's model_id before removal and cleaning the tree
   from that snapshot. Regression test:
   `tests/test_dp_routing.rs::test_dp_remove_worker_cleans_cache_aware_tree`.
2. **Removing a model's last worker skipped tree cleanup.** The code looked
   the policy up through `PolicyRegistry` *after* notifying it of the
   removal; dropping the last worker removes the model's policy entry, so
   the lookup found nothing even though the policy instance (shared with
   the default policy) was still alive and its tree still held the tenant.
   Fixed by cleaning the tree before notifying the registry.

`flush_cache` is unrelated to tree memory: the regular router forwards it
to workers (clearing *worker-side* KV caches), and the router-manager
variant is a stub that never touches local trees. There is no endpoint that
clears the local approximate tree short of removing workers.

## Sizing guidance

For a deployment with `W` workers across `M` model trees under a cgroup of
`C` bytes, with the insert-time cap active:

```
tree heap ≈ M × W × max_tree_size × ~2.4 bytes/char   (≈2 KB prompts)
```

Leave room for the router baseline, in-flight request buffers, and the
high-water effect above. For the production shape that OOM'd (1 GiB
cgroup, 8 workers, one model): `max_tree_size` of 2^26 budgets ~1.2 GiB of
tree heap at fill — oversubscribed by design. Either lower
`max_tree_size` (e.g. 2^24 ≈ 19 MiB/tenant ≈ 150 MiB total) or raise the
cgroup; the default stays 2^26 upstream-compatible, so deployments must set
it explicitly for small cgroups.

## Related documentation fixes in this fork

- `docs/load_balancing/README.md` documented `max_tree_size` as "maximum
  nodes per radix tree" with the internal `CacheAwareConfig::default()`
  values (10000/30 s); the CLI and `RouterArgs` defaults are 2^26/120 s.
  The table now states the unit (chars per tenant) and both default sets.
- `py_src/vllm_router/router.py`'s `Router.__init__` docstring listed
  defaults from an older release (startup 300 s, cache_threshold 0.5,
  abs 32, rel 1.0001, eviction 60 s, payload 256 MB, tree 2^24); it now
  matches `RouterArgs` (600 s, 0.3, 64, 1.5, 120 s, 512 MB, 2^26).
- The policy header comment in `src/policies/cache_aware.rs` described
  `max_tree_size` as nodes per tree; it now describes the per-tenant char
  budget and the insert-time skip.
- `CacheAwarePolicy::get_tenant_char_counts` exposes the per-tenant char
  counts for a model's tree so operators can see how close each worker is
  to its budget.
