# fastprp — Fast Pseudo-Random Permutations for Small Domains

Rust implementation of Stefanov & Shi (2012), "FastPRP: Fast Pseudo-Random
Permutations for Small Domains."  Provides bijective, key-controlled
permutations over arbitrary domains {0, 1, ..., N-1} with efficient
single-element lookup (no need to materialise the full table).

## Quick start

```toml
# Cargo.toml
[dependencies]
fastprp = { path = "../fastprp" }   # or git url once published
rayon   = "1.10"                     # only needed for parallel batch
```

```rust
use fastprp::FastPrp;

let key = [0u8; 16]; // 128-bit AES key
let n   = 1_000_000; // domain size

let prp = FastPrp::new(&key, n);

// Single-element encrypt / decrypt  (O(sqrt(N) log N) each)
let y = prp.permute(42);       // 42  -> some y in [0, N)
let x = prp.unpermute(y);      // y   -> 42
assert_eq!(x, 42);
```

## API reference

### Construction

```rust
// Default cache stride  s = 2*sqrt(N).
// Cache memory ≈ O(N / s * log N) ≈ O(sqrt(N) * log N).
let prp = FastPrp::new(key: &[u8; 16], n: u64) -> FastPrp;

// Custom stride — smaller stride = more cache memory, faster per-lookup.
//   stride = sqrt(N)/4  →  ~3.7 MB cache, ~2x faster lookups
//   stride = 2*sqrt(N)  →  ~400 KB cache  (default)
let prp = FastPrp::with_stride(key: &[u8; 16], n: u64, stride: u64) -> FastPrp;
```

`FastPrp` is `Sync + Send` — a single instance can be shared across threads
for concurrent `permute()` / `unpermute()` calls (all read-only).

### Single-element operations

```rust
prp.permute(x: u64) -> u64      // PRP encrypt:  x  -> y
prp.unpermute(y: u64) -> u64    // PRP decrypt:  y  -> x
```

Both are O(sqrt(N) * log N) using the internal counter cache.  No heap
allocation per call.  Thread-safe (`&self`).

### 4-way batched lookup

```rust
prp.permute_4(inputs: [u64; 4]) -> [u64; 4]
```

Processes 4 independent permutations with:
- Pipelined AES bit-lookups (4 blocks encrypted in one `encrypt_blocks` call)
- Shared `ones_before_alpha` scan across the two C0 queries per element (25%
  fewer AES blocks than 4x individual `permute`)
- Small-partition optimisation: when the partition shrinks below `2 * stride`,
  precomputes AES blocks for the range and answers C0 queries from the buffer

**Recommended pattern for many lookups:**

```rust
use rayon::prelude::*;

let inputs: Vec<u64> = (0..16384).collect();
let results: Vec<u64> = inputs
    .par_chunks(4)
    .flat_map(|chunk| {
        let mut batch = [0u64; 4];
        for (i, &v) in chunk.iter().enumerate() { batch[i] = v; }
        let out = prp.permute_4(batch);
        out[..chunk.len()].to_vec()
    })
    .collect();
```

### Full-domain batch permute

```rust
prp.batch_permute() -> Vec<u64>
```

Returns a table where `result[x] = permute(x)` for all x in {0, ..., N-1}.
Uses the Section 3.2 radix-sort view: level-by-level stable partitioning with
bulk AES generation, branchless writes, u32 internal arrays, and double
buffering.  O(N log N) time, O(N) memory.

**This is inherently single-threaded** — depth d+1 depends on depth d.
Parallelising the scattered writes makes it *slower* due to cache contention.

### Inspecting the instance

```rust
prp.domain_size()     -> u64    // N
prp.cache_stride()    -> u64    // s
prp.cache_size_bytes() -> usize // total cache memory
```

## Recommended patterns

### Many ad-hoc lookups (different keys)

Use `permute()` / `unpermute()` directly.  Each `FastPrp::new()` builds
the counter cache in O(N / stride * log N) time.  For N = 2^27 with
default stride this takes ~4 seconds; with `stride = sqrt(N)/4` it takes
~15 seconds but per-lookup drops from ~300 us to ~100 us.

### Offline full-table generation (one key, all N elements)

Split into independent buckets and parallelise:

```rust
use rayon::prelude::*;

let total_n: u64 = 1 << 27;
let num_buckets: u64 = 200;
let bucket_size = total_n / num_buckets;

let tables: Vec<Vec<u64>> = (0..num_buckets)
    .into_par_iter()
    .map(|b| {
        let mut k = master_key;
        k[0] = (b & 0xFF) as u8;
        k[1] = ((b >> 8) & 0xFF) as u8;
        let prp = FastPrp::new(&k, bucket_size);
        prp.batch_permute()
    })
    .collect();
```

Typical performance at N = 2^27 total:

| Buckets | Bucket size | Wall-clock (24 cores) |
|---------|-------------|-----------------------|
| 80      | 1.68M       | 1.0 s                 |
| 200     | 671K        | 0.85 s                |
| 400     | 336K        | 0.72 s                |

Sweet spot: bucket working set fits in L2 cache (~300K–700K elements).

### Streaming the result

`batch_permute()` returns `result[x] = y`.  To iterate in *output* order
(position 0, 1, 2, ...) you need the inverse table:

```rust
let fwd = prp.batch_permute();          // fwd[x] = y
let mut inv = vec![0u64; fwd.len()];    // inv[y] = x
for (x, &y) in fwd.iter().enumerate() {
    inv[y as usize] = x as u64;
}
// Now inv[0], inv[1], ... gives the element at each output position.
```

## Performance reference (Apple M-series, single core)

| N     | `new()` | `permute()` | `permute_4()` | `batch_permute()` |
|-------|---------|-------------|----------------|-------------------|
| 2^14  | <1 ms   | ~8 us       | ~27 us (4 elem)| ~1 ms             |
| 2^20  | ~60 ms  | ~53 us      | ~180 us        | ~87 ms            |
| 2^27  | ~4 s    | ~297 us     | ~1.1 ms        | ~14.7 s           |

With `stride = sqrt(N)/4` (larger cache):

| N     | `new()` | `permute()` | Cache size |
|-------|---------|-------------|------------|
| 2^27  | ~15 s   | ~107 us     | 3.7 MB     |

## Security

The construction is a PRP (pseudo-random permutation) with security
reducible to AES-128.  The max recursion depth is set to `16 * ln(N)`,
giving negligible (< 2^{-80}) probability that any partition remains
unresolved.

Different keys produce independent permutations.  The same key always
produces the same permutation (deterministic, no randomness beyond the key).
