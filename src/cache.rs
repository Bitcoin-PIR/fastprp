use crate::bitstring::BitstringGen;

/// Partition-boundary cache (Section 4.2: Counter Alignment).
///
/// At each depth d, stores cumulative C1(β_d, 0, pos) at every partition
/// boundary position. This makes C0(β_d, alpha, len) a pure O(1) lookup
/// when both alpha and alpha+len are partition boundaries — which they
/// always are in the permute/unpermute recursion.
///
/// Also provides cumulative_ones(d, alpha) so that the stride cache only
/// needs one scan (for alpha+x) instead of three.
pub struct PartitionCache {
    /// For each depth: sorted Vec of (position, cumulative_C1_from_0_to_pos).
    depths: Vec<Vec<(u64, u32)>>,
}

impl PartitionCache {
    /// Build by simulating the partition tree (boundary tracking only, no element shuffling).
    pub fn new(gen: &BitstringGen, n: u64, num_depths: u32) -> Self {
        let mut depths: Vec<Vec<(u64, u32)>> = Vec::new();
        let mut partitions: Vec<(u64, u64)> = vec![(0, n)];

        for d in 0..num_depths {
            if partitions.iter().all(|&(_, l)| l <= 1) {
                break;
            }

            // Boundaries are contiguous: [alpha_0, alpha_1, ..., alpha_k, N].
            // Scan β_d left-to-right, accumulating C1 at each boundary.
            let num_bounds = partitions.len() + 1;
            let mut entries: Vec<(u64, u32)> = Vec::with_capacity(num_bounds);

            let mut prev_pos = 0u64;
            let mut cum_c1 = 0u32;

            for &(alpha, _) in &partitions {
                if alpha > prev_pos {
                    cum_c1 += gen.count_ones_range(d, prev_pos, alpha - prev_pos) as u32;
                }
                entries.push((alpha, cum_c1));
                prev_pos = alpha;
            }
            // Final boundary at N.
            let &(last_a, last_l) = partitions.last().unwrap();
            let end = last_a + last_l;
            if end > prev_pos {
                cum_c1 += gen.count_ones_range(d, prev_pos, end - prev_pos) as u32;
            }
            entries.push((end, cum_c1));

            // Split partitions for depth d+1.
            let mut new_partitions: Vec<(u64, u64)> =
                Vec::with_capacity(partitions.len() * 2);

            for (i, &(alpha, len)) in partitions.iter().enumerate() {
                if len <= 1 {
                    new_partitions.push((alpha, len));
                    continue;
                }
                let c1 = (entries[i + 1].1 - entries[i].1) as u64;
                let c0 = len - c1;
                if c0 > 0 { new_partitions.push((alpha, c0)); }
                if c1 > 0 { new_partitions.push((alpha + c0, c1)); }
            }

            depths.push(entries);
            partitions = new_partitions;
        }

        Self { depths }
    }

    /// Look up cumulative C1(β_d, 0, pos). Returns None if pos is not a boundary.
    #[inline]
    pub fn cumulative_ones(&self, d: u32, pos: u64) -> Option<u64> {
        let d = d as usize;
        if d >= self.depths.len() { return None; }
        let entries = &self.depths[d];
        match entries.binary_search_by_key(&pos, |&(p, _)| p) {
            Ok(i) => Some(entries[i].1 as u64),
            Err(_) => None,
        }
    }

    /// Compute C0(β_d, alpha, len) from cached boundaries. O(1).
    /// Returns None if alpha or alpha+len is not a cached boundary.
    #[inline]
    pub fn c0(&self, d: u32, alpha: u64, len: u64) -> Option<u64> {
        let c1_a = self.cumulative_ones(d, alpha)?;
        let c1_e = self.cumulative_ones(d, alpha + len)?;
        Some(len - (c1_e - c1_a))
    }

    /// Total memory in bytes.
    pub fn size_bytes(&self) -> usize {
        self.depths.iter().map(|v| v.len() * 12).sum()
    }
}

/// Cached counters for efficient C1 computation.
///
/// Stores C1(β_d, 0, s*i) for each depth d and stride index i,
/// where s is the cache stride.
pub struct CounterCache {
    /// Cache stride (interval between cached counters)
    pub stride: u64,
    /// Number of depths cached
    pub num_depths: u32,
    /// Domain size
    pub n: u64,
    /// Cached C1 values: cache[d][i] = C1(β_d, 0, stride * (i+1))
    /// i.e., cumulative count of ones from position 0 to stride*(i+1)-1
    cache: Vec<Vec<u32>>,
}

impl CounterCache {
    /// Build the counter cache for the given bitstring generator.
    /// stride = 2 * sqrt(N) by default.
    pub fn new(gen: &BitstringGen, stride: u64) -> Self {
        let n = gen.domain_size();
        // Cache depths up to log2(N/stride)
        let num_depths = if n > stride {
            ((n as f64) / (stride as f64)).log2().ceil() as u32 + 1
        } else {
            1
        };
        // Add some extra depths for safety
        let num_depths = num_depths + 4;
        let entries_per_depth = (n / stride) as usize;

        let mut cache = Vec::with_capacity(num_depths as usize);

        for d in 0..num_depths {
            let mut depth_cache = Vec::with_capacity(entries_per_depth);
            let mut cumulative: u64 = 0;
            for i in 0..entries_per_depth {
                let start = (i as u64) * stride;
                let ones = gen.count_ones_range(d, start, stride);
                cumulative += ones;
                depth_cache.push(cumulative as u32);
            }
            cache.push(depth_cache);
        }

        Self {
            stride,
            num_depths,
            n,
            cache,
        }
    }

    /// Get cached cumulative C1(β_d, 0, stride * (i+1)).
    /// Returns 0 if i < 0 (conceptually).
    #[inline]
    fn get_cached(&self, d: u32, i: usize) -> u64 {
        if d >= self.num_depths || i >= self.cache[d as usize].len() {
            return 0; // Fall back to uncached
        }
        self.cache[d as usize][i] as u64
    }

    /// Total cache memory usage in bytes.
    pub fn size_bytes(&self) -> usize {
        self.cache.iter().map(|v| v.len() * 4).sum()
    }

    /// Check if depth d is cached.
    #[inline]
    pub fn is_depth_cached(&self, d: u32) -> bool {
        d < self.num_depths && !self.cache[d as usize].is_empty()
    }

    /// Compute cumulative C1(β_d, 0, pos) using the stride cache + remainder scan.
    /// Used when the partition cache provides c1_alpha but we still need c1(alpha+x).
    #[inline]
    pub fn cumulative_ones(&self, gen: &BitstringGen, d: u32, pos: u64) -> u64 {
        if pos == 0 { return 0; }
        if !self.is_depth_cached(d) {
            return gen.count_ones_range(d, 0, pos);
        }
        let s = self.stride;
        let full = pos / s;
        let cached = if full > 0 {
            self.get_cached(d, (full - 1) as usize)
        } else {
            0
        };
        let rem_start = full * s;
        if rem_start < pos {
            cached + gen.count_ones_range(d, rem_start, pos - rem_start)
        } else {
            cached
        }
    }

    /// Compute C1(β_d, α, count) using cached values.
    /// Returns the number of 1-bits in β_d[α..α+count].
    pub fn c1(&self, gen: &BitstringGen, d: u32, alpha: u64, count: u64) -> u64 {
        if count == 0 {
            return 0;
        }

        if !self.is_depth_cached(d) {
            return gen.count_ones_range(d, alpha, count);
        }

        let s = self.stride;
        let end = alpha + count;

        // Find the cached boundaries that bracket [alpha, end)
        // cached[i] = C1(β_d, 0, s*(i+1))

        // Cumulative ones from 0 to alpha
        let ones_before_alpha = if alpha == 0 {
            0
        } else {
            let full_strides_alpha = alpha / s;
            let cached_alpha = if full_strides_alpha > 0 {
                self.get_cached(d, (full_strides_alpha - 1) as usize)
            } else {
                0
            };
            let remainder_start = full_strides_alpha * s;
            if remainder_start < alpha {
                cached_alpha + gen.count_ones_range(d, remainder_start, alpha - remainder_start)
            } else {
                cached_alpha
            }
        };

        // Cumulative ones from 0 to end
        let ones_before_end = if end == 0 {
            0
        } else {
            let full_strides_end = end / s;
            let cached_end = if full_strides_end > 0 {
                self.get_cached(d, (full_strides_end - 1) as usize)
            } else {
                0
            };
            let remainder_start = full_strides_end * s;
            if remainder_start < end {
                cached_end + gen.count_ones_range(d, remainder_start, end - remainder_start)
            } else {
                cached_end
            }
        };

        ones_before_end - ones_before_alpha
    }

    /// Compute C0(β_d, α, count) using cached values.
    #[inline]
    pub fn c0(&self, gen: &BitstringGen, d: u32, alpha: u64, count: u64) -> u64 {
        count - self.c1(gen, d, alpha, count)
    }

    /// Compute C0(β_d, α, x) and C0(β_d, α, len) in one call,
    /// sharing the `ones_before_alpha` scan that both need.
    /// Returns (c0_x, c0_len).
    pub fn c0_pair(&self, gen: &BitstringGen, d: u32, alpha: u64, x: u64, len: u64) -> (u64, u64) {
        if !self.is_depth_cached(d) {
            return (
                x - gen.count_ones_range(d, alpha, x),
                len - gen.count_ones_range(d, alpha, len),
            );
        }

        let s = self.stride;

        // Shared: ones_before_alpha — computed ONCE instead of twice.
        let ones_before_alpha = if alpha == 0 {
            0
        } else {
            let full = alpha / s;
            let cached = if full > 0 {
                self.get_cached(d, (full - 1) as usize)
            } else {
                0
            };
            let rem_start = full * s;
            if rem_start < alpha {
                cached + gen.count_ones_range(d, rem_start, alpha - rem_start)
            } else {
                cached
            }
        };

        // ones_before(alpha + x)
        let end_x = alpha + x;
        let ones_before_end_x = {
            let full = end_x / s;
            let cached = if full > 0 {
                self.get_cached(d, (full - 1) as usize)
            } else {
                0
            };
            let rem_start = full * s;
            if rem_start < end_x {
                cached + gen.count_ones_range(d, rem_start, end_x - rem_start)
            } else {
                cached
            }
        };

        // ones_before(alpha + len)
        let end_len = alpha + len;
        let ones_before_end_len = {
            let full = end_len / s;
            let cached = if full > 0 {
                self.get_cached(d, (full - 1) as usize)
            } else {
                0
            };
            let rem_start = full * s;
            if rem_start < end_len {
                cached + gen.count_ones_range(d, rem_start, end_len - rem_start)
            } else {
                cached
            }
        };

        let c1_x = ones_before_end_x - ones_before_alpha;
        let c1_len = ones_before_end_len - ones_before_alpha;
        (x - c1_x, len - c1_len)
    }

    /// Find the k-th one bit (1-indexed) in β_d starting at position `alpha`.
    /// Uses binary search over cached values for efficiency.
    pub fn c1_inv(&self, gen: &BitstringGen, d: u32, alpha: u64, k: u64) -> u64 {
        if !self.is_depth_cached(d) || k == 0 {
            return gen.find_kth_one(d, alpha, k);
        }

        let s = self.stride;

        // We need C1(β_d, 0, alpha) to convert to absolute
        let ones_before_alpha = if alpha == 0 {
            0
        } else {
            let full = alpha / s;
            let cached = if full > 0 {
                self.get_cached(d, (full - 1) as usize)
            } else {
                0
            };
            let rem_start = full * s;
            if rem_start < alpha {
                cached + gen.count_ones_range(d, rem_start, alpha - rem_start)
            } else {
                cached
            }
        };

        let target = ones_before_alpha + k;

        // Binary search in cache for the stride containing the target-th one
        let cache_len = self.cache[d as usize].len();
        let mut lo = 0usize;
        let mut hi = cache_len;

        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if (self.cache[d as usize][mid] as u64) < target {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }

        // The target-th one bit is in stride block `lo`
        let block_start = (lo as u64) * s;
        let ones_at_block_start = if lo > 0 {
            self.cache[d as usize][lo - 1] as u64
        } else {
            0
        };
        let remaining = target - ones_at_block_start;

        // Linear scan within this stride block
        let offset = gen.find_kth_one(d, block_start, remaining);
        block_start + offset - alpha
    }

    /// Find the k-th zero bit (1-indexed) in β_d starting at position `alpha`.
    pub fn c0_inv(&self, gen: &BitstringGen, d: u32, alpha: u64, k: u64) -> u64 {
        if !self.is_depth_cached(d) || k == 0 {
            return gen.find_kth_zero(d, alpha, k);
        }

        let s = self.stride;

        // Count zeros before alpha
        let zeros_before_alpha = if alpha == 0 {
            0
        } else {
            alpha - {
                let full = alpha / s;
                let cached = if full > 0 {
                    self.get_cached(d, (full - 1) as usize)
                } else {
                    0
                };
                let rem_start = full * s;
                if rem_start < alpha {
                    cached + gen.count_ones_range(d, rem_start, alpha - rem_start)
                } else {
                    cached
                }
            }
        };

        let target_zeros = zeros_before_alpha + k;

        // Binary search: find the stride block where the target-th zero is
        let cache_len = self.cache[d as usize].len();
        let mut lo = 0usize;
        let mut hi = cache_len;

        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let total_bits = ((mid + 1) as u64) * s;
            let ones = self.cache[d as usize][mid] as u64;
            let zeros = total_bits - ones;
            if zeros < target_zeros {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }

        let block_start = (lo as u64) * s;
        let zeros_at_block_start = if lo > 0 {
            let total_bits = (lo as u64) * s;
            let ones = self.cache[d as usize][lo - 1] as u64;
            total_bits - ones
        } else {
            0
        };
        let remaining = target_zeros - zeros_at_block_start;

        let offset = gen.find_kth_zero(d, block_start, remaining);
        block_start + offset - alpha
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cached_c1_matches_raw() {
        let key = [42u8; 16];
        let n = 1024u64;
        let gen = BitstringGen::new(&key, n);
        let cache = CounterCache::new(&gen, 64);

        for d in 0..3 {
            for alpha in [0, 10, 64, 100, 500] {
                for count in [1, 10, 50, 100, 200] {
                    if alpha + count <= n {
                        let raw = gen.count_ones_range(d, alpha, count);
                        let cached = cache.c1(&gen, d, alpha, count);
                        assert_eq!(raw, cached, "d={d} alpha={alpha} count={count}");
                    }
                }
            }
        }
    }

    #[test]
    fn test_cached_c0_matches_raw() {
        let key = [42u8; 16];
        let n = 1024u64;
        let gen = BitstringGen::new(&key, n);
        let cache = CounterCache::new(&gen, 64);

        for d in 0..3 {
            let raw = gen.count_zeros_range(d, 0, n);
            let cached = cache.c0(&gen, d, 0, n);
            assert_eq!(raw, cached, "d={d}");
        }
    }

    #[test]
    fn test_cached_c1_inv() {
        let key = [42u8; 16];
        let n = 512u64;
        let gen = BitstringGen::new(&key, n);
        let cache = CounterCache::new(&gen, 32);

        for d in 0..2 {
            let total_ones = gen.count_ones_range(d, 0, n);
            for k in 1..=total_ones.min(20) {
                let raw = gen.find_kth_one(d, 0, k);
                let cached = cache.c1_inv(&gen, d, 0, k);
                assert_eq!(raw, cached, "d={d} k={k}");
            }
        }
    }

    #[test]
    fn test_cached_c0_inv() {
        let key = [42u8; 16];
        let n = 512u64;
        let gen = BitstringGen::new(&key, n);
        let cache = CounterCache::new(&gen, 32);

        for d in 0..2 {
            let total_zeros = gen.count_zeros_range(d, 0, n);
            for k in 1..=total_zeros.min(20) {
                let raw = gen.find_kth_zero(d, 0, k);
                let cached = cache.c0_inv(&gen, d, 0, k);
                assert_eq!(raw, cached, "d={d} k={k}");
            }
        }
    }
}
