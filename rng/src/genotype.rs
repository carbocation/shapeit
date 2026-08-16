use core::slice;

const ABI_VERSION: u32 = 1;
pub(crate) const STATUS_OK: u32 = 0;
pub(crate) const STATUS_NULL_POINTER: u32 = 1;
pub(crate) const STATUS_INVALID_DIMENSIONS: u32 = 2;
pub(crate) const STATUS_OUT_OF_BOUNDS: u32 = 3;
pub(crate) const STATUS_INTEGER_OVERFLOW: u32 = 4;

const MAX_AMBIGUOUS_PER_SEGMENT: usize = 22;
const MASK_INIT: u64 = u64::MAX;
const MASK_SCAFFOLD: u64 = 0x00aa_00aa_00aa_00aa;
const MASK_UNFOLD0: u64 = 0x55aa_55aa_55aa_55aa;
const MASK_UNFOLD1: u64 = 0x3333_cccc_3333_cccc;
const MASK_UNFOLD2: u64 = 0x0f0f_0f0f_f0f0_f0f0;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct GraphSizes {
    segments: usize,
    ambiguous: usize,
    missing: usize,
}

pub(crate) struct LogicalRng {
    seed: u64,
    domain: u32,
    iteration: u32,
    item: u64,
    next_block: u64,
    words: [u32; 4],
    next_word: usize,
}

impl LogicalRng {
    pub(crate) fn new(seed: u64, domain: u32, iteration: u32, item: u64) -> Self {
        Self {
            seed,
            domain,
            iteration,
            item,
            next_block: 0,
            words: [0; 4],
            next_word: 4,
        }
    }

    #[inline]
    fn next_u32(&mut self) -> u32 {
        if self.next_word == self.words.len() {
            self.words = super::application_block(
                self.seed,
                self.domain,
                self.iteration,
                self.item,
                self.next_block,
            );
            self.next_block = self.next_block.wrapping_add(1);
            self.next_word = 0;
        }
        let value = self.words[self.next_word];
        self.next_word += 1;
        value
    }

    #[inline]
    fn next_f64(&mut self) -> f64 {
        let bits = (u64::from(self.next_u32()) << 32) | u64::from(self.next_u32());
        ((bits >> 11) as f64) * (1.0 / 9_007_199_254_740_992.0)
    }

    #[inline]
    pub(crate) fn next_bounded(&mut self, range: u32) -> u32 {
        debug_assert!(range != 0);
        let mut value = self.next_u32();
        let mut product = u64::from(value) * u64::from(range);
        let mut low = product as u32;
        if low < range {
            let threshold = range.wrapping_neg() % range;
            while low < threshold {
                value = self.next_u32();
                product = u64::from(value) * u64::from(range);
                low = product as u32;
            }
        }
        (product >> 32) as u32
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct SampleLayout {
    transitions: usize,
    missing: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct GenotypeWindowV1 {
    pub(crate) start_locus: i32,
    pub(crate) start_segment: i32,
    pub(crate) start_ambiguous: i32,
    pub(crate) start_missing: i32,
    pub(crate) start_transition: i32,
    pub(crate) stop_locus: i32,
    pub(crate) stop_segment: i32,
    pub(crate) stop_ambiguous: i32,
    pub(crate) stop_missing: i32,
    pub(crate) stop_transition: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct GenotypeStorageViewV1 {
    transition_count: usize,
    transition_mask: *const u8,
    transition_mask_length: usize,
    transition_probabilities: *const f32,
    transition_probabilities_length: usize,
    missing_probabilities: *const f32,
    missing_probabilities_length: usize,
    storage_events: u32,
}

pub struct GenotypeStorageV1 {
    transition_count: usize,
    transition_mask: Vec<u8>,
    transition_indexes: Vec<u32>,
    transition_probabilities: Vec<f32>,
    missing_probabilities: Vec<f32>,
    storage_events: u32,
}

impl GenotypeStorageV1 {
    fn new(transition_probabilities: &[f64], missing_probabilities: &[f32]) -> Result<Self, u32> {
        if transition_probabilities.len() > u32::MAX as usize {
            return Err(STATUS_INTEGER_OVERFLOW);
        }
        let mut transition_mask = vec![0u8; transition_probabilities.len().div_ceil(8)];
        let mut transition_indexes = Vec::new();
        let mut stored_probabilities = Vec::new();
        for (index, &probability) in transition_probabilities.iter().enumerate() {
            if probability >= 1e-6 {
                transition_mask[index >> 3] |= 1 << (index & 7);
                transition_indexes.push(index as u32);
                stored_probabilities.push(probability as f32);
            }
        }
        Ok(Self {
            transition_count: transition_probabilities.len(),
            transition_mask,
            transition_indexes,
            transition_probabilities: stored_probabilities,
            missing_probabilities: missing_probabilities.to_vec(),
            storage_events: 1,
        })
    }

    fn update(
        &mut self,
        transition_probabilities: &[f64],
        missing_probabilities: &[f32],
    ) -> Result<(), u32> {
        if transition_probabilities.len() != self.transition_count
            || missing_probabilities.len() != self.missing_probabilities.len()
        {
            return Err(STATUS_INVALID_DIMENSIONS);
        }
        let storage_events = self
            .storage_events
            .checked_add(1)
            .ok_or(STATUS_INTEGER_OVERFLOW)?;
        for (&index, stored) in self
            .transition_indexes
            .iter()
            .zip(self.transition_probabilities.iter_mut())
        {
            *stored = (f64::from(*stored) + transition_probabilities[index as usize]) as f32;
        }
        for (stored, &current) in self
            .missing_probabilities
            .iter_mut()
            .zip(missing_probabilities)
        {
            *stored += current;
        }
        self.storage_events = storage_events;
        Ok(())
    }

    fn view(&self) -> GenotypeStorageViewV1 {
        GenotypeStorageViewV1 {
            transition_count: self.transition_count,
            transition_mask: self.transition_mask.as_ptr(),
            transition_mask_length: self.transition_mask.len(),
            transition_probabilities: self.transition_probabilities.as_ptr(),
            transition_probabilities_length: self.transition_probabilities.len(),
            missing_probabilities: self.missing_probabilities.as_ptr(),
            missing_probabilities_length: self.missing_probabilities.len(),
            storage_events: self.storage_events,
        }
    }
}

#[inline]
fn variant_nibble(variants: &[u8], locus: usize) -> u8 {
    (variants[locus >> 1] >> ((locus & 1) << 2)) & 0x0f
}

#[inline]
fn set_variant_nibble(variants: &mut [u8], locus: usize, value: u8) {
    let shift = (locus & 1) << 2;
    variants[locus >> 1] = (variants[locus >> 1] & !(0x0f << shift)) | (value << shift);
}

#[derive(Clone, Copy)]
enum PedigreeMode {
    Trio,
    Father,
    Mother,
}

fn scaffold_pedigree(
    child: &mut [u8],
    father: Option<&[u8]>,
    mother: Option<&[u8]>,
    variant_count: usize,
    mode: PedigreeMode,
    counts: &mut [u32],
) {
    for locus in 0..variant_count {
        let mut child_code = variant_nibble(child, locus);
        let child_graph = graph_code(child_code);
        let father_code = father.map(|variants| variant_nibble(variants, locus));
        let mother_code = mother.map(|variants| variant_nibble(variants, locus));
        if child_graph == 2 {
            let father_homozygous = father_code.is_some_and(|code| graph_code(code) == 0);
            let mother_homozygous = mother_code.is_some_and(|code| graph_code(code) == 0);
            match mode {
                PedigreeMode::Trio if father_homozygous && mother_homozygous => {
                    let father_allele = father_code.unwrap() & 4 != 0;
                    let mother_allele = mother_code.unwrap() & 4 != 0;
                    if father_allele != mother_allele {
                        child_code |= 3;
                        counts[2] = counts[2].wrapping_add(1);
                        if father_allele {
                            child_code = (child_code | 4) & !8;
                        } else {
                            child_code = (child_code & !4) | 8;
                        }
                    } else {
                        counts[0] = counts[0].wrapping_add(1);
                    }
                }
                PedigreeMode::Trio | PedigreeMode::Father if father_homozygous => {
                    let father_allele = father_code.unwrap() & 4 != 0;
                    child_code |= 3;
                    counts[2] = counts[2].wrapping_add(1);
                    if father_allele {
                        child_code = (child_code | 4) & !8;
                    } else {
                        child_code = (child_code & !4) | 8;
                    }
                }
                PedigreeMode::Trio | PedigreeMode::Mother if mother_homozygous => {
                    let mother_allele = mother_code.unwrap() & 4 != 0;
                    child_code |= 3;
                    counts[2] = counts[2].wrapping_add(1);
                    if mother_allele {
                        child_code = (child_code & !4) | 8;
                    } else {
                        child_code = (child_code | 4) & !8;
                    }
                }
                _ => counts[3] = counts[3].wrapping_add(1),
            }
            counts[1] = counts[1].wrapping_add(1);
            set_variant_nibble(child, locus, child_code);
        } else if child_graph == 0 {
            let child_allele = child_code & 4 != 0;
            if matches!(mode, PedigreeMode::Trio | PedigreeMode::Father)
                && father_code
                    .is_some_and(|code| graph_code(code) == 0 && (code & 4 != 0) != child_allele)
            {
                counts[0] = counts[0].wrapping_add(1);
            }
            if matches!(mode, PedigreeMode::Trio | PedigreeMode::Mother)
                && mother_code
                    .is_some_and(|code| graph_code(code) == 0 && (code & 4 != 0) != child_allele)
            {
                counts[0] = counts[0].wrapping_add(1);
            }
            counts[1] = counts[1].wrapping_add(1);
        } else if child_graph == 1 && matches!(mode, PedigreeMode::Trio) {
            let father_allele = father_code.unwrap() & 4 != 0;
            let mother_allele = mother_code.unwrap() & 4 != 0;
            if father_allele != mother_allele {
                child_code |= 3;
            } else {
                child_code &= !3;
            }
            if father_allele {
                child_code |= 4;
            } else {
                child_code &= !4;
            }
            if mother_allele {
                child_code |= 8;
            } else {
                child_code &= !8;
            }
            set_variant_nibble(child, locus, child_code);
        }
    }
}

#[inline]
fn graph_code(variant: u8) -> u8 {
    variant & 3
}

fn required_variant_bytes(variant_count: usize) -> Result<usize, u32> {
    variant_count
        .checked_add(1)
        .map(|value| value >> 1)
        .ok_or(STATUS_INTEGER_OVERFLOW)
}

fn byte_ranges_overlap(
    first: *const u8,
    first_length: usize,
    second: *const u8,
    second_length: usize,
) -> Result<bool, u32> {
    if first_length == 0 || second_length == 0 {
        return Ok(false);
    }
    let first_start = first as usize;
    let second_start = second as usize;
    let first_stop = first_start
        .checked_add(first_length)
        .ok_or(STATUS_INTEGER_OVERFLOW)?;
    let second_stop = second_start
        .checked_add(second_length)
        .ok_or(STATUS_INTEGER_OVERFLOW)?;
    Ok(first_start < second_stop && second_start < first_stop)
}

fn graph_sizes(variants: &[u8], variant_count: usize) -> GraphSizes {
    let mut relative_unfolded = 0usize;
    let mut relative_variants = 0usize;
    let mut relative_scaffolded = false;
    let mut relative_ambiguous = 0usize;
    let mut segments = 1usize;
    let mut ambiguous = 0usize;
    let mut missing = 0usize;
    let mut locus = 0usize;

    while locus < variant_count {
        let code = graph_code(variant_nibble(variants, locus));
        let heterozygous = code == 2;
        let scaffolded = code == 3;
        let predicted_unfolded = relative_unfolded
            + usize::from(heterozygous)
            + usize::from(relative_scaffolded || scaffolded);
        if predicted_unfolded == 4
            || relative_variants == u16::MAX as usize
            || relative_ambiguous == MAX_AMBIGUOUS_PER_SEGMENT
        {
            relative_unfolded = 0;
            relative_variants = 0;
            relative_scaffolded = false;
            relative_ambiguous = 0;
            segments += 1;
            continue;
        }
        relative_unfolded += usize::from(heterozygous);
        relative_scaffolded |= scaffolded;
        relative_ambiguous += usize::from(heterozygous || scaffolded);
        ambiguous += usize::from(heterozygous || scaffolded);
        missing += usize::from(code == 1);
        relative_variants += 1;
        locus += 1;
    }
    GraphSizes {
        segments,
        ambiguous,
        missing,
    }
}

fn build_graph(
    variants: &[u8],
    variant_count: usize,
    segment_lengths: &mut [u16],
    ambiguous: &mut [u8],
    diplotypes: &mut [u64],
) -> u32 {
    let mut ordered_segments = vec![false; segment_lengths.len()];
    let mut relative_unfolded = 0usize;
    let mut relative_variants = 0usize;
    let mut relative_scaffolded = false;
    let mut relative_ambiguous = 0usize;
    let mut segment = 0usize;
    let mut locus = 0usize;

    while locus < variant_count {
        let code = graph_code(variant_nibble(variants, locus));
        let heterozygous = code == 2;
        let scaffolded = code == 3;
        let predicted_unfolded = relative_unfolded
            + usize::from(heterozygous)
            + usize::from(relative_scaffolded || scaffolded);
        if predicted_unfolded == 4
            || relative_variants == u16::MAX as usize
            || relative_ambiguous == MAX_AMBIGUOUS_PER_SEGMENT
        {
            segment_lengths[segment] = relative_variants as u16;
            ordered_segments[segment] = relative_scaffolded;
            relative_unfolded = 0;
            relative_variants = 0;
            relative_scaffolded = false;
            relative_ambiguous = 0;
            segment += 1;
            continue;
        }
        relative_unfolded += usize::from(heterozygous);
        relative_scaffolded |= scaffolded;
        relative_ambiguous += usize::from(heterozygous || scaffolded);
        relative_variants += 1;
        locus += 1;
    }
    segment_lengths[segment] = relative_variants as u16;
    ordered_segments[segment] = relative_scaffolded;

    let mut absolute_locus = 0usize;
    let mut absolute_ambiguous = 0usize;
    let mut transitions = 0u32;
    let mut previous_diplotypes = 1u32;
    for (segment, &length) in segment_lengths.iter().enumerate() {
        let mut unfolded = usize::from(ordered_segments[segment]);
        let mut diplotype = if ordered_segments[segment] {
            MASK_SCAFFOLD
        } else {
            MASK_INIT
        };
        for relative_locus in 0..usize::from(length) {
            let variant = variant_nibble(variants, absolute_locus + relative_locus);
            match graph_code(variant) {
                2 => {
                    ambiguous[absolute_ambiguous] = match unfolded {
                        0 => 0xaa,
                        1 => 0xcc,
                        2 => 0xf0,
                        _ => unreachable!("segment contains too many unfolded genotypes"),
                    };
                    diplotype &= match unfolded {
                        0 => MASK_UNFOLD0,
                        1 => MASK_UNFOLD1,
                        2 => MASK_UNFOLD2,
                        _ => unreachable!("segment contains too many unfolded genotypes"),
                    };
                    unfolded += 1;
                    absolute_ambiguous += 1;
                }
                3 => {
                    let haplotype0 = (variant & 4) != 0;
                    let haplotype1 = (variant & 8) != 0;
                    let mut code = 0u8;
                    for haplotype in 0..8 {
                        let allele = if haplotype & 1 == 0 {
                            haplotype0
                        } else {
                            haplotype1
                        };
                        code |= u8::from(allele) << haplotype;
                    }
                    ambiguous[absolute_ambiguous] = code;
                    absolute_ambiguous += 1;
                }
                _ => {}
            }
        }
        diplotypes[segment] = diplotype;
        let current_diplotypes = diplotype.count_ones();
        transitions =
            transitions.wrapping_add(previous_diplotypes.wrapping_mul(current_diplotypes));
        previous_diplotypes = current_diplotypes;
        absolute_locus += usize::from(length);
    }
    transitions
}

fn validate_sample_layout(
    variants: &[u8],
    variant_count: usize,
    ambiguous_length: usize,
    diplotypes: &[u64],
    segment_lengths: &[u16],
    transition_probabilities_length: usize,
    missing_probabilities_length: usize,
) -> Result<SampleLayout, u32> {
    if diplotypes.is_empty() || diplotypes.len() != segment_lengths.len() {
        return Err(STATUS_INVALID_DIMENSIONS);
    }
    let mut loci = 0usize;
    let mut ambiguous = 0usize;
    let mut missing = 0usize;
    for &length in segment_lengths {
        loci = loci
            .checked_add(usize::from(length))
            .ok_or(STATUS_INTEGER_OVERFLOW)?;
    }
    if loci != variant_count {
        return Err(STATUS_INVALID_DIMENSIONS);
    }
    for locus in 0..variant_count {
        match graph_code(variant_nibble(variants, locus)) {
            1 => missing += 1,
            2 | 3 => ambiguous += 1,
            _ => {}
        }
    }
    if ambiguous != ambiguous_length {
        return Err(STATUS_INVALID_DIMENSIONS);
    }
    let required_missing = missing.checked_mul(8).ok_or(STATUS_INTEGER_OVERFLOW)?;
    if required_missing > missing_probabilities_length {
        return Err(STATUS_OUT_OF_BOUNDS);
    }
    let mut previous = 1usize;
    let mut transitions = 0usize;
    for &diplotype in diplotypes {
        let current = diplotype.count_ones() as usize;
        if current == 0 {
            return Err(STATUS_INVALID_DIMENSIONS);
        }
        transitions = transitions
            .checked_add(
                previous
                    .checked_mul(current)
                    .ok_or(STATUS_INTEGER_OVERFLOW)?,
            )
            .ok_or(STATUS_INTEGER_OVERFLOW)?;
        previous = current;
    }
    if transitions > transition_probabilities_length {
        return Err(STATUS_OUT_OF_BOUNDS);
    }
    Ok(SampleLayout {
        transitions,
        missing,
    })
}

#[inline]
fn sample_probabilities(probabilities: &[f64], total: f64, rng: &mut LogicalRng) -> usize {
    let mut cumulative = probabilities[0];
    let draw = rng.next_f64() * total;
    for index in 0..probabilities.len() - 1 {
        if draw < cumulative {
            return index;
        }
        cumulative += probabilities[index + 1];
    }
    probabilities.len() - 1
}

#[inline]
fn diplotype_code(mask: u64, index: usize) -> u8 {
    let mut active = mask;
    for _ in 0..index {
        active &= active - 1;
    }
    active.trailing_zeros() as u8
}

fn sample_forward(
    diplotypes: &[u64],
    transition_probabilities: &[f64],
    sampled: &mut [u8],
    rng: &mut LogicalRng,
) {
    let mut probabilities = [0.0f64; 64];
    let mut previous_sampled = 0usize;
    let mut previous_count = 1usize;
    let mut transition_offset = 0usize;
    for (segment, &diplotype) in diplotypes.iter().enumerate() {
        let current_count = diplotype.count_ones() as usize;
        let start = transition_offset + previous_sampled * current_count;
        let mut total = 0.0f64;
        for relative in 0..current_count {
            probabilities[relative] = transition_probabilities[start + relative];
            total += probabilities[relative];
        }
        previous_sampled = sample_probabilities(&probabilities, total, rng);
        sampled[segment] = diplotype_code(diplotype, previous_sampled);
        transition_offset += previous_count * current_count;
        previous_count = current_count;
    }
}

fn sample_backward(
    diplotypes: &[u64],
    transition_probabilities: &[f64],
    transitions: usize,
    sampled: &mut [u8],
    rng: &mut LogicalRng,
) {
    let mut probabilities = [0.0f64; 64 * 64];
    let mut next_sampled = None;
    let mut next_count = diplotypes.last().unwrap().count_ones() as usize;
    let mut transition_offset = transitions;
    for segment in (0..diplotypes.len().saturating_sub(1)).rev() {
        let current_count = diplotypes[segment].count_ones() as usize;
        transition_offset -= next_count * current_count;
        if let Some(next_sampled_index) = next_sampled {
            let mut total = 0.0f64;
            for relative in 0..current_count {
                probabilities[relative] = transition_probabilities
                    [transition_offset + next_sampled_index + relative * next_count];
                total += probabilities[relative];
            }
            let current_sampled = sample_probabilities(&probabilities[..64], total, rng);
            sampled[segment] = diplotype_code(diplotypes[segment], current_sampled);
            next_sampled = Some(current_sampled);
        } else {
            let block_length = next_count * current_count;
            let mut total = 0.0f64;
            for relative in 0..block_length {
                probabilities[relative] = transition_probabilities[transition_offset + relative];
                total += probabilities[relative];
            }
            let joint_sampled = sample_probabilities(&probabilities, total, rng);
            sampled[segment + 1] =
                diplotype_code(diplotypes[segment + 1], joint_sampled % next_count);
            let current_sampled = joint_sampled / next_count;
            sampled[segment] = diplotype_code(diplotypes[segment], current_sampled);
            next_sampled = Some(current_sampled);
        }
        next_count = current_count;
    }
}

#[inline]
fn set_haplotype(variants: &mut [u8], locus: usize, haplotype: usize, allele: bool) {
    let byte = &mut variants[locus >> 1];
    let shift = ((locus & 1) << 2) + 2 + haplotype;
    let mask = 1u8 << shift;
    *byte = (*byte & !mask) | (u8::from(allele) << shift);
}

struct ApplySample<'a> {
    variants: &'a mut [u8],
    ambiguous: &'a [u8],
    segment_lengths: &'a [u16],
    sampled: &'a [u8],
    missing_probabilities: &'a [f32],
    haploid: bool,
}

fn apply_sample(parameters: ApplySample<'_>, rng: &mut LogicalRng) {
    let ApplySample {
        variants,
        ambiguous,
        segment_lengths,
        sampled,
        missing_probabilities,
        haploid,
    } = parameters;
    let mut absolute_locus = 0usize;
    let mut ambiguous_index = 0usize;
    let mut missing_index = 0usize;
    for (segment, &length) in segment_lengths.iter().enumerate() {
        let haplotype0 = usize::from(sampled[segment] >> 3);
        let haplotype1 = usize::from(sampled[segment] & 7);
        for _ in 0..length {
            let code = graph_code(variant_nibble(variants, absolute_locus));
            if code == 1 {
                let start = missing_index * 8;
                if haploid {
                    let probability0 = missing_probabilities[start + haplotype0];
                    let probability1 = missing_probabilities[start + haplotype1];
                    let probability00 = (1.0 - probability0) * (1.0 - probability1);
                    let probability11 = probability0 * probability1;
                    let allele = rng.next_f64()
                        <= f64::from(probability11 / (probability00 + probability11));
                    set_haplotype(variants, absolute_locus, 0, allele);
                    set_haplotype(variants, absolute_locus, 1, allele);
                } else {
                    let allele0 =
                        rng.next_f64() <= f64::from(missing_probabilities[start + haplotype0]);
                    let allele1 =
                        rng.next_f64() <= f64::from(missing_probabilities[start + haplotype1]);
                    set_haplotype(variants, absolute_locus, 0, allele0);
                    set_haplotype(variants, absolute_locus, 1, allele1);
                }
                missing_index += 1;
            }
            if code > 1 {
                let graph = ambiguous[ambiguous_index];
                set_haplotype(
                    variants,
                    absolute_locus,
                    0,
                    ((graph >> haplotype0) & 1) != 0,
                );
                set_haplotype(
                    variants,
                    absolute_locus,
                    1,
                    ((graph >> haplotype1) & 1) != 0,
                );
                ambiguous_index += 1;
            }
            absolute_locus += 1;
        }
    }
}

fn solve_path(
    diplotypes: &[u64],
    transition_count: usize,
    stored_transition_indexes: &[u32],
    stored_transition_probabilities: &[f32],
) -> Vec<u8> {
    let mut maximum_probabilities: Vec<Vec<f64>> = Vec::with_capacity(diplotypes.len());
    let mut maximum_indexes: Vec<Vec<usize>> = Vec::with_capacity(diplotypes.len());
    let mut previous_count = 1usize;
    let mut transition_offset = 0usize;
    let mut stored_relative = 0usize;

    for (segment, &diplotype) in diplotypes.iter().enumerate() {
        let current_count = diplotype.count_ones() as usize;
        let mut probabilities = vec![0.0f64; current_count];
        let mut indexes = vec![0usize; current_count];
        for relative_transition in 0..previous_count * current_count {
            let previous = relative_transition / current_count;
            let current = relative_transition % current_count;
            let absolute_transition = transition_offset + relative_transition;
            let stored = stored_relative < stored_transition_indexes.len()
                && stored_transition_indexes[stored_relative] as usize == absolute_transition;
            let transition_probability = if stored {
                let value = f64::from(stored_transition_probabilities[stored_relative]);
                stored_relative += 1;
                value
            } else {
                1e-6
            };
            let previous_probability = if segment == 0 {
                1.0
            } else {
                maximum_probabilities[segment - 1][previous]
            };
            let probability = previous_probability * transition_probability;
            if probability > probabilities[current] {
                probabilities[current] = probability;
                indexes[current] = previous;
            }
        }
        let mut total = 0.0f64;
        for &probability in &probabilities {
            total += probability;
        }
        for probability in &mut probabilities {
            *probability /= total;
        }
        maximum_probabilities.push(probabilities);
        maximum_indexes.push(indexes);
        transition_offset += previous_count * current_count;
        previous_count = current_count;
    }
    debug_assert_eq!(transition_offset, transition_count);
    debug_assert_eq!(stored_relative, stored_transition_indexes.len());

    let final_probabilities = maximum_probabilities.last().unwrap();
    let mut best = 0usize;
    for index in 1..final_probabilities.len() {
        if final_probabilities[index] > final_probabilities[best] {
            best = index;
        }
    }
    let mut sampled = vec![0u8; diplotypes.len()];
    let final_segment = diplotypes.len() - 1;
    sampled[final_segment] = diplotype_code(diplotypes[final_segment], best);
    for segment in (0..final_segment).rev() {
        best = maximum_indexes[segment + 1][best];
        sampled[segment] = diplotype_code(diplotypes[segment], best);
    }
    sampled
}

struct ApplySolution<'a> {
    variants: &'a mut [u8],
    ambiguous: &'a [u8],
    segment_lengths: &'a [u16],
    sampled: &'a [u8],
    missing_probabilities: &'a [f32],
    storage_events: u32,
    haploid: bool,
}

fn apply_solution(parameters: ApplySolution<'_>) {
    let ApplySolution {
        variants,
        ambiguous,
        segment_lengths,
        sampled,
        missing_probabilities,
        storage_events,
        haploid,
    } = parameters;
    let event_count = storage_events as f32;
    let mut absolute_locus = 0usize;
    let mut ambiguous_index = 0usize;
    let mut missing_index = 0usize;
    for (segment, &length) in segment_lengths.iter().enumerate() {
        let haplotype0 = usize::from(sampled[segment] >> 3);
        let haplotype1 = usize::from(sampled[segment] & 7);
        for _ in 0..length {
            let code = graph_code(variant_nibble(variants, absolute_locus));
            if code == 1 {
                let start = missing_index * 8;
                let probability0 = missing_probabilities[start + haplotype0];
                let probability1 = missing_probabilities[start + haplotype1];
                if haploid {
                    let normalized0 = probability0 / event_count;
                    let normalized1 = probability1 / event_count;
                    let probability00 = (1.0 - normalized0) * (1.0 - normalized1);
                    let probability11 = normalized0 * normalized1;
                    let allele = probability11 > probability00;
                    set_haplotype(variants, absolute_locus, 0, allele);
                    set_haplotype(variants, absolute_locus, 1, allele);
                } else {
                    let threshold = 0.5 * event_count;
                    set_haplotype(variants, absolute_locus, 0, probability0 >= threshold);
                    set_haplotype(variants, absolute_locus, 1, probability1 >= threshold);
                }
                missing_index += 1;
            }
            if code > 1 {
                let graph = ambiguous[ambiguous_index];
                set_haplotype(
                    variants,
                    absolute_locus,
                    0,
                    ((graph >> haplotype0) & 1) != 0,
                );
                set_haplotype(
                    variants,
                    absolute_locus,
                    1,
                    ((graph >> haplotype1) & 1) != 0,
                );
                ambiguous_index += 1;
            }
            absolute_locus += 1;
        }
    }
}

#[derive(Clone, Copy)]
struct SegmentCoordinates {
    locus_start: usize,
    locus_count: usize,
    ambiguous_start: usize,
    ambiguous_count: usize,
    missing_start: usize,
    missing_count: usize,
    transition_start: usize,
    transition_count: usize,
    start_centimorgans: f64,
    stop_centimorgans: f64,
}

fn segment_coordinates(
    variants: &[u8],
    variant_count: usize,
    diplotypes: &[u64],
    segment_lengths: &[u16],
    segment_start_centimorgans: &[f64],
    segment_stop_centimorgans: &[f64],
) -> Result<Vec<SegmentCoordinates>, u32> {
    if diplotypes.is_empty()
        || diplotypes.len() != segment_lengths.len()
        || diplotypes.len() != segment_start_centimorgans.len()
        || diplotypes.len() != segment_stop_centimorgans.len()
    {
        return Err(STATUS_INVALID_DIMENSIONS);
    }
    let mut coordinates = Vec::with_capacity(diplotypes.len());
    let mut locus = 0usize;
    let mut ambiguous = 0usize;
    let mut missing = 0usize;
    let mut transition = 0usize;
    let mut previous_diplotypes = 1usize;
    for segment in 0..diplotypes.len() {
        let locus_count = usize::from(segment_lengths[segment]);
        if locus_count == 0 {
            return Err(STATUS_INVALID_DIMENSIONS);
        }
        let stop = locus
            .checked_add(locus_count)
            .ok_or(STATUS_INTEGER_OVERFLOW)?;
        if stop > variant_count {
            return Err(STATUS_OUT_OF_BOUNDS);
        }
        let mut ambiguous_count = 0usize;
        let mut missing_count = 0usize;
        for absolute_locus in locus..stop {
            match graph_code(variant_nibble(variants, absolute_locus)) {
                1 => missing_count += 1,
                2 | 3 => ambiguous_count += 1,
                _ => {}
            }
        }
        let current_diplotypes = diplotypes[segment].count_ones() as usize;
        if current_diplotypes == 0 {
            return Err(STATUS_INVALID_DIMENSIONS);
        }
        let transition_count = previous_diplotypes
            .checked_mul(current_diplotypes)
            .ok_or(STATUS_INTEGER_OVERFLOW)?;
        coordinates.push(SegmentCoordinates {
            locus_start: locus,
            locus_count,
            ambiguous_start: ambiguous,
            ambiguous_count,
            missing_start: missing,
            missing_count,
            transition_start: transition,
            transition_count,
            start_centimorgans: segment_start_centimorgans[segment],
            stop_centimorgans: segment_stop_centimorgans[segment],
        });
        locus = stop;
        ambiguous = ambiguous
            .checked_add(ambiguous_count)
            .ok_or(STATUS_INTEGER_OVERFLOW)?;
        missing = missing
            .checked_add(missing_count)
            .ok_or(STATUS_INTEGER_OVERFLOW)?;
        transition = transition
            .checked_add(transition_count)
            .ok_or(STATUS_INTEGER_OVERFLOW)?;
        previous_diplotypes = current_diplotypes;
    }
    if locus != variant_count {
        return Err(STATUS_INVALID_DIMENSIONS);
    }
    Ok(coordinates)
}

fn split_windows(
    minimum_centimorgans: f64,
    left: usize,
    right: usize,
    segments: &[SegmentCoordinates],
    rng: &mut LogicalRng,
) -> Option<Vec<(usize, usize)>> {
    let number_of_segments = right - left + 1;
    let number_of_variants =
        segments[right].locus_start + segments[right].locus_count - segments[left].locus_start;
    let length_centimorgans = segments[right].stop_centimorgans - segments[left].start_centimorgans;
    if number_of_segments < 4
        || number_of_variants < 100
        || length_centimorgans < minimum_centimorgans
    {
        return None;
    }

    let random_span = (number_of_segments / 2) as u32;
    let split = rng.next_bounded(random_span) as usize + number_of_segments / 4 + 1;
    let left_windows = split_windows(minimum_centimorgans, left, left + split, segments, rng);
    let right_windows = split_windows(minimum_centimorgans, left + split, right, segments, rng);
    match (left_windows, right_windows) {
        (Some(mut left_windows), Some(right_windows)) => {
            left_windows.extend(right_windows);
            Some(left_windows)
        }
        _ => Some(vec![(left, right)]),
    }
}

fn signed_stop(start: usize, count: usize) -> Result<i32, u32> {
    let stop = i64::try_from(start)
        .map_err(|_| STATUS_INTEGER_OVERFLOW)?
        .checked_add(i64::try_from(count).map_err(|_| STATUS_INTEGER_OVERFLOW)?)
        .and_then(|value| value.checked_sub(1))
        .ok_or(STATUS_INTEGER_OVERFLOW)?;
    i32::try_from(stop).map_err(|_| STATUS_INTEGER_OVERFLOW)
}

fn output_window(
    start: usize,
    stop: usize,
    segments: &[SegmentCoordinates],
) -> Result<GenotypeWindowV1, u32> {
    let start_segment = segments[start];
    let stop_segment = segments[stop];
    let as_i32 = |value: usize| i32::try_from(value).map_err(|_| STATUS_INTEGER_OVERFLOW);
    Ok(GenotypeWindowV1 {
        start_locus: as_i32(start_segment.locus_start)?,
        start_segment: as_i32(start)?,
        start_ambiguous: as_i32(start_segment.ambiguous_start)?,
        start_missing: as_i32(start_segment.missing_start)?,
        start_transition: as_i32(
            start_segment
                .transition_start
                .checked_add(start_segment.transition_count)
                .ok_or(STATUS_INTEGER_OVERFLOW)?,
        )?,
        stop_locus: signed_stop(stop_segment.locus_start, stop_segment.locus_count)?,
        stop_segment: as_i32(stop)?,
        stop_ambiguous: signed_stop(stop_segment.ambiguous_start, stop_segment.ambiguous_count)?,
        stop_missing: signed_stop(stop_segment.missing_start, stop_segment.missing_count)?,
        stop_transition: signed_stop(stop_segment.transition_start, stop_segment.transition_count)?,
    })
}

pub(crate) struct WindowInputs<'a> {
    pub(crate) variants: &'a [u8],
    pub(crate) variant_count: usize,
    pub(crate) diplotypes: &'a [u64],
    pub(crate) segment_lengths: &'a [u16],
    pub(crate) segment_start_centimorgans: &'a [f64],
    pub(crate) segment_stop_centimorgans: &'a [f64],
    pub(crate) minimum_window_centimorgans: f32,
}

pub(crate) fn build_windows(
    inputs: WindowInputs<'_>,
    rng: &mut LogicalRng,
) -> Result<Vec<GenotypeWindowV1>, u32> {
    let segments = segment_coordinates(
        inputs.variants,
        inputs.variant_count,
        inputs.diplotypes,
        inputs.segment_lengths,
        inputs.segment_start_centimorgans,
        inputs.segment_stop_centimorgans,
    )?;
    if segments.len() > i32::MAX as usize {
        return Err(STATUS_INTEGER_OVERFLOW);
    }
    let ranges = split_windows(
        f64::from(inputs.minimum_window_centimorgans),
        0,
        segments.len() - 1,
        &segments,
        rng,
    )
    .unwrap_or_else(|| vec![(0, segments.len() - 1)]);
    let mut output = Vec::with_capacity(ranges.len());
    for (start, stop) in ranges {
        output.push(output_window(start, stop, &segments)?);
    }
    Ok(output)
}

#[no_mangle]
pub extern "C" fn shapeit_genotype_abi_version() -> u32 {
    ABI_VERSION
}

#[no_mangle]
/// Return the exact output sizes for one packed genotype graph.
///
/// # Safety
///
/// `variants` must be readable for `variants_length` bytes when `variant_count`
/// is nonzero. All three output pointers must be writable.
pub unsafe extern "C" fn shapeit_genotype_graph_sizes_v1(
    variants: *const u8,
    variants_length: usize,
    variant_count: usize,
    segment_count: *mut usize,
    ambiguous_count: *mut usize,
    missing_count: *mut usize,
) -> u32 {
    if segment_count.is_null() || ambiguous_count.is_null() || missing_count.is_null() {
        return STATUS_NULL_POINTER;
    }
    let required = match required_variant_bytes(variant_count) {
        Ok(value) => value,
        Err(status) => return status,
    };
    if required > variants_length {
        return STATUS_OUT_OF_BOUNDS;
    }
    if required != 0 && variants.is_null() {
        return STATUS_NULL_POINTER;
    }
    let variants = if required == 0 {
        &[]
    } else {
        slice::from_raw_parts(variants, required)
    };
    let sizes = graph_sizes(variants, variant_count);
    *segment_count = sizes.segments;
    *ambiguous_count = sizes.ambiguous;
    *missing_count = sizes.missing;
    STATUS_OK
}

#[no_mangle]
/// Build one complete common-phasing genotype graph.
///
/// # Safety
///
/// Input and output buffers must be valid for their stated lengths and must not
/// overlap. Output sizes must match `shapeit_genotype_graph_sizes_v1`.
pub unsafe extern "C" fn shapeit_genotype_graph_build_v1(
    variants: *const u8,
    variants_length: usize,
    variant_count: usize,
    segment_lengths: *mut u16,
    segment_lengths_length: usize,
    ambiguous: *mut u8,
    ambiguous_length: usize,
    diplotypes: *mut u64,
    diplotypes_length: usize,
    transition_count: *mut u32,
) -> u32 {
    if transition_count.is_null() {
        return STATUS_NULL_POINTER;
    }
    let required = match required_variant_bytes(variant_count) {
        Ok(value) => value,
        Err(status) => return status,
    };
    if required > variants_length {
        return STATUS_OUT_OF_BOUNDS;
    }
    if required != 0 && variants.is_null() {
        return STATUS_NULL_POINTER;
    }
    let variants = if required == 0 {
        &[]
    } else {
        slice::from_raw_parts(variants, required)
    };
    let sizes = graph_sizes(variants, variant_count);
    if segment_lengths_length != sizes.segments
        || ambiguous_length != sizes.ambiguous
        || diplotypes_length != sizes.segments
    {
        return STATUS_INVALID_DIMENSIONS;
    }
    if segment_lengths.is_null()
        || diplotypes.is_null()
        || (ambiguous_length != 0 && ambiguous.is_null())
    {
        return STATUS_NULL_POINTER;
    }
    let segment_lengths = slice::from_raw_parts_mut(segment_lengths, segment_lengths_length);
    let ambiguous = if ambiguous_length == 0 {
        &mut []
    } else {
        slice::from_raw_parts_mut(ambiguous, ambiguous_length)
    };
    let diplotypes = slice::from_raw_parts_mut(diplotypes, diplotypes_length);
    *transition_count = build_graph(
        variants,
        variant_count,
        segment_lengths,
        ambiguous,
        diplotypes,
    );
    STATUS_OK
}

#[no_mangle]
/// Apply trio or duo Mendelian scaffolding to one packed child genotype.
///
/// # Safety
///
/// Required parent buffers must be readable for the packed variant length,
/// `child_variants` must be writable for that length, and `counts` must be
/// writable for four `u32` values. Mutable buffers must not overlap a parent.
pub unsafe extern "C" fn shapeit_genotype_pedigree_scaffold_v1(
    child_variants: *mut u8,
    child_variants_length: usize,
    variant_count: usize,
    father_variants: *const u8,
    father_variants_length: usize,
    mother_variants: *const u8,
    mother_variants_length: usize,
    pedigree_mode: u32,
    counts: *mut u32,
    counts_length: usize,
) -> u32 {
    if counts.is_null() {
        return STATUS_NULL_POINTER;
    }
    let required = match required_variant_bytes(variant_count) {
        Ok(value) => value,
        Err(status) => return status,
    };
    if required > child_variants_length || counts_length < 4 {
        return STATUS_OUT_OF_BOUNDS;
    }
    if required != 0 && child_variants.is_null() {
        return STATUS_NULL_POINTER;
    }
    let mode = match pedigree_mode {
        0 => PedigreeMode::Trio,
        1 => PedigreeMode::Father,
        2 => PedigreeMode::Mother,
        _ => return STATUS_INVALID_DIMENSIONS,
    };
    let father_required = matches!(mode, PedigreeMode::Trio | PedigreeMode::Father);
    let mother_required = matches!(mode, PedigreeMode::Trio | PedigreeMode::Mother);
    if required != 0
        && ((father_required && father_variants.is_null())
            || (mother_required && mother_variants.is_null()))
    {
        return STATUS_NULL_POINTER;
    }
    if (father_required && required > father_variants_length)
        || (mother_required && required > mother_variants_length)
    {
        return STATUS_OUT_OF_BOUNDS;
    }
    let counts_bytes = 4 * core::mem::size_of::<u32>();
    let child = child_variants.cast_const();
    let counts_bytes_pointer = counts.cast::<u8>().cast_const();
    let child_counts_overlap =
        byte_ranges_overlap(child, required, counts_bytes_pointer, counts_bytes);
    let father_child_overlap = if father_required {
        byte_ranges_overlap(child, required, father_variants, required)
    } else {
        Ok(false)
    };
    let mother_child_overlap = if mother_required {
        byte_ranges_overlap(child, required, mother_variants, required)
    } else {
        Ok(false)
    };
    let father_counts_overlap = if father_required {
        byte_ranges_overlap(
            father_variants,
            required,
            counts_bytes_pointer,
            counts_bytes,
        )
    } else {
        Ok(false)
    };
    let mother_counts_overlap = if mother_required {
        byte_ranges_overlap(
            mother_variants,
            required,
            counts_bytes_pointer,
            counts_bytes,
        )
    } else {
        Ok(false)
    };
    for overlap in [
        child_counts_overlap,
        father_child_overlap,
        mother_child_overlap,
        father_counts_overlap,
        mother_counts_overlap,
    ] {
        match overlap {
            Ok(true) => return STATUS_INVALID_DIMENSIONS,
            Err(status) => return status,
            Ok(false) => {}
        }
    }

    let child = if required == 0 {
        &mut []
    } else {
        slice::from_raw_parts_mut(child_variants, required)
    };
    let father = if father_required {
        Some(if required == 0 {
            &[]
        } else {
            slice::from_raw_parts(father_variants, required)
        })
    } else {
        None
    };
    let mother = if mother_required {
        Some(if required == 0 {
            &[]
        } else {
            slice::from_raw_parts(mother_variants, required)
        })
    } else {
        None
    };
    let counts = slice::from_raw_parts_mut(counts, 4);
    scaffold_pedigree(child, father, mother, variant_count, mode, counts);
    STATUS_OK
}

#[no_mangle]
/// Preserve the established packed-bit reset of ambiguous haploid genotypes.
///
/// # Safety
///
/// `variants` must be writable for the packed variant length and `reset_count`
/// must be writable for one `u32`; the two outputs must not overlap.
pub unsafe extern "C" fn shapeit_genotype_reset_haploid_hets_v1(
    variants: *mut u8,
    variants_length: usize,
    variant_count: usize,
    reset_count: *mut u32,
) -> u32 {
    if reset_count.is_null() {
        return STATUS_NULL_POINTER;
    }
    let required = match required_variant_bytes(variant_count) {
        Ok(value) => value,
        Err(status) => return status,
    };
    if required > variants_length {
        return STATUS_OUT_OF_BOUNDS;
    }
    if required != 0 && variants.is_null() {
        return STATUS_NULL_POINTER;
    }
    match byte_ranges_overlap(
        variants.cast_const(),
        required,
        reset_count.cast::<u8>().cast_const(),
        core::mem::size_of::<u32>(),
    ) {
        Ok(true) => return STATUS_INVALID_DIMENSIONS,
        Err(status) => return status,
        Ok(false) => {}
    }

    let variants = if required == 0 {
        &mut []
    } else {
        slice::from_raw_parts_mut(variants, required)
    };
    let mut count = 0u32;
    for locus in 0..variant_count {
        let code = variant_nibble(variants, locus);
        if graph_code(code) > 1 {
            set_variant_nibble(variants, locus, code | 1);
            count = count.wrapping_add(1);
        }
    }
    *reset_count = count;
    STATUS_OK
}

#[no_mangle]
/// Sample one complete genotype graph and update its packed haplotype alleles.
///
/// RNG coordinates identify a fresh logical Philox stream. The function
/// consumes that stream in the same order as the established forward/backward
/// sampler and missing-genotype imputation.
///
/// # Safety
///
/// Every buffer must be valid for its stated length. The mutable packed-variant
/// buffer must not overlap any immutable input. Invalid layouts are rejected
/// before any allele is modified.
pub unsafe extern "C" fn shapeit_genotype_sample_v1(
    variants: *mut u8,
    variants_length: usize,
    variant_count: usize,
    ambiguous: *const u8,
    ambiguous_length: usize,
    diplotypes: *const u64,
    diplotypes_length: usize,
    segment_lengths: *const u16,
    segment_lengths_length: usize,
    transition_probabilities: *const f64,
    transition_probabilities_length: usize,
    missing_probabilities: *const f32,
    missing_probabilities_length: usize,
    haploid: u8,
    seed: u64,
    domain: u32,
    iteration: u32,
    item: u64,
) -> u32 {
    if variants.is_null()
        || diplotypes.is_null()
        || segment_lengths.is_null()
        || transition_probabilities.is_null()
        || (ambiguous_length != 0 && ambiguous.is_null())
        || (missing_probabilities_length != 0 && missing_probabilities.is_null())
    {
        return STATUS_NULL_POINTER;
    }
    if haploid > 1 {
        return STATUS_INVALID_DIMENSIONS;
    }
    let required_variants = match required_variant_bytes(variant_count) {
        Ok(value) => value,
        Err(status) => return status,
    };
    if required_variants > variants_length {
        return STATUS_OUT_OF_BOUNDS;
    }
    let variants = slice::from_raw_parts_mut(variants, variants_length);
    let ambiguous = if ambiguous_length == 0 {
        &[]
    } else {
        slice::from_raw_parts(ambiguous, ambiguous_length)
    };
    let diplotypes = slice::from_raw_parts(diplotypes, diplotypes_length);
    let segment_lengths = slice::from_raw_parts(segment_lengths, segment_lengths_length);
    let transition_probabilities =
        slice::from_raw_parts(transition_probabilities, transition_probabilities_length);
    let missing_probabilities = if missing_probabilities_length == 0 {
        &[]
    } else {
        slice::from_raw_parts(missing_probabilities, missing_probabilities_length)
    };
    let layout = match validate_sample_layout(
        variants,
        variant_count,
        ambiguous.len(),
        diplotypes,
        segment_lengths,
        transition_probabilities.len(),
        missing_probabilities.len(),
    ) {
        Ok(value) => value,
        Err(status) => return status,
    };
    let missing_probabilities = &missing_probabilities[..layout.missing * 8];

    let mut rng = LogicalRng::new(seed, domain, iteration, item);
    let mut sampled = vec![0u8; diplotypes.len()];
    if rng.next_f64() < 0.5 {
        sample_forward(
            diplotypes,
            &transition_probabilities[..layout.transitions],
            &mut sampled,
            &mut rng,
        );
    } else {
        sample_backward(
            diplotypes,
            &transition_probabilities[..layout.transitions],
            layout.transitions,
            &mut sampled,
            &mut rng,
        );
    }
    apply_sample(
        ApplySample {
            variants,
            ambiguous,
            segment_lengths,
            sampled: &sampled,
            missing_probabilities,
            haploid: haploid != 0,
        },
        &mut rng,
    );
    STATUS_OK
}

#[no_mangle]
/// Select the maximum-probability stored path and apply it to packed alleles.
///
/// Stored transition indexes must be strictly increasing and correspond
/// one-for-one with the compressed stored probabilities.
///
/// # Safety
///
/// Every buffer must be valid for its stated length. The mutable packed-variant
/// buffer must not overlap any immutable input. Invalid layouts are rejected
/// before any allele is modified.
pub unsafe extern "C" fn shapeit_genotype_solve_v1(
    variants: *mut u8,
    variants_length: usize,
    variant_count: usize,
    ambiguous: *const u8,
    ambiguous_length: usize,
    diplotypes: *const u64,
    diplotypes_length: usize,
    segment_lengths: *const u16,
    segment_lengths_length: usize,
    stored_transition_indexes: *const u32,
    stored_transition_indexes_length: usize,
    stored_transition_probabilities: *const f32,
    stored_transition_probabilities_length: usize,
    missing_probabilities: *const f32,
    missing_probabilities_length: usize,
    haploid: u8,
    storage_events: u32,
) -> u32 {
    if diplotypes.is_null()
        || segment_lengths.is_null()
        || (stored_transition_indexes_length != 0 && stored_transition_indexes.is_null())
        || (stored_transition_probabilities_length != 0
            && stored_transition_probabilities.is_null())
        || (ambiguous_length != 0 && ambiguous.is_null())
        || (missing_probabilities_length != 0 && missing_probabilities.is_null())
    {
        return STATUS_NULL_POINTER;
    }
    if haploid > 1
        || storage_events == 0
        || stored_transition_indexes_length != stored_transition_probabilities_length
    {
        return STATUS_INVALID_DIMENSIONS;
    }
    let required_variants = match required_variant_bytes(variant_count) {
        Ok(value) => value,
        Err(status) => return status,
    };
    if required_variants > variants_length {
        return STATUS_OUT_OF_BOUNDS;
    }
    if variants_length != 0 && variants.is_null() {
        return STATUS_NULL_POINTER;
    }
    let variants = if variants_length == 0 {
        &mut []
    } else {
        slice::from_raw_parts_mut(variants, variants_length)
    };
    let ambiguous = if ambiguous_length == 0 {
        &[]
    } else {
        slice::from_raw_parts(ambiguous, ambiguous_length)
    };
    let diplotypes = slice::from_raw_parts(diplotypes, diplotypes_length);
    let segment_lengths = slice::from_raw_parts(segment_lengths, segment_lengths_length);
    let stored_transition_indexes = if stored_transition_indexes_length == 0 {
        &[]
    } else {
        slice::from_raw_parts(stored_transition_indexes, stored_transition_indexes_length)
    };
    let stored_transition_probabilities = if stored_transition_probabilities_length == 0 {
        &[]
    } else {
        slice::from_raw_parts(
            stored_transition_probabilities,
            stored_transition_probabilities_length,
        )
    };
    let missing_probabilities = if missing_probabilities_length == 0 {
        &[]
    } else {
        slice::from_raw_parts(missing_probabilities, missing_probabilities_length)
    };
    let layout = match validate_sample_layout(
        variants,
        variant_count,
        ambiguous.len(),
        diplotypes,
        segment_lengths,
        usize::MAX,
        missing_probabilities.len(),
    ) {
        Ok(value) => value,
        Err(status) => return status,
    };
    if layout.transitions > u32::MAX as usize {
        return STATUS_INTEGER_OVERFLOW;
    }
    let mut previous = None;
    for &index in stored_transition_indexes {
        if index as usize >= layout.transitions || previous.is_some_and(|value| index <= value) {
            return STATUS_INVALID_DIMENSIONS;
        }
        previous = Some(index);
    }

    let sampled = solve_path(
        diplotypes,
        layout.transitions,
        stored_transition_indexes,
        stored_transition_probabilities,
    );
    apply_solution(ApplySolution {
        variants,
        ambiguous,
        segment_lengths,
        sampled: &sampled,
        missing_probabilities: &missing_probabilities[..layout.missing * 8],
        storage_events,
        haploid: haploid != 0,
    });
    STATUS_OK
}

#[no_mangle]
/// Accumulate one main-iteration probability set in Rust-owned storage.
///
/// A null storage pointer allocates the persistent mask from transition
/// probabilities at or above `1e-6`. Later calls preserve that mask and apply
/// the established float accumulation order.
///
/// # Safety
///
/// `storage` must be writable and contain null or a live storage object from
/// this function. Non-empty probability buffers must be readable.
pub unsafe extern "C" fn shapeit_genotype_storage_update_v1(
    storage: *mut *mut GenotypeStorageV1,
    transition_probabilities: *const f64,
    transition_count: usize,
    missing_probabilities: *const f32,
    missing_probabilities_length: usize,
) -> u32 {
    if storage.is_null()
        || (transition_count != 0 && transition_probabilities.is_null())
        || (missing_probabilities_length != 0 && missing_probabilities.is_null())
    {
        return STATUS_NULL_POINTER;
    }
    let transition_probabilities = if transition_count == 0 {
        &[]
    } else {
        slice::from_raw_parts(transition_probabilities, transition_count)
    };
    let missing_probabilities = if missing_probabilities_length == 0 {
        &[]
    } else {
        slice::from_raw_parts(missing_probabilities, missing_probabilities_length)
    };
    if (*storage).is_null() {
        let value = match GenotypeStorageV1::new(transition_probabilities, missing_probabilities) {
            Ok(value) => value,
            Err(status) => return status,
        };
        *storage = Box::into_raw(Box::new(value));
        STATUS_OK
    } else {
        match (*(*storage)).update(transition_probabilities, missing_probabilities) {
            Ok(()) => STATUS_OK,
            Err(status) => status,
        }
    }
}

#[no_mangle]
/// Free Rust-owned genotype probability storage. Null is accepted.
///
/// # Safety
///
/// `storage` must be null or a live pointer returned by the update function,
/// and it must be freed at most once.
pub unsafe extern "C" fn shapeit_genotype_storage_free_v1(storage: *mut GenotypeStorageV1) {
    if !storage.is_null() {
        drop(Box::from_raw(storage));
    }
}

#[no_mangle]
/// Borrow the serialized mask and accumulated probability arrays.
///
/// # Safety
///
/// `storage` must be live and `view` writable. Borrowed pointers remain valid
/// only until the next storage update or free.
pub unsafe extern "C" fn shapeit_genotype_storage_borrow_v1(
    storage: *const GenotypeStorageV1,
    view: *mut GenotypeStorageViewV1,
) -> u32 {
    if storage.is_null() || view.is_null() {
        return STATUS_NULL_POINTER;
    }
    *view = (*storage).view();
    STATUS_OK
}

#[no_mangle]
/// Solve a genotype graph directly from Rust-owned accumulated storage.
///
/// # Safety
///
/// Graph buffers follow `shapeit_genotype_solve_v1`; `storage` must be live and
/// must not overlap the graph buffers.
pub unsafe extern "C" fn shapeit_genotype_solve_storage_v1(
    variants: *mut u8,
    variants_length: usize,
    variant_count: usize,
    ambiguous: *const u8,
    ambiguous_length: usize,
    diplotypes: *const u64,
    diplotypes_length: usize,
    segment_lengths: *const u16,
    segment_lengths_length: usize,
    storage: *const GenotypeStorageV1,
    haploid: u8,
) -> u32 {
    if storage.is_null() {
        return STATUS_NULL_POINTER;
    }
    let storage = &*storage;
    shapeit_genotype_solve_v1(
        variants,
        variants_length,
        variant_count,
        ambiguous,
        ambiguous_length,
        diplotypes,
        diplotypes_length,
        segment_lengths,
        segment_lengths_length,
        storage.transition_indexes.as_ptr(),
        storage.transition_indexes.len(),
        storage.transition_probabilities.as_ptr(),
        storage.transition_probabilities.len(),
        storage.missing_probabilities.as_ptr(),
        storage.missing_probabilities.len(),
        haploid,
        storage.storage_events,
    )
}

#[no_mangle]
/// Build all HMM windows for one common-phasing genotype graph.
///
/// RNG coordinates identify a fresh logical Philox stream. Recursive split
/// decisions consume it in the established depth-first order.
///
/// # Safety
///
/// Every input buffer must be readable for its stated length. `windows` must be
/// writable for `windows_capacity` records and `windows_length` must be
/// writable. Invalid layouts are rejected before output is modified.
pub unsafe extern "C" fn shapeit_genotype_windows_v1(
    variants: *const u8,
    variants_length: usize,
    variant_count: usize,
    diplotypes: *const u64,
    diplotypes_length: usize,
    segment_lengths: *const u16,
    segment_lengths_length: usize,
    segment_start_centimorgans: *const f64,
    segment_start_centimorgans_length: usize,
    segment_stop_centimorgans: *const f64,
    segment_stop_centimorgans_length: usize,
    minimum_window_centimorgans: f32,
    seed: u64,
    domain: u32,
    iteration: u32,
    item: u64,
    windows: *mut GenotypeWindowV1,
    windows_capacity: usize,
    windows_length: *mut usize,
) -> u32 {
    if windows_length.is_null()
        || (variants_length != 0 && variants.is_null())
        || (diplotypes_length != 0 && diplotypes.is_null())
        || (segment_lengths_length != 0 && segment_lengths.is_null())
        || (segment_start_centimorgans_length != 0 && segment_start_centimorgans.is_null())
        || (segment_stop_centimorgans_length != 0 && segment_stop_centimorgans.is_null())
        || (windows_capacity != 0 && windows.is_null())
    {
        return STATUS_NULL_POINTER;
    }
    let required_variants = match required_variant_bytes(variant_count) {
        Ok(value) => value,
        Err(status) => return status,
    };
    if required_variants > variants_length {
        return STATUS_OUT_OF_BOUNDS;
    }
    let variants = if variants_length == 0 {
        &[]
    } else {
        slice::from_raw_parts(variants, variants_length)
    };
    let diplotypes = if diplotypes_length == 0 {
        &[]
    } else {
        slice::from_raw_parts(diplotypes, diplotypes_length)
    };
    let segment_lengths = if segment_lengths_length == 0 {
        &[]
    } else {
        slice::from_raw_parts(segment_lengths, segment_lengths_length)
    };
    let segment_start_centimorgans = if segment_start_centimorgans_length == 0 {
        &[]
    } else {
        slice::from_raw_parts(
            segment_start_centimorgans,
            segment_start_centimorgans_length,
        )
    };
    let segment_stop_centimorgans = if segment_stop_centimorgans_length == 0 {
        &[]
    } else {
        slice::from_raw_parts(segment_stop_centimorgans, segment_stop_centimorgans_length)
    };
    let mut rng = LogicalRng::new(seed, domain, iteration, item);
    let output = match build_windows(
        WindowInputs {
            variants,
            variant_count,
            diplotypes,
            segment_lengths,
            segment_start_centimorgans,
            segment_stop_centimorgans,
            minimum_window_centimorgans,
        },
        &mut rng,
    ) {
        Ok(value) => value,
        Err(status) => return status,
    };
    if output.len() > windows_capacity {
        return STATUS_OUT_OF_BOUNDS;
    }
    if !output.is_empty() {
        let windows = slice::from_raw_parts_mut(windows, windows_capacity);
        windows[..output.len()].copy_from_slice(&output);
    }
    *windows_length = output.len();
    STATUS_OK
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pack(variants: &[u8]) -> Vec<u8> {
        let mut packed = vec![0u8; (variants.len() + 1) >> 1];
        for (locus, &variant) in variants.iter().enumerate() {
            packed[locus >> 1] |= variant << ((locus & 1) << 2);
        }
        packed
    }

    fn build(variants: &[u8]) -> (GraphSizes, Vec<u16>, Vec<u8>, Vec<u64>, u32) {
        let packed = pack(variants);
        let sizes = graph_sizes(&packed, variants.len());
        let mut lengths = vec![0u16; sizes.segments];
        let mut ambiguous = vec![0u8; sizes.ambiguous];
        let mut diplotypes = vec![0u64; sizes.segments];
        let transitions = build_graph(
            &packed,
            variants.len(),
            &mut lengths,
            &mut ambiguous,
            &mut diplotypes,
        );
        (sizes, lengths, ambiguous, diplotypes, transitions)
    }

    #[test]
    fn storage_preserves_mask_format_and_cpp_float_accumulation() {
        let first_transitions = [0.0f64, 1e-6, 0.25, 9e-7, 1.0];
        let first_missing = [0.125f32, 0.5];
        let mut storage = GenotypeStorageV1::new(&first_transitions, &first_missing).unwrap();
        assert_eq!(storage.transition_mask, [0b0001_0110]);
        assert_eq!(storage.transition_indexes, [1, 2, 4]);
        assert_eq!(storage.storage_events, 1);

        let next_transitions = [0.75f64, 0.333_333_333_333, 0.125, 2.0, 1e-8];
        let next_missing = [0.25f32, 0.125];
        let expected_transitions: Vec<f32> = storage
            .transition_indexes
            .iter()
            .zip(storage.transition_probabilities.iter())
            .map(|(&index, &stored)| (f64::from(stored) + next_transitions[index as usize]) as f32)
            .collect();
        storage.update(&next_transitions, &next_missing).unwrap();
        assert_eq!(storage.transition_probabilities, expected_transitions);
        assert_eq!(storage.missing_probabilities, [0.375, 0.625]);
        assert_eq!(storage.storage_events, 2);

        let view = storage.view();
        assert_eq!(view.transition_count, first_transitions.len());
        assert_eq!(view.transition_mask_length, 1);
        assert_eq!(view.transition_probabilities_length, 3);
        assert_eq!(view.missing_probabilities_length, 2);
        assert_eq!(view.storage_events, 2);
    }

    #[test]
    fn graph_builder_preserves_unfolding_and_segment_limits() {
        let variants = [2, 2, 2, 2, 1, 0];
        let (sizes, lengths, ambiguous, diplotypes, transitions) = build(&variants);
        assert_eq!(
            sizes,
            GraphSizes {
                segments: 2,
                ambiguous: 4,
                missing: 1
            }
        );
        assert_eq!(lengths, [3, 3]);
        assert_eq!(ambiguous, [0xaa, 0xcc, 0xf0, 0xaa]);
        let expected0 = MASK_INIT & MASK_UNFOLD0 & MASK_UNFOLD1 & MASK_UNFOLD2;
        let expected1 = MASK_INIT & MASK_UNFOLD0;
        assert_eq!(diplotypes, [expected0, expected1]);
        assert_eq!(
            transitions,
            expected0.count_ones() + expected0.count_ones() * expected1.count_ones()
        );
    }

    #[test]
    fn graph_builder_preserves_scaffold_order_and_alleles() {
        let variants = [3 | 4, 2, 2, 2];
        let (sizes, lengths, ambiguous, diplotypes, _) = build(&variants);
        assert_eq!(
            sizes,
            GraphSizes {
                segments: 2,
                ambiguous: 4,
                missing: 0
            }
        );
        assert_eq!(lengths, [3, 1]);
        assert_eq!(ambiguous, [0x55, 0xcc, 0xf0, 0xaa]);
        assert_eq!(
            diplotypes,
            [
                MASK_SCAFFOLD & MASK_UNFOLD1 & MASK_UNFOLD2,
                MASK_INIT & MASK_UNFOLD0
            ]
        );
    }

    #[test]
    fn pedigree_scaffolding_preserves_trio_and_duo_rules() {
        let father = pack(&[12, 12, 6, 0, 12]);
        let mother = pack(&[0, 6, 12, 12, 0]);
        let mut child = pack(&[6, 10, 6, 12, 1]);
        let mut counts = [0u32; 4];
        let status = unsafe {
            shapeit_genotype_pedigree_scaffold_v1(
                child.as_mut_ptr(),
                child.len(),
                5,
                father.as_ptr(),
                father.len(),
                mother.as_ptr(),
                mother.len(),
                0,
                counts.as_mut_ptr(),
                counts.len(),
            )
        };
        assert_eq!(status, STATUS_OK);
        assert_eq!(child, pack(&[7, 7, 11, 12, 7]));
        assert_eq!(counts, [1, 4, 3, 0]);

        let parent = pack(&[0]);
        let mut paternal = pack(&[6]);
        let mut maternal = pack(&[6]);
        let mut paternal_counts = [0u32; 4];
        let mut maternal_counts = [0u32; 4];
        let paternal_status = unsafe {
            shapeit_genotype_pedigree_scaffold_v1(
                paternal.as_mut_ptr(),
                paternal.len(),
                1,
                parent.as_ptr(),
                parent.len(),
                core::ptr::null(),
                0,
                1,
                paternal_counts.as_mut_ptr(),
                paternal_counts.len(),
            )
        };
        let maternal_status = unsafe {
            shapeit_genotype_pedigree_scaffold_v1(
                maternal.as_mut_ptr(),
                maternal.len(),
                1,
                core::ptr::null(),
                0,
                parent.as_ptr(),
                parent.len(),
                2,
                maternal_counts.as_mut_ptr(),
                maternal_counts.len(),
            )
        };
        assert_eq!(paternal_status, STATUS_OK);
        assert_eq!(maternal_status, STATUS_OK);
        assert_eq!(paternal, pack(&[11]));
        assert_eq!(maternal, pack(&[7]));
        assert_eq!(paternal_counts, [0, 1, 1, 0]);
        assert_eq!(maternal_counts, [0, 1, 1, 0]);
    }

    #[test]
    fn haploid_reset_preserves_existing_bitwise_missing_semantics() {
        let mut variants = pack(&[2, 3, 0, 1]);
        let mut reset_count = 0u32;
        let status = unsafe {
            shapeit_genotype_reset_haploid_hets_v1(
                variants.as_mut_ptr(),
                variants.len(),
                4,
                &mut reset_count,
            )
        };
        assert_eq!(status, STATUS_OK);
        assert_eq!(variants, pack(&[3, 3, 0, 1]));
        assert_eq!(reset_count, 2);
    }

    #[test]
    fn graph_abi_rejects_wrong_output_sizes_without_writing() {
        let variants = pack(&[2, 0]);
        let mut lengths = [0xa5a5u16; 2];
        let mut ambiguous = [0xa5u8; 1];
        let mut diplotypes = [0xa5a5_a5a5_a5a5_a5a5u64; 2];
        let mut transitions = 0xa5a5_a5a5u32;
        let status = unsafe {
            shapeit_genotype_graph_build_v1(
                variants.as_ptr(),
                variants.len(),
                2,
                lengths.as_mut_ptr(),
                lengths.len(),
                ambiguous.as_mut_ptr(),
                ambiguous.len(),
                diplotypes.as_mut_ptr(),
                diplotypes.len(),
                &mut transitions,
            )
        };
        assert_eq!(status, STATUS_INVALID_DIMENSIONS);
        assert_eq!(lengths, [0xa5a5; 2]);
        assert_eq!(ambiguous, [0xa5]);
        assert_eq!(diplotypes, [0xa5a5_a5a5_a5a5_a5a5; 2]);
        assert_eq!(transitions, 0xa5a5_a5a5);
    }

    #[test]
    fn graph_sampler_applies_sampled_diplotypes_without_changing_graph_codes() {
        let mut variants = [0x22u8];
        let ambiguous = [0xaa, 0xcc];
        let diplotypes = [1u64 << 9, 1u64 << 18];
        let lengths = [1u16, 1u16];
        let transitions = [1.0f64, 1.0f64];
        let status = unsafe {
            shapeit_genotype_sample_v1(
                variants.as_mut_ptr(),
                variants.len(),
                2,
                ambiguous.as_ptr(),
                ambiguous.len(),
                diplotypes.as_ptr(),
                diplotypes.len(),
                lengths.as_ptr(),
                lengths.len(),
                transitions.as_ptr(),
                transitions.len(),
                core::ptr::null(),
                0,
                0,
                15_052_011,
                3,
                7,
                11,
            )
        };
        assert_eq!(status, STATUS_OK);
        assert_eq!(variants, [0xee]);
    }

    #[test]
    fn graph_solver_applies_stored_path_and_missing_consensus() {
        let mut variants = [0x21u8];
        let ambiguous = [0xaau8];
        let diplotypes = [1u64 << 9];
        let lengths = [2u16];
        let stored_indexes = [0u32];
        let stored_probabilities = [1.0f32];
        let mut missing_probabilities = [0.0f32; 8];
        missing_probabilities[1] = 1.0;
        let status = unsafe {
            shapeit_genotype_solve_v1(
                variants.as_mut_ptr(),
                variants.len(),
                2,
                ambiguous.as_ptr(),
                ambiguous.len(),
                diplotypes.as_ptr(),
                diplotypes.len(),
                lengths.as_ptr(),
                lengths.len(),
                stored_indexes.as_ptr(),
                stored_indexes.len(),
                stored_probabilities.as_ptr(),
                stored_probabilities.len(),
                missing_probabilities.as_ptr(),
                missing_probabilities.len(),
                0,
                2,
            )
        };
        assert_eq!(status, STATUS_OK);
        assert_eq!(variants, [0xed]);
    }

    #[test]
    fn graph_solver_preserves_haploid_missing_consensus_rule() {
        let mut variants = [0x01u8];
        let diplotypes = [1u64 << 9];
        let lengths = [1u16];
        let stored_indexes = [0u32];
        let stored_probabilities = [1.0f32];
        let mut missing_probabilities = [0.0f32; 8];
        missing_probabilities[1] = 1.5;
        let status = unsafe {
            shapeit_genotype_solve_v1(
                variants.as_mut_ptr(),
                variants.len(),
                1,
                core::ptr::null(),
                0,
                diplotypes.as_ptr(),
                diplotypes.len(),
                lengths.as_ptr(),
                lengths.len(),
                stored_indexes.as_ptr(),
                stored_indexes.len(),
                stored_probabilities.as_ptr(),
                stored_probabilities.len(),
                missing_probabilities.as_ptr(),
                missing_probabilities.len(),
                1,
                2,
            )
        };
        assert_eq!(status, STATUS_OK);
        assert_eq!(variants, [0x0d]);
    }

    #[test]
    fn window_builder_preserves_graph_coordinate_conventions() {
        let variants = [0u8; 50];
        let diplotypes = [1u64; 4];
        let lengths = [25u16; 4];
        let start_centimorgans = [0.0f64, 1.0, 2.0, 3.0];
        let stop_centimorgans = [0.9f64, 1.9, 2.9, 4.0];
        let mut windows = [GenotypeWindowV1::default(); 4];
        let mut window_count = 0usize;
        let status = unsafe {
            shapeit_genotype_windows_v1(
                variants.as_ptr(),
                variants.len(),
                100,
                diplotypes.as_ptr(),
                diplotypes.len(),
                lengths.as_ptr(),
                lengths.len(),
                start_centimorgans.as_ptr(),
                start_centimorgans.len(),
                stop_centimorgans.as_ptr(),
                stop_centimorgans.len(),
                1.0,
                15_052_011,
                2,
                3,
                7,
                windows.as_mut_ptr(),
                windows.len(),
                &mut window_count,
            )
        };
        assert_eq!(status, STATUS_OK);
        assert_eq!(window_count, 1);
        assert_eq!(
            windows[0],
            GenotypeWindowV1 {
                start_locus: 0,
                start_segment: 0,
                start_ambiguous: 0,
                start_missing: 0,
                start_transition: 1,
                stop_locus: 99,
                stop_segment: 3,
                stop_ambiguous: -1,
                stop_missing: -1,
                stop_transition: 3,
            }
        );
    }
}
