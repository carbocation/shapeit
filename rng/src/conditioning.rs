use core::{mem, slice};

use crate::bitmatrix::heterozygote_overlap;
use crate::genotype::{
    build_windows, GenotypeGraphV1, GenotypeWindowV1, LogicalRng, WindowInputs,
    STATUS_INTEGER_OVERFLOW, STATUS_INVALID_DIMENSIONS, STATUS_NULL_POINTER, STATUS_OK,
    STATUS_OUT_OF_BOUNDS,
};
use crate::ibd2::{shapeit_ibd2_tracks_push_v1, Ibd2TrackV1, Ibd2TracksV1};

const ABI_VERSION: u32 = 1;
const STATUS_INSUFFICIENT_STATES: u32 = 5;
const FALLBACK_HAPLOTYPES: usize = 100;

#[repr(C)]
pub struct ConditioningBuildV1 {
    abi_version: u32,
    struct_size: usize,

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

    selected_sites: *const u8,
    selected_sites_length: usize,
    site_grouping: *const i32,
    site_grouping_length: usize,
    pbwt_neighbors: *const i32,
    pbwt_neighbors_length: usize,
    pbwt_depth: usize,
    pbwt_group_count: usize,

    target_individual: usize,
    target_individual_count: usize,
    haplotype_count: usize,
    haploid_individuals: *const u8,
    haploid_individuals_length: usize,

    haplotypes: *const u8,
    haplotypes_length: usize,
    haplotype_stride: usize,
    maximum_heterozygote_mismatch: f32,

    window_seed: u64,
    window_domain: u32,
    window_iteration: u32,
    window_item: u64,
    fallback_seed: u64,
    fallback_domain: u32,
    fallback_iteration: u32,
    fallback_item: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ConditioningGraphBuildV1 {
    pub(crate) abi_version: u32,
    pub(crate) struct_size: usize,
    pub(crate) graph: *const GenotypeGraphV1,
    pub(crate) centimorgans: *const f64,
    pub(crate) centimorgans_length: usize,
    pub(crate) minimum_window_centimorgans: f32,
    pub(crate) selected_sites: *const u8,
    pub(crate) selected_sites_length: usize,
    pub(crate) site_grouping: *const i32,
    pub(crate) site_grouping_length: usize,
    pub(crate) pbwt_neighbors: *const i32,
    pub(crate) pbwt_neighbors_length: usize,
    pub(crate) pbwt_depth: usize,
    pub(crate) pbwt_group_count: usize,
    pub(crate) target_individual: usize,
    pub(crate) target_individual_count: usize,
    pub(crate) haplotype_count: usize,
    pub(crate) haploid_individuals: *const u8,
    pub(crate) haploid_individuals_length: usize,
    pub(crate) haplotypes: *const u8,
    pub(crate) haplotypes_length: usize,
    pub(crate) haplotype_stride: usize,
    pub(crate) maximum_heterozygote_mismatch: f32,
    pub(crate) window_seed: u64,
    pub(crate) window_domain: u32,
    pub(crate) window_iteration: u32,
    pub(crate) window_item: u64,
    pub(crate) fallback_seed: u64,
    pub(crate) fallback_domain: u32,
    pub(crate) fallback_iteration: u32,
    pub(crate) fallback_item: u64,
}

pub type ConditioningTrackV1 = Ibd2TrackV1;

#[derive(Default)]
pub struct ConditioningJobV1 {
    pub(crate) windows: Vec<GenotypeWindowV1>,
    pub(crate) states: Vec<Vec<u32>>,
    pub(crate) tracks: Vec<ConditioningTrackV1>,
    pub(crate) used_fallback: Vec<bool>,
    seen: Vec<u32>,
    seen_epoch: u32,
    ordering: Vec<u32>,
    pub(crate) subset_haplotypes: Vec<u8>,
    pub(crate) single_scratch: Vec<f32>,
    pub(crate) double_scratch: Vec<f64>,
    pub(crate) alpha_locus_scratch: Vec<i32>,
    pub(crate) index_scratch: Vec<usize>,
    pub(crate) transition_probabilities: Vec<f64>,
    pub(crate) missing_probabilities: Vec<f32>,
    segment_start_centimorgans: Vec<f64>,
    segment_stop_centimorgans: Vec<f64>,
}

struct ConditioningInputs<'a> {
    windows: Vec<GenotypeWindowV1>,
    selected_sites: &'a [u8],
    site_grouping: &'a [i32],
    pbwt_neighbors: &'a [i32],
    pbwt_depth: usize,
    pbwt_group_count: usize,
    target_individual: usize,
    target_individual_count: usize,
    haplotype_count: usize,
    haploid_individuals: &'a [u8],
    haplotypes: &'a [u8],
    haplotype_stride: usize,
    maximum_heterozygote_mismatch: f32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ConditioningSharedLayout {
    variant_count: usize,
    selected_sites_address: usize,
    selected_sites_length: usize,
    site_grouping_address: usize,
    site_grouping_length: usize,
    pbwt_neighbors_address: usize,
    pbwt_neighbors_length: usize,
    pbwt_depth: usize,
    pbwt_group_count: usize,
    target_individual_count: usize,
    haplotype_count: usize,
    haploid_individuals_address: usize,
    haploid_individuals_length: usize,
    haplotypes_address: usize,
    haplotypes_length: usize,
    haplotype_stride: usize,
    pbwt_offset: usize,
}

impl ConditioningSharedLayout {
    #[inline]
    fn matches_inputs(&self, inputs: &ConditioningInputs<'_>) -> bool {
        self.selected_sites_address == inputs.selected_sites.as_ptr() as usize
            && self.selected_sites_length == inputs.selected_sites.len()
            && self.site_grouping_address == inputs.site_grouping.as_ptr() as usize
            && self.site_grouping_length == inputs.site_grouping.len()
            && self.pbwt_neighbors_address == inputs.pbwt_neighbors.as_ptr() as usize
            && self.pbwt_neighbors_length == inputs.pbwt_neighbors.len()
            && self.pbwt_depth == inputs.pbwt_depth
            && self.pbwt_group_count == inputs.pbwt_group_count
            && self.target_individual_count == inputs.target_individual_count
            && self.haplotype_count == inputs.haplotype_count
            && self.haploid_individuals_address == inputs.haploid_individuals.as_ptr() as usize
            && self.haploid_individuals_length == inputs.haploid_individuals.len()
            && self.haplotypes_address == inputs.haplotypes.as_ptr() as usize
            && self.haplotypes_length == inputs.haplotypes.len()
            && self.haplotype_stride == inputs.haplotype_stride
    }

    #[inline]
    fn matches_graph_build(&self, parameters: &ConditioningGraphBuildV1) -> bool {
        self.selected_sites_address == parameters.selected_sites as usize
            && self.selected_sites_length == parameters.selected_sites_length
            && self.site_grouping_address == parameters.site_grouping as usize
            && self.site_grouping_length == parameters.site_grouping_length
            && self.pbwt_neighbors_address == parameters.pbwt_neighbors as usize
            && self.pbwt_neighbors_length == parameters.pbwt_neighbors_length
            && self.pbwt_depth == parameters.pbwt_depth
            && self.pbwt_group_count == parameters.pbwt_group_count
            && self.target_individual_count == parameters.target_individual_count
            && self.haplotype_count == parameters.haplotype_count
            && self.haploid_individuals_address == parameters.haploid_individuals as usize
            && self.haploid_individuals_length == parameters.haploid_individuals_length
            && self.haplotypes_address == parameters.haplotypes as usize
            && self.haplotypes_length == parameters.haplotypes_length
            && self.haplotype_stride == parameters.haplotype_stride
    }
}

#[cfg(test)]
std::thread_local! {
    static CONDITIONING_SHARED_VALIDATION_SCANS: core::cell::Cell<usize> = const {
        core::cell::Cell::new(0)
    };
}

#[cfg(test)]
fn record_conditioning_shared_validation_scan() {
    CONDITIONING_SHARED_VALIDATION_SCANS.with(|scans| scans.set(scans.get() + 1));
}

#[cfg(not(test))]
#[inline]
fn record_conditioning_shared_validation_scan() {}

#[cfg(test)]
pub(crate) fn reset_conditioning_shared_validation_scans() {
    CONDITIONING_SHARED_VALIDATION_SCANS.with(|scans| scans.set(0));
}

#[cfg(test)]
pub(crate) fn conditioning_shared_validation_scans() -> usize {
    CONDITIONING_SHARED_VALIDATION_SCANS.with(core::cell::Cell::get)
}

#[inline]
unsafe fn const_slice<'a, T>(pointer: *const T, length: usize) -> &'a [T] {
    if length == 0 {
        &[]
    } else {
        slice::from_raw_parts(pointer, length)
    }
}

#[inline]
fn require_pointer<T>(pointer: *const T, length: usize) -> Result<(), u32> {
    if length != 0 && pointer.is_null() {
        Err(STATUS_NULL_POINTER)
    } else {
        Ok(())
    }
}

fn validate_conditioning_shared(
    inputs: &ConditioningInputs<'_>,
) -> Result<ConditioningSharedLayout, u32> {
    record_conditioning_shared_validation_scan();
    if inputs.selected_sites.len() != inputs.site_grouping.len()
        || inputs.target_individual >= inputs.target_individual_count
        || inputs.target_individual_count == 0
        || inputs.pbwt_depth == 0
        || inputs.pbwt_group_count == 0
        || inputs.haplotype_stride == 0
        || inputs.haploid_individuals.len() < inputs.target_individual_count
        || inputs.haplotype_count > i32::MAX as usize
    {
        return Err(STATUS_INVALID_DIMENSIONS);
    }
    if inputs.selected_sites.iter().any(|&value| value > 1)
        || inputs.haploid_individuals[..inputs.target_individual_count]
            .iter()
            .any(|&value| value > 1)
    {
        return Err(STATUS_INVALID_DIMENSIONS);
    }
    for &group in inputs.site_grouping {
        if group < 0 || group as usize >= inputs.pbwt_group_count {
            return Err(STATUS_OUT_OF_BOUNDS);
        }
    }
    let target_haplotype_count = inputs
        .target_individual_count
        .checked_mul(2)
        .ok_or(STATUS_INTEGER_OVERFLOW)?;
    if target_haplotype_count > inputs.haplotype_count {
        return Err(STATUS_INVALID_DIMENSIONS);
    }
    let pbwt_offset = inputs
        .pbwt_group_count
        .checked_mul(target_haplotype_count)
        .ok_or(STATUS_INTEGER_OVERFLOW)?;
    let required_neighbors = inputs
        .pbwt_depth
        .checked_mul(pbwt_offset)
        .ok_or(STATUS_INTEGER_OVERFLOW)?;
    if required_neighbors > inputs.pbwt_neighbors.len() {
        return Err(STATUS_OUT_OF_BOUNDS);
    }
    let required_haplotypes = inputs
        .haplotype_count
        .checked_mul(inputs.haplotype_stride)
        .ok_or(STATUS_INTEGER_OVERFLOW)?;
    if required_haplotypes > inputs.haplotypes.len() {
        return Err(STATUS_OUT_OF_BOUNDS);
    }
    Ok(ConditioningSharedLayout {
        variant_count: inputs.selected_sites.len(),
        selected_sites_address: inputs.selected_sites.as_ptr() as usize,
        selected_sites_length: inputs.selected_sites.len(),
        site_grouping_address: inputs.site_grouping.as_ptr() as usize,
        site_grouping_length: inputs.site_grouping.len(),
        pbwt_neighbors_address: inputs.pbwt_neighbors.as_ptr() as usize,
        pbwt_neighbors_length: inputs.pbwt_neighbors.len(),
        pbwt_depth: inputs.pbwt_depth,
        pbwt_group_count: inputs.pbwt_group_count,
        target_individual_count: inputs.target_individual_count,
        haplotype_count: inputs.haplotype_count,
        haploid_individuals_address: inputs.haploid_individuals.as_ptr() as usize,
        haploid_individuals_length: inputs.haploid_individuals.len(),
        haplotypes_address: inputs.haplotypes.as_ptr() as usize,
        haplotypes_length: inputs.haplotypes.len(),
        haplotype_stride: inputs.haplotype_stride,
        pbwt_offset,
    })
}

fn collect_conditioning_states(
    inputs: ConditioningInputs<'_>,
    fallback_rng: &mut LogicalRng,
    job: &mut ConditioningJobV1,
) -> Result<(), u32> {
    let validation = validate_conditioning_shared(&inputs)?;
    collect_conditioning_states_prevalidated(inputs, &validation, fallback_rng, job)
}

fn collect_conditioning_states_prevalidated(
    inputs: ConditioningInputs<'_>,
    validation: &ConditioningSharedLayout,
    fallback_rng: &mut LogicalRng,
    job: &mut ConditioningJobV1,
) -> Result<(), u32> {
    if !validation.matches_inputs(&inputs)
        || inputs.target_individual >= validation.target_individual_count
    {
        return Err(STATUS_INVALID_DIMENSIONS);
    }
    let ConditioningInputs {
        windows,
        selected_sites,
        site_grouping,
        pbwt_neighbors,
        pbwt_depth,
        pbwt_group_count,
        target_individual,
        target_individual_count,
        haplotype_count,
        haploid_individuals,
        haplotypes,
        haplotype_stride,
        maximum_heterozygote_mismatch,
    } = inputs;
    let pbwt_offset = validation.pbwt_offset;

    let target_haplotype0 = target_individual * 2;
    let target_haplotype1 = target_haplotype0 + 1;
    let ConditioningJobV1 {
        windows: job_windows,
        states,
        tracks,
        used_fallback,
        seen,
        seen_epoch,
        ordering,
        ..
    } = job;
    *job_windows = windows;
    if states.len() < job_windows.len() {
        states.resize_with(job_windows.len(), Vec::new);
    }
    for window_states in states.iter_mut() {
        window_states.clear();
    }
    states.truncate(job_windows.len());
    tracks.clear();
    used_fallback.clear();
    if seen.len() != haplotype_count {
        seen.clear();
        seen.resize(haplotype_count, 0);
        *seen_epoch = 0;
    }

    for (window_index, window) in job_windows.iter().copied().enumerate() {
        *seen_epoch = seen_epoch.wrapping_add(1);
        if *seen_epoch == 0 {
            seen.fill(0);
            *seen_epoch = 1;
        }
        let start_locus = window.start_locus as usize;
        let stop_locus = window.stop_locus as usize;
        if start_locus > stop_locus || stop_locus >= selected_sites.len() {
            return Err(STATUS_OUT_OF_BOUNDS);
        }
        let window_states = &mut states[window_index];
        for locus in start_locus..=stop_locus {
            if selected_sites[locus] == 0 {
                continue;
            }
            let group = site_grouping[locus] as usize;
            for depth in 0..pbwt_depth {
                let depth_offset = depth * pbwt_offset;
                for target_haplotype in [target_haplotype0, target_haplotype1] {
                    let neighbor =
                        pbwt_neighbors[depth_offset + target_haplotype * pbwt_group_count + group];
                    if neighbor < 0 {
                        continue;
                    }
                    let neighbor = neighbor as usize;
                    if neighbor >= haplotype_count {
                        return Err(STATUS_OUT_OF_BOUNDS);
                    }
                    if seen[neighbor] != *seen_epoch {
                        seen[neighbor] = *seen_epoch;
                        window_states.push(neighbor as u32);
                    }
                }
            }
        }
        window_states.sort_unstable();

        let mut remove = vec![false; window_states.len()];
        for index in 1..window_states.len() {
            let individual0 = window_states[index - 1] as usize / 2;
            let individual1 = window_states[index] as usize / 2;
            if individual0 == individual1
                && individual0 < target_individual_count
                && haploid_individuals[individual0] == 0
            {
                let overlap = heterozygote_overlap(
                    haplotypes,
                    haplotype_stride,
                    target_individual,
                    individual0,
                    start_locus,
                    stop_locus,
                )?;
                if overlap > maximum_heterozygote_mismatch {
                    remove[index - 1] = true;
                    remove[index] = true;
                    tracks.push(ConditioningTrackV1 {
                        individual: individual0 as i32,
                        from: window.start_locus,
                        to: window.stop_locus,
                    });
                }
            }
        }
        if remove.iter().any(|&value| value) {
            let mut index = 0usize;
            window_states.retain(|_| {
                let retain = !remove[index];
                index += 1;
                retain
            });
        }

        let fallback = window_states.len() < 2;
        if fallback {
            if ordering.len() != haplotype_count {
                ordering.resize(haplotype_count, 0);
            }
            for (index, value) in ordering.iter_mut().enumerate() {
                *value = index as u32;
            }
            for index in (1..ordering.len()).rev() {
                let swap = fallback_rng.next_bounded((index + 1) as u32) as usize;
                ordering.swap(index, swap);
            }
            let mut added = 0usize;
            for &random_state in ordering.iter() {
                if random_state as usize / 2 != target_individual {
                    window_states.push(random_state);
                    added += 1;
                    if added == FALLBACK_HAPLOTYPES {
                        break;
                    }
                }
            }
            window_states.sort_unstable();
            window_states.dedup();
            if window_states.len() < 2 {
                return Err(STATUS_INSUFFICIENT_STATES);
            }
        }
        used_fallback.push(fallback);
    }
    Ok(())
}

#[no_mangle]
pub extern "C" fn shapeit_conditioning_abi_version() -> u32 {
    ABI_VERSION
}

#[no_mangle]
/// Build one complete common-phasing conditioning job.
///
/// The returned opaque job owns all windows and conditioning-state vectors. A
/// live job supplied through `job` is rebuilt in place so worker-local scratch
/// and nested vector capacities are retained.
///
/// # Safety
///
/// Every pointer in `parameters` must be valid for its stated length. `job`
/// must be writable and contain either null or a live job returned by this
/// function. On success the caller owns the job and must eventually free it
/// with `shapeit_conditioning_job_free_v1`.
pub unsafe extern "C" fn shapeit_conditioning_job_build_v1(
    parameters: *const ConditioningBuildV1,
    job: *mut *mut ConditioningJobV1,
) -> u32 {
    if parameters.is_null() || job.is_null() {
        return STATUS_NULL_POINTER;
    }
    conditioning_job_build(&*parameters, job, None)
}

unsafe fn conditioning_job_build(
    parameters: &ConditioningBuildV1,
    job: *mut *mut ConditioningJobV1,
    shared_validation: Option<&ConditioningSharedLayout>,
) -> u32 {
    if parameters.abi_version != ABI_VERSION
        || parameters.struct_size < mem::size_of::<ConditioningBuildV1>()
    {
        return STATUS_INVALID_DIMENSIONS;
    }
    for result in [
        require_pointer(parameters.variants, parameters.variants_length),
        require_pointer(parameters.diplotypes, parameters.diplotypes_length),
        require_pointer(
            parameters.segment_lengths,
            parameters.segment_lengths_length,
        ),
        require_pointer(
            parameters.segment_start_centimorgans,
            parameters.segment_start_centimorgans_length,
        ),
        require_pointer(
            parameters.segment_stop_centimorgans,
            parameters.segment_stop_centimorgans_length,
        ),
        require_pointer(parameters.selected_sites, parameters.selected_sites_length),
        require_pointer(parameters.site_grouping, parameters.site_grouping_length),
        require_pointer(parameters.pbwt_neighbors, parameters.pbwt_neighbors_length),
        require_pointer(
            parameters.haploid_individuals,
            parameters.haploid_individuals_length,
        ),
        require_pointer(parameters.haplotypes, parameters.haplotypes_length),
    ] {
        if let Err(status) = result {
            return status;
        }
    }
    let required_variants = match parameters.variant_count.checked_add(1) {
        Some(value) => value >> 1,
        None => return STATUS_INTEGER_OVERFLOW,
    };
    if required_variants > parameters.variants_length
        || parameters.selected_sites_length != parameters.variant_count
        || parameters.site_grouping_length != parameters.variant_count
    {
        return STATUS_OUT_OF_BOUNDS;
    }

    let variants = const_slice(parameters.variants, parameters.variants_length);
    let diplotypes = const_slice(parameters.diplotypes, parameters.diplotypes_length);
    let segment_lengths = const_slice(
        parameters.segment_lengths,
        parameters.segment_lengths_length,
    );
    let segment_start_centimorgans = const_slice(
        parameters.segment_start_centimorgans,
        parameters.segment_start_centimorgans_length,
    );
    let segment_stop_centimorgans = const_slice(
        parameters.segment_stop_centimorgans,
        parameters.segment_stop_centimorgans_length,
    );
    let selected_sites = const_slice(parameters.selected_sites, parameters.selected_sites_length);
    let site_grouping = const_slice(parameters.site_grouping, parameters.site_grouping_length);
    let pbwt_neighbors = const_slice(parameters.pbwt_neighbors, parameters.pbwt_neighbors_length);
    let haploid_individuals = const_slice(
        parameters.haploid_individuals,
        parameters.haploid_individuals_length,
    );
    let haplotypes = const_slice(parameters.haplotypes, parameters.haplotypes_length);

    let mut window_rng = LogicalRng::new(
        parameters.window_seed,
        parameters.window_domain,
        parameters.window_iteration,
        parameters.window_item,
    );
    let windows = match build_windows(
        WindowInputs {
            variants,
            variant_count: parameters.variant_count,
            diplotypes,
            segment_lengths,
            segment_start_centimorgans,
            segment_stop_centimorgans,
            minimum_window_centimorgans: parameters.minimum_window_centimorgans,
        },
        &mut window_rng,
    ) {
        Ok(value) => value,
        Err(status) => return status,
    };
    let mut fallback_rng = LogicalRng::new(
        parameters.fallback_seed,
        parameters.fallback_domain,
        parameters.fallback_iteration,
        parameters.fallback_item,
    );
    let mut allocated = if (*job).is_null() {
        Some(Box::new(ConditioningJobV1::default()))
    } else {
        None
    };
    {
        let target = if let Some(value) = allocated.as_deref_mut() {
            value
        } else {
            &mut *(*job)
        };
        let inputs = ConditioningInputs {
            windows,
            selected_sites,
            site_grouping,
            pbwt_neighbors,
            pbwt_depth: parameters.pbwt_depth,
            pbwt_group_count: parameters.pbwt_group_count,
            target_individual: parameters.target_individual,
            target_individual_count: parameters.target_individual_count,
            haplotype_count: parameters.haplotype_count,
            haploid_individuals,
            haplotypes,
            haplotype_stride: parameters.haplotype_stride,
            maximum_heterozygote_mismatch: parameters.maximum_heterozygote_mismatch,
        };
        let collected = match shared_validation {
            Some(validation) => collect_conditioning_states_prevalidated(
                inputs,
                validation,
                &mut fallback_rng,
                target,
            ),
            None => collect_conditioning_states(inputs, &mut fallback_rng, target),
        };
        if let Err(status) = collected {
            return status;
        }
    }
    if let Some(value) = allocated {
        *job = Box::into_raw(value);
    }
    STATUS_OK
}

/// Validate the target-independent buffers shared by every sample in one
/// common-phasing iteration. The returned token is bound to their identities
/// and dimensions and must not outlive the immutable borrow promised by the
/// iteration ABI.
pub(crate) unsafe fn validate_conditioning_graph_job_shared_v1(
    parameters: &ConditioningGraphBuildV1,
    variant_count: usize,
) -> Result<ConditioningSharedLayout, u32> {
    if parameters.abi_version != ABI_VERSION
        || parameters.struct_size < mem::size_of::<ConditioningGraphBuildV1>()
    {
        return Err(STATUS_INVALID_DIMENSIONS);
    }
    for result in [
        require_pointer(parameters.centimorgans, parameters.centimorgans_length),
        require_pointer(parameters.selected_sites, parameters.selected_sites_length),
        require_pointer(parameters.site_grouping, parameters.site_grouping_length),
        require_pointer(parameters.pbwt_neighbors, parameters.pbwt_neighbors_length),
        require_pointer(
            parameters.haploid_individuals,
            parameters.haploid_individuals_length,
        ),
        require_pointer(parameters.haplotypes, parameters.haplotypes_length),
    ] {
        result?;
    }
    if parameters.centimorgans_length < variant_count
        || parameters.selected_sites_length != variant_count
        || parameters.site_grouping_length != variant_count
    {
        return Err(STATUS_OUT_OF_BOUNDS);
    }
    let inputs = ConditioningInputs {
        windows: Vec::new(),
        selected_sites: const_slice(parameters.selected_sites, parameters.selected_sites_length),
        site_grouping: const_slice(parameters.site_grouping, parameters.site_grouping_length),
        pbwt_neighbors: const_slice(parameters.pbwt_neighbors, parameters.pbwt_neighbors_length),
        pbwt_depth: parameters.pbwt_depth,
        pbwt_group_count: parameters.pbwt_group_count,
        target_individual: parameters.target_individual,
        target_individual_count: parameters.target_individual_count,
        haplotype_count: parameters.haplotype_count,
        haploid_individuals: const_slice(
            parameters.haploid_individuals,
            parameters.haploid_individuals_length,
        ),
        haplotypes: const_slice(parameters.haplotypes, parameters.haplotypes_length),
        haplotype_stride: parameters.haplotype_stride,
        maximum_heterozygote_mismatch: parameters.maximum_heterozygote_mismatch,
    };
    let validation = validate_conditioning_shared(&inputs)?;
    debug_assert_eq!(validation.variant_count, variant_count);
    Ok(validation)
}

/// Build one sample job after its iteration-shared buffers were validated.
/// Graph-local segment structure is still checked on every invocation.
pub(crate) unsafe fn conditioning_graph_job_build_prevalidated_v1(
    parameters: &ConditioningGraphBuildV1,
    job: *mut *mut ConditioningJobV1,
    shared_validation: &ConditioningSharedLayout,
) -> u32 {
    if job.is_null() {
        return STATUS_NULL_POINTER;
    }
    conditioning_graph_job_build(parameters, job, Some(shared_validation))
}

#[no_mangle]
/// Build a conditioning job directly from a Rust-owned genotype graph.
///
/// Segment boundary cM arrays are derived from the full-variant cM input and
/// retained as worker-local Rust workspace across job rebuilds.
///
/// # Safety
///
/// `parameters` must be valid and every non-empty buffer readable for its
/// stated length. `job` must be writable and contain null or a live job
/// returned by a conditioning-job constructor.
pub unsafe extern "C" fn shapeit_conditioning_graph_job_build_v1(
    parameters: *const ConditioningGraphBuildV1,
    job: *mut *mut ConditioningJobV1,
) -> u32 {
    if parameters.is_null() || job.is_null() {
        return STATUS_NULL_POINTER;
    }
    conditioning_graph_job_build(&*parameters, job, None)
}

unsafe fn conditioning_graph_job_build(
    parameters: &ConditioningGraphBuildV1,
    job: *mut *mut ConditioningJobV1,
    shared_validation: Option<&ConditioningSharedLayout>,
) -> u32 {
    if parameters.abi_version != ABI_VERSION
        || parameters.struct_size < mem::size_of::<ConditioningGraphBuildV1>()
    {
        return STATUS_INVALID_DIMENSIONS;
    }
    if parameters.graph.is_null() {
        return STATUS_NULL_POINTER;
    }
    if let Err(status) = require_pointer(parameters.centimorgans, parameters.centimorgans_length) {
        return status;
    }
    let graph = &*parameters.graph;
    if !graph.is_built() {
        return STATUS_INVALID_DIMENSIONS;
    }
    let (variant_count, _, _) = graph.hmm_dimensions();
    if shared_validation.is_some_and(|validation| {
        validation.variant_count != variant_count || !validation.matches_graph_build(parameters)
    }) {
        return STATUS_INVALID_DIMENSIONS;
    }
    if parameters.centimorgans_length < variant_count {
        return STATUS_OUT_OF_BOUNDS;
    }
    let centimorgans = const_slice(parameters.centimorgans, parameters.centimorgans_length);
    let (variants, _, segment_lengths, diplotypes) = graph.hmm_arrays();
    let mut validated_loci = 0usize;
    for &length in segment_lengths {
        if length == 0 {
            return STATUS_INVALID_DIMENSIONS;
        }
        validated_loci = match validated_loci.checked_add(length as usize) {
            Some(value) => value,
            None => return STATUS_INTEGER_OVERFLOW,
        };
    }
    if validated_loci != variant_count {
        return STATUS_INVALID_DIMENSIONS;
    }

    let mut segment_starts = if (*job).is_null() {
        Vec::new()
    } else {
        mem::take(&mut (**job).segment_start_centimorgans)
    };
    let mut segment_stops = if (*job).is_null() {
        Vec::new()
    } else {
        mem::take(&mut (**job).segment_stop_centimorgans)
    };
    segment_starts.clear();
    segment_stops.clear();
    segment_starts.reserve(segment_lengths.len());
    segment_stops.reserve(segment_lengths.len());
    let mut locus = 0usize;
    for &length in segment_lengths {
        segment_starts.push(centimorgans[locus]);
        locus += length as usize;
        segment_stops.push(centimorgans[locus - 1]);
    }

    let build = ConditioningBuildV1 {
        abi_version: parameters.abi_version,
        struct_size: mem::size_of::<ConditioningBuildV1>(),
        variants: variants.as_ptr(),
        variants_length: variants.len(),
        variant_count,
        diplotypes: diplotypes.as_ptr(),
        diplotypes_length: diplotypes.len(),
        segment_lengths: segment_lengths.as_ptr(),
        segment_lengths_length: segment_lengths.len(),
        segment_start_centimorgans: segment_starts.as_ptr(),
        segment_start_centimorgans_length: segment_starts.len(),
        segment_stop_centimorgans: segment_stops.as_ptr(),
        segment_stop_centimorgans_length: segment_stops.len(),
        minimum_window_centimorgans: parameters.minimum_window_centimorgans,
        selected_sites: parameters.selected_sites,
        selected_sites_length: parameters.selected_sites_length,
        site_grouping: parameters.site_grouping,
        site_grouping_length: parameters.site_grouping_length,
        pbwt_neighbors: parameters.pbwt_neighbors,
        pbwt_neighbors_length: parameters.pbwt_neighbors_length,
        pbwt_depth: parameters.pbwt_depth,
        pbwt_group_count: parameters.pbwt_group_count,
        target_individual: parameters.target_individual,
        target_individual_count: parameters.target_individual_count,
        haplotype_count: parameters.haplotype_count,
        haploid_individuals: parameters.haploid_individuals,
        haploid_individuals_length: parameters.haploid_individuals_length,
        haplotypes: parameters.haplotypes,
        haplotypes_length: parameters.haplotypes_length,
        haplotype_stride: parameters.haplotype_stride,
        maximum_heterozygote_mismatch: parameters.maximum_heterozygote_mismatch,
        window_seed: parameters.window_seed,
        window_domain: parameters.window_domain,
        window_iteration: parameters.window_iteration,
        window_item: parameters.window_item,
        fallback_seed: parameters.fallback_seed,
        fallback_domain: parameters.fallback_domain,
        fallback_iteration: parameters.fallback_iteration,
        fallback_item: parameters.fallback_item,
    };
    let status = conditioning_job_build(&build, job, shared_validation);
    if !(*job).is_null() {
        (**job).segment_start_centimorgans = segment_starts;
        (**job).segment_stop_centimorgans = segment_stops;
    }
    status
}

#[no_mangle]
/// Free an opaque conditioning job. A null pointer is accepted.
///
/// # Safety
///
/// `job` must be null or a live pointer returned by
/// `shapeit_conditioning_job_build_v1`, and it must be freed at most once.
pub unsafe extern "C" fn shapeit_conditioning_job_free_v1(job: *mut ConditioningJobV1) {
    if !job.is_null() {
        drop(Box::from_raw(job));
    }
}

#[no_mangle]
/// Return the number of windows in an opaque job, or zero for null.
///
/// # Safety
///
/// `job` must be null or point to a live conditioning job.
pub unsafe extern "C" fn shapeit_conditioning_job_window_count_v1(
    job: *const ConditioningJobV1,
) -> usize {
    if job.is_null() {
        0
    } else {
        (*job).windows.len()
    }
}

#[no_mangle]
/// Borrow one window and its conditioning states from an opaque job.
///
/// # Safety
///
/// `job` and every output pointer must be valid. The state pointer remains
/// valid only until the job is rebuilt or freed.
pub unsafe extern "C" fn shapeit_conditioning_job_window_v1(
    job: *const ConditioningJobV1,
    index: usize,
    window: *mut GenotypeWindowV1,
    states: *mut *const u32,
    states_length: *mut usize,
    used_fallback: *mut u8,
) -> u32 {
    if job.is_null()
        || window.is_null()
        || states.is_null()
        || states_length.is_null()
        || used_fallback.is_null()
    {
        return STATUS_NULL_POINTER;
    }
    let job = &*job;
    if index >= job.windows.len() {
        return STATUS_OUT_OF_BOUNDS;
    }
    *window = job.windows[index];
    *states = job.states[index].as_ptr();
    *states_length = job.states[index].len();
    *used_fallback = u8::from(job.used_fallback[index]);
    STATUS_OK
}

#[no_mangle]
/// Borrow all newly detected IBD2 tracks from an opaque job.
///
/// # Safety
///
/// `job` and both output pointers must be valid. The track pointer remains
/// valid only until the job is rebuilt or freed.
pub unsafe extern "C" fn shapeit_conditioning_job_tracks_v1(
    job: *const ConditioningJobV1,
    tracks: *mut *const ConditioningTrackV1,
    tracks_length: *mut usize,
) -> u32 {
    if job.is_null() || tracks.is_null() || tracks_length.is_null() {
        return STATUS_NULL_POINTER;
    }
    let job = &*job;
    *tracks = job.tracks.as_ptr();
    *tracks_length = job.tracks.len();
    STATUS_OK
}

#[no_mangle]
/// Append this job's detected IBD2 tracks directly to a Rust registry.
///
/// # Safety
///
/// `job` and `registry` must be live. The caller must serialize registry writes.
pub unsafe extern "C" fn shapeit_conditioning_job_push_tracks_v1(
    job: *const ConditioningJobV1,
    registry: *mut Ibd2TracksV1,
    source_individual: usize,
) -> u32 {
    if job.is_null() || registry.is_null() {
        return STATUS_NULL_POINTER;
    }
    shapeit_ibd2_tracks_push_v1(
        registry,
        source_individual,
        (*job).tracks.as_ptr(),
        (*job).tracks.len(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::ptr;

    #[test]
    fn empty_pbwt_uses_exact_complete_fallback_panel() {
        let variants = [0u8; 50];
        let diplotypes = [1u64; 4];
        let segment_lengths = [25u16; 4];
        let start_centimorgans = [0.0f64, 1.0, 2.0, 3.0];
        let stop_centimorgans = [0.9f64, 1.9, 2.9, 4.0];
        let mut selected_sites = [0u8; 100];
        let site_grouping = [0i32; 100];
        let pbwt_neighbors = [-1i32; 2];
        let haploid_individuals = [0u8];
        let haplotypes = [0u8; 52];
        let parameters = ConditioningBuildV1 {
            abi_version: ABI_VERSION,
            struct_size: mem::size_of::<ConditioningBuildV1>(),
            variants: variants.as_ptr(),
            variants_length: variants.len(),
            variant_count: 100,
            diplotypes: diplotypes.as_ptr(),
            diplotypes_length: diplotypes.len(),
            segment_lengths: segment_lengths.as_ptr(),
            segment_lengths_length: segment_lengths.len(),
            segment_start_centimorgans: start_centimorgans.as_ptr(),
            segment_start_centimorgans_length: start_centimorgans.len(),
            segment_stop_centimorgans: stop_centimorgans.as_ptr(),
            segment_stop_centimorgans_length: stop_centimorgans.len(),
            minimum_window_centimorgans: 1.0,
            selected_sites: selected_sites.as_ptr(),
            selected_sites_length: selected_sites.len(),
            site_grouping: site_grouping.as_ptr(),
            site_grouping_length: site_grouping.len(),
            pbwt_neighbors: pbwt_neighbors.as_ptr(),
            pbwt_neighbors_length: pbwt_neighbors.len(),
            pbwt_depth: 1,
            pbwt_group_count: 1,
            target_individual: 0,
            target_individual_count: 1,
            haplotype_count: 4,
            haploid_individuals: haploid_individuals.as_ptr(),
            haploid_individuals_length: haploid_individuals.len(),
            haplotypes: haplotypes.as_ptr(),
            haplotypes_length: haplotypes.len(),
            haplotype_stride: 13,
            maximum_heterozygote_mismatch: 0.75,
            window_seed: 15_052_011,
            window_domain: 2,
            window_iteration: 3,
            window_item: 0,
            fallback_seed: 15_052_011,
            fallback_domain: 8,
            fallback_iteration: 3,
            fallback_item: 0,
        };
        let mut job = ptr::null_mut();
        reset_conditioning_shared_validation_scans();
        let status = unsafe { shapeit_conditioning_job_build_v1(&parameters, &mut job) };
        assert_eq!(status, STATUS_OK);
        assert!(!job.is_null());
        let original_job = job;
        let status = unsafe { shapeit_conditioning_job_build_v1(&parameters, &mut job) };
        assert_eq!(status, STATUS_OK);
        assert_eq!(job, original_job);
        assert_eq!(conditioning_shared_validation_scans(), 2);

        selected_sites[99] = 2;
        assert_eq!(selected_sites[99], 2);
        let status = unsafe { shapeit_conditioning_job_build_v1(&parameters, &mut job) };
        assert_eq!(status, STATUS_INVALID_DIMENSIONS);
        assert_eq!(job, original_job);
        assert_eq!(conditioning_shared_validation_scans(), 3);
        let mut window = GenotypeWindowV1::default();
        let mut states = ptr::null();
        let mut states_length = 0usize;
        let mut used_fallback = 0u8;
        let status = unsafe {
            shapeit_conditioning_job_window_v1(
                job,
                0,
                &mut window,
                &mut states,
                &mut states_length,
                &mut used_fallback,
            )
        };
        assert_eq!(status, STATUS_OK);
        assert_eq!(window.start_locus, 0);
        assert_eq!(window.stop_locus, 99);
        assert_eq!(used_fallback, 1);
        assert_eq!(
            unsafe { slice::from_raw_parts(states, states_length) },
            [2, 3]
        );
        unsafe { shapeit_conditioning_job_free_v1(job) };
    }

    #[test]
    fn public_graph_builder_revalidates_shared_buffers() {
        let variants = [0u8];
        let mut graph = ptr::null_mut();
        let status = unsafe {
            crate::genotype::shapeit_genotype_graph_create_v1(
                variants.as_ptr(),
                variants.len(),
                1,
                &mut graph,
            )
        };
        assert_eq!(status, STATUS_OK);

        let centimorgans = [0.0f64];
        let mut selected_sites = [0u8];
        let site_grouping = [0i32];
        let pbwt_neighbors = [-1i32; 4];
        let haploid_individuals = [0u8; 2];
        let haplotypes = [0u8; 4];
        let parameters = ConditioningGraphBuildV1 {
            abi_version: ABI_VERSION,
            struct_size: mem::size_of::<ConditioningGraphBuildV1>(),
            graph,
            centimorgans: centimorgans.as_ptr(),
            centimorgans_length: centimorgans.len(),
            minimum_window_centimorgans: 1.0,
            selected_sites: selected_sites.as_ptr(),
            selected_sites_length: selected_sites.len(),
            site_grouping: site_grouping.as_ptr(),
            site_grouping_length: site_grouping.len(),
            pbwt_neighbors: pbwt_neighbors.as_ptr(),
            pbwt_neighbors_length: pbwt_neighbors.len(),
            pbwt_depth: 1,
            pbwt_group_count: 1,
            target_individual: 0,
            target_individual_count: 2,
            haplotype_count: 4,
            haploid_individuals: haploid_individuals.as_ptr(),
            haploid_individuals_length: haploid_individuals.len(),
            haplotypes: haplotypes.as_ptr(),
            haplotypes_length: haplotypes.len(),
            haplotype_stride: 1,
            maximum_heterozygote_mismatch: 0.75,
            window_seed: 15_052_011,
            window_domain: 2,
            window_iteration: 3,
            window_item: 0,
            fallback_seed: 15_052_011,
            fallback_domain: 8,
            fallback_iteration: 3,
            fallback_item: 0,
        };
        let mut job = ptr::null_mut();
        reset_conditioning_shared_validation_scans();
        let status = unsafe { shapeit_conditioning_graph_job_build_v1(&parameters, &mut job) };
        assert_eq!(status, STATUS_OK);
        assert_eq!(conditioning_shared_validation_scans(), 1);
        let original_job = job;

        selected_sites[0] = 2;
        assert_eq!(selected_sites[0], 2);
        let status = unsafe { shapeit_conditioning_graph_job_build_v1(&parameters, &mut job) };
        assert_eq!(status, STATUS_INVALID_DIMENSIONS);
        assert_eq!(job, original_job);
        assert_eq!(conditioning_shared_validation_scans(), 2);

        unsafe {
            shapeit_conditioning_job_free_v1(job);
            crate::genotype::shapeit_genotype_graph_free_v1(graph);
        }
    }
}
