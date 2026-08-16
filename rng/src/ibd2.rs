use core::slice;

const ABI_VERSION: u32 = 1;
const STATUS_OK: u32 = 0;
const STATUS_NULL_POINTER: u32 = 1;
const STATUS_INVALID_DIMENSIONS: u32 = 2;
const STATUS_OUT_OF_BOUNDS: u32 = 3;

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Ibd2TrackV1 {
    pub(crate) individual: i32,
    pub(crate) from: i32,
    pub(crate) to: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Ibd2StatsV1 {
    pub(crate) individuals: usize,
    pub(crate) tracks: usize,
    pub(crate) merged: usize,
}

pub struct Ibd2TracksV1 {
    centimorgans: Vec<f32>,
    tracks: Vec<Vec<Ibd2TrackV1>>,
    collapsed: bool,
}

impl Ibd2TracksV1 {
    pub(crate) fn new(individual_count: usize, centimorgans: &[f32]) -> Result<Self, u32> {
        if individual_count == 0
            || individual_count > i32::MAX as usize
            || centimorgans.is_empty()
            || centimorgans.len() > i32::MAX as usize
            || centimorgans.iter().any(|value| !value.is_finite())
        {
            return Err(STATUS_INVALID_DIMENSIONS);
        }
        Ok(Self {
            centimorgans: centimorgans.to_vec(),
            tracks: vec![Vec::new(); individual_count],
            collapsed: true,
        })
    }

    pub(crate) fn individual_count(&self) -> usize {
        self.tracks.len()
    }

    pub(crate) fn is_collapsed(&self) -> bool {
        self.collapsed
    }

    #[inline]
    pub(crate) fn allows(&self, haplotype0: usize, haplotype1: usize, locus: usize) -> bool {
        let individual0 = haplotype0 / 2;
        let individual1 = haplotype1 / 2;
        let source = core::cmp::min(individual0, individual1);
        let target = core::cmp::max(individual0, individual1);
        if source == target {
            return false;
        }
        for track in &self.tracks[source] {
            let tracked_individual = track.individual as usize;
            if tracked_individual > target {
                break;
            }
            if tracked_individual == target
                && track.from as usize <= locus
                && locus <= track.to as usize
            {
                return false;
            }
        }
        true
    }

    fn expand(&self, mut track: Ibd2TrackV1) -> Ibd2TrackV1 {
        let left_centimorgans = self.centimorgans[track.from as usize];
        while track.from > 1 && left_centimorgans - self.centimorgans[track.from as usize] < 4.0 {
            track.from -= 1;
        }

        let right_centimorgans = self.centimorgans[track.to as usize];
        while track.to as usize + 1 < self.centimorgans.len()
            && self.centimorgans[track.to as usize] - right_centimorgans < 4.0
        {
            track.to += 1;
        }
        track
    }

    pub(crate) fn push(
        &mut self,
        source_individual: usize,
        input: &[Ibd2TrackV1],
    ) -> Result<(), u32> {
        if source_individual >= self.tracks.len() {
            return Err(STATUS_OUT_OF_BOUNDS);
        }
        let mut expanded = Vec::with_capacity(input.len());
        for &track in input {
            if track.individual < 0
                || track.individual as usize >= self.tracks.len()
                || track.from < 0
                || track.from > track.to
                || track.to as usize >= self.centimorgans.len()
            {
                return Err(STATUS_OUT_OF_BOUNDS);
            }
            let track = self.expand(track);
            expanded.push((
                core::cmp::min(source_individual, track.individual as usize),
                Ibd2TrackV1 {
                    individual: core::cmp::max(source_individual, track.individual as usize) as i32,
                    from: track.from,
                    to: track.to,
                },
            ));
        }
        for (source, track) in expanded {
            self.tracks[source].push(track);
        }
        if !input.is_empty() {
            self.collapsed = false;
        }
        Ok(())
    }

    pub(crate) fn collapse(&mut self) -> Ibd2StatsV1 {
        let mut stats = Ibd2StatsV1::default();
        for tracks in &mut self.tracks {
            tracks.sort_unstable_by_key(|track| (track.individual, track.from));
            let mut collapsed: Vec<Ibd2TrackV1> = Vec::with_capacity(tracks.len());
            for track in tracks.drain(..) {
                if let Some(previous) = collapsed.last_mut() {
                    if previous.individual == track.individual
                        && track.to >= previous.from
                        && track.from <= previous.to
                    {
                        let inclusive0 = previous.from <= track.from && previous.to >= track.to;
                        let inclusive1 = track.from <= previous.from && track.to >= previous.to;
                        previous.from = core::cmp::min(previous.from, track.from);
                        previous.to = core::cmp::max(previous.to, track.to);
                        stats.merged += usize::from(!inclusive0 && !inclusive1);
                        continue;
                    }
                }
                collapsed.push(track);
            }
            *tracks = collapsed;
            stats.tracks += tracks.len();
            stats.individuals += usize::from(!tracks.is_empty());
        }
        self.collapsed = true;
        stats
    }
}

#[no_mangle]
pub extern "C" fn shapeit_ibd2_abi_version() -> u32 {
    ABI_VERSION
}

#[no_mangle]
/// Allocate a Rust-owned IBD2 registry and copy its genetic-map coordinates.
///
/// # Safety
///
/// `centimorgans` and `registry` must be valid for their stated accesses. On
/// success the caller owns the registry and must eventually free it.
pub unsafe extern "C" fn shapeit_ibd2_tracks_create_v1(
    individual_count: usize,
    centimorgans: *const f32,
    centimorgans_length: usize,
    registry: *mut *mut Ibd2TracksV1,
) -> u32 {
    if centimorgans.is_null() || registry.is_null() {
        return STATUS_NULL_POINTER;
    }
    let centimorgans = slice::from_raw_parts(centimorgans, centimorgans_length);
    let value = match Ibd2TracksV1::new(individual_count, centimorgans) {
        Ok(value) => Box::new(value),
        Err(status) => return status,
    };
    *registry = Box::into_raw(value);
    STATUS_OK
}

#[no_mangle]
/// Free a Rust-owned IBD2 registry. Null is accepted.
///
/// # Safety
///
/// `registry` must be null or a live registry returned by the create function,
/// and it must be freed at most once.
pub unsafe extern "C" fn shapeit_ibd2_tracks_free_v1(registry: *mut Ibd2TracksV1) {
    if !registry.is_null() {
        drop(Box::from_raw(registry));
    }
}

#[no_mangle]
/// Expand and append newly detected tracks for one source individual.
///
/// # Safety
///
/// `registry` must be live. A non-empty `tracks` buffer must be valid for its
/// stated length.
pub unsafe extern "C" fn shapeit_ibd2_tracks_push_v1(
    registry: *mut Ibd2TracksV1,
    source_individual: usize,
    tracks: *const Ibd2TrackV1,
    tracks_length: usize,
) -> u32 {
    if registry.is_null() || (tracks_length != 0 && tracks.is_null()) {
        return STATUS_NULL_POINTER;
    }
    let tracks = if tracks_length == 0 {
        &[]
    } else {
        slice::from_raw_parts(tracks, tracks_length)
    };
    match (*registry).push(source_individual, tracks) {
        Ok(()) => STATUS_OK,
        Err(status) => status,
    }
}

#[no_mangle]
/// Sort and collapse all accumulated IBD2 tracks and return reporting counts.
///
/// # Safety
///
/// `registry` must be live and `stats` writable.
pub unsafe extern "C" fn shapeit_ibd2_tracks_collapse_v1(
    registry: *mut Ibd2TracksV1,
    stats: *mut Ibd2StatsV1,
) -> u32 {
    if registry.is_null() || stats.is_null() {
        return STATUS_NULL_POINTER;
    }
    *stats = (*registry).collapse();
    STATUS_OK
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_expands_collapses_and_excludes_exact_loci() {
        let centimorgans: Vec<f32> = (0..10).map(|value| value as f32).collect();
        let mut registry = Ibd2TracksV1 {
            centimorgans,
            tracks: vec![Vec::new(); 3],
            collapsed: true,
        };
        registry
            .push(
                0,
                &[
                    Ibd2TrackV1 {
                        individual: 1,
                        from: 5,
                        to: 5,
                    },
                    Ibd2TrackV1 {
                        individual: 1,
                        from: 6,
                        to: 6,
                    },
                ],
            )
            .unwrap();
        assert!(!registry.is_collapsed());
        let stats = registry.collapse();
        assert_eq!(stats.individuals, 1);
        assert_eq!(stats.tracks, 1);
        assert_eq!(registry.tracks[0][0].from, 1);
        assert_eq!(registry.tracks[0][0].to, 9);
        assert!(!registry.allows(0, 2, 1));
        assert!(!registry.allows(0, 2, 9));
        assert!(registry.allows(0, 4, 5));
        assert!(!registry.allows(0, 1, 5));
    }
}
