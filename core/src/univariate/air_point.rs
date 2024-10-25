/// A struct that houses a pair of points.
///
/// This struct is useful for getting combined methods on a view of an AIR, consisting of values
/// in the `local` and `next` rows.
#[repr(C)]
pub struct AirPoint<T> {
    local: T,
    next: T,
}
