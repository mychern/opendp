use crate::core::FfiResult;
use crate::error::*;
use crate::ffi::any::{AnyObject, Downcast};
use crate::ffi::util::{self, c_bool, to_bool};

#[cfg(feature = "contrib-continual")]
use super::continual::{BaselineContinualToeplitz, ContinualRelease, MonotonicContinualToeplitz};

#[cfg(feature = "contrib-continual")]
type ToeplitzAtom = i64;

#[cfg(feature = "contrib-continual")]
enum ToeplitzHandle {
    Baseline(BaselineContinualToeplitz<ToeplitzAtom>),
    Monotonic(MonotonicContinualToeplitz<ToeplitzAtom>),
}

#[cfg(feature = "contrib-continual")]
impl ToeplitzHandle {
    fn new(scale: f64, monotonic: bool) -> Fallible<Self> {
        if monotonic {
            MonotonicContinualToeplitz::new(scale).map(Self::Monotonic)
        } else {
            BaselineContinualToeplitz::new(scale).map(Self::Baseline)
        }
    }

    fn append_count_on_new_timestamp(&mut self, value: ToeplitzAtom) -> Fallible<usize> {
        match self {
            Self::Baseline(mech) => mech.append_count_on_new_timestamp(value),
            Self::Monotonic(mech) => mech.append_count_on_new_timestamp(value),
        }
    }

    fn fetch_privacy_preserving_sub_interval_sum(
        &mut self,
        start_time: usize,
        end_time: usize,
    ) -> Fallible<ToeplitzAtom> {
        match self {
            Self::Baseline(mech) => {
                mech.fetch_privacy_preserving_sub_interval_sum(start_time, end_time)
            }
            Self::Monotonic(mech) => {
                mech.fetch_privacy_preserving_sub_interval_sum(start_time, end_time)
            }
        }
    }
}

#[cfg(feature = "contrib-continual")]
fn with_handle<T, F>(mechanism: *mut AnyObject, f: F) -> Fallible<T>
where
    F: FnOnce(&mut ToeplitzHandle) -> Fallible<T>,
{
    let mechanism = try_as_mut_ref!(mechanism);
    let handle = try_!(mechanism.downcast_mut::<ToeplitzHandle>());
    f(handle)
}

#[cfg(feature = "contrib-continual")]
fn fallible_any(result: Fallible<AnyObject>) -> FfiResult<*mut AnyObject> {
    match result {
        Ok(obj) => FfiResult::Ok(util::into_raw(obj)),
        Err(err) => FfiResult::from(err),
    }
}

#[cfg(feature = "contrib-continual")]
#[unsafe(no_mangle)]
pub extern "C" fn opendp_measurements__toeplitz_continual_new_i64(
    scale: f64,
    enforce_monotonicity: c_bool,
) -> FfiResult<*mut AnyObject> {
    let result: Fallible<AnyObject> = ToeplitzHandle::new(scale, to_bool(enforce_monotonicity))
        .map(AnyObject::new);
    fallible_any(result)
}

#[cfg(feature = "contrib-continual")]
#[unsafe(no_mangle)]
pub extern "C" fn opendp_measurements__toeplitz_append_count_on_new_timestamp_i64(
    mechanism: *mut AnyObject,
    value: ToeplitzAtom,
) -> FfiResult<*mut AnyObject> {
    let result: Fallible<AnyObject> = with_handle(mechanism, |handle| {
        handle
            .append_count_on_new_timestamp(value)
            .map(AnyObject::new)
    });
    fallible_any(result)
}

#[cfg(feature = "contrib-continual")]
#[unsafe(no_mangle)]
pub extern "C" fn opendp_measurements__toeplitz_fetch_privacy_preserving_sub_interval_sum_i64(
    mechanism: *mut AnyObject,
    start_time: usize,
    end_time: usize,
) -> FfiResult<*mut AnyObject> {
    let result: Fallible<AnyObject> = with_handle(mechanism, |handle| {
        handle
            .fetch_privacy_preserving_sub_interval_sum(start_time, end_time)
            .map(AnyObject::new)
    });
    fallible_any(result)
}

#[cfg(not(feature = "contrib-continual"))]
#[allow(unused_variables)]
#[unsafe(no_mangle)]
pub extern "C" fn opendp_measurements__toeplitz_continual_new_i64(
    scale: f64,
    enforce_monotonicity: c_bool,
) -> FfiResult<*mut AnyObject> {
    fallible!(FFI, "Toeplitz continual API requires the `contrib-continual` feature").into()
}

#[cfg(not(feature = "contrib-continual"))]
#[allow(unused_variables)]
#[unsafe(no_mangle)]
pub extern "C" fn opendp_measurements__toeplitz_append_count_on_new_timestamp_i64(
    mechanism: *mut AnyObject,
    value: i64,
) -> FfiResult<*mut AnyObject> {
    fallible!(FFI, "Toeplitz continual API requires the `contrib-continual` feature").into()
}

#[cfg(not(feature = "contrib-continual"))]
#[allow(unused_variables)]
#[unsafe(no_mangle)]
pub extern "C" fn opendp_measurements__toeplitz_fetch_privacy_preserving_sub_interval_sum_i64(
    mechanism: *mut AnyObject,
    start_time: usize,
    end_time: usize,
) -> FfiResult<*mut AnyObject> {
    fallible!(FFI, "Toeplitz continual API requires the `contrib-continual` feature").into()
}
