mod kernel;

// use crate::kernel::cputensor;
// use kernel::gputensor;
// use kernel::{add_kernel, gelu, mul_kernel, pow_kernel};
use kernel::benchmark;
fn main() {
    // mul_kernel::launch(&cubecl::Device::default());
    // add_kernel::launch(&cubecl::Device::default());
    // pow_kernel::launch(&cubecl::Device::default());
    // gelu::launch(&cubecl::Device::default());
    // gputensor::launch::<f32>(&cubecl::Device::default());
    benchmark::launch::<f32>(&cubecl::Device::default());
}
