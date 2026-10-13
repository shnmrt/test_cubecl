mod kernel;

use kernel::{add_kernel, mul_kernel, pow_kernel};

fn main() {
    // mul_kernel::launch(&cubecl::Device::default());
    // add_kernel::launch(&cubecl::Device::default());
    pow_kernel::launch(&cubecl::Device::default());
}
