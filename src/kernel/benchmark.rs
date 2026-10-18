use std::marker::PhantomData;

use cubecl::{Device, benchmark::Benchmark, future, prelude::*};

use crate::kernel::gputensor::GpuTensor;

pub struct ReductionBench<F: Float + CubeElement> {
    input_shape: Vec<usize>,
    client: Client,
    _f: PhantomData<F>,
}

impl<F: Float + CubeElement> Benchmark for ReductionBench<F> {
    type Input = GpuTensor<F>;
    type Output = GpuTensor<F>;

    fn prepare(&self) -> Self::Input {
        GpuTensor::<F>::arange(self.input_shape.clone(), &self.client)
    }

    fn name(&self) -> String {
        format!("{}-reduction-{:?}", self.client.name(), self.input_shape).to_lowercase()
    }

    fn sync(&self) {
        future::block_on(self.client.sync()).expect("Failed to synchronize device")
    }

    fn execute(&self, input: Self::Input) -> Result<Self::Output, String> {
        let output_shape: Vec<usize> = vec![self.input_shape[0]];
        let output = GpuTensor::<F>::empty(output_shape, &self.client);

        unsafe {
            reduce_matrix::launch_unchecked::<F>(
                &self.client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(1),
                input.as_arg(),
                output.as_arg(),
            );
        }
        Ok(output)
    }
}

#[cube(launch_unchecked)]
fn reduce_matrix<F: Float>(input: &Tensor<F>, output: &mut Tensor<F>) {
    for row in 0..input.shape(0) {
        let mut acc = F::new(0.0_f32);
        for col in 0..input.shape(1) {
            acc += input[row * input.stride(0) + col];
        }
        output[row] = acc;
    }
}

pub fn launch<F: Float + CubeElement>(device: &Device) {
    let client = device.client();

    let bench1 = ReductionBench::<F> {
        input_shape: vec![512, 8 * 1024],
        client: client.clone(),
        _f: PhantomData,
    };

    let bench2 = ReductionBench::<F> {
        input_shape: vec![128, 32 * 1024],
        client: client.clone(),
        _f: PhantomData,
    };

    for bench in [bench1, bench2] {
        println!("{}", bench.name());
        println!(
            "{}",
            bench
                .run(cubecl::profile::TimingMethod::System)
                .expect("Benchmark failed")
        );
    }
}
