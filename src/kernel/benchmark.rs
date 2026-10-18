use std::marker::PhantomData;

use cubecl::{Device, benchmark::Benchmark, future, prelude::*};

use crate::kernel::gputensor::GpuTensor;

pub struct ReductionBench<F: Float + CubeElement> {
    input_shape: Vec<usize>,
    client: Client,
    _f: PhantomData<F>,
}

const VECTOR_SIZE: usize = 4;

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
        let output_shape: Vec<usize> = self.input_shape[..2].to_vec();
        let output = GpuTensor::<F>::empty(output_shape, &self.client);

        unsafe {
            reduce_matrix::launch_unchecked::<F>(
                &self.client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_2d(self.input_shape[0] as u32, self.input_shape[1] as u32), // Add parallelization on the first
                VECTOR_SIZE,
                // dimension
                input.as_arg(),
                output.as_arg(),
            );
        }
        Ok(output)
    }
}

#[cube(launch_unchecked)]
fn reduce_matrix<F: Float, N: Size>(input: &Tensor<Vector<F, N>>, output: &mut Tensor<F>) {
    // non-paralization
    // for row in 0..input.shape(0) {
    //     let mut acc = F::new(0.0_f32);
    //     for col in 0..input.shape(1) {
    //         acc += input[row * input.stride(0) + col];
    //     }
    //     output[row] = acc;
    // }
    // parallelization
    // let row = UNIT_POS_X as usize;
    // let mut acc = F::new(0.0_f32);
    // for col in 0..input.shape(1) {
    //     acc += input[row * input.stride(0) + col];
    // }
    // output[row] = acc;
    // vectorization + parallelization
    // let row = UNIT_POS_X as usize;
    // let offset = row * input.stride(0) / N::value();
    // let mut acc = Vector::<F, N>::new(F::new(0.0_f32));
    // for i in 0..input.shape(1) / N::value() {
    //     acc += input[offset + i]
    // }
    // output[row] = acc.vector_sum()
    // parallel reduction 3d
    let x = UNIT_POS_X as usize;
    let y = UNIT_POS_Y as usize;
    let offset = (x * input.stride(0) + y * input.stride(1)) / N::value();
    let mut acc = Vector::<F, N>::new(F::new(0.0_f32));
    for i in 0..input.shape(2) / N::value() {
        acc += input[offset + i];
    }
    let row = x * output.stride(0) + y;
    output[row] = acc.vector_sum();
}

pub fn launch<F: Float + CubeElement>(device: &Device) {
    let client = device.client();

    let bench1 = ReductionBench::<F> {
        input_shape: vec![64, 256, 1024],
        client: client.clone(),
        _f: PhantomData,
    };

    let bench2 = ReductionBench::<F> {
        input_shape: vec![64, 64, 4096],
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

// No parallelization
// wgpu<wgsl>-reduction-[512, 8192]
//
// ―――――――― Result ―――――――――
//   Timing      system
//   Samples     15
//   Mean        477.290ms
//   Variance    123.000ns
//   Median      477.374ms
//   Min         476.408ms
//   Max         477.916ms
// ―――――――――――――――――――――――――
// wgpu<wgsl>-reduction-[128, 32768]
//
// ―――――――― Result ―――――――――
//   Timing      system
//   Samples     15
//   Mean        474.774ms
//   Variance    104.000ns
//   Median      474.837ms
//   Min         474.114ms
//   Max         475.147ms
// ―――――――――――――――――――――――――
//

// with parallelization
// wgpu<wgsl>-reduction-[512, 8192]
// ―――――――― Result ―――――――――
//   Timing      system
//   Samples     15
//   Mean        5.516ms
//   Variance    290.000ns
//   Median      5.464ms
//   Min         4.857ms
//   Max         6.331ms
// ―――――――――――――――――――――――――
// wgpu<wgsl>-reduction-[128, 32768]
//
// ―――――――― Result ―――――――――
//   Timing      system
//   Samples     15
//   Mean        6.390ms
//   Variance    239.000ns
//   Median      6.440ms
//   Min         5.643ms
//   Max         7.246ms
// ―――――――――――――――――――――――――
// parallelization + vectorization
// wgpu<wgsl>-reduction-[512, 8192]
//
// ―――――――― Result ―――――――――
//   Timing      system
//   Samples     15
//   Mean        4.150ms
//   Variance    10.000ns
//   Median      4.131ms
//   Min         4.063ms
//   Max         4.518ms
// ―――――――――――――――――――――――――
// wgpu<wgsl>-reduction-[128, 32768]
//
// ―――――――― Result ―――――――――
//   Timing      system
//   Samples     15
//   Mean        5.173ms
//   Variance    1.141µs
//   Median      5.611ms
//   Min         2.911ms
//   Max         6.599ms
// ―――――――――――――――――――――――――
// 3d parallel reduction
//wgpu<wgsl>-reduction-[64, 256, 1024]
//
// ―――――――― Result ―――――――――
//   Timing      system
//   Samples     15
//   Mean        87.366µs
//   Variance    0.000ns
//   Median      88.166µs
//   Min         67.333µs
//   Max         92.125µs
// ―――――――――――――――――――――――――
// wgpu<wgsl>-reduction-[64, 64, 4096]
//
// ―――――――― Result ―――――――――
//   Timing      system
//   Samples     15
//   Mean        75.039µs
//   Variance    0.000ns
//   Median      73.125µs
//   Min         72.250µs
//   Max         98.875µs
// ―――――――――――――――――――――――――
//
