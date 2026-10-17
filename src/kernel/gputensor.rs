use std::marker::PhantomData;

use cubecl::{Device, prelude::*, server::Handle, std::tensor::compact_strides};

/// Simple GpuTensor
#[derive(Debug, Clone)]
pub struct GpuTensor<F: Float + CubeElement> {
    data: Handle,
    shape: Vec<usize>,
    _f: PhantomData<F>,
}

impl<F: Float + CubeElement> GpuTensor<F> {
    /// Create a GpuTensor with a shape filled by number in order
    pub fn arange(shape: Vec<usize>, client: &Client) -> Self {
        let size = shape.iter().product();
        let data: Vec<F> = (0..size).map(|i| F::from_int(i as i64)).collect();
        let data = client.create_from_slice(F::as_bytes(&data));

        Self {
            data,
            shape,
            _f: PhantomData,
        }
    }

    /// Create an empty GpuTensor with a shape
    pub fn empty(shape: Vec<usize>, client: &Client) -> Self {
        let size = shape.iter().product::<usize>() * core::mem::size_of::<F>();
        let data = client.empty(size);
        Self {
            data,
            shape,
            _f: PhantomData,
        }
    }

    /// Pass the tensor's storage, and strides to a kernel
    pub fn as_arg(&self) -> TensorArg {
        unsafe {
            TensorArg::from_raw_parts(
                self.data.clone(),
                compact_strides(&self.shape),
                self.shape.clone().into(),
            )
        }
    }
    /// Return the data from the client
    pub fn read(self, client: &Client) -> Vec<F> {
        let bytes = client
            .read_one(self.data)
            .expect("Failed to read the tensor");
        F::from_bytes(&bytes).to_vec()
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

    let input = GpuTensor::<F>::arange(vec![3, 3], &client);
    let output = GpuTensor::<F>::empty(vec![3], &client);

    unsafe {
        reduce_matrix::launch_unchecked::<F>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            input.as_arg(),
            output.as_arg(),
        );
    }

    println!(
        "Execution reduction with runtime {:?} => {:?}",
        client.name(),
        output.read(&client)
    );
}
