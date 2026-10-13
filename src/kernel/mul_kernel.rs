use cubecl::Device;
use cubecl::prelude::*;

#[cube(launch_unchecked)]
fn mul_kernel<F: Float>(a: &[F], b: &[F], out: &mut [F]) {
    // ABSOLUTE_POS = global thread index
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = a[ABSOLUTE_POS] * b[ABSOLUTE_POS]
    }
}

pub fn launch(device: &Device) {
    let client = device.client();
    const N: usize = 5_000_000;

    let a = (0..N).map(|i| i as f32).collect::<Vec<f32>>();
    let b = (0..N).rev().map(|i| (i) as f32).collect::<Vec<f32>>();

    // upload inputs allocate output on the GPU
    let a_handle = client.create_from_slice(f32::as_bytes(&a));
    let b_handle = client.create_from_slice(f32::as_bytes(&b));
    let out_handle = client.empty(N * core::mem::size_of::<f32>());

    // launch config
    let threads_per_cube: u32 = 256;
    // let num_cubes: u32 = ((N as u32) + threads_per_cube - 1) / threads_per_cube;
    let num_cubes = (N as u32).div_ceil(threads_per_cube);

    unsafe {
        mul_kernel::launch_unchecked::<f32>(
            &client,
            CubeCount::Static(num_cubes, 1, 1),
            CubeDim::new(&client, threads_per_cube as usize),
            BufferArg::from_raw_parts(a_handle, N),
            BufferArg::from_raw_parts(b_handle, N),
            BufferArg::from_raw_parts(out_handle.clone(), N),
        );
    };

    // Download results
    let bytes = client.read_one(out_handle).unwrap();
    let result = f32::from_bytes(&bytes);

    // Sanity Check
    let start_idx = result.len().saturating_sub(5);
    let last_five = &result[start_idx..];
    println!("starting index : {:?}", last_five);
    // println!("{:?}", result[start_idx..]);
}
