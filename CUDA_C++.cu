#include "helper_math.h"
#include "utils.h"

#define SQRT3 1.73205080757f

inline __device__ __host__ float clamp(float f, float a, float b)
{
    return fmaxf(a, fminf(f, b));
}

inline __host__ __device__ float calc_dt(float t, float exp_step_factor, int max_samples, int grid_size, float scale){
    return clamp(t*exp_step_factor, SQRT3/max_samples, SQRT3*2*scale/grid_size);
}

__global__ void raymarching_train_kernel(
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> rays_o,
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> rays_d,
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> hits_t,
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> depth_mask,
    torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> sigma_mask,
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> d_c_i,
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> cost_intervial,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> img_idxs,
    const int cascades,
    const int grid_size,
    const float scale,
    const float exp_step_factor,
    const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> noise,
    const int max_samples,
    int* __restrict__ counter,
    torch::PackedTensorAccessor64<int64_t, 2, torch::RestrictPtrTraits> rays_a,
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> xyzs,
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> dirs,
    torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> deltas,
    torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> ts
){
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rays_o.size(0)) return;

    float distance = 0;
    float depth = 0;
    float p = 0;
    float sigma = 0;
    float high = cost_intervial[d_c_i[r][2]][1], low = cost_intervial[d_c_i[r][2]][0]; 
    distance = high - low, depth = d_c_i[r][0], p=d_c_i[r][1];
    float con_interval = clamp(distance * (1.001 - p), distance/1780, distance);
    sigma = con_interval/6;
    sigma_mask[r = sigma;
    const uint32_t grid_size3 = grid_size*grid_size*grid_size;
    const float grid_size_inv = 1.0f/grid_size;


    float t1 = hits_t[r][0], t2 = hits_t[r][1];

    if (t1>=0) { // only perturb the starting t
        // const float dt = calc_dt(t1, exp_step_factor, max_samples, grid_size, scale);
        float dt = calc_dt(t1, exp_step_factor, max_samples, grid_size, scale);
        t1 += dt*noise[r];
    }

    // first pass: compute the number of samples on the ray
    float t = t1; int N_samples = 0;

    // if t1 < 0 (no hit) this loop will be skipped (N_samples will be 0)
    while (t1<=t && t<t2 && N_samples<max_samples){
        float dt = calc_dt(t, exp_step_factor, max_samples, grid_size, scale);
        if (t>depth -3*sigma && t <depth+3*sigma)
            {
                depth_mask[s][2] = 1;
            }
        t += dt;
        N_samples++;
    }


    // second pass: write to output
    const int start_idx = atomicAdd(counter, N_samples);
    const int ray_count = atomicAdd(counter+1, 1);

    rays_a[ray_count][0] = r;
    rays_a[ray_count][1] = start_idx; rays_a[ray_count][2] = N_samples;
    t = t1; int samples = 0;
    while (t<t2 && samples<N_samples){
        
        float dt = calc_dt(t, exp_step_factor, max_samples, grid_size, scale);
        const int s = start_idx + samples;
        depth_mask[s][0] = t;
        depth_mask[s][1] = dt;
        if (t>depth -3*sigma && t <depth+3*sigma)
        {
            depth_mask[s][2] = 1;
        }
        xyzs[s][0] = x; xyzs[s][1] = y; xyzs[s][2] = z;
        dirs[s][0] = dx; dirs[s][1] = dy; dirs[s][2] = dz;
        ts[s] = t; deltas[s] = dt;
        t += dt; samples++;
    }
}


std::vector<torch::Tensor> raymarching_train_cu(
    const torch::Tensor rays_o,
    const torch::Tensor rays_d,
    const torch::Tensor hits_t,
    const torch::Tensor d_c_i,
    const torch::Tensor cost_intervial,
    const torch::Tensor img_idxs,
    const int cascades,
    const float scale,
    const float exp_step_factor,
    const torch::Tensor noise,
    const int grid_size,
    const int max_samples
){
    const int N_rays = rays_o.size(0);

    // count the number of samples and the number of rays processed
    auto counter = torch::zeros({2}, torch::dtype(torch::kInt32).device(rays_o.device()));
    // ray attributes: ray_idx, start_idx, N_samples
    auto rays_a = torch::zeros({N_rays, 3},
                        torch::dtype(torch::kLong).device(rays_o.device()));
    auto sigma_mask = torch::zeros({N_rays}, rays_o.options());
    auto depth_mask = torch::zeros({N_rays*max_samples,3}, rays_o.options());
    auto xyzs = torch::zeros({N_rays*max_samples, 3}, rays_o.options());
    auto dirs = torch::zeros({N_rays*max_samples, 3}, rays_o.options());
    auto deltas = torch::zeros({N_rays*max_samples}, rays_o.options());
    auto ts = torch::zeros({N_rays*max_samples}, rays_o.options());

    const int threads = 256, blocks = (N_rays+threads-1)/threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(rays_o.type(), "raymarching_train_cu", 
    ([&] {
        raymarching_train_kernel<<<blocks, threads>>>(
            rays_o.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            rays_d.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            hits_t.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            depth_mask.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            sigma_mask.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
            d_c_i.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            cost_intervial.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            img_idxs.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>(),
            cascades,
            grid_size,
            scale,
            exp_step_factor,
            noise.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
            max_samples,
            counter.data_ptr<int>(),
            rays_a.packed_accessor64<int64_t, 2, torch::RestrictPtrTraits>(),
            xyzs.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            dirs.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            deltas.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
            ts.packed_accessor32<float, 1, torch::RestrictPtrTraits>()
        );
    }));

    return {rays_a, xyzs, dirs, deltas, ts, counter, depth_mask, sigma_mask};
}



template <typename scalar_t>
__global__ void s_2_loss_fw_kernel(
    const torch::PackedTensorAccessor64<int64_t, 2, torch::RestrictPtrTraits> rays_a,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> ws,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> depth_mask,
    torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> sigma_mask,
    torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> area_prior,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> depth_gt,
    torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> depth_loss
){
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= rays_a.size(0)) return;

    const int ray_idx = rays_a[n][0], start_idx = rays_a[n][1], N_samples = rays_a[n][2];
    if(ray_idx>=1024){
        int depth_idx = ray_idx - 1024;
        float loss_sum = 0;

        int samples = 0;
        float mean_value = depth_gt[depth_idx];
        float judge = 0;

        float sigma = sigma_mask[depth_idx];
        
        while (samples < N_samples) {
            const int s = start_idx + samples;
            float t = depth_mask[s][0];
            float dt = depth_mask[s][1];
            float start = t;
            float area_sum = 0;
            float A = 1/(sigma * sqrt(2*M_PI));
            float B1 = (-1.0/2) * (((start-mean_value)/sigma) * ((start-mean_value)/sigma));
            float C1 = exp(B1);
            area_sum =  A*C1 * dt;
            
            if(depth_mask[s][2]>0)
            {
                judge += ws[s] * t;
            }
            area_prior[s] = area_sum;
            loss_sum += -log(ws[s]+0.0001)*area_sum;


            samples++;    

            if(judge>mean_value-3*sigma && judge<mean_value+3*sigma)
            {
                depth_loss[ray_idx]=0;
            }
            else{
                depth_loss[ray_idx] = loss_sum;
            }
        }
    }
  

}


std::vector<torch::Tensor> s_2_loss_fw_cu(
    const torch::Tensor ws,
    const torch::Tensor rays_a,
    const torch::Tensor depth_mask,
    const torch::Tensor depth_gt,
    const torch::Tensor sigma_mask
){
    const int N_rays = rays_a.size(0);
    const int N = ws.size(0);
    auto area_prior = torch::zeros({N}, ws.options());
    auto depth_loss = torch::zeros({N_rays}, ws.options());
    const int threads = 256, blocks = (N_rays+threads-1)/threads;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(ws.type(), "s_2_loss_fw_cu", 
    ([&] {
        s_2_loss_fw_kernel<scalar_t><<<blocks, threads>>>( 
            rays_a.packed_accessor64<int64_t, 2, torch::RestrictPtrTraits>(),
            ws.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            depth_mask.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            sigma_mask.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            area_prior.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            depth_gt.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            depth_loss.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>()
        );
    }));

    return {area_prior, depth_loss};
}


template <typename scalar_t>
__global__ void s_2_loss_bw_kernel(
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> dL_dloss,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> ws,
    const torch::PackedTensorAccessor64<int64_t, 2, torch::RestrictPtrTraits> rays_a,
    torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> dL_dws,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> area_prior,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> depth_mask,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> depth_gt,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> sigma_mask
){
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= rays_a.size(0)) return;

    const int ray_idx = rays_a[n][0], start_idx = rays_a[n][1], N_samples = rays_a[n][2];
    const int end_idx = start_idx+N_samples-1;

    if (ray_idx>=1024){
        int depth_idx = ray_idx - 1024;
        for (int s=start_idx; s<=end_idx; s++){
            if(dL_dloss[ray_idx] == 0){continue;}

            float var = -dL_dloss[ray_idx] * (1/(ws[s]+0.0001)) * area_prior[s];
            dL_dws[s] = var;
            
        }

    }
    
}


torch::Tensor s_2_loss_bw_cu(
    const torch::Tensor dL_dloss,
    const torch::Tensor ws,
    const torch::Tensor rays_a,
    const torch::Tensor area_prior,
    const torch::Tensor depth_mask,
    const torch::Tensor depth_gt,
    const torch::Tensor sigma_mask
){
    const int N_rays = rays_a.size(0), N = ws.size(0);

    auto dL_dws = torch::zeros({N}, dL_dloss.options());

    const int threads = 256, blocks = (N_rays+threads-1)/threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(ws.type(), "s_2_loss_bw_cu", 
    ([&] {
        s_2_loss_bw_kernel<scalar_t><<<blocks, threads>>>(
            dL_dloss.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            ws.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            rays_a.packed_accessor64<int64_t, 2, torch::RestrictPtrTraits>(),
            dL_dws.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            area_prior.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            depth_mask.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            depth_gt.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            sigma_mask.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>()
        );
    }));

    return dL_dws;
}