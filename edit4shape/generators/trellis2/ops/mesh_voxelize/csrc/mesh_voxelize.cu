/**
 * mesh_voxelize.cu — GPU mesh voxelization with exact triangle–box test
 *
 * Uses the Schwarz-Seidel / Akenine-Möller triangle-box overlap test:
 *   1. Plane vs box (normal-plane overlap)
 *   2. XY / YZ / ZX projection edge tests
 *
 * Two-pass approach:
 *   Pass 1 (count):  AABB candidates per face → prefix-sum → total buffer size
 *   Pass 2 (write):  re-enumerate AABB, run exact test, atomicAdd to write
 *
 * Dedup is done in Python via torch.unique(dim=0).
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

/* ------------------------------------------------------------------ */
/*  Schwarz-Seidel triangle–box overlap test (device)                 */
/*  Box = [px, px+vs_x] x [py, py+vs_y] x [pz, pz+vs_z]            */
/*  Triangle vertices v0, v1, v2 in the same coordinate system.       */
/* ------------------------------------------------------------------ */
__device__ __forceinline__ bool tri_box_overlap(
    float v0x, float v0y, float v0z,
    float v1x, float v1y, float v1z,
    float v2x, float v2y, float v2z,
    float px,  float py,  float pz,
    float vsx, float vsy, float vsz)
{
    // edge vectors
    float e0x = v1x - v0x, e0y = v1y - v0y, e0z = v1z - v0z;
    float e1x = v2x - v1x, e1y = v2y - v1y, e1z = v2z - v1z;

    // face normal
    float nx = e0y * e1z - e0z * e1y;
    float ny = e0z * e1x - e0x * e1z;
    float nz = e0x * e1y - e0y * e1x;

    // --- Test 1: plane vs box ---
    float cx = (nx > 0.f) ? vsx : 0.f;
    float cy = (ny > 0.f) ? vsy : 0.f;
    float cz = (nz > 0.f) ? vsz : 0.f;
    float d1 = nx*(cx-v0x+px) + ny*(cy-v0y+py) + nz*(cz-v0z+pz);
    float d2 = nx*(vsx-cx-v0x+px) + ny*(vsy-cy-v0y+py) + nz*(vsz-cz-v0z+pz);
    if (d1 * d2 > 0.f) return false;

    // --- Test 2: XY projection ---
    float e2x = v0x - v2x, e2y = v0y - v2y;
    float mul_xy = (nz < 0.f) ? -1.f : 1.f;
    float nxy_e0x = -mul_xy * e0y, nxy_e0y = mul_xy * e0x;
    float nxy_e1x = -mul_xy * e1y, nxy_e1y = mul_xy * e1x;
    float nxy_e2x = -mul_xy * e2y, nxy_e2y = mul_xy * e2x;

    float dxy_e0 = -(nxy_e0x * v0x + nxy_e0y * v0y) + fmaxf(nxy_e0x, 0.f)*vsx + fmaxf(nxy_e0y, 0.f)*vsy;
    float dxy_e1 = -(nxy_e1x * v1x + nxy_e1y * v1y) + fmaxf(nxy_e1x, 0.f)*vsx + fmaxf(nxy_e1y, 0.f)*vsy;
    float dxy_e2 = -(nxy_e2x * v2x + nxy_e2y * v2y) + fmaxf(nxy_e2x, 0.f)*vsx + fmaxf(nxy_e2y, 0.f)*vsy;

    if (nxy_e0x * px + nxy_e0y * py + dxy_e0 < 0.f) return false;
    if (nxy_e1x * px + nxy_e1y * py + dxy_e1 < 0.f) return false;
    if (nxy_e2x * px + nxy_e2y * py + dxy_e2 < 0.f) return false;

    // --- Test 3: YZ projection ---
    float e1z_ = e1z, e0z_ = e0z, e2z_ = v0z - v2z;
    float mul_yz = (nx < 0.f) ? -1.f : 1.f;
    float nyz_e0x = -mul_yz * e0z_, nyz_e0y = mul_yz * e0y;
    float nyz_e1x = -mul_yz * e1z_, nyz_e1y = mul_yz * e1y;
    float nyz_e2x = -mul_yz * e2z_, nyz_e2y = mul_yz * e2y;

    float dyz_e0 = -(nyz_e0x * v0y + nyz_e0y * v0z) + fmaxf(nyz_e0x, 0.f)*vsy + fmaxf(nyz_e0y, 0.f)*vsz;
    float dyz_e1 = -(nyz_e1x * v1y + nyz_e1y * v1z) + fmaxf(nyz_e1x, 0.f)*vsy + fmaxf(nyz_e1y, 0.f)*vsz;
    float dyz_e2 = -(nyz_e2x * v2y + nyz_e2y * v2z) + fmaxf(nyz_e2x, 0.f)*vsy + fmaxf(nyz_e2y, 0.f)*vsz;

    if (nyz_e0x * py + nyz_e0y * pz + dyz_e0 < 0.f) return false;
    if (nyz_e1x * py + nyz_e1y * pz + dyz_e1 < 0.f) return false;
    if (nyz_e2x * py + nyz_e2y * pz + dyz_e2 < 0.f) return false;

    // --- Test 4: ZX projection ---
    float mul_zx = (ny < 0.f) ? -1.f : 1.f;
    float nzx_e0x = -mul_zx * e0x, nzx_e0y = mul_zx * e0z_;
    float nzx_e1x = -mul_zx * e1x, nzx_e1y = mul_zx * e1z_;
    float nzx_e2x = -mul_zx * e2x, nzx_e2y = mul_zx * e2z_;

    float dzx_e0 = -(nzx_e0x * v0z + nzx_e0y * v0x) + fmaxf(nzx_e0x, 0.f)*vsz + fmaxf(nzx_e0y, 0.f)*vsx;
    float dzx_e1 = -(nzx_e1x * v1z + nzx_e1y * v1x) + fmaxf(nzx_e1x, 0.f)*vsz + fmaxf(nzx_e1y, 0.f)*vsx;
    float dzx_e2 = -(nzx_e2x * v2z + nzx_e2y * v2x) + fmaxf(nzx_e2x, 0.f)*vsz + fmaxf(nzx_e2y, 0.f)*vsx;

    if (nzx_e0x * pz + nzx_e0y * px + dzx_e0 < 0.f) return false;
    if (nzx_e1x * pz + nzx_e1y * px + dzx_e1 < 0.f) return false;
    if (nzx_e2x * pz + nzx_e2y * px + dzx_e2 < 0.f) return false;

    return true;
}

/* ------------------------------------------------------------------ */
/*  Pass 1: count exact voxels per face                               */
/* ------------------------------------------------------------------ */
__global__ void count_face_voxels_kernel(
    const float* __restrict__ vertices,
    const int*   __restrict__ faces,
    int*         __restrict__ counts,
    int num_faces,
    int grid_size,
    float mn_x, float mn_y, float mn_z,
    float vsx, float vsy, float vsz)
{
    int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= num_faces) return;

    int i0 = faces[f * 3 + 0];
    int i1 = faces[f * 3 + 1];
    int i2 = faces[f * 3 + 2];

    // triangle verts in "shifted" space (origin = aabb_min)
    float v0x = vertices[i0*3+0] - mn_x, v0y = vertices[i0*3+1] - mn_y, v0z = vertices[i0*3+2] - mn_z;
    float v1x = vertices[i1*3+0] - mn_x, v1y = vertices[i1*3+1] - mn_y, v1z = vertices[i1*3+2] - mn_z;
    float v2x = vertices[i2*3+0] - mn_x, v2y = vertices[i2*3+1] - mn_y, v2z = vertices[i2*3+2] - mn_z;

    // AABB of triangle in voxel indices
    float inv_vx = 1.f / vsx, inv_vy = 1.f / vsy, inv_vz = 1.f / vsz;
    int lo_x = max(0, (int)floorf(fminf(fminf(v0x, v1x), v2x) * inv_vx));
    int lo_y = max(0, (int)floorf(fminf(fminf(v0y, v1y), v2y) * inv_vy));
    int lo_z = max(0, (int)floorf(fminf(fminf(v0z, v1z), v2z) * inv_vz));
    int hi_x = min(grid_size - 1, (int)floorf(fmaxf(fmaxf(v0x, v1x), v2x) * inv_vx));
    int hi_y = min(grid_size - 1, (int)floorf(fmaxf(fmaxf(v0y, v1y), v2y) * inv_vy));
    int hi_z = min(grid_size - 1, (int)floorf(fmaxf(fmaxf(v0z, v1z), v2z) * inv_vz));

    int cnt = 0;
    for (int x = lo_x; x <= hi_x; x++) {
        float px = x * vsx;
        for (int y = lo_y; y <= hi_y; y++) {
            float py = y * vsy;
            for (int z = lo_z; z <= hi_z; z++) {
                float pz = z * vsz;
                if (tri_box_overlap(v0x,v0y,v0z, v1x,v1y,v1z, v2x,v2y,v2z,
                                    px,py,pz, vsx,vsy,vsz))
                    cnt++;
            }
        }
    }
    counts[f] = cnt;
}

/* ------------------------------------------------------------------ */
/*  Pass 2: write exact voxels                                        */
/* ------------------------------------------------------------------ */
__global__ void write_face_voxels_kernel(
    const float*   __restrict__ vertices,
    const int*     __restrict__ faces,
    const int64_t* __restrict__ offsets,
    int*           __restrict__ output,   // (total, 3)
    int num_faces,
    int grid_size,
    float mn_x, float mn_y, float mn_z,
    float vsx, float vsy, float vsz)
{
    int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= num_faces) return;

    int i0 = faces[f * 3 + 0];
    int i1 = faces[f * 3 + 1];
    int i2 = faces[f * 3 + 2];

    float v0x = vertices[i0*3+0] - mn_x, v0y = vertices[i0*3+1] - mn_y, v0z = vertices[i0*3+2] - mn_z;
    float v1x = vertices[i1*3+0] - mn_x, v1y = vertices[i1*3+1] - mn_y, v1z = vertices[i1*3+2] - mn_z;
    float v2x = vertices[i2*3+0] - mn_x, v2y = vertices[i2*3+1] - mn_y, v2z = vertices[i2*3+2] - mn_z;

    float inv_vx = 1.f / vsx, inv_vy = 1.f / vsy, inv_vz = 1.f / vsz;
    int lo_x = max(0, (int)floorf(fminf(fminf(v0x, v1x), v2x) * inv_vx));
    int lo_y = max(0, (int)floorf(fminf(fminf(v0y, v1y), v2y) * inv_vy));
    int lo_z = max(0, (int)floorf(fminf(fminf(v0z, v1z), v2z) * inv_vz));
    int hi_x = min(grid_size - 1, (int)floorf(fmaxf(fmaxf(v0x, v1x), v2x) * inv_vx));
    int hi_y = min(grid_size - 1, (int)floorf(fmaxf(fmaxf(v0y, v1y), v2y) * inv_vy));
    int hi_z = min(grid_size - 1, (int)floorf(fmaxf(fmaxf(v0z, v1z), v2z) * inv_vz));

    int64_t base = offsets[f];
    int idx = 0;
    for (int x = lo_x; x <= hi_x; x++) {
        float px = x * vsx;
        for (int y = lo_y; y <= hi_y; y++) {
            float py = y * vsy;
            for (int z = lo_z; z <= hi_z; z++) {
                float pz = z * vsz;
                if (tri_box_overlap(v0x,v0y,v0z, v1x,v1y,v1z, v2x,v2y,v2z,
                                    px,py,pz, vsx,vsy,vsz)) {
                    output[(base + idx) * 3 + 0] = x;
                    output[(base + idx) * 3 + 1] = y;
                    output[(base + idx) * 3 + 2] = z;
                    idx++;
                }
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/*  C++ / Python wrappers                                             */
/* ------------------------------------------------------------------ */

torch::Tensor count_face_voxels(
    torch::Tensor vertices,   // (V, 3) float32 CUDA
    torch::Tensor faces,      // (F, 3) int32   CUDA
    int grid_size,
    torch::Tensor aabb)       // (2, 3) float32 CUDA
{
    TORCH_CHECK(vertices.is_cuda(),  "vertices must be on CUDA");
    TORCH_CHECK(faces.is_cuda(),     "faces must be on CUDA");
    TORCH_CHECK(vertices.dtype() == torch::kFloat32);
    TORCH_CHECK(faces.dtype()    == torch::kInt32);

    const int F = faces.size(0);
    auto counts = torch::empty({F}, vertices.options().dtype(torch::kInt32));
    if (F == 0) return counts;

    float mn_x = aabb[0][0].item<float>();
    float mn_y = aabb[0][1].item<float>();
    float mn_z = aabb[0][2].item<float>();
    float mx_x = aabb[1][0].item<float>();
    float mx_y = aabb[1][1].item<float>();
    float mx_z = aabb[1][2].item<float>();

    float vsx = (mx_x - mn_x) / grid_size;
    float vsy = (mx_y - mn_y) / grid_size;
    float vsz = (mx_z - mn_z) / grid_size;

    const int threads = 256;
    const int blocks  = (F + threads - 1) / threads;

    count_face_voxels_kernel<<<blocks, threads>>>(
        vertices.data_ptr<float>(),
        faces.data_ptr<int>(),
        counts.data_ptr<int>(),
        F, grid_size,
        mn_x, mn_y, mn_z,
        vsx, vsy, vsz);

    return counts;
}

torch::Tensor write_face_voxels(
    torch::Tensor vertices,
    torch::Tensor faces,
    torch::Tensor offsets,
    int64_t total_candidates,
    int grid_size,
    torch::Tensor aabb)
{
    TORCH_CHECK(offsets.dtype() == torch::kInt64);

    const int F = faces.size(0);
    auto output = torch::empty({total_candidates, 3},
                               vertices.options().dtype(torch::kInt32));
    if (F == 0 || total_candidates == 0) return output;

    float mn_x = aabb[0][0].item<float>();
    float mn_y = aabb[0][1].item<float>();
    float mn_z = aabb[0][2].item<float>();
    float mx_x = aabb[1][0].item<float>();
    float mx_y = aabb[1][1].item<float>();
    float mx_z = aabb[1][2].item<float>();

    float vsx = (mx_x - mn_x) / grid_size;
    float vsy = (mx_y - mn_y) / grid_size;
    float vsz = (mx_z - mn_z) / grid_size;

    const int threads = 256;
    const int blocks  = (F + threads - 1) / threads;

    write_face_voxels_kernel<<<blocks, threads>>>(
        vertices.data_ptr<float>(),
        faces.data_ptr<int>(),
        offsets.data_ptr<int64_t>(),
        output.data_ptr<int>(),
        F, grid_size,
        mn_x, mn_y, mn_z,
        vsx, vsy, vsz);

    return output;
}

// ---------- pybind ----------

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("count_face_voxels", &count_face_voxels,
          "Count exact (triangle-box) voxels per face");
    m.def("write_face_voxels", &write_face_voxels,
          "Write exact voxels to flat buffer");
}
