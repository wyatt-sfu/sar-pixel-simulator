"""
This module contains the FDTD field update equations implemented on the GPU.

See the CPU version for more comments refering to the reference material
these update equations are sourced from.
"""
from string import Template

kernel_templates = Template('''
__global__ void ex_update($FPTYPE * __restrict__ ex,
                          const $FPTYPE * __restrict__ hy,
                          const $FPTYPE * __restrict__ hz,
                          const $FPTYPE * __restrict__ ca,
                          const $FPTYPE * __restrict__ cb)
{
    for (int z = blockIdx.z * blockDim.z + threadIdx.z;
            z < $NZ - 1;
            z += blockDim.z * gridDim.z) {
        if (z < 1) {
            continue;
        }
        for (int y = blockIdx.y * blockDim.y + threadIdx.y;
                y < $NY - 1;
                y += blockDim.y * gridDim.y) {
            if (y < 1) {
                continue;
            }
            for (int x = blockIdx.x * blockDim.x + threadIdx.x;
                    x < $NX - 1;
                    x += blockDim.x * gridDim.x) {
                const size_t off = (z * $NY * $NX) + (y * $NX) + x;
                const size_t offDown = ((z - 1) * $NY * $NX) + (y * $NX) + x;
                const size_t offFront = (z * $NX * $NY) + ((y - 1) * $NX) + x;

                const $FPTYPE scaled = ex[off] * ca[off];
                const $FPTYPE delta_hz = hz[off] - hz[offFront];
                const $FPTYPE delta_hy = hy[off] - hy[offDown];
                ex[off] = scaled + (cb[off] * (delta_hz - delta_hy));
            }
        }
    }
}

__global__ void ey_update($FPTYPE * __restrict__ ey,
                          const $FPTYPE * __restrict__ hx,
                          const $FPTYPE * __restrict__ hz,
                          const $FPTYPE * __restrict__ ca,
                          const $FPTYPE * __restrict__ cb)
{
    for (int z = blockIdx.z * blockDim.z + threadIdx.z;
            z < $NZ - 1;
            z += blockDim.z * gridDim.z) {
        if (z < 1) {
            continue;
        }
        for (int y = blockIdx.y * blockDim.y + threadIdx.y;
                y < $NY - 1;
                y += blockDim.y * gridDim.y) {
            for (int x = blockIdx.x * blockDim.x + threadIdx.x;
                    x < $NX - 1;
                    x += blockDim.x * gridDim.x) {
                if (x < 1) {
                    continue;
                }
                const size_t off = (z * $NY * $NX) + (y * $NX) + x;
                const size_t offDown = ((z - 1) * $NY * $NX) + (y * $NX) + x;
                const size_t offLeft = (z * $NX * $NY) + (y * $NX) + x - 1;

                const $FPTYPE scaled = ey[off] * ca[off];
                const $FPTYPE delta_hx = hx[off] - hx[offDown];
                const $FPTYPE delta_hz = hz[off] - hz[offLeft];
                ey[off] = scaled + (cb[off] * (delta_hx - delta_hz));
            }
        }
    }
}

__global__ void ez_update($FPTYPE * __restrict__ ez,
                          const $FPTYPE * __restrict__ hx,
                          const $FPTYPE * __restrict__ hy,
                          const $FPTYPE * __restrict__ ca,
                          const $FPTYPE * __restrict__ cb)
{
    for (int z = blockIdx.z * blockDim.z + threadIdx.z;
            z < $NZ - 1;
            z += blockDim.z * gridDim.z) {
        for (int y = blockIdx.y * blockDim.y + threadIdx.y;
                y < $NY - 1;
                y += blockDim.y * gridDim.y) {
            if (y < 1) {
                continue;
            }
            for (int x = blockIdx.x * blockDim.x + threadIdx.x;
                    x < $NX - 1;
                    x += blockDim.x * gridDim.x) {
                if (x < 1) {
                    continue;
                }
                const size_t off = (z * $NY * $NX) + (y * $NX) + x;
                const size_t offFront = (z * $NY * $NX) + ((y - 1) * $NX) + x;
                const size_t offLeft = (z * $NX * $NY) + (y * $NX) + x - 1;

                const $FPTYPE scaled = ez[off] * ca[off];
                const $FPTYPE delta_hx = hx[off] - hx[offFront];
                const $FPTYPE delta_hy = hy[off] - hy[offLeft];
                ez[off] = scaled + (cb[off] * (delta_hy - delta_hx));
            }
        }
    }
}

__global__ void cpml_e_down($FPTYPE * __restrict__ ex,
                            $FPTYPE * __restrict__ ey,
                            const $FPTYPE * __restrict__ hx,
                            const $FPTYPE * __restrict__ hy,
                            $FPTYPE * __restrict__ psiExz_d,
                            $FPTYPE * __restrict__ psiEyz_d,
                            const $FPTYPE * __restrict__ be_d,
                            const $FPTYPE * __restrict__ ce_d,
                            const $FPTYPE * __restrict__ cb,
                            const $FPTYPE dx)
{
    for (int z = blockIdx.z * blockDim.z + threadIdx.z;
            z < $NPML;
            z += blockDim.z * gridDim.z) {
        if (z < 1) {
            continue;
        }
        for (int y = blockIdx.y * blockDim.y + threadIdx.y;
                y < $NY - 1;
                y += blockDim.y * gridDim.y) {
            for (int x = blockIdx.x * blockDim.x + threadIdx.x;
                    x < $NX - 1;
                    x += blockDim.x * gridDim.x) {
                const size_t off = (z * $NY * $NX) + (y * $NX) + x;
                const size_t offDown = ((z - 1) * $NX * $NY) + (y * $NX) + x;

                psiEyz_d[off] = (be_d[off] * psiEyz_d[off])
                    + (ce_d[off] * (hx[off] - hx[offDown]) / dx);
                psiExz_d[off] = (be_d[off] * psiExz_d[off])
                    + (ce_d[off] * (hy[off] - hy[offDown]) / dx);
                ey[off] += cb[off] * dx * psiEyz_d[off];
                ex[off] -= cb[off] * dx * psiExz_d[off];
            }
        }
    }
}

__global__ void cpml_e_up($FPTYPE * __restrict__ ex,
                          $FPTYPE * __restrict__ ey,
                          const $FPTYPE * __restrict__ hx,
                          const $FPTYPE * __restrict__ hy,
                          $FPTYPE * __restrict__ psiExz_u,
                          $FPTYPE * __restrict__ psiEyz_u,
                          const $FPTYPE * __restrict__ be_u,
                          const $FPTYPE * __restrict__ ce_u,
                          const $FPTYPE * __restrict__ cb,
                          const $FPTYPE dx)
{
    for (int z = blockIdx.z * blockDim.z + threadIdx.z;
            z < $NPML - 1;
            z += blockDim.z * gridDim.z) {
        for (int y = blockIdx.y * blockDim.y + threadIdx.y;
                y < $NY - 1;
                y += blockDim.y * gridDim.y) {
            for (int x = blockIdx.x * blockDim.x + threadIdx.x;
                    x < $NX - 1;
                    x += blockDim.x * gridDim.x) {
                const size_t zf = $NZ - $NPML + z;
                const size_t off = (z * $NY * $NX) + (y * $NX) + x;
                const size_t offFull = (zf * $NY * $NX) + (y * $NX) + x;
                const size_t offFullDown = ((zf - 1) * $NX * $NY) + (y * $NX)
                    + x;

                psiEyz_u[off] = (be_u[off] * psiEyz_u[off])
                    + (ce_u[off] * (hx[offFull] - hx[offFullDown]) / dx);
                psiExz_u[off] = (be_u[off] * psiExz_u[off])
                    + (ce_u[off] * (hy[offFull] - hy[offFullDown]) / dx);
                ey[offFull] += cb[offFull] * dx * psiEyz_u[off];
                ex[offFull] -= cb[offFull] * dx * psiExz_u[off];
            }
        }
    }
}

__global__ void cpml_e_left($FPTYPE * __restrict__ ey,
                            $FPTYPE * __restrict__ ez,
                            const $FPTYPE * __restrict__ hy,
                            const $FPTYPE * __restrict__ hz,
                            $FPTYPE * __restrict__ psiEyx_l,
                            $FPTYPE * __restrict__ psiEzx_l,
                            const $FPTYPE * __restrict__ be_l,
                            const $FPTYPE * __restrict__ ce_l,
                            const $FPTYPE * __restrict__ cb,
                            const $FPTYPE dx)
{
    for (int z = blockIdx.z * blockDim.z + threadIdx.z;
            z < $NZ - 1;
            z += blockDim.z * gridDim.z) {
        for (int y = blockIdx.y * blockDim.y + threadIdx.y;
                y < $NY - 1;
                y += blockDim.y * gridDim.y) {
            for (int x = blockIdx.x * blockDim.x + threadIdx.x;
                    x < $NPML;
                    x += blockDim.x * gridDim.x) {
                if (x < 1) {
                    continue;
                }
                const size_t offCpml = (z * $NY * $NPML) + (y * $NPML) + x;
                const size_t off = (z * $NY * $NX) + (y * $NX) + x;
                const size_t offLeft = (z * $NX * $NY) + (y * $NX) + x - 1;

                psiEzx_l[offCpml] = (be_l[offCpml] * psiEzx_l[offCpml])
                    + (ce_l[offCpml] * (hy[off] - hy[offLeft]) / dx);
                psiEyx_l[offCpml] = (be_l[offCpml] * psiEyx_l[offCpml])
                    + (ce_l[offCpml] * (hz[off] - hz[offLeft]) / dx);
                ez[off] += cb[off] * dx * psiEzx_l[offCpml];
                ey[off] -= cb[off] * dx * psiEyx_l[offCpml];
            }
        }
    }
}

__global__ void cpml_e_right($FPTYPE * __restrict__ ey,
                             $FPTYPE * __restrict__ ez,
                             const $FPTYPE * __restrict__ hy,
                             const $FPTYPE * __restrict__ hz,
                             $FPTYPE * __restrict__ psiEyx_r,
                             $FPTYPE * __restrict__ psiEzx_r,
                             const $FPTYPE * __restrict__ be_r,
                             const $FPTYPE * __restrict__ ce_r,
                             const $FPTYPE * __restrict__ cb,
                             const $FPTYPE dx)
{
    for (int z = blockIdx.z * blockDim.z + threadIdx.z;
            z < $NZ - 1;
            z += blockDim.z * gridDim.z) {
        for (int y = blockIdx.y * blockDim.y + threadIdx.y;
                y < $NY - 1;
                y += blockDim.y * gridDim.y) {
            for (int x = blockIdx.x * blockDim.x + threadIdx.x;
                    x < $NPML - 1;
                    x += blockDim.x * gridDim.x) {
                const size_t xf = $NX - $NPML + x;
                const size_t offCpml = (z * $NY * $NPML) + (y * $NPML) + x;
                const size_t offFull = (z * $NY * $NX) + (y * $NX) + xf;
                const size_t offFullLeft = (z * $NX * $NY) + (y * $NX) + xf
                    - 1;

                psiEzx_r[offCpml] = (be_r[offCpml] * psiEzx_r[offCpml])
                    + (ce_r[offCpml] * (hy[offFull] - hy[offFullLeft]) / dx);
                psiEyx_r[offCpml] = (be_r[offCpml] * psiEyx_r[offCpml])
                    + (ce_r[offCpml] * (hz[offFull] - hz[offFullLeft]) / dx);
                ez[offFull] += cb[offFull] * dx * psiEzx_r[offCpml];
                ey[offFull] -= cb[offFull] * dx * psiEyx_r[offCpml];
            }
        }
    }
}

__global__ void cpml_e_front($FPTYPE * __restrict__ ex,
                             $FPTYPE * __restrict__ ez,
                             const $FPTYPE * __restrict__ hx,
                             const $FPTYPE * __restrict__ hz,
                             $FPTYPE * __restrict__ psiExy_f,
                             $FPTYPE * __restrict__ psiEzy_f,
                             const $FPTYPE * __restrict__ be_f,
                             const $FPTYPE * __restrict__ ce_f,
                             const $FPTYPE * __restrict__ cb,
                             const $FPTYPE dx)
{
    for (int z = blockIdx.z * blockDim.z + threadIdx.z;
            z < $NZ - 1;
            z += blockDim.z * gridDim.z) {
        for (int y = blockIdx.y * blockDim.y + threadIdx.y;
                y < $NPML;
                y += blockDim.y * gridDim.y) {
            if (y < 1) {
                continue;
            }
            for (int x = blockIdx.x * blockDim.x + threadIdx.x;
                    x < $NX - 1;
                    x += blockDim.x * gridDim.x) {
                const size_t offCpml = (z * $NPML * $NX) + (y * $NX) + x;
                const size_t off = (z * $NY * $NX) + (y * $NX) + x;
                const size_t offFront = (z * $NX * $NY) + ((y - 1) * $NX) + x;

                psiExy_f[offCpml] = (be_f[offCpml] * psiExy_f[offCpml])
                    + (ce_f[offCpml] * (hz[off] - hz[offFront]) / dx);
                psiEzy_f[offCpml] = (be_f[offCpml] * psiEzy_f[offCpml])
                    + (ce_f[offCpml] * (hx[off] - hx[offFront]) / dx);
                ex[off] += cb[off] * dx * psiExy_f[offCpml];
                ez[off] -= cb[off] * dx * psiEzy_f[offCpml];
            }
        }
    }
}

__global__ void cpml_e_back($FPTYPE * __restrict__ ex,
                            $FPTYPE * __restrict__ ez,
                            const $FPTYPE * __restrict__ hx,
                            const $FPTYPE * __restrict__ hz,
                            $FPTYPE * __restrict__ psiExy_b,
                            $FPTYPE * __restrict__ psiEzy_b,
                            const $FPTYPE * __restrict__ be_b,
                            const $FPTYPE * __restrict__ ce_b,
                            const $FPTYPE * __restrict__ cb,
                            const $FPTYPE dx)
{
    for (int z = blockIdx.z * blockDim.z + threadIdx.z;
            z < $NZ - 1;
            z += blockDim.z * gridDim.z) {
        for (int y = blockIdx.y * blockDim.y + threadIdx.y;
                y < $NPML - 1;
                y += blockDim.y * gridDim.y) {
            for (int x = blockIdx.x * blockDim.x + threadIdx.x;
                    x < $NX - 1;
                    x += blockDim.x * gridDim.x) {
                const size_t yf = $NY - $NPML + y;
                const size_t offCpml = (z * $NPML * $NX) + (y * $NX) + x;
                const size_t offFull = (z * $NY * $NX) + (yf * $NX) + x;
                const size_t offFullFront = (z * $NX * $NY) + ((yf - 1) * $NX)
                    + x;

                psiExy_b[offCpml] = (be_b[offCpml] * psiExy_b[offCpml])
                    + (ce_b[offCpml] * (hz[offFull] - hz[offFullFront]) / dx);
                psiEzy_b[offCpml] = (be_b[offCpml] * psiEzy_b[offCpml])
                    + (ce_b[offCpml] * (hx[offFull] - hx[offFullFront]) / dx);
                ex[offFull] += cb[offFull] * dx * psiExy_b[offCpml];
                ez[offFull] -= cb[offFull] * dx * psiEzy_b[offCpml];
            }
        }
    }
}

__global__ void hx_update($FPTYPE * __restrict__ hx,
                          const $FPTYPE * __restrict__ ey,
                          const $FPTYPE * __restrict__ ez,
                          const $FPTYPE dtm0dx)
{
    for (int z = blockIdx.z * blockDim.z + threadIdx.z;
            z < $NZ - 1;
            z += blockDim.z * gridDim.z) {
        for (int y = blockIdx.y * blockDim.y + threadIdx.y;
                y < $NY - 1;
                y += blockDim.y * gridDim.y) {
            for (int x = blockIdx.x * blockDim.x + threadIdx.x;
                    x < $NX - 1;
                    x += blockDim.x * gridDim.x) {
                if (x < 1) {
                    continue;
                }
                const size_t off = (z * $NY * $NX) + (y * $NX) + x;
                const size_t offUp = ((z + 1) * $NX * $NY) + (y * $NX) + x;
                const size_t offBack = (z * $NY * $NX) + ((y + 1) * $NX) + x;

                const $FPTYPE delta_ez = ez[offBack] - ez[off];
                const $FPTYPE delta_ey = ey[offUp] - ey[off];
                hx[off] -= dtm0dx * (delta_ez - delta_ey);
            }
        }
    }
}

__global__ void hy_update($FPTYPE * __restrict__ hy,
                          const $FPTYPE * __restrict__ ex,
                          const $FPTYPE * __restrict__ ez,
                          const $FPTYPE dtm0dx)
{
    for (int z = blockIdx.z * blockDim.z + threadIdx.z;
            z < $NZ - 1;
            z += blockDim.z * gridDim.z) {
        for (int y = blockIdx.y * blockDim.y + threadIdx.y;
                y < $NY - 1;
                y += blockDim.y * gridDim.y) {
            if (y < 1) {
                continue;
            }
            for (int x = blockIdx.x * blockDim.x + threadIdx.x;
                    x < $NX - 1;
                    x += blockDim.x * gridDim.x) {
                const size_t off = (z * $NY * $NX) + (y * $NX) + x;
                const size_t offUp = ((z + 1) * $NX * $NY) + (y * $NX) + x;
                const size_t offRight = (z * $NY * $NX) + (y * $NX) + x + 1;

                const $FPTYPE delta_ex = ex[offUp] - ex[off];
                const $FPTYPE delta_ez = ez[offRight] - ez[off];
                hy[off] -= dtm0dx * (delta_ex - delta_ez);
            }
        }
    }
}

__global__ void hz_update($FPTYPE * __restrict__ hz,
                          const $FPTYPE * __restrict__ ex,
                          const $FPTYPE * __restrict__ ey,
                          const $FPTYPE dtm0dx)
{
    for (int z = blockIdx.z * blockDim.z + threadIdx.z;
            z < $NZ - 1;
            z += blockDim.z * gridDim.z) {
        if (z < 1) {
            continue;
        }
        for (int y = blockIdx.y * blockDim.y + threadIdx.y;
                y < $NY - 1;
                y += blockDim.y * gridDim.y) {
            for (int x = blockIdx.x * blockDim.x + threadIdx.x;
                    x < $NX - 1;
                    x += blockDim.x * gridDim.x) {
                const size_t off = (z * $NY * $NX) + (y * $NX) + x;
                const size_t offBack = (z * $NX * $NY) + ((y + 1) * $NX) + x;
                const size_t offRight = (z * $NY * $NX) + (y * $NX) + x + 1;

                const $FPTYPE delta_ex = ex[offBack] - ex[off];
                const $FPTYPE delta_ey = ey[offRight] - ey[off];
                hz[off] -= dtm0dx * (delta_ey - delta_ex);
            }
        }
    }
}

__global__ void cpml_h_down(const $FPTYPE * __restrict__ ex,
                            const $FPTYPE * __restrict__ ey,
                            $FPTYPE * __restrict__ hx,
                            $FPTYPE * __restrict__ hy,
                            $FPTYPE * __restrict__ psiHxz_d,
                            $FPTYPE * __restrict__ psiHyz_d,
                            const $FPTYPE * __restrict__ bh_d,
                            const $FPTYPE * __restrict__ ch_d,
                            const $FPTYPE dtm0,
                            const $FPTYPE dx)
{
    for (int z = blockIdx.z * blockDim.z + threadIdx.z;
            z < $NPML - 1;
            z += blockDim.z * gridDim.z) {
        for (int y = blockIdx.y * blockDim.y + threadIdx.y;
                y < $NY - 1;
                y += blockDim.y * gridDim.y) {
            for (int x = blockIdx.x * blockDim.x + threadIdx.x;
                    x < $NX - 1;
                    x += blockDim.x * gridDim.x) {
                const size_t off = (z * $NY * $NX) + (y * $NX) + x;
                const size_t offUp = ((z + 1) * $NX * $NY) + (y * $NX) + x;

                psiHyz_d[off] = (bh_d[off] * psiHyz_d[off])
                    + (ch_d[off] * (ex[offUp] - ex[off]) / dx);
                psiHxz_d[off] = (bh_d[off] * psiHxz_d[off])
                    + (ch_d[off] * (ey[offUp] - ey[off]) / dx);
                hy[off] -= dtm0 * psiHyz_d[off];
                hx[off] += dtm0 * psiHxz_d[off];
            }
        }
    }
}

__global__ void cpml_h_up(const $FPTYPE * __restrict__ ex,
                          const $FPTYPE * __restrict__ ey,
                          $FPTYPE * __restrict__ hx,
                          $FPTYPE * __restrict__ hy,
                          $FPTYPE * __restrict__ psiHxz_u,
                          $FPTYPE * __restrict__ psiHyz_u,
                          const $FPTYPE * __restrict__ bh_u,
                          const $FPTYPE * __restrict__ ch_u,
                          const $FPTYPE dtm0,
                          const $FPTYPE dx)
{
    for (int z = blockIdx.z * blockDim.z + threadIdx.z;
            z < $NPML - 1;
            z += blockDim.z * gridDim.z) {
        for (int y = blockIdx.y * blockDim.y + threadIdx.y;
                y < $NY - 1;
                y += blockDim.y * gridDim.y) {
            for (int x = blockIdx.x * blockDim.x + threadIdx.x;
                    x < $NX - 1;
                    x += blockDim.x * gridDim.x) {
                const size_t zf = $NZ - $NPML + z;
                const size_t off = (z * $NY * $NX) + (y * $NX) + x;
                const size_t offFull = (zf * $NY * $NX) + (y * $NX) + x;
                const size_t offFullUp = ((zf + 1) * $NX * $NY) + (y * $NX)
                    + x;

                psiHyz_u[off] = (bh_u[off] * psiHyz_u[off])
                    + (ch_u[off] * (ex[offFullUp] - ex[offFull]) / dx);
                psiHxz_u[off] = (bh_u[off] * psiHxz_u[off])
                    + (ch_u[off] * (ey[offFullUp] - ey[offFull]) / dx);
                hy[offFull] -= dtm0 * psiHyz_u[off];
                hx[offFull] += dtm0 * psiHxz_u[off];
            }
        }
    }
}

__global__ void cpml_h_left(const $FPTYPE * __restrict__ ey,
                            const $FPTYPE * __restrict__ ez,
                            $FPTYPE * __restrict__ hy,
                            $FPTYPE * __restrict__ hz,
                            $FPTYPE * __restrict__ psiHyx_l,
                            $FPTYPE * __restrict__ psiHzx_l,
                            const $FPTYPE * __restrict__ bh_l,
                            const $FPTYPE * __restrict__ ch_l,
                            const $FPTYPE dtm0,
                            const $FPTYPE dx)
{
    for (int z = blockIdx.z * blockDim.z + threadIdx.z;
            z < $NZ - 1;
            z += blockDim.z * gridDim.z) {
        for (int y = blockIdx.y * blockDim.y + threadIdx.y;
                y < $NY - 1;
                y += blockDim.y * gridDim.y) {
            for (int x = blockIdx.x * blockDim.x + threadIdx.x;
                    x < $NPML - 1;
                    x += blockDim.x * gridDim.x) {
                const size_t offCpml = (z * $NY * $NPML) + (y * $NPML) + x;
                const size_t off = (z * $NY * $NX) + (y * $NX) + x;
                const size_t offRight = (z * $NX * $NY) + (y * $NX) + x + 1;

                psiHzx_l[offCpml] = (bh_l[offCpml] * psiHzx_l[offCpml])
                    + (ch_l[offCpml] * (ey[offRight] - ey[off]) / dx);
                psiHyx_l[offCpml] = (bh_l[offCpml] * psiHyx_l[offCpml])
                    + (ch_l[offCpml] * (ez[offRight] - ez[off]) / dx);
                hz[off] -= dtm0 * psiHzx_l[offCpml];
                hy[off] += dtm0 * psiHyx_l[offCpml];
            }
        }
    }
}

__global__ void cpml_h_right(const $FPTYPE * __restrict__ ey,
                            const $FPTYPE * __restrict__ ez,
                            $FPTYPE * __restrict__ hy,
                            $FPTYPE * __restrict__ hz,
                            $FPTYPE * __restrict__ psiHyx_r,
                            $FPTYPE * __restrict__ psiHzx_r,
                            const $FPTYPE * __restrict__ bh_r,
                            const $FPTYPE * __restrict__ ch_r,
                            const $FPTYPE dtm0,
                            const $FPTYPE dx)
{
    for (int z = blockIdx.z * blockDim.z + threadIdx.z;
            z < $NZ - 1;
            z += blockDim.z * gridDim.z) {
        for (int y = blockIdx.y * blockDim.y + threadIdx.y;
                y < $NY - 1;
                y += blockDim.y * gridDim.y) {
            for (int x = blockIdx.x * blockDim.x + threadIdx.x;
                    x < $NPML - 1;
                    x += blockDim.x * gridDim.x) {
                const size_t xf = $NX - $NPML + x;
                const size_t offCpml = (z * $NY * $NPML) + (y * $NPML) + x;
                const size_t offFull = (z * $NY * $NX) + (y * $NX) + xf;
                const size_t offFullRight = (z * $NX * $NY) + (y * $NX) + xf
                    + 1;

                psiHzx_r[offCpml] = (bh_r[offCpml] * psiHzx_r[offCpml])
                    + (ch_r[offCpml] * (ey[offFullRight] - ey[offFull]) / dx);
                psiHyx_r[offCpml] = (bh_r[offCpml] * psiHyx_r[offCpml])
                    + (ch_r[offCpml] * (ez[offFullRight] - ez[offFull]) / dx);
                hz[offFull] -= dtm0 * psiHzx_r[offCpml];
                hy[offFull] += dtm0 * psiHyx_r[offCpml];
            }
        }
    }
}

__global__ void cpml_h_front(const $FPTYPE * __restrict__ ex,
                             const $FPTYPE * __restrict__ ez,
                             $FPTYPE * __restrict__ hx,
                             $FPTYPE * __restrict__ hz,
                             $FPTYPE * __restrict__ psiHxy_f,
                             $FPTYPE * __restrict__ psiHzy_f,
                             const $FPTYPE * __restrict__ bh_f,
                             const $FPTYPE * __restrict__ ch_f,
                             const $FPTYPE dtm0,
                             const $FPTYPE dx)
{
    for (int z = blockIdx.z * blockDim.z + threadIdx.z;
            z < $NZ - 1;
            z += blockDim.z * gridDim.z) {
        for (int y = blockIdx.y * blockDim.y + threadIdx.y;
                y < $NPML - 1;
                y += blockDim.y * gridDim.y) {
            for (int x = blockIdx.x * blockDim.x + threadIdx.x;
                    x < $NX - 1;
                    x += blockDim.x * gridDim.x) {
                const size_t offCpml = (z * $NPML * $NX) + (y * $NX) + x;
                const size_t off = (z * $NY * $NX) + (y * $NX) + x;
                const size_t offBack = (z * $NX * $NY) + ((y + 1) * $NX) + x;

                psiHxy_f[offCpml] = (bh_f[offCpml] * psiHxy_f[offCpml])
                    + (ch_f[offCpml] * (ez[offBack] - ez[off]) / dx);
                psiHzy_f[offCpml] = (bh_f[offCpml] * psiHzy_f[offCpml])
                    + (ch_f[offCpml] * (ex[offBack] - ex[off]) / dx);
                hx[off] -= dtm0 * psiHxy_f[offCpml];
                hz[off] += dtm0 * psiHzy_f[offCpml];
            }
        }
    }
}

__global__ void cpml_h_back(const $FPTYPE * __restrict__ ex,
                            const $FPTYPE * __restrict__ ez,
                            $FPTYPE * __restrict__ hx,
                            $FPTYPE * __restrict__ hz,
                            $FPTYPE * __restrict__ psiHxy_b,
                            $FPTYPE * __restrict__ psiHzy_b,
                            const $FPTYPE * __restrict__ bh_b,
                            const $FPTYPE * __restrict__ ch_b,
                            const $FPTYPE dtm0,
                            const $FPTYPE dx)
{
    for (int z = blockIdx.z * blockDim.z + threadIdx.z;
            z < $NZ - 1;
            z += blockDim.z * gridDim.z) {
        for (int y = blockIdx.y * blockDim.y + threadIdx.y;
                y < $NPML - 1;
                y += blockDim.y * gridDim.y) {
            for (int x = blockIdx.x * blockDim.x + threadIdx.x;
                    x < $NX - 1;
                    x += blockDim.x * gridDim.x) {
                const size_t yf = $NY - $NPML + y;
                const size_t offCpml = (z * $NPML * $NX) + (y * $NX) + x;
                const size_t offFull = (z * $NY * $NX) + (yf * $NX) + x;
                const size_t offFullBack = (z * $NX * $NY) + ((yf + 1) * $NX)
                    + x;

                psiHxy_b[offCpml] = (bh_b[offCpml] * psiHxy_b[offCpml])
                    + (ch_b[offCpml] * (ez[offFullBack] - ez[offFull]) / dx);
                psiHzy_b[offCpml] = (bh_b[offCpml] * psiHzy_b[offCpml])
                    + (ch_b[offCpml] * (ex[offFullBack] - ex[offFull]) / dx);
                hx[offFull] -= dtm0 * psiHxy_b[offCpml];
                hz[offFull] += dtm0 * psiHzy_b[offCpml];
            }
        }
    }
}

__global__ void inject_incident_e_field($FPTYPE * __restrict__ hx,
                                        $FPTYPE * __restrict__ hy,
                                        const $FPTYPE * __restrict__ ex_inc,
                                        const $FPTYPE * __restrict__ ey_inc,
                                        const $FPTYPE db,
                                        int tfsf_z)
{
    for (int y = blockIdx.y * blockDim.y + threadIdx.y;
            y < $NY;
            y += blockDim.y * gridDim.y) {
        for (int x = blockIdx.x * blockDim.x + threadIdx.x;
                x < $NX - 1;
                x += blockDim.x * gridDim.x) {
            const size_t offField = (tfsf_z * $NY * $NX) + (y * $NX) + x;
            const size_t offInc = (y * $NX) + x;
            hx[offField] += db * ey_inc[offInc];
            hy[offField] -= db * ex_inc[offInc];
        }
    }
}

__global__ void inject_incident_h_field($FPTYPE * __restrict__ ex,
                                        $FPTYPE * __restrict__ ey,
                                        const $FPTYPE * __restrict__ hx_inc,
                                        const $FPTYPE * __restrict__ hy_inc,
                                        const $FPTYPE dte0dx,
                                        int tfsf_z)
{
    for (int y = blockIdx.y * blockDim.y + threadIdx.y;
            y < $NY;
            y += blockDim.y * gridDim.y) {
        for (int x = blockIdx.x * blockDim.x + threadIdx.x;
                x < $NX - 1;
                x += blockDim.x * gridDim.x) {
            const size_t offField = (tfsf_z * $NY * $NX) + (y * $NX) + x;
            const size_t offInc = (y * $NX) + x;
            ey[offField] += dte0dx * hx_inc[offInc];
            ex[offField] -= dte0dx * hy_inc[offInc];
        }
    }
}

__global__ void phasors_x_plane(const $FPTYPE * __restrict__ ey,
                                const $FPTYPE * __restrict__ ez,
                                const $FPTYPE * __restrict__ hy,
                                const $FPTYPE * __restrict__ hz,
                                $FPTYPE * __restrict__ ey_ph,
                                $FPTYPE * __restrict__ ez_ph,
                                $FPTYPE * __restrict__ hy_ph,
                                $FPTYPE * __restrict__ hz_ph,
                                $FPTYPE omega,
                                int phasor_x,
                                int t,
                                $FPTYPE dt)
{
    const $FPTYPE tE = (($FPTYPE) t);
    const $FPTYPE tH = (($FPTYPE) t) - 0.5;
    $FPTYPE expFactorECos = 0.0;
    $FPTYPE expFactorESin = 0.0;
    $FPTYPE expFactorHCos = 0.0;
    $FPTYPE expFactorHSin = 0.0;

    ${SINCOS}(-omega * tE * dt, &expFactorESin, &expFactorECos);
    ${SINCOS}(-omega * tH * dt, &expFactorHSin, &expFactorHCos);

    for (int z = blockIdx.z * blockDim.z + threadIdx.z;
            z < $NZ_PH;
            z += blockDim.z * gridDim.z) {
        if (z < 1) {
                continue;
        }
        for (int y = blockIdx.y * blockDim.y + threadIdx.y;
                y < $NY_PH;
                y += blockDim.y * gridDim.y) {
            if (y < 1) {
                continue;
            }

            int z_field = z + $NPML;
            int y_field = y + $NPML;

            const size_t offField = (z_field * $NY * $NX) + (y_field * $NX)
                                    + phasor_x;
            const size_t offFront = (z_field * $NY * $NX)
                                    + ((y_field - 1) * $NX) + phasor_x;
            const size_t offDown = ((z_field - 1) * $NY * $NX)
                                   + (y_field * $NX) + phasor_x;
            const size_t offLeft = (z_field * $NY * $NX) + (y_field * $NX)
                                    + phasor_x - 1;
            const size_t offLeftDown = ((z_field - 1) * $NY * $NX)
                                        + (y_field * $NX) + phasor_x - 1;
            const size_t offLeftFront = (z_field * $NY * $NX)
                                       + ((y_field - 1) * $NX)
                                       + (phasor_x - 1);
            const size_t offPhasorRe = 2 * ((z * $NY_PH) + y);
            const size_t offPhasorIm = 2 * ((z * $NY_PH) + y) + 1;

            const $FPTYPE eyMean = 0.5 * (ey[offField] + ey[offFront]);
            const $FPTYPE ezMean = 0.5 * (ez[offField] + ez[offDown]);
            const $FPTYPE hyMean = 0.25 * (hy[offField] + hy[offLeft]
                                           + hy[offDown] + hy[offLeftDown]);
            const $FPTYPE hzMean = 0.25 * (hz[offField] + hz[offLeft]
                                           + hz[offFront] + hz[offLeftFront]);

            ey_ph[offPhasorRe] += eyMean * expFactorECos;
            ey_ph[offPhasorIm] += eyMean * expFactorESin;
            ez_ph[offPhasorRe] += ezMean * expFactorECos;
            ez_ph[offPhasorIm] += ezMean * expFactorESin;
            hy_ph[offPhasorRe] += hyMean * expFactorHCos;
            hy_ph[offPhasorIm] += hyMean * expFactorHSin;
            hz_ph[offPhasorRe] += hzMean * expFactorHCos;
            hz_ph[offPhasorIm] += hzMean * expFactorHSin;
        }
    }
}

__global__ void phasors_y_plane(const $FPTYPE * __restrict__ ex,
                                const $FPTYPE * __restrict__ ez,
                                const $FPTYPE * __restrict__ hx,
                                const $FPTYPE * __restrict__ hz,
                                $FPTYPE * __restrict__ ex_ph,
                                $FPTYPE * __restrict__ ez_ph,
                                $FPTYPE * __restrict__ hx_ph,
                                $FPTYPE * __restrict__ hz_ph,
                                $FPTYPE omega,
                                int phasor_y,
                                int t,
                                $FPTYPE dt)
{
    const $FPTYPE tE = (($FPTYPE) t);
    const $FPTYPE tH = (($FPTYPE) t) - 0.5;
    $FPTYPE expFactorECos = 0.0;
    $FPTYPE expFactorESin = 0.0;
    $FPTYPE expFactorHCos = 0.0;
    $FPTYPE expFactorHSin = 0.0;

    ${SINCOS}(-omega * tE * dt, &expFactorESin, &expFactorECos);
    ${SINCOS}(-omega * tH * dt, &expFactorHSin, &expFactorHCos);

    for (int z = blockIdx.z * blockDim.z + threadIdx.z;
            z < $NZ_PH;
            z += blockDim.z * gridDim.z) {
        if (z < 1) {
                continue;
        }
        for (int x = blockIdx.x * blockDim.x + threadIdx.x;
                x < $NX_PH;
                x += blockDim.x * gridDim.x) {
            if (x < 1) {
                continue;
            }

            int z_field = z + $NPML;
            int x_field = x + $NPML;

            const size_t offField = (z_field * $NY * $NX) + (phasor_y * $NX)
                                    + x_field;
            const size_t offLeft = (z_field * $NY * $NX) + (phasor_y * $NX)
                                   + x_field - 1;
            const size_t offFront = (z_field * $NY * $NX)
                                    + ((phasor_y - 1) * $NX)
                                    + x_field;
            const size_t offDown = ((z_field - 1) * $NY * $NX)
                                   + (phasor_y * $NX)
                                   + x_field;
            const size_t offFrontDown = ((z_field - 1) * $NY * $NX)
                                        + ((phasor_y - 1) * $NX)
                                        + x_field;
            const size_t offFrontLeft = (z_field * $NY * $NX)
                                       + ((phasor_y - 1) * $NX)
                                       + (x_field - 1);
            const size_t offPhasorRe = 2 * ((z * $NX_PH) + x);
            const size_t offPhasorIm = 2 * ((z * $NX_PH) + x) + 1;

            const $FPTYPE exMean = 0.5 * (ex[offField] + ex[offLeft]);
            const $FPTYPE ezMean = 0.5 * (ez[offField] + ez[offDown]);
            const $FPTYPE hxMean = 0.25 * (hx[offField] + hx[offFront]
                                           + hx[offDown] + hx[offFrontDown]);
            const $FPTYPE hzMean = 0.25 * (hz[offField] + hz[offFront]
                                           + hz[offLeft] + hz[offFrontLeft]);

            ex_ph[offPhasorRe] += exMean * expFactorECos;
            ex_ph[offPhasorIm] += exMean * expFactorESin;
            ez_ph[offPhasorRe] += ezMean * expFactorECos;
            ez_ph[offPhasorIm] += ezMean * expFactorESin;
            hx_ph[offPhasorRe] += hxMean * expFactorHCos;
            hx_ph[offPhasorIm] += hxMean * expFactorHSin;
            hz_ph[offPhasorRe] += hzMean * expFactorHCos;
            hz_ph[offPhasorIm] += hzMean * expFactorHSin;
        }
    }
}

__global__ void phasors_z_plane(const $FPTYPE * __restrict__ ex,
                                const $FPTYPE * __restrict__ ey,
                                const $FPTYPE * __restrict__ hx,
                                const $FPTYPE * __restrict__ hy,
                                $FPTYPE * __restrict__ ex_ph,
                                $FPTYPE * __restrict__ ey_ph,
                                $FPTYPE * __restrict__ hx_ph,
                                $FPTYPE * __restrict__ hy_ph,
                                $FPTYPE omega,
                                int phasor_z,
                                int t,
                                $FPTYPE dt)
{
    const $FPTYPE tE = (($FPTYPE) t);
    const $FPTYPE tH = (($FPTYPE) t) - 0.5;
    $FPTYPE expFactorECos = 0.0;
    $FPTYPE expFactorESin = 0.0;
    $FPTYPE expFactorHCos = 0.0;
    $FPTYPE expFactorHSin = 0.0;

    ${SINCOS}(-omega * tE * dt, &expFactorESin, &expFactorECos);
    ${SINCOS}(-omega * tH * dt, &expFactorHSin, &expFactorHCos);

    for (int y = blockIdx.y * blockDim.y + threadIdx.y;
            y < $NY_PH;
            y += blockDim.y * gridDim.y) {
        if (y < 1) {
                continue;
        }
        for (int x = blockIdx.x * blockDim.x + threadIdx.x;
                x < $NX_PH;
                x += blockDim.x * gridDim.x) {
            if (x < 1) {
                continue;
            }

            int y_field = y + $NPML;
            int x_field = x + $NPML;

            const size_t offField = (phasor_z * $NY * $NX) + (y_field * $NX)
                                    + x_field;
            const size_t offLeft = (phasor_z * $NY * $NX) + (y_field * $NX)
                                   + x_field - 1;
            const size_t offFront = (phasor_z * $NY * $NX)
                                    + ((y_field - 1) * $NX)
                                    + x_field;
            const size_t offDown = ((phasor_z - 1) * $NY * $NX)
                                   + (y_field * $NX)
                                   + x_field;
            const size_t offDownFront = ((phasor_z - 1) * $NY * $NX)
                                        + ((y_field - 1) * $NX)
                                        + x_field;
            const size_t offDownLeft = ((phasor_z - 1) * $NY * $NX)
                                       + (y_field * $NX)
                                       + (x_field - 1);
            const size_t offPhasorRe = 2 * ((y * $NX_PH) + x);
            const size_t offPhasorIm = 2 * ((y * $NX_PH) + x) + 1;

            const $FPTYPE exMean = 0.5 * (ex[offField] + ex[offLeft]);
            const $FPTYPE eyMean = 0.5 * (ey[offField] + ey[offFront]);
            const $FPTYPE hxMean = 0.25 * (hx[offField] + hx[offFront]
                                           + hx[offDown] + hx[offDownFront]);
            const $FPTYPE hyMean = 0.25 * (hy[offField] + hy[offLeft]
                                           + hy[offDown] + hy[offDownLeft]);

            ex_ph[offPhasorRe] += exMean * expFactorECos;
            ex_ph[offPhasorIm] += exMean * expFactorESin;
            ey_ph[offPhasorRe] += eyMean * expFactorECos;
            ey_ph[offPhasorIm] += eyMean * expFactorESin;
            hx_ph[offPhasorRe] += hxMean * expFactorHCos;
            hx_ph[offPhasorIm] += hxMean * expFactorHSin;
            hy_ph[offPhasorRe] += hyMean * expFactorHCos;
            hy_ph[offPhasorIm] += hyMean * expFactorHSin;
        }
    }
}
''')
