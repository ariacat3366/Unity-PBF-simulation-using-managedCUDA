//function kernel
__device__ float length(float3 r) {
    return r.x*r.x + r.y*r.y + r.z*r.z;
}
__device__ float3 mul_float3(float3 r1, float3 r2) {
    return make_float3(r1.x * r2.x,  r1.y * r2.y,  r1.z * r2.z);
}
__device__ float3 add_float3(float3 r1, float3 r2) {
    return make_float3(r1.x + r2.x,  r1.y + r2.y,  r1.z + r2.z);
}
__device__ float3 dif_float3(float3 r1, float3 r2) {
    return make_float3(r1.x - r2.x,  r1.y - r2.y,  r1.z - r2.z);
}
__device__ float3 scale_float3(float s, float3 r) {
    r.x *= s;
    r.y *= s;
    r.z *= s;
    return r;
}
__device__ float Kernel_Poly6(float3 r, float h) {
	float PI = 3.14159;
	return 315.0f / (64 * PI * pow(h, 9)) * pow(pow(h, 2) - length(r), 3);
}
__device__ float3 Gradient_Kernel_Poly6(float3 r, float h) {
	float PI = 3.14159;
	return make_float3(
            r.x * -945.0f / ( 32.0f * PI * pow(h,9) ) * pow(pow(h, 2) - length(r), 2),
            r.y * -945.0f / ( 32.0f * PI * pow(h,9) ) * pow(pow(h, 2) - length(r), 2),
            r.z * -945.0f / ( 32.0f * PI * pow(h,9) ) * pow(pow(h, 2) - length(r), 2));
}
__device__ float Lap_Kernel_Poly6(float3 r, float h) {
	float PI = 3.14159;
	return 945.0f / (8 * PI * pow(h, 9)) * (pow(h, 2) - length(r)) * (length(r) - 3 / 4 * (pow(h, 2) - length(r)));
}
__device__ float3 Gradient_Kernel_Spiky(float3 r, float h) {
	float PI = 3.14159;
    float _r = sqrt(length(r));
    float v = -45.0f / (PI * pow(h, 6) * _r) * pow(h - _r, 2);
	return make_float3(r.x*v, r.y*v, r.z*v);
}
__device__ float Lap_Kernel_Viscosity(float3 r, float h) {
	float PI = 3.14159;
	return 45.0f / (PI * pow(h, 5)) * (1 - sqrt(length(r)) / h);
}


//SPH particle struct
struct pPBF {
	float3 pos;
    float3 vel;
    float m;
	float rho;
    float lambda;
	float col;
};


extern "C" __global__ void
PBF_1(pPBF *p, int *np, const float h, const int N)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x; 
    if (idx > N) return;

    pPBF _p = p[idx];

    // calc rho

    int i;
    int num_of_np = 200; // the number of neighbor particle
    int npnum = idx*num_of_np, _npnum = 0;
    float _rho = 0.0f;

    for (i = 0; i < N; ++i)
    {
        if (i == idx) continue;
        float3 r = dif_float3(_p.pos, p[i].pos);
        if (length(r) <= h*h)
        {
            _rho += p[i].m * Kernel_Poly6(r, h);

            // neighbor
            if (_npnum >= num_of_np) continue;
            np[npnum + _npnum] = i;
            ++_npnum;
        }
    }

    // neighbor
    for (; _npnum < num_of_np; ++_npnum)
    {
        np[npnum + _npnum] = -1;
    }

    p[idx].rho = _rho + 0.0001f;
    p[idx].col = 1.0f / (p[idx].rho/2.0f + 1.0f);

    return;
}

