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



//PBF particle struct
struct pPBF {
	float3 pos;
    float3 vel;
    float m;
	float rho;
    float lambda;
	float col;
};

extern "C" __global__ void
PBF_3(pPBF *p, int *np, const float h, const int N, const float _x, const float _z)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x; 
    if (idx > N) return;

    pPBF _p = p[idx];

    float3 d_p = make_float3(0,0,0);
    float rho_0 = 1.0f;
    float lambda_corr = 0.0f;
    int i;

    // delta_p
    
    /*
    for (i = 0; i < N; ++i)
    {
        pPBF __p = p[i];
        float3 r = dif_float3(_p.pos, __p.pos);
        if (i == idx) continue;
        if (length(r) > h*h) continue;
        
        //d_p = add_float3(d_p, scale_float3((_p.lambda + __p.lambda)*__p.m, Gradient_Kernel_Spiky(r, h)));
        lambda_corr = -0.001f * pow(Kernel_Poly6(r, h)/2.0f, 4);
        d_p = add_float3(d_p, scale_float3(_p.lambda + __p.lambda + lambda_corr, Gradient_Kernel_Spiky(r, h)));
    }
    */    
    int num_of_np = 200;
    for (i = 0; i < num_of_np; ++i)
    {
        int _np = idx * num_of_np + i;
        if (np[_np] < 0) break;

        pPBF __p = p[np[_np]];
        float3 r = dif_float3(_p.pos, __p.pos);
        
        lambda_corr = -0.001f * pow(Kernel_Poly6(r, h)/1.0f, 4);
        d_p = add_float3(d_p, scale_float3(_p.lambda + __p.lambda + lambda_corr, Gradient_Kernel_Spiky(r, h)));
    }

    //_p.pos = add_float3(_p.pos, scale_float3(1.0f/rho_0, d_p));
    _p.pos = add_float3(_p.pos, d_p);
    


    //wall

    float modify = 0.4f;
    _p.pos = make_float3(
        _p.pos.x <= 0 ? -1.0f * _p.pos.x * modify : _p.pos.x >= _x ? _x - (_p.pos.x - _x) * modify : _p.pos.x,
        _p.pos.y <= 0 ? -1.0f * _p.pos.y * modify : _p.pos.y,
        _p.pos.z <= 0 ? -1.0f * _p.pos.z * modify : _p.pos.z >= _z ? _z - (_p.pos.z - _z) * modify : _p.pos.z
    );

    // update x
    
    p[idx].pos = _p.pos;

    return;
}
