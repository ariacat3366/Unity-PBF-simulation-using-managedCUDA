using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using System.IO;
using System.Runtime.InteropServices;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.VectorTypes;
using ManagedCuda.NVRTC;

public class PBF : CudaGenerator
{
    // 粒子の状態
    struct pPBF
    {
        public Vector3 pos;
        public Vector3 vel;
        public float m;
        public float rho;
        public float lambda;
        public float col;
        public pPBF(Vector3 _pos, Vector3 _vel, float _m, float _rho, float _lambda)
        {
            pos = _pos;
            vel = _vel;
            m = _m;
            rho = 0;
            lambda = _lambda;
            col = 0.5f;
        }
    }

    // 粒子数
    public int NUM_OF_P;
    // 近傍半径
    public float h;
    // イテレート回数
    public float iter;

    // レンダリングシェーダー
    public Shader PBFShader;

    // マテリアル
    Material PBFMaterial;
    Material SSFRMaterial;

    // コンピュートバッファ
    ComputeBuffer pBuf;
    ComputeBuffer posBuf;
    ComputeBuffer colBuf;

    // CUDA Variable
    CudaDeviceVariable<pPBF> d_pPBF_particle;
    CudaDeviceVariable<Vector3> d_pPBF_pos;
    CudaDeviceVariable<float> d_pPBF_col;
    CudaDeviceVariable<int> d_np; // neighbor particle


    // host Variable
    pPBF[] h_pPBF_particle;
    Vector3[] h_pPBF_pos;
    float[] h_pPBF_col;

    //重力加速度
    public float g;
    //水の一辺の長さ
    public int water_len;
    // 粒子間距離
    public float scale;
    // delta time
    float dt;
    float t;
    // 壁
    public float wall_x;
    float _wall_x;
    public float wall_z;

    // Wave
    public bool wave;


    // Use this for initialization
    void Start()
    {
        InitializeCUDA();

        PBFMaterial = new Material(PBFShader);

        h_pPBF_particle = new pPBF[NUM_OF_P];
        h_pPBF_pos = new Vector3[NUM_OF_P];
        h_pPBF_col = new float[NUM_OF_P];

        dt = Time.deltaTime;
        _wall_x = wall_x;
        
        //particle
        for (int i = 0; i < NUM_OF_P; i++)            
        {
            int x, y, z;
            x = i % water_len;
            z = (i / water_len) % water_len;
            y = (i / water_len / water_len);

            h_pPBF_particle[i] = new pPBF(
            new Vector3(x, y, z) * scale + new Vector3(0, 1.0f, 0) + new Vector3(3f, 0f, 3f),
            new Vector3(0, 0, 0),
            0.15f,
            0f,
            0f
            );
            h_pPBF_pos[i] = h_pPBF_particle[i].pos;
            h_pPBF_col[i] = h_pPBF_particle[i].col;
        }        

        d_pPBF_particle = h_pPBF_particle;
        d_pPBF_pos = h_pPBF_pos;
        d_pPBF_col = h_pPBF_col;
        d_np = new int[200 * NUM_OF_P];

        pBuf = new ComputeBuffer(NUM_OF_P, Marshal.SizeOf(typeof(pPBF)));
        posBuf = new ComputeBuffer(NUM_OF_P, Marshal.SizeOf(typeof(Vector3)));
        colBuf = new ComputeBuffer(NUM_OF_P, Marshal.SizeOf(typeof(float)));


        int threadsPerBlock = 1024;
        int blocksPerGrid = (NUM_OF_P + threadsPerBlock - 1) / threadsPerBlock;
        cudaKernel[0].BlockDimensions = new dim3(threadsPerBlock, 1, 1);
        cudaKernel[0].GridDimensions = new dim3(blocksPerGrid, 1, 1);
        cudaKernel[1].BlockDimensions = new dim3(threadsPerBlock, 1, 1);
        cudaKernel[1].GridDimensions = new dim3(blocksPerGrid, 1, 1);
        cudaKernel[2].BlockDimensions = new dim3(threadsPerBlock, 1, 1);
        cudaKernel[2].GridDimensions = new dim3(blocksPerGrid, 1, 1);
        cudaKernel[3].BlockDimensions = new dim3(threadsPerBlock, 1, 1);
        cudaKernel[3].GridDimensions = new dim3(blocksPerGrid, 1, 1);

    }

    void OnDisable()
    {
        d_pPBF_particle.Dispose();
        d_pPBF_pos.Dispose();
        d_pPBF_col.Dispose();
        pBuf.Release();
        posBuf.Release();
        colBuf.Release();
    }

    // Update is called once per frame
    void Update()
    {
        
        if (wave)
        {
            _wall_x = (float)(wall_x + 8 * Math.Sin(t * 1.2f));
            t += dt;
        }

        for (int i = 0; i < iter; ++i)
        {
            // rho
            cudaKernel[0].Run(d_pPBF_particle.DevicePointer, d_np.DevicePointer, h, NUM_OF_P);
            // lambda
            cudaKernel[1].Run(d_pPBF_particle.DevicePointer, d_np.DevicePointer, h, NUM_OF_P);
            // update x
            cudaKernel[2].Run(d_pPBF_particle.DevicePointer, d_np.DevicePointer, h, NUM_OF_P, _wall_x, wall_z);
        }
        // apply v and x  :: F_ext
        cudaKernel[3].Run(d_pPBF_particle.DevicePointer, d_pPBF_pos.DevicePointer, d_pPBF_col.DevicePointer, g, dt, NUM_OF_P);

        d_pPBF_pos.CopyToHost(h_pPBF_pos);
        d_pPBF_col.CopyToHost(h_pPBF_col);

        //Debug.Log("Pos:" + h_pPBF_pos[10]);
        

        posBuf.SetData(h_pPBF_pos);
        colBuf.SetData(h_pPBF_col);


    }

    void OnRenderObject()
    {
        // テクスチャ、バッファをマテリアルに設定
        // レンダリングを開始
        if (false)
        {
            SSFRMaterial.SetBuffer("pPBFBuf", posBuf);
            SSFRMaterial.SetPass(0);
            Graphics.DrawProcedural(MeshTopology.Points, NUM_OF_P);
        }
        else
        {
            PBFMaterial.SetBuffer("posBuf", posBuf);
            PBFMaterial.SetBuffer("colBuf", colBuf);
            //PBFMaterial.SetBuffer("pBuf", pBuf);            
            PBFMaterial.SetPass(0);
            Graphics.DrawProcedural(MeshTopology.Points, NUM_OF_P);
        }
    }
}

