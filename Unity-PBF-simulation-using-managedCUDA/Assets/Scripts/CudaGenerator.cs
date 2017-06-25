using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using System.IO;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.VectorTypes;
using ManagedCuda.NVRTC;

public class CudaGenerator : MonoBehaviour
{
    public string[] cudafiles;
    public string[] CompileOption;

    protected CudaKernel[] cudaKernel;
    protected CudaContext ctx;

    protected void InitializeCUDA()
    {
        string[] filetext = new string[cudafiles.Length];
        cudaKernel = new CudaKernel[cudafiles.Length];
        ctx = new CudaContext(0);

        for (int i = 0; i < cudafiles.Length; ++i)
        {
            filetext[i] = File.ReadAllText(@"Assets\Scripts\CUDA\" + cudafiles[i] + ".cu");
            Debug.Log(filetext[i]);

            CudaRuntimeCompiler rtc = new CudaRuntimeCompiler(filetext[i], cudafiles[i]);
            rtc.Compile(CompileOption);
            Debug.Log(rtc.GetLogAsString());

            byte[] ptx = rtc.GetPTX();
            rtc.Dispose();

            cudaKernel[i] = ctx.LoadKernelPTX(ptx, cudafiles[i]);
        }
    }

    void Start()
    {
        InitializeCUDA();
    }

    void OnDestroy()
    {
        ctx.Dispose();
    }
}
