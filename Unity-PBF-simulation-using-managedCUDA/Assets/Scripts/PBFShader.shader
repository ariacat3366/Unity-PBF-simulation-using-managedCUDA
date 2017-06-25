Shader "Custom/PBFShader" {
	SubShader{
		// アルファを使う
		ZWrite On
		Blend SrcAlpha OneMinusSrcAlpha

		Pass{
		CGPROGRAM

		// シェーダーモデルは5.0を指定
#pragma target 5.0

		// シェーダー関数を設定
#pragma vertex vert
#pragma geometry geom
#pragma fragment frag

#include "UnityCG.cginc"

	struct pPBF {
		float3 pos;
		float3 vel;
		float m;
		float rho;
		float lambda;
		float col;
	};

	StructuredBuffer<pPBF> pBuf;
	StructuredBuffer<float3> posBuf;
	StructuredBuffer<float> colBuf;

	// 頂点シェーダからの出力
	struct v2f {
		float4 pos : SV_POSITION;
		float2 uv : TEXCOORD0;
		float4 col : COLOR;
	};

	// 頂点シェーダ
	v2f vert(uint id : SV_VertexID)
	{
		v2f output;
		output.pos = float4(posBuf[id], 1.0);
		//output.pos = float4(pBuf[id].pos, 1.0);
		output.uv = float2(0, 0);
		output.col = float4(float3(0.2,0.8,1.0) * colBuf[id], 1);
		//output.col = float4(float3(0.2,0.8,1.0) * pBuf[id].col, 1);
		return output;
	}

	// ジオメトリシェーダ
	[maxvertexcount(4)]
	void geom(point v2f input[1], inout TriangleStream<v2f> outStream)
	{
		float4 pos = input[0].pos;
		float4 col = input[0].col;

		float4x4 billboardMatrix = UNITY_MATRIX_V;
		billboardMatrix._m03 =
		billboardMatrix._m13 =
		billboardMatrix._m23 =
		billboardMatrix._m33 = 0;
		
		v2f o;

		//(-1,-1)(-1,1)(1,-1)(1,1)
		for (int x = -1; x < 2; x += 2) {
			for (int y = -1; y < 2; y += 2) {
				float4 cpos = pos + mul(float4(x, y, 0, 0)*0.3, billboardMatrix);
				o.pos = mul(UNITY_MATRIX_VP, cpos);
				o.uv = float2(x + 1, y + 1) / 2;
				o.col = float4(1, 1, 1, 1) * col;

				outStream.Append(o);
			}
		}

		// トライアングルストリップを終了
		outStream.RestartStrip();
	}
	// ピクセルシェーダー
	fixed4 frag(v2f i) : COLOR
	{
		// 出力はテクスチャカラーと頂点色
		float4 col = i.col;
		if (length(i.uv - float2(0.5,0.5)) > 0.3) discard;

		// 色を返す
		return col;
	}

	ENDCG
	}
	}
}
