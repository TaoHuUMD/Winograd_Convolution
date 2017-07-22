#ifndef PUBLIC_TOOL_H
#define PUBLIC_TOOL_H

#include <fstream>
#include <iostream>
#include "mathlib.h"

namespace PUBLIC_TOOL{

	template<typename Dtype>
	Dtype max(Dtype a, Dtype b) {
		if (a > b) return a;
		else return b;
	}

	template<typename Dtype>
	Dtype min(Dtype a, Dtype b) {
		if (a < b) return a;
		else return b;
	}

	 void dlm_cpu_gemm(const CBLAS_TRANSPOSE TransA,
		const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
		const float alpha, const float* A, const float* B, const float beta,
		float* C) {
		int lda = (TransA == CblasNoTrans) ? K : M;
		int ldb = (TransB == CblasNoTrans) ? N : K;
		cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
			ldb, beta, C, N);
	}
	
	 void dlm_cpu_gemm(const CBLAS_TRANSPOSE TransA,
		 const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
		 const double alpha, const double* A, const double* B, const double beta,
		 double* C) {
		 int lda = (TransA == CblasNoTrans) ? K : M;
		 int ldb = (TransB == CblasNoTrans) ? N : K;
		 cblas_dgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
			 ldb, beta, C, N);
	 }


};

#endif			