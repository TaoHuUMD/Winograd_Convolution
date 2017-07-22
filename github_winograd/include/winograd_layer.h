#ifndef WINOGRAD_LAYER_H
#define WINOGRAD_LAYER_H

#include <memory>
#include "winograd_kernel.h"
#include "tool.h"
//winograd for cpu inference

// default 3x3
const int KERNEL_SIZE = 3;

namespace WINOGRAD_KERNEL
{

	template <typename Dtype>
	class WinogradLayer {

	private:

		int m_group_;
		int m_batchSize;

		int m_bottom_dim_;// par size
		int m_top_dim_;

		// The following variables are initialized in WeightAlign
		int tile_h_in_, tile_w_in_; /* input tile size */
		int tile_h_out_, tile_w_out_; /* output tile size */
		int ntiles_h_, ntiles_w_; /* number of tiles */

		int conv_in_channels_; //ic
		int conv_out_channels_;//oc

		int m_iH;
		int m_iW;

		int m_oH;
		int m_oW;

		int m_kH;
		int m_kW;
		int m_sH;
		int m_sW;

		int m_pad;
		bool m_bias;

	private:

		Dtype* m_inputOrg;
		const Dtype* m_weightOrg;

		Dtype* m_winogradWeight; // support NCHW storage
		Dtype* m_winogradInput;

		Dtype* m_col_buff;//buffer

		WINOGRAD_ALG m_alg;

	public:

		WinogradLayer(WINOGRAD_ALG alg, int batch_size, int iH, int iW, int iC, int kH, int kW, int sH, int sW, int oC, int pad, bool bias = true) : m_alg(alg) {

#if DEBUG_WINOGRAD
			assert(kH == kW, "kernel 3x3 is the best choice, some errors may occur for other kernels");
#endif
			m_iH = iH;
			m_iW = iW;
			conv_in_channels_ = iC;
			m_kH = kH;
			m_kW = kW;
			m_sH = sH;
			m_sW = sW;
			conv_out_channels_ = oC;
			m_pad = pad; // pad_h = pad_w
			m_bias = bias;

			m_batchSize = batch_size;
			m_group_ = 1;

			m_bottom_dim_ = 0;// default batch =1
			m_top_dim_ = 0;

			m_winogradWeight = NULL;
			m_winogradInput = NULL;


			// Output width.
			m_oW = (m_iW + m_pad * 2 - m_kW) / m_sW + 1;
			m_oH = (m_iH + m_pad * 2 - m_kH) / m_sH + 1;

			if (alg == WT_8X8_F_6X6_3X3) {

				tile_h_in_ = 8;
				tile_w_in_ = 8; /* input tile size */

				tile_h_out_ = tile_h_in_ - m_kH + 1;
				tile_w_out_ = tile_w_in_ - m_kW + 1; /* output tile size */

				ntiles_h_ = (PUBLIC_TOOL::max(m_iH + m_pad - tile_h_in_ + 1, m_oH) + tile_h_out_ - 1) / tile_h_out_;
				ntiles_w_ = (PUBLIC_TOOL::max(m_iW + m_pad - tile_w_in_ + 1, m_oW) + tile_w_out_ - 1) / tile_w_out_;

			}
			else if (alg == WT_6X6_F_4X4_3X3) {

				tile_h_in_ = 6;
				tile_w_in_ = 6; /* input tile size */

				tile_h_out_ = tile_h_in_ - m_kH + 1;
				tile_w_out_ = tile_w_in_ - m_kW + 1; /* output tile size */

				ntiles_h_ = (PUBLIC_TOOL::max(m_iH + m_pad - tile_h_in_ + 1, m_oH) + tile_h_out_ - 1) / tile_h_out_;
				ntiles_w_ = (PUBLIC_TOOL::max(m_iW + m_pad - tile_w_in_ + 1, m_oW) + tile_w_out_ - 1) / tile_w_out_;

			}
			else throw("convolution algorithm error!");

		}

		template <typename Dtype>
		const std::shared_ptr<Dtype> get_inference_cpu(Dtype* data, const Dtype* par, Dtype* col_buff) {

			m_inputOrg = data;
			m_weightOrg = par;
			m_col_buff = col_buff;


			std::shared_ptr<Dtype> resOut = std::shared_ptr<Dtype>(new Dtype[m_oH*m_oW*conv_out_channels_]);

			//trans weight to winograd domain
			trans_weight2wiongrad();


			for (int n = 0; n < m_batchSize; n++) {

				//trans input to winograd domain
				trans_input2winograd(m_inputOrg + n*m_bottom_dim_, m_col_buff);


				// Convolution in Winograd domain
				winograd_conv();


				// Transform back to time domain	
				trans2spatial(resOut.get() + n*this->m_top_dim_);

				//bias
				if (this->m_bias) {

					int base = conv_in_channels_ * conv_out_channels_ * m_kW * m_kH;

					const Dtype* bias = &par[base];

					this->forward_cpu_bias(resOut.get() + n * this->m_top_dim_, bias);
				}
			}

			return  resOut;
		}


	public:
		~WinogradLayer() {
			/*if (!m_winogradInput) delete[] m_winogradInput;
			if (!m_winogradWeight) delete[] m_winogradWeight;*/
		}


	private:

		void trans_weight2wiongrad() {// weight: hwcn --> cn hw

			// transform weights to Winograd domain
			if (!m_winogradWeight) m_winogradWeight = new Dtype[conv_in_channels_*conv_out_channels_*tile_h_in_*tile_w_in_];

			PUBLIC_TOOL::dlm_cpu_gemm(CblasNoTrans, CblasTrans,
				tile_h_in_*tile_w_in_, (conv_in_channels_ / m_group_)*conv_out_channels_, m_kH*m_kW,
				(Dtype)1,
				Winograd_Kron::getInstance(m_alg, WINOGRAD_G)->get().get(),
				m_weightOrg,
				(Dtype)0,
				m_winogradWeight);			

		}

		template <typename Dtype>
		void trans_input2winograd(const Dtype *data, Dtype *col_buff) {
			// Transform input to Winograd domain

			winograd_input_im2col_cpu(data, col_buff);


			int M = this->conv_in_channels_*ntiles_h_*ntiles_w_;

			if (!m_winogradInput) m_winogradInput = new Dtype[tile_h_in_*tile_w_in_*M];

			PUBLIC_TOOL::dlm_cpu_gemm(CblasTrans, CblasTrans,
				tile_h_in_*tile_w_in_, M, tile_h_in_*tile_w_in_,
				(Dtype)1,
				Winograd_Kron::getInstance(m_alg, WINOGRAD_B)->get().get(),
				col_buff,
				(Dtype)0, this->m_winogradInput);

		}

		void winograd_conv() {

			// Convolution in Winograd domain
			for (int j = 0; j < tile_h_in_*tile_w_in_; ++j) {
				for (int g = 0; g < this->m_group_; ++g) {
					PUBLIC_TOOL::dlm_cpu_gemm(CblasNoTrans, CblasNoTrans,
						this->conv_out_channels_ / this->m_group_, ntiles_h_*ntiles_w_, this->conv_in_channels_ / this->m_group_,
						(Dtype)1,
						m_winogradWeight + (j*this->m_group_ + g)*(this->conv_out_channels_ / this->m_group_)*(this->conv_in_channels_ / this->m_group_),
						m_winogradInput + (j*this->m_group_ + g)*(this->conv_in_channels_ / this->m_group_)*ntiles_h_*ntiles_w_,
						(Dtype)0, m_col_buff + (j*this->m_group_ + g)*(this->conv_out_channels_ / this->m_group_)*ntiles_h_*ntiles_w_);
				}
			}
			// col_buff has (tile_h_in*tile_w_in) x (conv_out_channels) x (ntiles_h*ntiles_w)
	 }

		template <typename Dtype>
		void trans2spatial(Dtype *data) {

			Dtype *winogradRes = new Dtype[this->conv_out_channels_*ntiles_h_*ntiles_w_*tile_h_out_*tile_w_out_];

			PUBLIC_TOOL::dlm_cpu_gemm(CblasTrans, CblasNoTrans,
				this->conv_out_channels_*ntiles_h_*ntiles_w_, tile_h_out_*tile_w_out_, tile_h_in_*tile_w_in_,
				(Dtype)1, m_col_buff,
				Winograd_Kron::getInstance(m_alg, WINOGRAD_A)->get().get(),
				(Dtype)0, winogradRes);

			winograd_output_col2im_cpu(winogradRes, data);

			delete[] winogradRes;
		}

		template<typename Dtype>
		void winograd_input_im2col_cpu(const Dtype *data, Dtype *col_buff)
		{
			int height = m_iH;
			int width = m_iW;
			int pad_h = m_pad, pad_w = m_pad;

			for (int c = 0; c < this->conv_in_channels_; ++c) {
				for (int tile_h = 0; tile_h < ntiles_h_; ++tile_h) {
					for (int tile_w = 0; tile_w < ntiles_w_; ++tile_w) {
						
						
						
						for (int y = 0; y < tile_h_in_; ++y) {
							for (int x = 0; x < tile_w_in_; ++x) {
								int in_y = tile_h*tile_h_out_ + y - pad_h;
								int in_x = tile_w*tile_w_out_ + x - pad_w;

								if (in_y < 0 || in_x < 0 || in_y >= height || in_x >= width) {
									col_buff[(((c*ntiles_h_ + tile_h)*ntiles_w_ + tile_w)*tile_h_in_ + y)*tile_w_in_ + x] = 0;
								}
								else {
									col_buff[(((c*ntiles_h_ + tile_h)*ntiles_w_ + tile_w)*tile_h_in_ + y)*tile_w_in_ + x] =
										data[(c*height + in_y)*width + in_x];
								}
							}
						}


					} // for each tile
				} // for each tile
			} // for each input channel
		}


		template <typename Dtype>
		void forward_cpu_bias(Dtype* output,
			const Dtype* bias) {

			int out_spatial_dim_ = m_oH * m_oW;

			for (int i = 0; i < conv_out_channels_; i++) {

				for (int j = 0; j < out_spatial_dim_; j++)
					output[i*out_spatial_dim_ + j] += bias[i];

			}

			//dlm_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_,
			//	out_spatial_dim_, 1, (Dtype)1., bias, bias_multiplier_.cpu_data(),
			//	(Dtype)1., output);
		}


		template<typename Dtype>
		void winograd_output_col2im_cpu(const Dtype *col_buff, Dtype *data)
		{
			const int output_h = m_iH, output_w = m_iW;

			for (int n = 0; n < this->conv_out_channels_; ++n) {
				for (int tile_h = 0; tile_h < ntiles_h_; ++tile_h) {
					for (int tile_w = 0; tile_w < ntiles_w_; ++tile_w) {
						for (int y = 0; y < tile_h_out_; ++y) {
							for (int x = 0; x < tile_w_out_; ++x) {
								int out_y = tile_h*tile_h_out_ + y;
								int out_x = tile_w*tile_w_out_ + x;

								if (out_y < output_h && out_x < output_w) {

									/*int  kk = 0;
									if ((((n*ntiles_h_ + tile_h)*ntiles_w_ + tile_w)*tile_h_out_ + y)*tile_w_out_ + x == 184604)
										kk++;

									cout << "dat: "<<(n*output_h + out_y)*output_w + out_x << " col : " << (((n*ntiles_h_ + tile_h)*ntiles_w_ + tile_w)*tile_h_out_ + y)*tile_w_out_ + x << endl;*/

									data[(n*output_h + out_y)*output_w + out_x] =
										col_buff[(((n*ntiles_h_ + tile_h)*ntiles_w_ + tile_w)*tile_h_out_ + y)*tile_w_out_ + x];
								}
							}
						}
					} // for each tile
				} // for each tile
			} // for each input channel
		}

	};
}

#endif