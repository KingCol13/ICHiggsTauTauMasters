#include <iostream>
#include <math.h>

#include "TFile.h"
#include "TKey.h"
#include "TTreeReader.h"
#include "TTreeReaderValue.h"
#include "Math/Vector4D.h"
#include "TVector3.h"
#include "TMatrixD.h"
#include "TVectorD.h"

/*

Compile:
g++ likelihoodMethod.cpp `root-config --libs --glibs --cflags --ldflags --auxlibs --auxcflags` -std=c++17 -o likelihoodMethod

modify path after -I and -L for root Include and lib directories accordingly
linker libs:
Core
RIO
Tree
TreePlayer

Find neutrino by maximising likelihood function.

*/

double normal_probability(double x, double mu, double sigma)
{
	return std::exp(-0.5*std::pow((x-mu)/sigma,2))/(sigma*std::sqrt(2*M_PI));
}

double multivariate_normal_probability(TVectorD x, TVectorD mean, TMatrixD covariance)
{
	double det_cov;
	TMatrixD inverse_covariance = covariance.Invert(&det_cov);

	return std::exp(-0.5*(x-mean)*(inverse_covariance*(x-mean)))/std::sqrt(std::pow(2*M_PI, mean.GetNrows())*std::abs(det_cov));
}
/**
	Returns likelihood of params given the measured quantities and covariances
	TODO: SV (use theta, phi for direction?), impact parameter
*/
double likelihood(double params[6],
				  ROOT::Math::PxPyPzEVector vis_1,
				  ROOT::Math::PxPyPzEVector vis_2,
				  TVectorD met, TMatrixD met_cov,
				  TVectorD sv_1, TMatrixD sv_cov_1,
				  TVectorD sv_2, TMatrixD sv_cov_2,
				  TVectorD ip_1, TMatrixD ip_cov_1,
				  TVectorD ip_2, TMatrixD ip_cov_2
)
{
	// params[0,1,2] = x1, y1, z1
	// params[3,4,5] = x2, y2, z2
	ROOT::Math::PxPyPzEVector nu_1(params[0], params[1], params[2], 0);
	ROOT::Math::PxPyPzEVector nu_2(params[3], params[4], params[5], 0);

	ROOT::Math::PxPyPzEVector tau_1 = vis_1 + nu_1;
	ROOT::Math::PxPyPzEVector tau_2 = vis_2 + nu_2;

	ROOT::Math::PxPyPzEVector higgs = tau_1 + tau_2;

	double pred_m_higgs = higgs.M();
	double pred_m_tau_1 = tau_1.M();
	double pred_m_tau_2 = tau_2.M();

	ROOT::Math::PxPyPzEVector sum_nu = nu_1+nu_2;
	double pred_met_x = sum_nu.px();
	double pred_met_y = sum_nu.py();

	return normal_probability(pred_m_higgs, 125, 1);
}

int main()
{
	std::cout << "Hello world!" << std::endl;
	return 0;
	
	double vis_px_1, vis_py_1, vis_pz_1, vis_e_1;
	double vis_px_2, vis_py_2, vis_pz_2, vis_e_2;
	double metx, mety;
	float metCov00, metCov01, metCov10, metCov11;
	double ip_x_1, ip_y_1, ip_z_1;
	double ip_x_2, ip_y_2, ip_z_2;
	double ipcov00_1, ipcov01_1, ipcov02_1, ipcov10_1, ipcov11_1, ipcov12_1, ipcov20_1, ipcov21_1, ipcov22_1;
	double ipcov00_2, ipcov01_2, ipcov02_2, ipcov10_2, ipcov11_2, ipcov12_2, ipcov20_2, ipcov21_2, ipcov22_2;
	double sv_x_1, sv_y_1, sv_z_1;
	double sv_x_2, sv_y_2, sv_z_2;
	double svcov00_1, svcov01_1, svcov02_1, svcov10_1, svcov11_1, svcov12_1, svcov20_1, svcov21_1, svcov22_1;
	double svcov00_2, svcov01_2, svcov02_2, svcov10_2, svcov11_2, svcov12_2, svcov20_2, svcov21_2, svcov22_2;
}
