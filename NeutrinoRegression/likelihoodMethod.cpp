#include <iostream>
#include <math.h>

#include <gsl/gsl_multimin.h>
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
g++ likelihoodMethod.cpp `root-config --libs --glibs --cflags --ldflags --auxlibs --auxcflags` -lgsl -std=c++17 -o likelihoodMethod

modify path after -I and -L for root Include and lib directories accordingly
linker libs:
Core
RIO
Tree
TreePlayer

Find neutrino by maximising likelihood function.

*/

struct Params
{
	ROOT::Math::PxPyPzEVector vis_1;
	ROOT::Math::PxPyPzEVector vis_2;
	TVectorD met;
	TMatrixD met_cov;
	TVectorD sv_1;
	TMatrixD sv_cov_1;
	TVectorD sv_2;
	TMatrixD sv_cov_2;
	TVectorD ip_1;
	TMatrixD ip_cov_1;
	TVectorD ip_2;
	TMatrixD ip_cov_2;
};

struct Particle
{
	double px;
	double py;
	double pz;
	double E;
};

/** Function to set particle data to be read from a_tree */
void setupParticle(TTree *a_tree, std::string a_name, Particle &a_particle, int a_fromTau)
{
	a_tree->SetBranchAddress((a_name+"_px_"+std::to_string(a_fromTau)).c_str(), &a_particle.px);
	a_tree->SetBranchAddress((a_name+"_py_"+std::to_string(a_fromTau)).c_str(), &a_particle.py);
	a_tree->SetBranchAddress((a_name+"_pz_"+std::to_string(a_fromTau)).c_str(), &a_particle.pz);
	a_tree->SetBranchAddress((a_name+"_E_"+std::to_string(a_fromTau)).c_str(), &a_particle.E);
}

ROOT::Math::PxPyPzEVector makeParticle(Particle part)
{
	return ROOT::Math::PxPyPzEVector(part.px, part.py, part.pz, part.E);
}

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

double log_normal_probability(double x, double mu, double sigma)
{
	return -0.5*std::pow((x-mu)/sigma,2) - std::log(sigma) - 0.5*std::log(2*M_PI);
}

double log_multivariate_normal_probability(TVectorD x, TVectorD mean, TMatrixD covariance)
{
	double det_cov;
	TMatrixD inverse_covariance = covariance.Invert(&det_cov);
	
	return -0.5*(x-mean)*(inverse_covariance*(x-mean)) - 0.5*std::log(det_cov) - 0.5*mean.GetNrows()*std::log(2*M_PI);
}

void svToDirection(TVectorD &dir, TMatrixD &dir_cov, TVectorD &sv, TMatrixD &sv_cov)
{
	double x = sv(0);
	double y = sv(1);
	double z = sv(2);
	
	dir(0) = std::atan(std::sqrt(x*x+y*y)/z);
	dir(1) = std::atan(y/x);
	
	double rho_sqr = x*x+y*y;
	double r_sqr = rho_sqr+z*z;
	dir_cov(0,0) = ((z*z*x*x*sv_cov(0,0)+z*z+y*y*sv_cov(1,1))/rho_sqr + rho_sqr*sv_cov(2,2))/(r_sqr*r_sqr);
	dir_cov(1,1) = (y*y*sv_cov(0,0) + x*x*sv_cov(1,1))/rho_sqr;
	//TODO: better than setting off-diagonals to 0
	dir_cov(0,1) = 0;
	dir_cov(1,0) = 0;
}

/**
	Returns likelihood of params given the measured quantities and covariances
	TODO: SV (use theta, phi for direction?), impact parameter
*/
double likelihood(double nu_p_vec[6],
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
	ROOT::Math::PxPyPzEVector nu_1(nu_p_vec[0], nu_p_vec[1], nu_p_vec[2], std::sqrt(nu_p_vec[0]*nu_p_vec[0] + nu_p_vec[1]*nu_p_vec[1] + nu_p_vec[2]*nu_p_vec[2]));
	ROOT::Math::PxPyPzEVector nu_2(nu_p_vec[3], nu_p_vec[4], nu_p_vec[5], std::sqrt(nu_p_vec[3]*nu_p_vec[3] + nu_p_vec[4]*nu_p_vec[4] + nu_p_vec[5]*nu_p_vec[5]));

	ROOT::Math::PxPyPzEVector tau_1 = vis_1 + nu_1;
	ROOT::Math::PxPyPzEVector tau_2 = vis_2 + nu_2;

	ROOT::Math::PxPyPzEVector higgs = tau_1 + tau_2;

	double pred_m_higgs = higgs.M();
	double pred_m_tau_1 = tau_1.M();
	double pred_m_tau_2 = tau_2.M();

	ROOT::Math::PxPyPzEVector sum_nu = nu_1+nu_2;
	TVectorD pred_met(2);
	pred_met(0) = sum_nu.px();
	pred_met(1) = sum_nu.py();
	
	double total = 0;
	// Masses
	total -= log_normal_probability(pred_m_higgs, 125, 1);
	total -= log_normal_probability(pred_m_tau_1, 1.777, 0.1);
	total -= log_normal_probability(pred_m_tau_2, 1.777, 0.1);
	
	// met
	total -= log_multivariate_normal_probability(pred_met, met, met_cov);
	
	// secondary vertex
	// TODO: work out if this is the best way
	TVectorD pred_dir_1(2);
	pred_dir_1(0) = std::atan(std::sqrt(tau_1.Px()*tau_1.Px()+tau_1.Py()*tau_1.Py())/tau_1.Pz());
	pred_dir_1(1) = std::atan(tau_1.Py()/tau_1.Px());
	TVectorD dir_1(2);
	TMatrixD dir_cov_1(2,2);
	svToDirection(dir_1, dir_cov_1, sv_1, sv_cov_1);
	
	TVectorD pred_dir_2(2);
	pred_dir_2(0) = std::atan(std::sqrt(tau_2.Px()*tau_2.Px()+tau_2.Py()*tau_2.Py())/tau_2.Pz());
	pred_dir_2(1) = std::atan(tau_2.Py()/tau_2.Px());
	TVectorD dir_2(2);
	TMatrixD dir_cov_2(2,2);
	svToDirection(dir_2, dir_cov_2, sv_2, sv_cov_2);
	
	//total -= log_multivariate_normal_probability(pred_dir_1, dir_1, dir_cov_1);
	//total -= log_multivariate_normal_probability(pred_dir_2, dir_2, dir_cov_2);
	
	return total;
}

double gsl_my_f(const gsl_vector *nu_p_gsl_vec, void *params)
{
	Params *myP = (Params *) params;
	double nu_p_vec[6];
	for(unsigned int i=0; i<6; i++)
	{
		nu_p_vec[i] = gsl_vector_get(nu_p_gsl_vec, i);
	}
	return likelihood(nu_p_vec,
										myP->vis_1,
										myP->vis_2,
										myP->met, myP->met_cov,
				  					myP->sv_1, myP->sv_cov_1,
				  					myP->sv_2, myP->sv_cov_2,
				  					myP->ip_1, myP->ip_cov_1,
				  					myP->ip_2, myP->ip_cov_2
									 );
}

int main()
{
	TFile file("ROOTfiles/MVAFILE_AllHiggs_tt.root", "READ");
	if (file.IsZombie())
	{
		std::cerr << "File didn't load correctly." << std::endl;
		return -1;
	}
	TTree *tree = static_cast<TTree*>(file.Get("ntuple"));
	
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
	int mva_dm_1, mva_dm_2;
	
	tree->SetBranchAddress("mva_dm_1", &mva_dm_1);
	tree->SetBranchAddress("mva_dm_2", &mva_dm_2);
	
	// Setup particles
	Particle pi_1, pi2_1, pi3_1, pi0_1;
	Particle pi_2, pi2_2, pi3_2, pi0_2;
	setupParticle(tree, "pi", pi_1, 1);
	setupParticle(tree, "pi", pi_2, 2);
	setupParticle(tree, "pi2", pi2_1, 1);
	setupParticle(tree, "pi2", pi2_2, 2);
	setupParticle(tree, "pi3", pi3_1, 1);
	setupParticle(tree, "pi3", pi3_2, 2);
	setupParticle(tree, "pi0", pi0_1, 1);
	setupParticle(tree, "pi0", pi0_2, 2);
	
	tree->SetBranchAddress("metx", &metx);
	tree->SetBranchAddress("metx", &metx);
	
	tree->SetBranchAddress("metx", &metx);
	tree->SetBranchAddress("mety", &mety);
	
	tree->SetBranchAddress("metcov00", &metCov00);
	tree->SetBranchAddress("metcov01", &metCov01);
	tree->SetBranchAddress("metcov10", &metCov10);
	tree->SetBranchAddress("metcov11", &metCov11);
	
	tree->SetBranchAddress("ip_x_1", &ip_x_1);
	tree->SetBranchAddress("ip_y_1", &ip_y_1);
	tree->SetBranchAddress("ip_z_1", &ip_z_1);
	tree->SetBranchAddress("ip_x_2", &ip_x_2);
	tree->SetBranchAddress("ip_y_2", &ip_y_2);
	tree->SetBranchAddress("ip_z_2", &ip_z_2);
	
	tree->SetBranchAddress("ipcov00_1", &ipcov00_1);
	tree->SetBranchAddress("ipcov01_1", &ipcov01_1);
	tree->SetBranchAddress("ipcov02_1", &ipcov02_1);
	tree->SetBranchAddress("ipcov10_1", &ipcov10_1);
	tree->SetBranchAddress("ipcov11_1", &ipcov11_1);
	tree->SetBranchAddress("ipcov12_1", &ipcov12_1);
	tree->SetBranchAddress("ipcov20_1", &ipcov20_1);
	tree->SetBranchAddress("ipcov21_1", &ipcov21_1);
	tree->SetBranchAddress("ipcov22_1", &ipcov22_1);

	tree->SetBranchAddress("ipcov00_2", &ipcov00_2);
	tree->SetBranchAddress("ipcov01_2", &ipcov01_2);
	tree->SetBranchAddress("ipcov02_2", &ipcov02_2);
	tree->SetBranchAddress("ipcov10_2", &ipcov10_2);
	tree->SetBranchAddress("ipcov11_2", &ipcov11_2);
	tree->SetBranchAddress("ipcov12_2", &ipcov12_2);
	tree->SetBranchAddress("ipcov20_2", &ipcov20_2);
	tree->SetBranchAddress("ipcov21_2", &ipcov21_2);
	tree->SetBranchAddress("ipcov22_2", &ipcov22_2);
	
	tree->SetBranchAddress("sv_x_1", &sv_x_1);
	tree->SetBranchAddress("sv_y_1", &sv_y_1);
	tree->SetBranchAddress("sv_z_1", &sv_z_1);
	tree->SetBranchAddress("sv_x_2", &sv_x_2);
	tree->SetBranchAddress("sv_y_2", &sv_y_2);
	tree->SetBranchAddress("sv_z_2", &sv_z_2);
	
	tree->SetBranchAddress("svcov00_1", &svcov00_1);
	tree->SetBranchAddress("svcov01_1", &svcov01_1);
	tree->SetBranchAddress("svcov02_1", &svcov02_1);
	tree->SetBranchAddress("svcov10_1", &svcov10_1);
	tree->SetBranchAddress("svcov11_1", &svcov11_1);
	tree->SetBranchAddress("svcov12_1", &svcov12_1);
	tree->SetBranchAddress("svcov20_1", &svcov20_1);
	tree->SetBranchAddress("svcov21_1", &svcov21_1);
	tree->SetBranchAddress("svcov22_1", &svcov22_1);
	
	tree->SetBranchAddress("svcov00_2", &svcov00_2);
	tree->SetBranchAddress("svcov01_2", &svcov01_2);
	tree->SetBranchAddress("svcov02_2", &svcov02_2);
	tree->SetBranchAddress("svcov10_2", &svcov10_2);
	tree->SetBranchAddress("svcov11_2", &svcov11_2);
	tree->SetBranchAddress("svcov12_2", &svcov12_2);
	tree->SetBranchAddress("svcov20_2", &svcov20_2);
	tree->SetBranchAddress("svcov21_2", &svcov21_2);
	tree->SetBranchAddress("svcov22_2", &svcov22_2);
	
	// Event loop
	//for (int i = 0, nEntries = tree->GetEntries(); i < nEntries; i++)
	for (int i = 0, nEntries = 1; i < nEntries; i++)
	{
		tree->GetEntry(i);
		
		TVectorD met(2);
		met(0) = metx;
		met(1) = mety;
		
		TMatrixD metCov(2,2);
		metCov(0,0) = metCov00;
		metCov(0,1) = metCov01;
		metCov(1,0) = metCov10;
		metCov(1,1) = metCov11;
		
		TVectorD ip_1(3);
		ip_1(0) = ip_x_1;
		ip_1(1) = ip_y_1;
		ip_1(2) = ip_z_1;
		
		TMatrixD ip_cov_1(3,3);
		ip_cov_1(0,0) = ipcov00_1;
		ip_cov_1(0,1) = ipcov01_1;
		ip_cov_1(0,2) = ipcov02_1;
		ip_cov_1(1,0) = ipcov10_1;
		ip_cov_1(1,1) = ipcov11_1;
		ip_cov_1(1,2) = ipcov12_1;
		ip_cov_1(2,0) = ipcov20_1;
		ip_cov_1(2,1) = ipcov21_1;
		ip_cov_1(2,2) = ipcov22_1;
		
		TVectorD ip_2(3);
		ip_2(0) = ip_x_2;
		ip_2(1) = ip_y_2;
		ip_2(2) = ip_z_2;
		
		TMatrixD ip_cov_2(3,3);
		ip_cov_2(0,0) = ipcov00_2;
		ip_cov_2(0,1) = ipcov01_2;
		ip_cov_2(0,2) = ipcov02_2;
		ip_cov_2(1,0) = ipcov10_2;
		ip_cov_2(1,1) = ipcov11_2;
		ip_cov_2(1,2) = ipcov12_2;
		ip_cov_2(2,0) = ipcov20_2;
		ip_cov_2(2,1) = ipcov21_2;
		ip_cov_2(2,2) = ipcov22_2;
		
		TVectorD sv_1(3);
		sv_1(0) = sv_x_1;
		sv_1(1) = sv_y_1;
		sv_1(2) = sv_z_1;
		
		TMatrixD sv_cov_1(3,3);
		sv_cov_1(0,0) = svcov00_1;
		sv_cov_1(0,1) = svcov01_1;
		sv_cov_1(0,2) = svcov02_1;
		sv_cov_1(1,0) = svcov10_1;
		sv_cov_1(1,1) = svcov11_1;
		sv_cov_1(1,2) = svcov12_1;
		sv_cov_1(2,0) = svcov20_1;
		sv_cov_1(2,1) = svcov21_1;
		sv_cov_1(2,2) = svcov22_1;
		
		TVectorD sv_2(3);
		sv_2(0) = sv_x_2;
		sv_2(1) = sv_y_2;
		sv_2(2) = sv_z_2;
		
		TMatrixD sv_cov_2(3,3);
		sv_cov_2(0,0) = svcov00_2;
		sv_cov_2(0,1) = svcov01_2;
		sv_cov_2(0,2) = svcov02_2;
		sv_cov_2(1,0) = svcov10_2;
		sv_cov_2(1,1) = svcov11_2;
		sv_cov_2(1,2) = svcov12_2;
		sv_cov_2(2,0) = svcov20_2;
		sv_cov_2(2,1) = svcov21_2;
		sv_cov_2(2,2) = svcov22_2;
		
		// Add up visible products for the first tau
		ROOT::Math::PxPyPzEVector vis_1 = makeParticle(pi_1);
		if (mva_dm_1 == 1 || mva_dm_1 == 2) // rho/pi+2pi0
		{
			vis_1+=makeParticle(pi0_1);
		}
		else if (mva_dm_1 == 10) // 3pi
		{
			vis_1+=makeParticle(pi2_1);
			vis_1+=makeParticle(pi3_1);
		}
		else if (mva_dm_1 == 11) // 3pi+pi0
		{
			vis_1+=makeParticle(pi2_1);
			vis_1+=makeParticle(pi3_1);
			vis_1+=makeParticle(pi0_1);
		}
		
		// Add up visible products for the second tau
		ROOT::Math::PxPyPzEVector vis_2 = makeParticle(pi_2);
		if (mva_dm_2 == 1 || mva_dm_2 == 2) // rho/pi+2pi0
		{
			vis_2+=makeParticle(pi0_2);
		}
		else if (mva_dm_2 == 10) // 3pi
		{
			vis_2+=makeParticle(pi2_2);
			vis_2+=makeParticle(pi3_2);
		}
		else if (mva_dm_2 == 11) // 3pi+pi0
		{
			vis_2+=makeParticle(pi2_2);
			vis_2+=makeParticle(pi3_2);
			vis_2+=makeParticle(pi0_2);
		}
		
		//double nu_vec[6] = {-7.37813482, 28.5213691 , 11.32233143, 28.69153254, 19.62560315, 39.85436348};
		
		Params par;
		par.vis_1 = vis_1;
		par.vis_2 = vis_2;
		par.met.ResizeTo(2);
		par.met = met;
		par.met_cov.ResizeTo(2,2);
		par.met_cov = metCov;
		par.sv_1.ResizeTo(3);
		par.sv_1 = sv_1;
		par.sv_cov_1.ResizeTo(3,3);
		par.sv_cov_1 = sv_cov_1;
		par.sv_2.ResizeTo(3);
		par.sv_2 = sv_2;
		par.sv_cov_2.ResizeTo(3,3);
		par.sv_cov_2 = sv_cov_2;
		par.ip_1.ResizeTo(3);
		par.ip_1 = ip_1;
		par.ip_cov_1.ResizeTo(3,3);
		par.ip_cov_1 = ip_cov_1;
		par.ip_2.ResizeTo(3);
		par.ip_2 = ip_2;
		par.ip_cov_2.ResizeTo(3,3);
		par.ip_cov_2 = ip_cov_2;
		
		const gsl_multimin_fminimizer_type *T = gsl_multimin_fminimizer_nmsimplex2;
		gsl_multimin_fminimizer *s = NULL;
		gsl_vector *step_size, *x;
		gsl_multimin_function minex_func;
		
		size_t iter = 0;
		int status;
		double size;
		
		x = gsl_vector_alloc (6);
		step_size = gsl_vector_alloc (6);
		
		//gsl_vector_set_all (x, 0);
		gsl_vector_set (x, 0, -7);
		gsl_vector_set (x, 1, 28);
		gsl_vector_set (x, 2, 11);
		gsl_vector_set (x, 3, 28);
		gsl_vector_set (x, 4, 19);
		gsl_vector_set (x, 5, 39);
		gsl_vector_set_all (step_size, 5.0);
		
		/* Initialize method and iterate */
		minex_func.n = 6;
		minex_func.f = gsl_my_f;
		minex_func.params = &par;
		
		s = gsl_multimin_fminimizer_alloc (T, 6);
		gsl_multimin_fminimizer_set (s, &minex_func, x, step_size);
		
		do
		{
			iter++;
			status = gsl_multimin_fminimizer_iterate(s);

			if (status)
				break;
			
			size = gsl_multimin_fminimizer_size (s);
			status = gsl_multimin_test_size (size, 1e-2);

			if (status == GSL_SUCCESS)
			{
				printf ("converged to minimum at\n");
			}
			
			std::cout << std::endl << "Iteration: " << iter << std::endl;
			std::cout << "x: ";
			for(unsigned int i=0; i<6; i++)
			{
				std::cout << gsl_vector_get(s->x, i) << ", ";
			}
			std::cout << std::endl;
			std::cout << "-ln(likelihood) = " << s->fval << std::endl;
		}
		while (status == GSL_CONTINUE && iter < 5);
		
		gsl_vector_free(x);
		gsl_vector_free(step_size);
		gsl_multimin_fminimizer_free (s);
		
		//std::cout << "Event: " << i << ", likelihood: " << likelihood(params, vis_1, vis_2, met, metCov, sv_1, sv_cov_1, sv_2, sv_cov_2, ip_1, ip_cov_1, ip_2, ip_cov_2) << std::endl;
	}
	
	return 0;
}
