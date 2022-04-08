#include "AlgorithmUniversal.h"

// #include "OptimLib/optim.hpp"
#ifdef R_BUILD
// [[Rcpp::depends(nloptr)]]
#include <nloptrAPI.h>
#else
#include <nlopt.h> // TODO: need to rewrite into python version
#endif
#include <stdio.h>
using namespace std;
using namespace Eigen;
// using namespace optim;

double abessUniversal::loss_function(UniversalData& active_data, VectorXd& y, VectorXd& weights, VectorXd& active_para, double& coef0, VectorXi& A,
    VectorXi& g_index, VectorXi& g_size, double lambda) 
{
    return active_data.loss(active_para, lambda);
}

bool abessUniversal::primary_model_fit(UniversalData& active_data, VectorXd& y, VectorXd& weights, VectorXd& active_para, double& coef0, double loss0,
    VectorXi& A, VectorXi& g_index, VectorXi& g_size) 
{    
    static int nlopt_run_times = 1;
    unsigned active_para_size = active_para.size();
    double value = 0.;
    nlopt_function f = active_data.get_nlopt_function(this->lambda_level);

    nlopt_opt opt = nlopt_create(NLOPT_LD_LBFGS, active_para_size);
    nlopt_set_min_objective(opt, f, &active_data);
    int result = nlopt_optimize(opt, active_para.data(), &value); // positive return values means success
    if (nlopt_run_times == 1) { // just for debug
        printf("nlopt's algorithm is %s\n", nlopt_algorithm_name(nlopt_get_algorithm(opt)));
    }
    nlopt_destroy(opt);

    printf("nlopt has run %d times\n", nlopt_run_times++);
    return result;
    
    //return true;

}

void abessUniversal::sacrifice(UniversalData& data, UniversalData& XA, VectorXd& y, VectorXd& para, VectorXd& beta_A, double& coef0, VectorXi& A, VectorXi& I, VectorXd& weights, VectorXi& g_index, VectorXi& g_size, int g_num, VectorXi& A_ind, VectorXd& sacrifice, VectorXi& U, VectorXi& U_ind, int num)
{
    for (int i = 0; i < A.size(); i++) {
        VectorXd gradient_group(g_size(A[i]));
        MatrixXd hessian_group(g_size(A[i]), g_size(A[i]));
        data.hessian(para, gradient_group, hessian_group, g_index(A[i]), g_size(A[i]), this->lambda_level);
        if (g_size(A[i]) == 1) {
            // optimize for frequent degradation situations
            sacrifice(A[i]) = beta(g_index(A[i])) * beta(g_index(A[i])) * hessian_group(0, 0);
        }
        else {
            sacrifice(A[i]) = (para.segment(g_index(A[i]), g_size(A[i])).transpose() * hessian_group * para.segment(g_index(A[i]), g_size(A[i])) / g_size(A[i])).eval()(0,0);
        }
    }
    for (int i = 0; i < I.size(); i++) {
        VectorXd gradient_group(g_size(I[i]));
        MatrixXd hessian_group(g_size(I[i]), g_size(I[i]));
        data.hessian(para, gradient_group, hessian_group, g_index(I[i]), g_size(I[i]), this->lambda_level);
        if (g_size(I[i]) == 1) {
            // Optimize for degradation situations, it often happens
            if (hessian_group(0, 0) < this->enough_small) {
                cout << "hessian is not positive definite!"; 
                sacrifice(I[i]) = gradient_group(0, 0) * gradient_group(0, 0) / this->enough_small;
            }
            else {
                sacrifice(I[i]) = gradient_group(0, 0) * gradient_group(0, 0) / hessian_group(0, 0);
            }
        }
        else {
            //? TODO: hessian may be not positive definite
            MatrixXd inv_hessian_group = hessian_group.ldlt().solve(MatrixXd::Identity(g_size(i), g_size(i)));
            sacrifice(I[i]) = (gradient_group.transpose() * inv_hessian_group * gradient_group / g_size(A[i])).eval()(0, 0);
        }
    }
}

double abessUniversal::effective_number_of_parameter(UniversalData& X, UniversalData& active_data, VectorXd& y, VectorXd& weights, VectorXd& beta, VectorXd& active_para, double& coef0)
{
    if (this->lambda_level == 0.) return active_data.cols();

    if (active_data.cols() == 0) return 0.;

    MatrixXd hessian(active_data.cols(), active_data.cols());
    active_data.hessian(active_para, hessian, this->lambda_level); //? TODO: need lambda or not?
    SelfAdjointEigenSolver<MatrixXd> adjoint_eigen_solver(hessian);
    double enp = 0.;
    for (int i = 0; i < adjoint_eigen_solver.eigenvalues().size(); i++) {
        enp += adjoint_eigen_solver.eigenvalues()(i) / (adjoint_eigen_solver.eigenvalues()(i) + this->lambda_level);
    }
    return enp;
}


