#include "AlgorithmUniversal.h"

#ifdef R_BUILD
// [[Rcpp::depends(nloptr)]]
#include <nloptrAPI.h>
#else
#include <nlopt.h> 
#endif

using namespace std;
using namespace Eigen;

double abessUniversal::loss_function(UniversalData& active_data, MatrixXd& y, VectorXd& weights, VectorXd& active_para, VectorXd& intercept, VectorXi& A,
    VectorXi& g_index, VectorXi& g_size, double lambda) 
{
    return active_data.loss(active_para, intercept, lambda);
}

bool abessUniversal::primary_model_fit(UniversalData& active_data, MatrixXd& y, VectorXd& weights, VectorXd& active_para, VectorXd& intercept, double loss0,
    VectorXi& A, VectorXi& g_index, VectorXi& g_size) 
{
    SPDLOG_DEBUG("optimization\ninit loss: {}\nintercept:{}\ncoefficient:{}", loss0, intercept.transpose(), active_para.transpose());    
    double value = 0.;
    unsigned optim_size = active_para.size() + intercept.size();
    VectorXd optim_para(optim_size);
    optim_para.head(intercept.size()) = intercept;
    optim_para.tail(active_para.size()) = active_para;
    nlopt_function f = active_data.get_nlopt_function(this->lambda_level);

    nlopt_opt opt = nlopt_create(NLOPT_LD_LBFGS, optim_size);
    nlopt_set_min_objective(opt, f, &active_data);
    nlopt_result result = nlopt_optimize(opt, optim_para.data(), &value); // positive return values means success
    nlopt_destroy(opt);

    bool success = result > 0;
    if(!success){
        SPDLOG_WARN("failed to optimize, state: {} ", nlopt_result_to_string(result));
    }
    intercept = optim_para.head(intercept.size());
    active_para = optim_para.tail(active_para.size());
    SPDLOG_DEBUG("optimization\nfinal loss: {}\nintercept:{}\ncoefficient:{}", value, intercept.transpose(), active_para.transpose());
    return success;
}

void abessUniversal::sacrifice(UniversalData& data, UniversalData& XA, MatrixXd& y, VectorXd& para, VectorXd& beta_A, VectorXd& intercept, VectorXi& A, VectorXi& I, VectorXd& weights, VectorXi& g_index, VectorXi& g_size, int g_num, VectorXi& A_ind, VectorXd& sacrifice, VectorXi& U, VectorXi& U_ind, int num)
{
    for (int i = 0; i < A.size(); i++) {
        VectorXd gradient_group(g_size(A[i]));
        MatrixXd hessian_group(g_size(A[i]), g_size(A[i]));
        data.hessian(para, intercept, gradient_group, hessian_group, g_index(A[i]), g_size(A[i]), this->lambda_level);
        if (g_size(A[i]) == 1) { // optimize for frequent degradation situations
            sacrifice(A[i]) = para(g_index(A[i])) * para(g_index(A[i])) * hessian_group(0, 0);
        }
        else {
            sacrifice(A[i]) = para.segment(g_index(A[i]), g_size(A[i])).transpose() * hessian_group * para.segment(g_index(A[i]), g_size(A[i]));
            sacrifice(A[i]) /= g_size(A[i]);
        }
    }
    for (int i = 0; i < I.size(); i++) {
        VectorXd gradient_group(g_size(I[i]));
        MatrixXd hessian_group(g_size(I[i]), g_size(I[i]));
        data.hessian(para, intercept, gradient_group, hessian_group, g_index(I[i]), g_size(I[i]), this->lambda_level);
        if (g_size(I[i]) == 1) { // Optimize for degradation situations, it often happens
            if (hessian_group(0, 0) < this->enough_small) {
                SPDLOG_ERROR("hessian is not positive definite:\n{}", hessian_group(0, 0));
                sacrifice(I[i]) = DBL_MAX; // TODO
            }
            else {
                sacrifice(I[i]) = gradient_group(0, 0) * gradient_group(0, 0) / hessian_group(0, 0);
            }
        }
        else {
            LLT<MatrixXd> hessian_group_llt(hessian_group);
            if (hessian_group_llt.info() == NumericalIssue){
                SPDLOG_ERROR("hessian is not positive definite:\n{}", hessian_group);
                sacrifice(I[i]) = DBL_MAX; // TODO
            }
            else{
                MatrixXd inv_hessian_group = hessian_group_llt.solve(MatrixXd::Identity(g_size(i), g_size(i)));
                sacrifice(I[i]) = gradient_group.transpose() * inv_hessian_group * gradient_group;
                sacrifice(I[i]) /= g_size(I[i]);
            }
        }
    }
}

double abessUniversal::effective_number_of_parameter(UniversalData& X, UniversalData& active_data, MatrixXd& y, VectorXd& weights, VectorXd& beta, VectorXd& active_para, VectorXd& intercept)
{
    if (this->lambda_level == 0.) return active_data.cols();

    if (active_data.cols() == 0) return 0.;

    MatrixXd hessian(active_data.cols(), active_data.cols());
    VectorXd g;
    active_data.hessian(active_para, intercept, g, hessian, 0, active_data.cols(), this->lambda_level);
    SelfAdjointEigenSolver<MatrixXd> adjoint_eigen_solver(hessian, EigenvaluesOnly);
    double enp = 0.;
    for (int i = 0; i < adjoint_eigen_solver.eigenvalues().size(); i++) {
        enp += adjoint_eigen_solver.eigenvalues()(i) / (adjoint_eigen_solver.eigenvalues()(i) + this->lambda_level);
    }
    return enp;
}

VectorXi abessUniversal::inital_screening(UniversalData &data, MatrixXd &y, VectorXd &init_para, VectorXd &init_intercept, VectorXi &init_active_set, VectorXi &I,VectorXd &sacrifice, VectorXd &weights, VectorXi &group_index, VectorXi &group_size, int &groups)
{
    if (sacrifice.size() == 0) {
        sacrifice = VectorXd::Zero(groups);
        for (int i = 0; i < groups; i++) {
            //sacrifice(i) = "optimal_para".segment(group_index(i), group_size(i)).squaredNorm();
            VectorXd active_para = init_para.segment(group_index(i), group_size(i));
            UniversalData active_data = data.slice_by_para(VectorXi::LinSpaced(group_size(i), group_index(i), group_size(i) + group_index(i) - 1));
            primary_model_fit(active_data, y, weights, active_para, init_intercept, 0, init_active_set, group_index, group_size);
            sacrifice(i) = active_para.squaredNorm();
        }
        // A_init
        for (int i = 0; i < init_active_set.size(); i++) {
            sacrifice(init_active_set(i)) = DBL_MAX - 1;
        }
        // alway_select
        for (int i = 0; i < this->always_select.size(); i++) {
            sacrifice(this->always_select(i)) = DBL_MAX;
        }
    }
    // get Active-set A according to max_k sacrifice
    Eigen::VectorXi A_new = max_k(sacrifice, this->sparsity_level);
    SPDLOG_DEBUG("init active set is :\n{}", A_new.transpose());
    return A_new;
}
