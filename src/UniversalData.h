#ifndef SRC_UNIVERSALDATA_H
#define SRC_UNIVERSALDATA_H

#ifdef R_BUILD
#include <RcppEigen.h>
// [[Rcpp::depends(RcppEigen)]]
// struct ExternData {}; //TODO
#else
#include <Eigen/Eigen>
#include <pybind11/pybind11.h>
using ExternData = pybind11::object;
#include <memory>
#endif

#include <functional>
#include <autodiff/forward/dual.hpp>

using Eigen::VectorXd;
using Eigen::VectorXi;
using Eigen::MatrixXd;
using Eigen::Matrix;
using Eigen::Index; 
using Eigen::Map;
using autodiff::dual;
using autodiff::dual2nd;
using VectorXdual = Matrix<dual,-1,1>;
using VectorXdual2nd = Matrix<dual2nd,-1,1>;
using std::function;
using std::shared_ptr;

using optim_function = function<double(const VectorXd&, VectorXd*, void*)>;
using nlopt_function = double (*)(unsigned n, const double* x, double* grad, void* f_data);

class UniversalModel;
// UniversalData includes everything about the statistic model like loss, constraints and the statistic data like samples, operations of data.
// In abess project, UniversalData will be an instantiation of T4 in template class algorithm, other instantiation of T4 often is matrix.
// Thus, UniversalData is just like a matrix in algorithm, and its instance is often denoted as 'x'.
// In order to work like matrix, UniversalData need the help of utility function like X_seg, slice.
class UniversalData {
    // complete_para: the initial para
    // activate_para: this is a concept of abess algorithm, which is considered to have an impact on the model
    // inactivate_para: similar to activate_para
    // effective_para: IMPORTANT! this is a concept of class UniversalData, its meaning depends on the instance of class, 
    //                  and used to simulate the division operation of data sets.
    //                  Its size(effective_size) is match for (set_by_para ? the size of this->data : this->effective_para_index.size()).
private:
    UniversalModel* model;
    Index sample_size;
    double lambda = 0.;  // L2 penalty coef
    // model_size and effective_para_index are useful just when function 'set_by_para' of model isn't setted
    Index model_size; // length of complete_para
    VectorXi effective_para_index;// complete_para[effective_para_index[i]] = effective_para[i], ohter location of complete_para is 0
    Index effective_size; // set_by_para ? the size of this->data : length of effective_para_index
    shared_ptr<ExternData> data;
public:
    UniversalData() = default;
    UniversalData(Index model_size, Index sample_size, ExternData& data, UniversalModel* model);
    UniversalData slice_by_para(const VectorXi& target_para_index);
    Index rows() const; // getter of sample_size
    Index cols() const; // getter of effective_para
    UniversalData slice_by_sample(const VectorXi& target_sample_index);
    optim_function get_optim_function(double lambda); // create a function which can be optimized by OptimLib 
    nlopt_function get_nlopt_function(double lambda); // create a function which can be optimized by nlopt
    double loss(const VectorXd& effective_para, const VectorXd& intercept, double lambda); // compute the loss with effective_para
    double gradient(const VectorXd& effective_para, const VectorXd& intercept, Map<VectorXd>& gradient, double lambda); // compute the gradient of effective_para
    void hessian(const VectorXd& effective_para, const VectorXd& intercept, VectorXd& gradient, MatrixXd& hessian, Index index, Index size, double lambda); // compute the hessian of sequence from index to (index+size-1) in effective_para                                                                                                                            
};

class UniversalModel{
    friend class UniversalData;
private:
    // size of para will be match for data
    function <double(VectorXd const& para, VectorXd const& intercept, ExternData const& data)> loss;
    function <dual(VectorXdual const& para, VectorXdual const& intercept, ExternData const& data)> gradient_autodiff;
    function <dual2nd(VectorXdual2nd const& para, VectorXdual2nd const& intercept, ExternData const& data)> hessian_autodiff;
    // only the derivative of intercept and para[compute_para_index[i]] need be computed, 
    // size of return will equal to the sum of compute_para_index and intercept. 
    // the derivative of intercept should be setted before para.
    function <VectorXd(VectorXd const& para, VectorXd const& intercept, ExternData const& data, VectorXi const& compute_para_index)> gradient_user_defined;
    // only the derivative of para[compute_para_index[i]] need be computed, size of gradient will equal to compute_para_index.
    // compute_para_index: compute_para[i] = para[compute_para_index[i]]
    // result need be setted in gradient IN-PLACE!
    function <void(VectorXd const& para, VectorXd const& intercept, ExternData const& data, VectorXi const& compute_para_index, VectorXd& gradient, MatrixXd& hessian)> hessian_user_defined;
    function <ExternData(ExternData const& old_data, VectorXi const& target_sample_index)> slice_by_sample;
    function <ExternData(ExternData const& old_data, VectorXi const& target_para_index)> slice_by_para;
    function <void(ExternData const* p)> deleter = [](ExternData const* p) { delete p; };
    //TODO: constraints
public:
    // register callback function
    void set_loss_of_model(function <double(VectorXd const&, VectorXd const&, ExternData const&)> const&);
    void set_gradient_autodiff(function <dual(VectorXdual const&, VectorXdual const&, ExternData const&)> const&);
    void set_hessian_autodiff(function <dual2nd(VectorXdual2nd const&, VectorXdual2nd const&, ExternData const&)> const&);
    void set_gradient_user_defined(function <VectorXd(VectorXd const&, VectorXd const&, ExternData const&, VectorXi const&)> const&);
    void set_hessian_user_defined(function <void(VectorXd const&, VectorXd const&, ExternData const&, VectorXi const&, VectorXd&, MatrixXd&)> const&);
    void set_slice_by_sample(function <ExternData(ExternData const&, VectorXi const&)> const&);
    void set_slice_by_para(function <ExternData(ExternData const&, VectorXi const&)> const&);
    void set_deleter(function <void(ExternData const&)> const&);
    void unset_slice_by_sample();
    void unset_slice_by_para();
    void unset_deleter();
};
#endif //SRC_UNIVERSALDATA_H
