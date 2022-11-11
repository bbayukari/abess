#include "UniversalData.h"
#include "utilities.h"
#include <autodiff/forward/dual/eigen.hpp>
#include <iostream>
using namespace std;
using Eigen::Map;
using Eigen::Matrix;

UniversalData::UniversalData(Eigen::Index model_size, Eigen::Index sample_size, ExternData& data, UniversalModel* model)
    : model(model), sample_size(sample_size), model_size(model_size), effective_size(model_size)
{
    if (!model->slice_by_para) {
        this->effective_para_index = VectorXi::LinSpaced(model_size, 0, model_size - 1);
    }
    this->data = shared_ptr<ExternData>(new ExternData(data));
}

UniversalData UniversalData::slice_by_para(const VectorXi& target_para_index)
{
    UniversalData tem(*this);
    if (model->slice_by_para) {
        tem.data = shared_ptr<ExternData>(new ExternData(model->slice_by_para(*data, target_para_index)), model->deleter);
    }
    else {
#if EIGEN_MAJOR_VERSION >= 4
        tem.effective_para_index = this->effective_para_index(target_para_index);
#else
        tem.effective_para_index = VectorXi(target_para_index.size());
        for (Eigen::Index i = 0; i < target_para_index.size(); i++) {
            tem.effective_para_index[i] = this->effective_para_index[target_para_index[i]];
        }
#endif
    }
    tem.effective_size = target_para_index.size();

    return tem;
}

UniversalData UniversalData::slice_by_sample(const VectorXi& target_sample_index)
{
    UniversalData tem(*this);
    tem.sample_size = target_sample_index.size();
    tem.data = shared_ptr<ExternData>(new ExternData(model->slice_by_sample(*data, target_sample_index)), model->deleter);
    return tem;
}

Eigen::Index UniversalData::cols() const
{
    return effective_size;
}

Eigen::Index UniversalData::rows() const
{
    return sample_size;
}

optim_function UniversalData::get_optim_function(double lambda)
{
    return [this,lambda](const VectorXd& effective_para_and_intercept, VectorXd* gradient, void* data) {
        VectorXd intercept = effective_para_and_intercept.head(effective_para_and_intercept.size() - this->effective_size);
        VectorXd effective_para = effective_para_and_intercept.tail(this->effective_size);
        if (gradient) {
            Map<VectorXd> grad(gradient->data(), gradient->size());
            return this->gradient(effective_para, intercept, grad, lambda);
        }
        else {
            return this->loss(effective_para, intercept, lambda);
        }
    };
}

nlopt_function UniversalData::get_nlopt_function(double lambda) 
{
    this->lambda = lambda;
    return [](unsigned n, const double* x, double* grad, void* f_data) {
        UniversalData* data = static_cast<UniversalData*>(f_data);
        Map<VectorXd const> intercept(x, n - data->effective_size);
        Map<VectorXd const> effective_para(x + n - data->effective_size, data->effective_size);
        if (grad) { // not use operator new
            Map<VectorXd> gradient(grad, n);
            return data->gradient(effective_para, intercept, gradient, data->lambda);
        }
        else {
            return data->loss(effective_para, intercept, data->lambda);
        }
    };
}

double UniversalData::loss(const VectorXd& effective_para, const VectorXd& intercept, double lambda)
{
    if(effective_para.size() != this->effective_size){
        SPDLOG_ERROR("the size of the effective parameters is different!");
        return DBL_MAX;
    }

    if (model->slice_by_para) {
        return model->loss(effective_para, intercept, *this->data) + lambda * effective_para.squaredNorm();
    }
    else {
        VectorXd complete_para = VectorXd::Zero(this->model_size);
#if EIGEN_MAJOR_VERSION >= 4
        complete_para(this->effective_para_index) = effective_para;
#else
        for (Eigen::Index i = 0; i < this->effective_size; i++) {
            complete_para[this->effective_para_index[i]] = effective_para[i];
        }
#endif
        return model->loss(complete_para, intercept, *this->data) + lambda * effective_para.squaredNorm();
    }
}

double UniversalData::gradient(const VectorXd& effective_para, const VectorXd& intercept, Map<VectorXd>& gradient, double lambda)
{
    if(effective_para.size() != this->effective_size){
        SPDLOG_ERROR("the size of the effective parameters is different!");
        return DBL_MAX;        
    }
    if(gradient.size() != this->effective_size + intercept.size()){
        SPDLOG_ERROR("Gradient Vector Size Wrong, the size of gradient is {}, effective_size is {}, intercept_size is {}", gradient.size(), effective_para.size(), intercept.size());
        return DBL_MAX;  
    }

    double value = 0.0;

    if (model->slice_by_para) {
        if (model->gradient_user_defined) {
            gradient = model->gradient_user_defined(effective_para, intercept, *this->data, VectorXi::LinSpaced(effective_size, 0, effective_size - 1));
            value = model->loss(effective_para, intercept, *this->data);
        }
        else {
            dual v;
            VectorXdual effective_para_dual = effective_para;
            VectorXdual intercept_dual = intercept;
            auto func = [this](VectorXdual const& compute_para, VectorXdual const& intercept) {
                return this->model->gradient_autodiff(compute_para, intercept, *this->data);
            };
            gradient.head(intercept.size()) = autodiff::gradient(func, wrt(intercept_dual), at(effective_para_dual, intercept_dual), v);
            gradient.tail(effective_size) = autodiff::gradient(func, wrt(effective_para_dual), at(effective_para_dual, intercept_dual), v);
            value = val(v);
        }
    }
    else {
        VectorXd complete_para = VectorXd::Zero(this->model_size);
#if EIGEN_MAJOR_VERSION >= 4
        complete_para(this->effective_para_index) = effective_para;
#else
        for (Eigen::Index i = 0; i < this->effective_size; i++) {
            complete_para[this->effective_para_index[i]] = effective_para[i];
        }
#endif
        if (model->gradient_user_defined) {
            gradient = model->gradient_user_defined(complete_para, intercept, *this->data, this->effective_para_index);
            value = model->loss(complete_para, intercept, *this->data);
        }
        else {
            dual v;
            VectorXdual effective_para_dual = effective_para;
            VectorXdual intercept_dual = intercept;
            auto func = [this, &complete_para](VectorXdual const& compute_para, VectorXdual const& intercept) {
                VectorXdual para = complete_para;
#if EIGEN_MAJOR_VERSION >= 4
                para(this->effective_para_index) = compute_para;
#else
                for (Eigen::Index i = 0; i < compute_para.size(); i++) {
                    para[this->effective_para_index[i]] = compute_para[i];
                }
#endif
                return this->model->gradient_autodiff(para, intercept, *this->data);
            };
            gradient.head(intercept.size()) = autodiff::gradient(func, wrt(intercept_dual), at(effective_para_dual, intercept_dual), v);
            gradient.tail(effective_size) = autodiff::gradient(func, wrt(effective_para_dual), at(effective_para_dual, intercept_dual), v);
            value = val(v);
        }
    }
   
    gradient.tail(effective_size) += 2 * lambda * effective_para;
    return value + lambda * effective_para.squaredNorm();
}



void UniversalData::hessian(const VectorXd& effective_para, const VectorXd& intercept, VectorXd& gradient, MatrixXd& hessian, Eigen::Index index, Eigen::Index size, double lambda)
{
    gradient.resize(size);
    hessian.resize(size, size);

    if(effective_para.size() != this->effective_size){
        SPDLOG_ERROR("the size of the effective parameters is different!");
        return;
    }

    VectorXi compute_para_index;
    VectorXd const* para_ptr;
    VectorXd complete_para;

    if (model->slice_by_para) {
        compute_para_index = VectorXi::LinSpaced(size, index, size + index - 1);
        para_ptr = &effective_para;
    }
    else {
        compute_para_index = this->effective_para_index.segment(index, size);
        complete_para = VectorXd::Zero(this->model_size);
#if EIGEN_MAJOR_VERSION >= 4
        complete_para(this->effective_para_index) = effective_para;
#else
        for (Eigen::Index i = 0; i < this->effective_size; i++) {
            complete_para[this->effective_para_index[i]] = effective_para[i];
        }
#endif
        para_ptr = &complete_para;
    }

    if (model->hessian_user_defined) {
        gradient = model->gradient_user_defined(*para_ptr, intercept, *this->data, compute_para_index).tail(size);
        hessian = model->hessian_user_defined(*para_ptr, intercept, *this->data, compute_para_index);
#if SPDLOG_ACTIVE_LEVEL <= SPDLOG_LEVEL_DEBUG
        if (!hessian.isApprox(hessian.transpose())){
            SPDLOG_ERROR("user defined hessian function return a non-symmetric matrix:\n{}", hessian);
            hessian = (hessian + hessian.transpose()) / 2.0;
        }
#endif        
    }
    else { // autodiff
        dual2nd v;
        VectorXdual2nd g;
        VectorXdual2nd compute_para = effective_para.segment(index, size);
        VectorXdual2nd intercept_dual = intercept;
        hessian = autodiff::hessian([this, para_ptr, &compute_para_index](VectorXdual2nd const& compute_para, VectorXdual2nd const& intercept_dual) {
            VectorXdual2nd para = *para_ptr;
            for (Eigen::Index i = 0; i < compute_para_index.size(); i++) {
                para[compute_para_index[i]] = compute_para[i];
            }
#if EIGEN_MAJOR_VERSION >= 4
            para(compute_para_index) = compute_para;
#else
            for (Eigen::Index i = 0; i < compute_para_index.size(); i++) {
                para[compute_para_index[i]] = compute_para[i];
            }
#endif
            return this->model->hessian_autodiff(para, intercept_dual, *this->data);
            }, wrt(compute_para), at(compute_para, intercept_dual), v, g);
        for (Eigen::Index i = 0; i < size; i++) {
            gradient[i] = val(g[i]);
        }
    }

    if (lambda != 0.0) {
        gradient += 2 * lambda * effective_para.segment(index, size);
        hessian += MatrixXd::Constant(size, size, 2 * lambda);
    }
}

void UniversalData::init_para(VectorXd & active_para, VectorXd & intercept, UniversalData const& active_data){
    if (model->init_para) {
        VectorXd complete_para = VectorXd::Zero(this->model_size);
#if EIGEN_MAJOR_VERSION >= 4
        complete_para(this->effective_para_index) = active_para;
#else
        for (Eigen::Index i = 0; i < this->effective_size; i++) {
            complete_para[this->effective_para_index[i]] = active_para[i];
        }
#endif
        tie(complete_para, intercept) = model->init_para(complete_para, intercept, *active_data.data, this->effective_para_index);
#if EIGEN_MAJOR_VERSION >= 4
        active_para = complete_para(this->effective_para_index);
#else
        for (Eigen::Index i = 0; i < this->effective_size; i++) {
            active_para[i] = complete_para[this->effective_para_index[i]];
        }
#endif
    }
}


void UniversalModel::set_loss_of_model(function <double(VectorXd const&, VectorXd const&, ExternData const&)> const& f)
{
    loss = f;
}

void UniversalModel::set_gradient_autodiff(function <dual(VectorXdual const&, VectorXdual const&, ExternData const&)> const& f) {
    gradient_autodiff = f;
    gradient_user_defined = nullptr;
}

void UniversalModel::set_hessian_autodiff(function <dual2nd(VectorXdual2nd const&, VectorXdual2nd const&, ExternData const&)> const& f) {
    hessian_autodiff = f;
    hessian_user_defined = nullptr;
}

void UniversalModel::set_gradient_user_defined(function <VectorXd(VectorXd const&, VectorXd const&, ExternData const&, VectorXi const&)> const& f)
{
    gradient_user_defined = f;
    gradient_autodiff = nullptr;
}

void UniversalModel::set_hessian_user_defined(function <MatrixXd(VectorXd const&, VectorXd const&, ExternData const&, VectorXi const&)> const& f)
{
    hessian_user_defined = f;
    hessian_autodiff = nullptr;
}

void UniversalModel::set_slice_by_sample(function <ExternData(ExternData const&, VectorXi const&)> const& f)
{
    slice_by_sample = f;
}

void UniversalModel::set_slice_by_para(function <ExternData(ExternData const&, VectorXi const&)> const& f)
{
    slice_by_para = f;
}

void UniversalModel::set_deleter(function<void(ExternData const&)> const& f)
{
    if (f) {
        deleter = [f](ExternData const* p) { f(*p); delete p; };
    }
    else {
        deleter = [](ExternData const* p) { delete p; };
    }
}

void UniversalModel::set_init_para(function <pair<VectorXd, VectorXd>(VectorXd const&, VectorXd const&, ExternData const&, VectorXi const&)> const& f)
{
    init_para = f;
}
