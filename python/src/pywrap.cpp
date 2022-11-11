#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <tuple>
#include "UniversalData.h"
#include "List.h"
#include "api.h"
#include "predefinedModel.h"

std::tuple<Eigen::MatrixXd, Eigen::VectorXd, double, double, double> pywrap_GLM(
    Eigen::MatrixXd x_Mat, Eigen::MatrixXd y_Mat, Eigen::VectorXd weight_Vec, int n, int p, int normalize_type,
    int algorithm_type, int model_type, int max_iter, int exchange_num, int path_type, bool is_warm_start, int ic_type,
    double ic_coef, int Kfold, Eigen::VectorXi gindex_Vec, Eigen::VectorXi sequence_Vec,
    Eigen::VectorXd lambda_sequence_Vec, Eigen::VectorXi cv_fold_id_Vec, int s_min, int s_max, double lambda_min,
    double lambda_max, int n_lambda, int screening_size, Eigen::VectorXi always_select_Vec,
    int primary_model_fit_max_iter, double primary_model_fit_epsilon, bool early_stop, bool approximate_Newton,
    int thread, bool covariance_update, bool sparse_matrix, int splicing_type, int sub_search,
    Eigen::VectorXi A_init_Vec) {
    List mylist =
        abessGLM_API(x_Mat, y_Mat, n, p, normalize_type, weight_Vec, algorithm_type, model_type, max_iter, exchange_num,
            path_type, is_warm_start, ic_type, ic_coef, Kfold, sequence_Vec, lambda_sequence_Vec, s_min, s_max,
            lambda_min, lambda_max, n_lambda, screening_size, gindex_Vec, always_select_Vec,
            primary_model_fit_max_iter, primary_model_fit_epsilon, early_stop, approximate_Newton, thread,
            covariance_update, sparse_matrix, splicing_type, sub_search, cv_fold_id_Vec, A_init_Vec);

    std::tuple<Eigen::MatrixXd, Eigen::VectorXd, double, double, double> output;
    int y_col = y_Mat.cols();
    if (y_col == 1 && model_type != 5 && model_type != 6) {
        Eigen::VectorXd beta;
        double coef0 = 0;
        double train_loss = 0;
        double test_loss = 0;
        double ic = 0;
        mylist.get_value_by_name("beta", beta);
        mylist.get_value_by_name("coef0", coef0);
        mylist.get_value_by_name("train_loss", train_loss);
        mylist.get_value_by_name("test_loss", test_loss);
        mylist.get_value_by_name("ic", ic);

        Eigen::MatrixXd beta_out(beta.size(), 1);
        beta_out.col(0) = beta;
        Eigen::VectorXd coef0_out(1);
        coef0_out(0) = coef0;
        output = std::make_tuple(beta_out, coef0_out, train_loss, test_loss, ic);
    } else {
        Eigen::MatrixXd beta;
        Eigen::VectorXd coef0;
        double train_loss = 0;
        double test_loss = 0;
        double ic = 0;
        mylist.get_value_by_name("beta", beta);
        mylist.get_value_by_name("coef0", coef0);
        mylist.get_value_by_name("train_loss", train_loss);
        mylist.get_value_by_name("test_loss", test_loss);
        mylist.get_value_by_name("ic", ic);

        output = std::make_tuple(beta, coef0, train_loss, test_loss, ic);
    }
    return output;
}

std::tuple<Eigen::MatrixXd, double, double, double, double> pywrap_PCA(
    Eigen::MatrixXd x_Mat, Eigen::VectorXd weight_Vec, int n, int p, int normalize_type, Eigen::MatrixXd sigma_Mat,
    int max_iter, int exchange_num, int path_type, bool is_warm_start, int ic_type, double ic_coef, int Kfold,
    Eigen::VectorXi gindex_Vec, Eigen::MatrixXi sequence_Mat, Eigen::VectorXi cv_fold_id_Vec, int s_min, int s_max,
    int screening_size, Eigen::VectorXi always_select_Vec, bool early_stop, int thread, bool sparse_matrix,
    int splicing_type, int sub_search, int pca_num, Eigen::VectorXi A_init_Vec) {
    List mylist = abessPCA_API(x_Mat, n, p, normalize_type, weight_Vec, sigma_Mat, max_iter, exchange_num, path_type,
                               is_warm_start, ic_type, ic_coef, Kfold, sequence_Mat, s_min, s_max, screening_size,
                               gindex_Vec, always_select_Vec, early_stop, thread, sparse_matrix, splicing_type,
                               sub_search, cv_fold_id_Vec, pca_num, A_init_Vec);

    Eigen::MatrixXd beta;
    if (pca_num == 1) {
        Eigen::VectorXd beta_temp;
        mylist.get_value_by_name("beta", beta_temp);
        // beta.resize(p, 1);
        beta = beta_temp;
    } else {
        // beta.resize(p, pca_num);
        mylist.get_value_by_name("beta", beta);
    }

    double coef0 = 0;
    double train_loss = 0;
    double test_loss = 0;
    double ic = 0;
    mylist.get_value_by_name("coef0", coef0);
    mylist.get_value_by_name("train_loss", train_loss);
    mylist.get_value_by_name("test_loss", test_loss);
    mylist.get_value_by_name("ic", ic);

    return std::make_tuple(beta, coef0, train_loss, test_loss, ic);
}

std::tuple<Eigen::VectorXd, double, double, double, double> pywrap_RPCA(
    Eigen::MatrixXd x_Mat, int n, int p, int normalize_type, int max_iter, int exchange_num, int path_type,
    bool is_warm_start, int ic_type, double ic_coef, Eigen::VectorXi gindex_Vec, Eigen::VectorXi sequence_Vec,
    Eigen::VectorXd lambda_sequence_Vec, int s_min, int s_max, double lambda_min, double lambda_max, int n_lambda,
    int screening_size, Eigen::VectorXi always_select_Vec, int primary_model_fit_max_iter,
    double primary_model_fit_epsilon, bool early_stop, int thread, bool sparse_matrix, int splicing_type,
    int sub_search, Eigen::VectorXi A_init_Vec) {
    List mylist =
        abessRPCA_API(x_Mat, n, p, max_iter, exchange_num, path_type, is_warm_start, ic_type, ic_coef, sequence_Vec,
                      lambda_sequence_Vec, s_min, s_max, lambda_min, lambda_max, n_lambda, screening_size,
                      primary_model_fit_max_iter, primary_model_fit_epsilon, gindex_Vec, always_select_Vec, early_stop,
                      thread, sparse_matrix, splicing_type, sub_search, A_init_Vec);

    Eigen::VectorXd beta;
    double coef0 = 0;
    double train_loss = 0;
    double test_loss = 0;
    double ic = 0;
    mylist.get_value_by_name("beta", beta);
    mylist.get_value_by_name("coef0", coef0);
    mylist.get_value_by_name("train_loss", train_loss);
    mylist.get_value_by_name("test_loss", test_loss);
    mylist.get_value_by_name("ic", ic);

    return std::make_tuple(beta, coef0, train_loss, test_loss, ic);
}

std::tuple<Eigen::VectorXd, Eigen::VectorXd, double, double, double>
pywrap_Universal(ExternData data, UniversalModel model, int model_size, int sample_size,int intercept_size, int max_iter,
    int exchange_num, int path_type, bool is_warm_start, int ic_type, double ic_coef, int Kfold, Eigen::VectorXi sequence, 
    Eigen::VectorXd lambda_seq, int s_min, int s_max, int screening_size, Eigen::VectorXi g_index, Eigen::VectorXi always_select, 
    int thread, int splicing_type, int sub_search, Eigen::VectorXi cv_fold_id, Eigen::VectorXi A_init)
{
    List mylist = abessUniversal_API(data, model, model_size, sample_size, intercept_size, max_iter, exchange_num,
        path_type, is_warm_start, ic_type, ic_coef, Kfold, sequence, lambda_seq, s_min, s_max,
        screening_size, g_index, always_select, thread, splicing_type, sub_search, cv_fold_id, A_init);
    Eigen::VectorXd beta;
    Eigen::VectorXd intercept;
    double train_loss = 0;
    double test_loss = 0;
    double ic = 0;
    mylist.get_value_by_name("beta", beta);
    mylist.get_value_by_name("coef0", intercept);
    mylist.get_value_by_name("train_loss", train_loss);
    mylist.get_value_by_name("test_loss", test_loss);
    mylist.get_value_by_name("ic", ic);
    return std::make_tuple(beta, intercept, train_loss, test_loss, ic);
}

PYBIND11_MODULE(pybind_cabess, m) {
    pybind11::class_<UniversalModel>(m, "UniversalModel").def(pybind11::init<>())
        .def("set_loss_of_model", &UniversalModel::set_loss_of_model)
        .def("set_gradient_autodiff", &UniversalModel::set_gradient_autodiff)
        .def("set_hessian_autodiff", &UniversalModel::set_hessian_autodiff)
        .def("set_gradient_user_defined", &UniversalModel::set_gradient_user_defined)
        .def("set_hessian_user_defined", &UniversalModel::set_hessian_user_defined)
        .def("set_slice_by_sample", &UniversalModel::set_slice_by_sample)
        .def("set_slice_by_para", &UniversalModel::set_slice_by_para)
        .def("set_deleter", &UniversalModel::set_deleter)
        .def("set_init_para", &UniversalModel::set_init_para);
    m.def("pywrap_GLM", &pywrap_GLM);
    m.def("pywrap_PCA", &pywrap_PCA);
    m.def("pywrap_RPCA", &pywrap_RPCA);
    m.def("pywrap_Universal", &pywrap_Universal);
    // predefine some function for test
    m.def("init_spdlog", &init_spdlog, pybind11::arg("console_log_level") = SPDLOG_LEVEL_OFF, pybind11::arg("file_log_level") = SPDLOG_LEVEL_INFO);
    pybind11::class_<PredefinedData>(m, "Data").def(pybind11::init<MatrixXd, MatrixXd>());
    m.def("loss_linear", &linear_model<double>);
    m.def("gradient_linear", &linear_model<dual>);
    m.def("hessian_linear", &linear_model<dual2nd>);
    m.def("loss_logistic", &logistic_model<double>);
    m.def("gradient_logistic", &logistic_model<dual>);
    m.def("hessian_logistic", &logistic_model<dual2nd>);
    m.def("slice_by_sample", &slice_by_sample);
    m.def("slice_by_para", &slice_by_para);
    m.def("deleter", &deleter);
}
