// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppEigen.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// abessGLM_API
List abessGLM_API(Eigen::MatrixXd x, Eigen::MatrixXd y, int n, int p, int normalize_type, Eigen::VectorXd weight, int algorithm_type, int model_type, int max_iter, int exchange_num, int path_type, bool is_warm_start, int ic_type, double ic_coef, int Kfold, Eigen::VectorXi sequence, Eigen::VectorXd lambda_seq, int s_min, int s_max, double lambda_min, double lambda_max, int nlambda, int screening_size, Eigen::VectorXi g_index, Eigen::VectorXi always_select, int primary_model_fit_max_iter, double primary_model_fit_epsilon, bool early_stop, bool approximate_Newton, int thread, bool covariance_update, bool sparse_matrix, int splicing_type, int sub_search, Eigen::VectorXi cv_fold_id);
RcppExport SEXP _abess_abessGLM_API(SEXP xSEXP, SEXP ySEXP, SEXP nSEXP, SEXP pSEXP, SEXP normalize_typeSEXP, SEXP weightSEXP, SEXP algorithm_typeSEXP, SEXP model_typeSEXP, SEXP max_iterSEXP, SEXP exchange_numSEXP, SEXP path_typeSEXP, SEXP is_warm_startSEXP, SEXP ic_typeSEXP, SEXP ic_coefSEXP, SEXP KfoldSEXP, SEXP sequenceSEXP, SEXP lambda_seqSEXP, SEXP s_minSEXP, SEXP s_maxSEXP, SEXP lambda_minSEXP, SEXP lambda_maxSEXP, SEXP nlambdaSEXP, SEXP screening_sizeSEXP, SEXP g_indexSEXP, SEXP always_selectSEXP, SEXP primary_model_fit_max_iterSEXP, SEXP primary_model_fit_epsilonSEXP, SEXP early_stopSEXP, SEXP approximate_NewtonSEXP, SEXP threadSEXP, SEXP covariance_updateSEXP, SEXP sparse_matrixSEXP, SEXP splicing_typeSEXP, SEXP sub_searchSEXP, SEXP cv_fold_idSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Eigen::MatrixXd >::type x(xSEXP);
    Rcpp::traits::input_parameter< Eigen::MatrixXd >::type y(ySEXP);
    Rcpp::traits::input_parameter< int >::type n(nSEXP);
    Rcpp::traits::input_parameter< int >::type p(pSEXP);
    Rcpp::traits::input_parameter< int >::type normalize_type(normalize_typeSEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXd >::type weight(weightSEXP);
    Rcpp::traits::input_parameter< int >::type algorithm_type(algorithm_typeSEXP);
    Rcpp::traits::input_parameter< int >::type model_type(model_typeSEXP);
    Rcpp::traits::input_parameter< int >::type max_iter(max_iterSEXP);
    Rcpp::traits::input_parameter< int >::type exchange_num(exchange_numSEXP);
    Rcpp::traits::input_parameter< int >::type path_type(path_typeSEXP);
    Rcpp::traits::input_parameter< bool >::type is_warm_start(is_warm_startSEXP);
    Rcpp::traits::input_parameter< int >::type ic_type(ic_typeSEXP);
    Rcpp::traits::input_parameter< double >::type ic_coef(ic_coefSEXP);
    Rcpp::traits::input_parameter< int >::type Kfold(KfoldSEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXi >::type sequence(sequenceSEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXd >::type lambda_seq(lambda_seqSEXP);
    Rcpp::traits::input_parameter< int >::type s_min(s_minSEXP);
    Rcpp::traits::input_parameter< int >::type s_max(s_maxSEXP);
    Rcpp::traits::input_parameter< double >::type lambda_min(lambda_minSEXP);
    Rcpp::traits::input_parameter< double >::type lambda_max(lambda_maxSEXP);
    Rcpp::traits::input_parameter< int >::type nlambda(nlambdaSEXP);
    Rcpp::traits::input_parameter< int >::type screening_size(screening_sizeSEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXi >::type g_index(g_indexSEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXi >::type always_select(always_selectSEXP);
    Rcpp::traits::input_parameter< int >::type primary_model_fit_max_iter(primary_model_fit_max_iterSEXP);
    Rcpp::traits::input_parameter< double >::type primary_model_fit_epsilon(primary_model_fit_epsilonSEXP);
    Rcpp::traits::input_parameter< bool >::type early_stop(early_stopSEXP);
    Rcpp::traits::input_parameter< bool >::type approximate_Newton(approximate_NewtonSEXP);
    Rcpp::traits::input_parameter< int >::type thread(threadSEXP);
    Rcpp::traits::input_parameter< bool >::type covariance_update(covariance_updateSEXP);
    Rcpp::traits::input_parameter< bool >::type sparse_matrix(sparse_matrixSEXP);
    Rcpp::traits::input_parameter< int >::type splicing_type(splicing_typeSEXP);
    Rcpp::traits::input_parameter< int >::type sub_search(sub_searchSEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXi >::type cv_fold_id(cv_fold_idSEXP);
    rcpp_result_gen = Rcpp::wrap(abessGLM_API(x, y, n, p, normalize_type, weight, algorithm_type, model_type, max_iter, exchange_num, path_type, is_warm_start, ic_type, ic_coef, Kfold, sequence, lambda_seq, s_min, s_max, lambda_min, lambda_max, nlambda, screening_size, g_index, always_select, primary_model_fit_max_iter, primary_model_fit_epsilon, early_stop, approximate_Newton, thread, covariance_update, sparse_matrix, splicing_type, sub_search, cv_fold_id));
    return rcpp_result_gen;
END_RCPP
}
// abessPCA_API
List abessPCA_API(Eigen::MatrixXd x, int n, int p, int normalize_type, Eigen::VectorXd weight, Eigen::MatrixXd sigma, int max_iter, int exchange_num, int path_type, bool is_warm_start, int ic_type, double ic_coef, int Kfold, Eigen::MatrixXi sequence, int s_min, int s_max, int screening_size, Eigen::VectorXi g_index, Eigen::VectorXi always_select, bool early_stop, int thread, bool sparse_matrix, int splicing_type, int sub_search, Eigen::VectorXi cv_fold_id, int pca_num);
RcppExport SEXP _abess_abessPCA_API(SEXP xSEXP, SEXP nSEXP, SEXP pSEXP, SEXP normalize_typeSEXP, SEXP weightSEXP, SEXP sigmaSEXP, SEXP max_iterSEXP, SEXP exchange_numSEXP, SEXP path_typeSEXP, SEXP is_warm_startSEXP, SEXP ic_typeSEXP, SEXP ic_coefSEXP, SEXP KfoldSEXP, SEXP sequenceSEXP, SEXP s_minSEXP, SEXP s_maxSEXP, SEXP screening_sizeSEXP, SEXP g_indexSEXP, SEXP always_selectSEXP, SEXP early_stopSEXP, SEXP threadSEXP, SEXP sparse_matrixSEXP, SEXP splicing_typeSEXP, SEXP sub_searchSEXP, SEXP cv_fold_idSEXP, SEXP pca_numSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Eigen::MatrixXd >::type x(xSEXP);
    Rcpp::traits::input_parameter< int >::type n(nSEXP);
    Rcpp::traits::input_parameter< int >::type p(pSEXP);
    Rcpp::traits::input_parameter< int >::type normalize_type(normalize_typeSEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXd >::type weight(weightSEXP);
    Rcpp::traits::input_parameter< Eigen::MatrixXd >::type sigma(sigmaSEXP);
    Rcpp::traits::input_parameter< int >::type max_iter(max_iterSEXP);
    Rcpp::traits::input_parameter< int >::type exchange_num(exchange_numSEXP);
    Rcpp::traits::input_parameter< int >::type path_type(path_typeSEXP);
    Rcpp::traits::input_parameter< bool >::type is_warm_start(is_warm_startSEXP);
    Rcpp::traits::input_parameter< int >::type ic_type(ic_typeSEXP);
    Rcpp::traits::input_parameter< double >::type ic_coef(ic_coefSEXP);
    Rcpp::traits::input_parameter< int >::type Kfold(KfoldSEXP);
    Rcpp::traits::input_parameter< Eigen::MatrixXi >::type sequence(sequenceSEXP);
    Rcpp::traits::input_parameter< int >::type s_min(s_minSEXP);
    Rcpp::traits::input_parameter< int >::type s_max(s_maxSEXP);
    Rcpp::traits::input_parameter< int >::type screening_size(screening_sizeSEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXi >::type g_index(g_indexSEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXi >::type always_select(always_selectSEXP);
    Rcpp::traits::input_parameter< bool >::type early_stop(early_stopSEXP);
    Rcpp::traits::input_parameter< int >::type thread(threadSEXP);
    Rcpp::traits::input_parameter< bool >::type sparse_matrix(sparse_matrixSEXP);
    Rcpp::traits::input_parameter< int >::type splicing_type(splicing_typeSEXP);
    Rcpp::traits::input_parameter< int >::type sub_search(sub_searchSEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXi >::type cv_fold_id(cv_fold_idSEXP);
    Rcpp::traits::input_parameter< int >::type pca_num(pca_numSEXP);
    rcpp_result_gen = Rcpp::wrap(abessPCA_API(x, n, p, normalize_type, weight, sigma, max_iter, exchange_num, path_type, is_warm_start, ic_type, ic_coef, Kfold, sequence, s_min, s_max, screening_size, g_index, always_select, early_stop, thread, sparse_matrix, splicing_type, sub_search, cv_fold_id, pca_num));
    return rcpp_result_gen;
END_RCPP
}
// abessRPCA_API
List abessRPCA_API(Eigen::MatrixXd x, int n, int p, int max_iter, int exchange_num, int path_type, bool is_warm_start, int ic_type, double ic_coef, Eigen::VectorXi sequence, Eigen::VectorXd lambda_seq, int s_min, int s_max, double lambda_min, double lambda_max, int nlambda, int screening_size, int primary_model_fit_max_iter, double primary_model_fit_epsilon, Eigen::VectorXi g_index, Eigen::VectorXi always_select, bool early_stop, int thread, bool sparse_matrix, int splicing_type, int sub_search);
RcppExport SEXP _abess_abessRPCA_API(SEXP xSEXP, SEXP nSEXP, SEXP pSEXP, SEXP max_iterSEXP, SEXP exchange_numSEXP, SEXP path_typeSEXP, SEXP is_warm_startSEXP, SEXP ic_typeSEXP, SEXP ic_coefSEXP, SEXP sequenceSEXP, SEXP lambda_seqSEXP, SEXP s_minSEXP, SEXP s_maxSEXP, SEXP lambda_minSEXP, SEXP lambda_maxSEXP, SEXP nlambdaSEXP, SEXP screening_sizeSEXP, SEXP primary_model_fit_max_iterSEXP, SEXP primary_model_fit_epsilonSEXP, SEXP g_indexSEXP, SEXP always_selectSEXP, SEXP early_stopSEXP, SEXP threadSEXP, SEXP sparse_matrixSEXP, SEXP splicing_typeSEXP, SEXP sub_searchSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Eigen::MatrixXd >::type x(xSEXP);
    Rcpp::traits::input_parameter< int >::type n(nSEXP);
    Rcpp::traits::input_parameter< int >::type p(pSEXP);
    Rcpp::traits::input_parameter< int >::type max_iter(max_iterSEXP);
    Rcpp::traits::input_parameter< int >::type exchange_num(exchange_numSEXP);
    Rcpp::traits::input_parameter< int >::type path_type(path_typeSEXP);
    Rcpp::traits::input_parameter< bool >::type is_warm_start(is_warm_startSEXP);
    Rcpp::traits::input_parameter< int >::type ic_type(ic_typeSEXP);
    Rcpp::traits::input_parameter< double >::type ic_coef(ic_coefSEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXi >::type sequence(sequenceSEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXd >::type lambda_seq(lambda_seqSEXP);
    Rcpp::traits::input_parameter< int >::type s_min(s_minSEXP);
    Rcpp::traits::input_parameter< int >::type s_max(s_maxSEXP);
    Rcpp::traits::input_parameter< double >::type lambda_min(lambda_minSEXP);
    Rcpp::traits::input_parameter< double >::type lambda_max(lambda_maxSEXP);
    Rcpp::traits::input_parameter< int >::type nlambda(nlambdaSEXP);
    Rcpp::traits::input_parameter< int >::type screening_size(screening_sizeSEXP);
    Rcpp::traits::input_parameter< int >::type primary_model_fit_max_iter(primary_model_fit_max_iterSEXP);
    Rcpp::traits::input_parameter< double >::type primary_model_fit_epsilon(primary_model_fit_epsilonSEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXi >::type g_index(g_indexSEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXi >::type always_select(always_selectSEXP);
    Rcpp::traits::input_parameter< bool >::type early_stop(early_stopSEXP);
    Rcpp::traits::input_parameter< int >::type thread(threadSEXP);
    Rcpp::traits::input_parameter< bool >::type sparse_matrix(sparse_matrixSEXP);
    Rcpp::traits::input_parameter< int >::type splicing_type(splicing_typeSEXP);
    Rcpp::traits::input_parameter< int >::type sub_search(sub_searchSEXP);
    rcpp_result_gen = Rcpp::wrap(abessRPCA_API(x, n, p, max_iter, exchange_num, path_type, is_warm_start, ic_type, ic_coef, sequence, lambda_seq, s_min, s_max, lambda_min, lambda_max, nlambda, screening_size, primary_model_fit_max_iter, primary_model_fit_epsilon, g_index, always_select, early_stop, thread, sparse_matrix, splicing_type, sub_search));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_abess_abessGLM_API", (DL_FUNC) &_abess_abessGLM_API, 35},
    {"_abess_abessPCA_API", (DL_FUNC) &_abess_abessPCA_API, 26},
    {"_abess_abessRPCA_API", (DL_FUNC) &_abess_abessRPCA_API, 26},
    {NULL, NULL, 0}
};

RcppExport void R_init_abess(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
