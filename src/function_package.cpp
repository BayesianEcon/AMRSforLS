// [[Rcpp::plugins(cpp11)]]
#include <iomanip>
#include <iostream>
#include <utility>
#include <vector>
#include <string>
#include <list>
#include <RcppDist.h>
#include <RcppArmadillo.h>
// [[Rcpp::depends( RcppDist , RcppArmadillo)]]
using namespace Rcpp;
using namespace arma;
using namespace std;

NumericVector cseq(double first, double last, double by)
{
  int n = (last - first) / by + 1;
  NumericVector result(n);
  result(0) = first;
  for (int i = 1; i < n; i++)
  {
    result(i) = result(i - 1) + by;
  }
  return result;
}

arma::mat mvrnormArma(int n, arma::vec mu, arma::mat sigma)
{
  int ncols = sigma.n_cols;
  arma::mat Y = arma::randn(n, ncols);
  return arma::repmat(mu, 1, n).t() + Y * arma::chol(sigma);
}

double mean_mat(NumericMatrix A)
{
  double n_i = A.nrow();
  double n_j = A.ncol();
  double result = 0;
  for (int i = 0; i < n_i; i++)
  {
    for (int j = 0; j < n_j; j++)
    {
      double aux = A(i, j);
      result = result + aux;
    }
  }
  return (result / (n_i * n_j));
}

double sd_mat(NumericMatrix A)
{
  double n_i = A.nrow();
  double n_j = A.ncol();
  arma::vec val((n_i * n_j - n_i) / 2);
  double count = 0;
  for (int i = 0; i < n_i; i++)
  {
    for (int j = 0; j < i; j++)
    {
      double aux = A(i, j);
      val[count] = aux;
      count = count + 1;
    }
  }
  double sd_res = arma::stddev(val);
  return (sd_res);
}

double clog_sum_exp(arma::vec x)
{
  auto b = max(x);
  double result = (b + log(sum(exp(x - b))));
  return result;
}

NumericVector csoftmax(NumericVector x)
{
  NumericVector result = exp(x - clog_sum_exp(x));
  return result;
}
NumericVector logistic(NumericVector x)
{
  NumericVector result = 1 / (1 + exp(-x));
  return result;
}

double logistic_d(double x)
{
  double result = 1 / (1 + exp(-x));
  return result;
}

vec arma_sort(vec x, vec y, vec z)
{
  // Order the elements of x by sorting y and z;
  // we order by y unless there's a tie, then order by z.
  // First create a vector of indices
  uvec idx = regspace<uvec>(0, x.size() - 1);
  // Then sort that vector by the values of y and z
  std::sort(idx.begin(), idx.end(), [&](int i, int j)
  {
    if ( y[i] == y[j] ) {
      return z[i] < z[j];
    }
    return y[i] < y[j]; });
  // And return x in that order
  return x(idx);
}

double logbivnorm(arma::vec x, arma::vec mean, arma::mat sigma)
{
  arma::vec diff = x - mean;
  arma::rowvec diff_t = diff.t();
  arma::mat inv_sigma = sigma;
  inv_sigma = inv(inv_sigma);
  double result = (-0.5 * diff_t * inv_sigma * diff).eval()(0, 0);
  return result;
};
static double const log2pi = std::log(2.0 * M_PI);

arma::vec dmvnrm_arma(arma::mat const &x,
                      arma::rowvec const &mean,
                      arma::mat const &sigma,
                      bool const logd = false)
{
  using arma::uword;
  uword const n = x.n_rows,
    xdim = x.n_cols;
  arma::vec out(n);
  arma::mat const rooti = arma::inv(trimatu(arma::chol(sigma)));
  double const rootisum = arma::sum(log(rooti.diag())),
    constants = -(double)xdim / 2.0 * log2pi,
    other_terms = rootisum + constants;
  arma::rowvec z;
  for (uword i = 0; i < n; i++)
  {
    z = (x.row(i) - mean) * rooti;
    out(i) = other_terms - 0.5 * arma::dot(z, z);
  }
  if (logd)
    return out;
  return exp(out);
}
/* C++ version of the dtrmv BLAS function */
void inplace_tri_mat_mult(arma::rowvec &x, arma::mat const &trimat)
{
  arma::uword const n = trimat.n_cols;
  for (unsigned j = n; j-- > 0;)
  {
    double tmp(0.);
    for (unsigned i = 0; i <= j; ++i)
      tmp += trimat.at(i, j) * x[i];
    x[j] = tmp;
  }
}

double a_ite(double ite, double N)
{
  double res = log(50 * sqrt(N) + ite) / (50 * sqrt(N) + ite);
  return (res);
}

NumericVector getRandomElementIndices(NumericVector belong, int N, int numClusters)
{
  NumericVector Indices = cseq(0, N - 1, 1);
  NumericVector Result(numClusters);

  for (int i = 0; i < numClusters; i++)
  {

    NumericVector Indices_sub = Indices[belong == 1 + i];
    NumericVector sam = sample(Indices_sub, 1);
    double sam_d = sam[0];
    Result(i) = sam_d;
  }

  return (Result);
}

arma::vec dmvnrm_arma_fast(arma::mat const &x,
                           arma::rowvec const &mean,
                           arma::mat const &sigma,
                           bool const logd = false)
{
  using arma::uword;
  uword const n = x.n_rows,
    xdim = x.n_cols;
  arma::vec out(n);
  arma::mat const rooti = arma::inv(trimatu(arma::chol(sigma)));
  double const rootisum = arma::sum(log(rooti.diag())),
    constants = -(double)xdim / 2.0 * log2pi,
    other_terms = rootisum + constants;
  arma::rowvec z;
  for (uword i = 0; i < n; i++)
  {
    z = (x.row(i) - mean);
    inplace_tri_mat_mult(z, rooti);
    out(i) = other_terms - 0.5 * arma::dot(z, z);
  }
  if (logd)
    return out;
  return exp(out);
}

NumericMatrix ecldistp2(NumericVector A, NumericVector B)
{
  int sizeA = A.length();
  NumericMatrix result(sizeA, sizeA);
  for (int i = 0; i < sizeA; i++)
  {
    double Ai = A(i);
    double Bi = B(i);
    for (int j = 0; j < sizeA; j++)
    {
      double Aj = A(j);
      double Bj = B(j);
      double aux = pow(Ai - Aj, 2) + pow(Bi - Bj, 2);
      result(i, j) = aux;
    }
  }
  return result;
};

NumericMatrix gath(NumericMatrix A)
{
  int n_col = A.ncol();
  int n_row = A.nrow();
  int n_flat = n_col * n_row - n_col;
  NumericVector ix(n_flat);
  NumericVector jx(n_flat);
  NumericVector result(n_flat);
  int m = 0;
  for (int i = 0; i < n_row; i++)
  {
    for (int j = 0; j < n_col; j++)
    {
      if (i != j)
      {
        double aux = A(i, j);
        result(m) = aux;
        ix(m) = i + 1;
        jx(m) = j + 1;
        m = m + 1;
      }
    }
  }
  NumericMatrix final(n_flat, 3);
  final(_, 0) = ix;
  final(_, 1) = jx;
  final(_, 2) = result;
  return final;
};

NumericMatrix gath_tri(NumericMatrix A)
{
  int n_col = A.ncol();
  int n_row = A.nrow();
  int n_flat = (n_col * n_row - n_col) / 2;
  NumericVector ix(n_flat);
  NumericVector jx(n_flat);
  NumericVector result(n_flat);
  int m = 0;
  for (int i = 0; i < n_row; i++)
  {
    for (int j = i; j < n_col; j++)
    {
      if (i != j)
      {
        double aux = A(i, j);
        result(m) = aux;
        ix(m) = i + 1;
        jx(m) = j + 1;
        m = m + 1;
      }
    }
  }
  NumericMatrix final(n_flat, 3);
  final(_, 0) = ix;
  final(_, 1) = jx;
  final(_, 2) = result;
  return final;
};

NumericVector vec_cpp(NumericMatrix A)
{
  double n = A.nrow();
  double m = A.ncol();
  NumericVector res(n * m);
  int ind = 0;
  for (int j = 0; j < m; j++)
  {
    for (int i = 0; i < n; i++)
    {
      double res_aux = A(i, j);
      res[ind] = res_aux;
      ind = ind + 1;
    }
  }
  return (res);
};

NumericVector in_operator(NumericVector range, NumericVector belong, NumericVector selected_blocks)
{
  NumericVector sample_v;

  for (int i = 0; i < belong.size(); i++)
  {

    double belong_i = belong(i);
    double sum_check = sum(selected_blocks == belong_i);

    if (sum_check > 0)
    {

      sample_v.push_back(range[i]);
    }
  }

  return sample_v;
}

//' Procrustes Trasnformation
//'
//' It applies Procrustes transformation of a set of n points laying in a d-dimensional space w.r.t. a reference set.
//'
//' @param X is an n x d matrix of reference points
//' @param Y is an n x d matrix of points to transform
//' @return is an n x d matrix of rotated points
//' @export
// [[Rcpp::export]]
 NumericMatrix procrustes_cpp(NumericMatrix X, NumericMatrix Y)
 {
   arma::mat X_aux = as<arma::mat>(X);

   double K = Y.cols();

   for (int k = 0; k < K; k++)
   {
     NumericVector zeta_demean = Y(_, k);
     zeta_demean = zeta_demean - mean(zeta_demean);
     Y(_, k) = clone(zeta_demean);
   }

   arma::mat Y_aux = as<arma::mat>(Y);
   arma::mat XY = X_aux.t() * Y_aux;
   arma::mat U;
   arma::vec s;
   arma::mat V;
   arma::svd(U, s, V, XY);
   arma::mat A = V * U.t();
   arma::mat result = Y_aux * A;
   return wrap(result);
 }

NumericMatrix procrustes_cpp_P(NumericMatrix X, NumericMatrix Y)
{
  arma::mat X_aux = as<arma::mat>(X);
  arma::mat Y_aux = as<arma::mat>(Y);
  arma::mat XY = X_aux.t() * Y_aux;
  arma::mat U;
  arma::vec s;
  arma::mat V;
  arma::svd(U, s, V, XY);
  arma::mat A = V * U.t();
  return wrap(A);
}

NumericMatrix dist_lcpp(NumericMatrix A, NumericMatrix B)
{
  int n = A.nrow();
  int m = B.nrow();
  int K = A.ncol();
  NumericMatrix result(n, m);
  for (int i = 0; i < n; i++)
  {
    for (int j = 0; j < m; j++)
    {
      if (j > i)
      {
        double aux = 0;
        for (int k = 0; k < K; k++)
        {
          double Ai = A(i, k);
          double Bj = B(j, k);
          aux = aux + pow(Ai - Bj, 2);
        }
        result(i, j) = sqrt(aux);
      }
    }
  }
  return result;
};

double betaSamplingEn(NumericVector vec_y, double beta, NumericMatrix dist, double beta_a, double beta_b, double N, double M, double prop_sigma, double distr_option)
{
  GetRNGstate();
  double beta_prop = R::rnorm(beta, prop_sigma);
  PutRNGstate();
  NumericVector pois(M);
  NumericVector pois_prop(M);
  double idx = 0;
  for (int i = 0; i < N; i++)
  {
    for (int j = i + 1; j < N; j++)
    {
      double d = dist(i, j);

      if (distr_option == 1)
      {
        double lambda_aux = exp(beta - pow(d, 2));
        double lambda_aux_prop = exp(beta_prop - pow(d, 2));

        double y_check = vec_y[idx];
        bool check_na = NumericVector::is_na(y_check);

        if (check_na == false)
        {

          double pois_aux = R::dpois(y_check, lambda_aux, true);
          double pois_prop_aux = R::dpois(y_check, lambda_aux_prop, true);
          pois(idx) = pois_aux;
          pois_prop(idx) = pois_prop_aux;
        };
      }
      else if (distr_option == 0)
      {

        double lambda_aux = exp(beta - pow(d, 2));
        lambda_aux = lambda_aux / (1 + lambda_aux);

        double lambda_aux_prop = exp(beta_prop - pow(d, 2));
        lambda_aux_prop = lambda_aux_prop / (1 + lambda_aux_prop);

        double y_check = vec_y[idx];
        bool check_na = NumericVector::is_na(y_check);

        if (check_na == false)
        {

          double pois_aux = R::dbinom(y_check, 1, lambda_aux, true);
          double pois_prop_aux = R::dbinom(y_check, 1, lambda_aux_prop, true);
          pois(idx) = pois_aux;
          pois_prop(idx) = pois_prop_aux;
        };
      }

      idx = idx + 1;
    }
  }
  double prior = R::dnorm(beta, beta_a, beta_b, true);
  double prior_prop = R::dnorm(beta_prop, beta_a, beta_b, true);
  double test = prior_prop - prior + sum(pois_prop - pois);
  GetRNGstate();
  double rand = R::runif(0, 1);
  PutRNGstate();
  if (test > log(rand))
  {
    beta = beta_prop;
  };
  return (beta);
}

List zetaSamplingEn(NumericMatrix y, double beta, NumericMatrix z_i, double zeta_a, double zeta_b, NumericVector lam_ad_z, NumericMatrix mu_mat_z, List Sigma_ad_z, NumericVector count_z, NumericVector sample, int N, int Nstar, int M, int K, NumericVector acc_vector, double acc_star, double ite, NumericMatrix zi_ref, double distr_option)
{
  //
  for (int i = 0; i < Nstar; i++)
  {
    //
    //
    int ind = sample[i];

    double count_z_i = count_z[ind];

    //
    NumericMatrix z_i_prop = clone(z_i);
    NumericVector z_i_x_aux = z_i_prop(ind, _);
    NumericVector z_i_x_pr = clone(z_i_x_aux);
    NumericVector z_i_x(K);

    NumericMatrix sigma = as<NumericMatrix>(Sigma_ad_z[ind]);

    double lm = lam_ad_z[ind];

    for (int k = 0; k < K; k++)
    {

      double sigma_k = sigma(k, k);
      GetRNGstate();
      double z_i_prop_x = R::rnorm(z_i_x_pr(k), sigma_k * lm);
      PutRNGstate();
      z_i_x(k) = z_i_prop_x;
    }

    z_i_prop(ind, _) = clone(z_i_x);
    NumericVector pois(N - 1);
    NumericVector pois_prop(N - 1);
    int idx = 0;
    for (int j = 0; j < N; j++)
    {
      if (j != ind)
      {
        double d = 0;
        double d_prop = 0;
        for (int k = 0; k < K; k++)
        {
          d = d + pow(z_i(ind, k) - z_i(j, k), 2);
          d_prop = d_prop + pow(z_i_prop(ind, k) - z_i_prop(j, k), 2);
        }
        if (distr_option == 1)
        {

          double lambda_aux = exp(beta - d);
          double lambda_aux_prop = exp(beta - d_prop);

          double y_check = y(ind, j);
          bool check_na = NumericVector::is_na(y_check);

          if (check_na == false)
          {

            double pois_aux = R::dpois(y_check, lambda_aux, true);
            double pois_prop_aux = R::dpois(y_check, lambda_aux_prop, true);
            pois[idx] = pois_aux;
            pois_prop[idx] = pois_prop_aux;
          }
        }
        else if (distr_option == 0)
        {

          double lambda_aux = exp(beta - d);
          lambda_aux = lambda_aux / (1 + lambda_aux);

          double lambda_aux_prop = exp(beta - d_prop);
          lambda_aux_prop = lambda_aux_prop / (1 + lambda_aux_prop);

          double y_check = y(ind, j);
          bool check_na = NumericVector::is_na(y_check);

          if (check_na == false)
          {

            double pois_aux = R::dbinom(y_check, 1, lambda_aux, true);
            double pois_prop_aux = R::dbinom(y_check, 1, lambda_aux_prop, true);
            pois[idx] = pois_aux;
            pois_prop[idx] = pois_prop_aux;
          }
        }

        idx = idx + 1;
      }
    }

    double prior = 0;
    double prior_prop = 0;
    for (int k = 0; k < K; k++)
    {
      double prior_aux = R::dnorm(z_i(ind, k), zeta_a, zeta_b, true);
      double prior_prop_aux = R::dnorm(z_i_x[k], zeta_a, zeta_b, true);
      prior = prior + prior_aux;
      prior_prop = prior_prop + prior_prop_aux;
    }

    double test = prior_prop - prior + sum(pois_prop - pois);
    GetRNGstate();
    double rand = R::runif(0, 1);
    PutRNGstate();

    if (test > log(rand))
    {
      z_i(ind, _) = clone(z_i_x);
    };

    double acceptance_z;
    if (exp(test) < 1)
    {
      acceptance_z = exp(test);
    }
    else
    {
      acceptance_z = 1;
    };

    count_z_i = count_z_i + 1;
    count_z(ind) = count_z_i;

    double gamm = 1 / pow(count_z_i + 1, 0.52);
    double log_lam = log(lm) + gamm * (acceptance_z - acc_star);

    NumericVector theta_ij_i = z_i(ind, _);
    NumericVector mu_mat_i = mu_mat_z(ind, _);
    NumericVector diff = (theta_ij_i - mu_mat_i);

    NumericVector mu_x = mu_mat_i + gamm * diff;

    mu_mat_z(ind, _) = mu_x;

    arma::vec dif_vec = diff;
    arma::rowvec dif_vec_t = dif_vec.t();

    arma::mat Sigma = as<arma::mat>(sigma);

    NumericMatrix Sigma_up = wrap(Sigma + gamm * (dif_vec * dif_vec_t - Sigma));

    Sigma_ad_z(ind) = Sigma_up;
    lam_ad_z(ind) = exp(log_lam);
    acc_vector(ind) = acceptance_z;
  }

  List result = List::create(Named("z_i") = z_i, Named("lam_ad_z") = lam_ad_z, Named("mu_mat_z") = mu_mat_z, Named("Sigma_ad_z") = Sigma_ad_z, Named("acc_vector") = acc_vector);
  return result;
};

List zetaSamplingEnR(NumericMatrix y, double beta, NumericMatrix z_i, double zeta_a, double zeta_b, NumericVector count_z, NumericVector sample, int N, int Nstar, int M, int K, NumericVector delta_vector, NumericVector acc_vector, double acc_star, NumericMatrix Acc_ite, double ite, NumericMatrix zi_ref, double distr_option)
{

  for (int i = 0; i < Nstar; i++)
  {
    //
    //
    int ind = sample[i];

    double count_z_i = count_z[ind];

    //
    NumericMatrix z_i_prop = clone(z_i);
    NumericVector z_i_x_aux = z_i_prop(ind, _);
    NumericVector z_i_x_pr = clone(z_i_x_aux);
    NumericVector z_i_x(K);

    double lm = delta_vector[ind];

    for (int k = 0; k < K; k++)
    {

      GetRNGstate();
      double z_i_prop_x = R::rnorm(z_i_x_pr(k), exp(2 * lm));
      PutRNGstate();
      z_i_x(k) = z_i_prop_x;
    }

    z_i_prop(ind, _) = clone(z_i_x);
    NumericVector pois(N - 1);
    NumericVector pois_prop(N - 1);

    int idx = 0;
    for (int j = 0; j < N; j++)
    {
      if (j != ind)
      {
        double d = 0;
        double d_prop = 0;
        for (int k = 0; k < K; k++)
        {
          d = d + pow(z_i(ind, k) - z_i(j, k), 2);
          d_prop = d_prop + pow(z_i_prop(ind, k) - z_i_prop(j, k), 2);
        }
        if (distr_option == 1)
        {

          double lambda_aux = exp(beta - d);
          double lambda_aux_prop = exp(beta - d_prop);

          double y_check = y(ind, j);
          bool check_na = NumericVector::is_na(y_check);

          if (check_na == false)
          {

            double pois_aux = R::dpois(y_check, lambda_aux, true);
            double pois_prop_aux = R::dpois(y_check, lambda_aux_prop, true);
            pois[idx] = pois_aux;
            pois_prop[idx] = pois_prop_aux;
          }
        }
        else if (distr_option == 0)
        {

          double lambda_aux = exp(beta - d);
          lambda_aux = lambda_aux / (1 + lambda_aux);

          double lambda_aux_prop = exp(beta - d_prop);
          lambda_aux_prop = lambda_aux_prop / (1 + lambda_aux_prop);

          double y_check = y(ind, j);
          bool check_na = NumericVector::is_na(y_check);

          if (check_na == false)
          {

            double pois_aux = R::dbinom(y_check, 1, lambda_aux, true);
            double pois_prop_aux = R::dbinom(y_check, 1, lambda_aux_prop, true);
            pois[idx] = pois_aux;
            pois_prop[idx] = pois_prop_aux;
          }
        }

        idx = idx + 1;
      }
    }

    double prior = 0;
    double prior_prop = 0;
    for (int k = 0; k < K; k++)
    {
      double prior_aux = R::dnorm(z_i(ind, k), zeta_a, zeta_b, true);
      double prior_prop_aux = R::dnorm(z_i_x[k], zeta_a, zeta_b, true);
      prior = prior + prior_aux;
      prior_prop = prior_prop + prior_prop_aux;
    }

    double test = prior_prop - prior + sum(pois_prop - pois);
    GetRNGstate();
    double rand = R::runif(0, 1);
    PutRNGstate();

    double acceptance_z = 0;

    if (test > log(rand))
    {
      z_i(ind, _) = clone(z_i_x);

      // acceptance_z = 1;
    }
    else
    {

      // acceptance_z = 0;
    };

    double lm_n;
    int ite_x = ite;

    if (((ite_x + 1) % 100) == 0)
    {

      NumericVector acc_h = Acc_ite(_, ind);
      NumericVector acc_h_sub = acc_h[cseq(ite_x - 100 + 1, ite_x - 1, 1)];

      if (mean(acc_h_sub) < acc_star)
      {
        lm_n = lm - 1 / (ite / 100);
      }
      else
      {

        lm_n = lm + 1 / (ite / 100);
      };

      delta_vector(ind) = lm_n;
    }

    if (exp(test) < 1)
    {
      acceptance_z = exp(test);
    }
    else
    {
      acceptance_z = 1;
    };

    count_z_i = count_z_i + 1;
    count_z(ind) = count_z_i;

    acc_vector(ind) = acceptance_z;
  }

  List result = List::create(Named("z_i") = z_i, Named("delta_vector") = delta_vector, Named("acc_vector") = acc_vector);
  return result;
};

NumericMatrix subsetMatrixByColumns(NumericMatrix A, NumericVector columnIndices)
{
  int N = A.nrow();
  int d = columnIndices.size();
  NumericMatrix subset(N, d);

  for (int i = 0; i < d; ++i)
  {
    int columnIndex = columnIndices[i]; // Subtract 1 to match 0-based indexing in C++
    subset(_, i) = A(_, columnIndex);
  }

  return subset;
}

arma::mat direct_sum(arma::mat A, arma::mat B)
{
  // Calculate the size of the resulting matrix
  int m = A.n_rows + B.n_rows;
  int n = A.n_cols + B.n_cols;
  // Create the resulting matrix and fill it with zeros
  arma::mat C = zeros<arma::mat>(m, n);
  // Copy the elements of A into the top-left corner of C
  C(arma::span(0, A.n_rows - 1), arma::span(0, A.n_cols - 1)) = A;
  // Copy the elements of B into the bottom-right corner of C
  C(arma::span(A.n_rows, m - 1), arma::span(A.n_cols, n - 1)) = B;
  return C;
}

double vector_norm(NumericVector x)
{
  double sum_squares = 0.0;
  int n = x.size();
  for (int i = 0; i < n; i++)
  {
    sum_squares += x[i] * x[i];
  }
  return sqrt(sum_squares);
}

NumericVector random_unit_vector(int d)
{
  // Generate d+1 random numbers between -1 and 1
  NumericVector coords(d + 1);
  for (int i = 0; i <= d; i++)
  {
    coords[i] = R::runif(-1, 1);
  }

  // Calculate the length of the vector
  double length = 0.0;
  for (int i = 0; i <= d; i++)
  {
    length += coords[i] * coords[i];
  }
  length = sqrt(length);

  // Normalize the vector to length 1
  for (int i = 0; i <= d; i++)
  {
    coords[i] /= length;
  }

  return coords;
}

NumericVector upper_triangle(NumericMatrix x)
{
  int n = x.nrow();
  NumericVector out(n * (n - 1) / 2);
  int idx = 0;

  for (int i = 0; i < n - 1; i++)
  {
    for (int j = i + 1; j < n; j++)
    {
      out[idx++] = x(i, j);
    }
  }

  return out;
}

#define PBSTR "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
#define PBWIDTH 60
void printProgress(double percentage)
{
  int val = (int)(percentage * 100);
  int lpad = (int)(percentage * PBWIDTH);
  int rpad = PBWIDTH - lpad;
  Rprintf("\r%3d%% [%.*s%*s]", val, lpad, PBSTR, rpad, "");
  //fflush(stdout);
};

//' ARMS Individual
//'
//' The function runs an MCMC algorithm to estimate Latent Space Models using Adaptive Multiple Random Scan.
//' The latent coordinates get probabilistically updated individually.
//'
//' @param y N x N adjacency matrix
//' @param beta   scalar, initial value for the intercept
//' @param beta_a scalar, prior mean for the intercept
//' @param beta_b scalar, prior standard deviation for the intercept
//' @param z_i, N x K matrix, initial value for the latent coordinates
//' @param zeta_a scalar, prior mean of the latent coordinates, usually set to 0
//' @param zeta_b scalar, prior standard deviation of the latent coordinates, usually set to 0
//' @param lam_ad_z N vector, lambda parameters, see Andrieu and Thoms 2008, A tutorial on adaptive MCMC
//' @param mu_mat_z N x K matrix, mu parameters, see Andrieu and Thoms 2008
//' @param Sigma_ad_z List of N K x K matrices,  see Andrieu and Thoms 2008
//' @param N scalar, number of nodes
//' @param K scalar, number of latent dimensions
//' @param prop_sigma_beta scalar, proposal standard deviation for the intercept beta
//' @param acc_target_z scalar, target acceptance rate for the latent coordinates, default 0.234
//' @param scan_every integer, how many iteration before the update of selection probabilities
//' @param Iterations integer, MCMC number of iterations
//' @param k_shift    scalar, shift of the decision rule sigmoid function
//' @param eq_option  binary, if 1 selection probabilities with equal default value
//' @param eq_prob    scalar, between 0 and 1, if eq_option = 1, this is the default value
//' @param ad_option  binary, if ad_option 0 we use the adaptive proposal by Andrieu and Thoms 2008 otherwise by Latuszynski, Roberts and Rosenthal 2013
//' @param distr_option binary, if 0 it runs the algorithm for a LS with Bernoulli likelihood, otherwise with Poisson likelihood, default 1
//' @return List of saved quantities  beta_ite intercpet chain,zeta_ite list of latent positions chains, P_ite matrix of selection probabilities chains, Acc_ite matrix of acceptance rates
//' @export
// [[Rcpp::export]]
 List MCMC_ADRS_AD(NumericMatrix y, double beta, double beta_a, double beta_b, NumericMatrix z_i, double zeta_a, double zeta_b, NumericVector lam_ad_z, NumericMatrix mu_mat_z, List Sigma_ad_z, int N, int K, double prop_sigma_beta, double acc_target_z = 0.234, int scan_every = 100, double Iterations = 1000, double k_shift = 0, double eq_option = 0, double eq_prob = 0.5, double ad_option = 0, double distr_option = 1)
 {

   int M = (N / 2.0) * (N - 1);
   NumericMatrix P_ite(Iterations, N);
   NumericMatrix Acc_ite(Iterations, N);
   NumericVector acc_vector_mean(N);
   NumericVector delta_vector(N);

   NumericVector count_z(N);
   NumericVector p_vector(N);
   p_vector.fill(1);

   if (scan_every >= Iterations)
   {
     stop("Error: variable:scan_every must be less than variable:Iterations");
   }

   NumericVector vec_y = upper_triangle(y);
   NumericVector vec_y_na = clone(vec_y);

   NumericMatrix zi_ref(N, K);
   NumericVector beta_ite(Iterations);

   List zeta_ite(Iterations);
   List zeta_iteM(K);

   for (int k = 0; k < K; k++)
   {

     NumericMatrix Zk(Iterations, N);
     zeta_iteM(k) = Zk;
   }

   NumericVector range = cseq(0, N - 1, 1); // 19

   for (double ite = 0; ite < Iterations; ite++)
   {

     double prct = (ite + 1) / Iterations;
     printProgress(prct);

     NumericMatrix dist = dist_lcpp(z_i, z_i);
     beta = betaSamplingEn(vec_y, beta, dist, beta_a, beta_b, N, M, prop_sigma_beta, distr_option);
     beta_ite(ite) = beta;

     ///////////////////////////////////
     NumericVector select_v(N);

     P_ite(ite, _) = p_vector;

     for (int i = 0; i < N; i++)
     {

       double p_i;

       if (eq_option == 0)
       {
         p_i = p_vector(i);
       }
       else
       {

         p_i = eq_prob;
       }

       GetRNGstate();
       double bin = R::rbinom(1, p_i);
       PutRNGstate();
       if (bin == 1)
       {
         select_v(i) = true;
       }
       else
       {
         select_v(i) = false;
       }
     }

     NumericVector sample_v = range[select_v == true];

     double Nstar = sample_v.length();

     NumericVector acc_vector(N);

     if (ite > 1)
     {
       acc_vector = Acc_ite(ite - 1, _);
     }

     if (ad_option == 0)
     {

       List z_i_aux_obj = zetaSamplingEn(y, beta, z_i, zeta_a, zeta_b, lam_ad_z, mu_mat_z, Sigma_ad_z, count_z, sample_v, N, Nstar, M, K, acc_vector, acc_target_z, ite, zi_ref, distr_option);

       // //
       NumericMatrix ziRes = z_i_aux_obj("z_i");
       z_i = clone(ziRes);

       NumericVector aux_vec_lambda = z_i_aux_obj("lam_ad_z");
       lam_ad_z = aux_vec_lambda;

       NumericVector aux_vec_acc = z_i_aux_obj("acc_vector");
       acc_vector = clone(aux_vec_acc);

       Acc_ite(ite, _) = acc_vector;

       NumericMatrix aux_mat_z = z_i_aux_obj("mu_mat_z");
       mu_mat_z = aux_mat_z;

       List aux_list_z = z_i_aux_obj("Sigma_ad_z");
       Sigma_ad_z = aux_list_z;
     }
     else
     {

       List z_i_aux_obj = zetaSamplingEnR(y, beta, z_i, zeta_a, zeta_b, count_z, sample_v, N, Nstar, M, K, delta_vector, acc_vector, acc_target_z, Acc_ite, ite, zi_ref, distr_option);

       NumericMatrix ziRes = z_i_aux_obj("z_i");
       z_i = clone(ziRes);

       NumericVector aux_vec_acc = z_i_aux_obj("acc_vector");
       acc_vector = clone(aux_vec_acc);

       Acc_ite(ite, _) = acc_vector;

       NumericVector aux_delta = z_i_aux_obj("delta_vector");
       delta_vector = clone(aux_delta);
     }

     //
     //
     for (int k = 0; k < K; k++)
     {
       NumericVector zeta_demean = z_i(_, k);
       zeta_demean = zeta_demean - mean(zeta_demean);
       z_i(_, k) = clone(zeta_demean);
     }

     //

     zeta_ite(ite) = clone(z_i);

     for (int k = 0; k < K; k++)
     {

       NumericVector z_i_aux = z_i(_, k);
       NumericMatrix Zk_aux = zeta_iteM(k);
       Zk_aux(ite, _) = clone(z_i_aux);
       zeta_iteM(k) = Zk_aux;
     }

     /// when it is time ////////////////////+

     int ite_x = ite;

     if (ite_x > scan_every)
     {
       if (((ite_x + 1) % scan_every) == 0)
       {

         NumericMatrix acc_aux_m = Acc_ite(Rcpp::Range(ite - scan_every + 1, ite), _);

         for (int i = 0; i < N; i++)
         {

           NumericVector acc_aux = acc_aux_m(_, i);
           acc_vector_mean(i) = mean(acc_aux);
         }

         // NumericVector sq_diff = abs(acc_vector_mean - 0.234);
         NumericVector sq_diff = (acc_vector_mean - acc_target_z);
         // p_vector  = 1/(1+exp(-1*( sq_diff - k_shift)  ));
         p_vector = 1 / (1 + exp(1 * (sq_diff)));
       }
     }
     //
   }

   List RESULT = List::create(Named("beta_ite") = beta_ite, Named("zeta_ite") = zeta_ite, Named("P_ite") = P_ite, Named("Acc_ite") = Acc_ite);
   return RESULT;
 }

//' ARMS Block
//'
//' The function runs an MCMC algorithm to estimate Latent Space Models using Adaptive Multiple Random Scan.
//' The latent coordinates get probabilistically updated in blocks.
//'
//' @param y N x N adjacency matrix
//' @param beta   scalar, initial value for the intercept
//' @param beta_a scalar, prior mean for the intercept
//' @param beta_b scalar, prior standard deviation for the intercept
//' @param z_i, N x K matrix, initial value for the latent coordinates
//' @param zeta_a scalar, prior mean of the latent coordinates, usually set to 0
//' @param zeta_b scalar, prior standard deviation of the latent coordinates, usually set to 0
//' @param lam_ad_z N vector, lambda parameters, see Andrieu and Thoms 2008, A tutorial on adaptive MCMC
//' @param mu_mat_z N x K matrix, mu parameters, see Andrieu and Thoms 2008
//' @param Sigma_ad_z List of N K x K matrices,  see Andrieu and Thoms 2008
//' @param N scalar, number of nodes
//' @param K scalar, number of latent dimensions
//' @param prop_sigma_beta scalar, proposal standard deviation for the intercept beta
//' @param acc_target_z scalar, target acceptance rate for the latent coordinates, default 0.234
//' @param belong N vector, it contains a index of group for each node, e.g. N = 5, c(1,1,1, 2, 2) creates two groups
//' @param scan_every integer, how many iteration before the update of selection probabilities
//' @param Iterations integer, MCMC number of iterations
//' @param k_shift    scalar, shift of the decision rule sigmoid function
//' @param eq_option  binary, if 1 selection probabilities with equal default value
//' @param ad_option  binary, if ad_option 0 we use the adaptive proposal by Andrieu and Thoms 2008 otherwise by Latuszynski, Roberts and Rosenthal 2013
//' @param distr_option binary, if 0 it runs the algorithm for a LS with Bernoulli likelihood, otherwise with Poisson likelihood, default 1
//' @return List of saved quantities  beta_ite intercpet chain,zeta_ite list of latent positions chains, P_ite matrix of selection probabilities chains, Acc_ite matrix of acceptance rates
//' @export
// [[Rcpp::export]]
 List MCMC_ADRS_BLOCK_AD(NumericMatrix y, double beta, double beta_a, double beta_b, NumericMatrix z_i, double zeta_a, double zeta_b, NumericVector lam_ad_z, NumericMatrix mu_mat_z, List Sigma_ad_z, int N, int K, double prop_sigma_beta, double acc_target_z, NumericVector belong, int scan_every = 100, double Iterations = 1000, double k_shift = 0.5, double ad_option = 0, double eq_option = 0, double distr_option = 1)
 {
   int M = (N/2.0)*(N-1);
   NumericMatrix Acc_ite(Iterations, N);
   NumericVector acc_vector_mean(N);
   NumericVector delta_vector(N);
   delta_vector.fill(0);

   NumericVector count_z(N);
   NumericVector list_blocks = sort_unique(belong);
   double N_blocks = list_blocks.length();

   NumericVector p_vector(N_blocks);
   p_vector.fill(1);
   NumericVector p_vector_norm = p_vector / sum(p_vector);
   NumericMatrix P_ite(Iterations, N_blocks);

   if (scan_every >= Iterations)
   {
     stop("Error: variable:scan_every must be less than variable:Iterations");
   }

   NumericVector vec_y = upper_triangle(y);
   NumericVector vec_y_na = clone(vec_y);

   NumericMatrix zi_ref(N, K);
   NumericVector beta_ite(Iterations);

   List zeta_ite(Iterations);
   List zeta_iteM(K);

   for (int k = 0; k < K; k++)
   {

     NumericMatrix Zk(Iterations, N);
     zeta_iteM(k) = Zk;
   }

   NumericVector range = cseq(0, N - 1, 1); // 19

   for (double ite = 0; ite < Iterations; ite++)
   {

     double prct = (ite + 1) / Iterations;
     printProgress(prct);

     NumericMatrix dist = dist_lcpp(z_i, z_i);
     beta = betaSamplingEn(vec_y, beta, dist, beta_a, beta_b, N, M, prop_sigma_beta, distr_option);
     beta_ite(ite) = beta;

     ///////////////////////////////////
     NumericVector select_v(N_blocks);

     // p_vector = p_vector/sum(p_vector);

     P_ite(ite, _) = p_vector_norm;

     for (int i = 0; i < N_blocks; i++)
     {

       double p_i = p_vector_norm(i);

       if (eq_option == 1)
       {

         p_i = 1 / N_blocks;
       }

       // double

       GetRNGstate();
       double bin = R::rbinom(1, p_i);
       PutRNGstate();
       if (bin == 1)
       {
         select_v(i) = true;
       }
       else
       {
         select_v(i) = false;
       }
     }

     NumericVector selected_blocks = list_blocks[select_v == true];

     NumericVector sample_v = in_operator(range, belong, selected_blocks);

     double Nstar = sample_v.length();

     NumericVector acc_vector(N);

     if (ite > 1)
     {
       acc_vector = Acc_ite(ite - 1, _);
     }

     if (ad_option == 0)
     {

       List z_i_aux_obj = zetaSamplingEn(y, beta, z_i, zeta_a, zeta_b, lam_ad_z, mu_mat_z, Sigma_ad_z, count_z, sample_v, N, Nstar, M, K, acc_vector, acc_target_z, ite, zi_ref, distr_option);

       if (ite == 1999)
       {
         zi_ref = clone(z_i);
       }

       // //
       NumericMatrix ziRes = z_i_aux_obj("z_i");
       z_i = clone(ziRes);

       NumericVector aux_vec_lambda = z_i_aux_obj("lam_ad_z");
       lam_ad_z = aux_vec_lambda;

       NumericVector aux_vec_acc = z_i_aux_obj("acc_vector");
       acc_vector = clone(aux_vec_acc);

       Acc_ite(ite, _) = acc_vector;

       NumericMatrix aux_mat_z = z_i_aux_obj("mu_mat_z");
       mu_mat_z = aux_mat_z;

       List aux_list_z = z_i_aux_obj("Sigma_ad_z");
       Sigma_ad_z = aux_list_z;
     }
     else
     {

       List z_i_aux_obj = zetaSamplingEnR(y, beta, z_i, zeta_a, zeta_b, count_z, sample_v, N, Nstar, M, K, delta_vector, acc_vector, acc_target_z, Acc_ite, ite, zi_ref, distr_option);

       NumericMatrix ziRes = z_i_aux_obj("z_i");
       z_i = clone(ziRes);

       NumericVector aux_vec_acc = z_i_aux_obj("acc_vector");
       acc_vector = clone(aux_vec_acc);

       Acc_ite(ite, _) = acc_vector;

       NumericVector aux_delta = z_i_aux_obj("delta_vector");
       delta_vector = clone(aux_delta);
     }

     //
     //
     for (int k = 0; k < K; k++)
     {
       NumericVector zeta_demean = z_i(_, k);
       zeta_demean = zeta_demean - mean(zeta_demean);
       z_i(_, k) = clone(zeta_demean);
     }
     //

     zeta_ite(ite) = clone(z_i);

     for (int k = 0; k < K; k++)
     {

       NumericVector z_i_aux = z_i(_, k);
       NumericMatrix Zk_aux = zeta_iteM(k);
       Zk_aux(ite, _) = clone(z_i_aux);
       zeta_iteM(k) = Zk_aux;
     }

     /// when it is time ////////////////////

     int ite_x = ite;

     if (ite_x > 100)
     {
       if (((ite_x + 1) % scan_every) == 0)
       {

         NumericMatrix acc_aux_m = Acc_ite(Rcpp::Range(ite - scan_every + 1, ite), _);

         for (int i = 0; i < N; i++)
         {

           NumericVector acc_aux = acc_aux_m(_, i);
           acc_vector_mean(i) = mean(acc_aux);
         }

         // NumericVector sq_diff = abs(acc_vector_mean - 0.234);
         NumericVector sq_diff = (acc_vector_mean - acc_target_z);
         // p_vector  = 1/(1+exp(-1*( sq_diff - k_shift)  ));
         NumericVector p_vector_i = 1 / (1 + exp(1 * (sq_diff)));

         for (int i = 0; i < selected_blocks.length(); i++)
         {

           double block_i = selected_blocks(i);
           NumericVector belong_selected = range[belong == block_i];

           NumericVector p_vector_i_selected = p_vector_i[belong_selected];
           double p_vector_i_selected_mean = mean(p_vector_i_selected);
           p_vector(i) = p_vector_i_selected_mean;
         }

         p_vector_norm = p_vector / sum(p_vector);
       }
     }
   }

   List RESULT = List::create(Named("beta_ite") = beta_ite, Named("zeta_ite") = zeta_ite, Named("P_ite") = P_ite, Named("Acc_ite") = Acc_ite);
   return RESULT;
 }

//' Systematic Gibbs
//'
//' The function runs an MCMC algorithm to estimate Latent Space Models using Systematic Random Scan.
//' The latent coordinates get all sequentially updated.
//'
//' @param y N x N adjacency matrix
//' @param beta   scalar, initial value for the intercept
//' @param beta_a scalar, prior mean for the intercept
//' @param beta_b scalar, prior standard deviation for the intercept
//' @param z_i, N x K matrix, initial value for the latent coordinates
//' @param zeta_a scalar, prior mean of the latent coordinates, usually set to 0
//' @param zeta_b scalar, prior standard deviation of the latent coordinates, usually set to 0
//' @param lam_ad_z N vector, lambda parameters, see Andrieu and Thoms 2008, A tutorial on adaptive MCMC
//' @param mu_mat_z N x K matrix, mu parameters, see Andrieu and Thoms 2008
//' @param Sigma_ad_z List of N K x K matrices,  see Andrieu and Thoms 2008
//' @param N scalar, number of nodes
//' @param K scalar, number of latent dimensions
//' @param prop_sigma_beta scalar, proposal standard deviation for the intercept beta
//' @param acc_target_z scalar, target acceptance rate for the latent coordinates, default 0.234
//' @param Iterations integer, MCMC number of iterations
//' @param ad_option  binary, if ad_option 0 we use the adaptive proposal by Andrieu and Thoms 2008 otherwise by Latuszynski, Roberts and Rosenthal 2013
//' @param distr_option binary, if 0 it runs the algorithm for a LS with Bernoulli likelihood, otherwise with Poisson likelihood, default 1
//' @return List of saved quantities  beta_ite intercpet chain,zeta_ite list of latent positions chains, P_ite matrix of selection probabilities chains, Acc_ite matrix of acceptance rates
//' @export
// [[Rcpp::export]]
 List MCMC_AD(NumericMatrix y, double beta, double beta_a, double beta_b, NumericMatrix z_i, double zeta_a, double zeta_b, NumericVector lam_ad_z, NumericMatrix mu_mat_z, List Sigma_ad_z, int N,  int K, double prop_sigma_beta, double acc_target_z, double Iterations = 1000, double ad_option = 0, double distr_option = 1)
 {
   int M = (N/2.0)*(N-1);
   NumericVector count_z(N);
   NumericMatrix Acc_ite(Iterations, N);
   NumericVector acc_vector_mean(N);
   NumericVector acc_vector(N);
   NumericVector delta_vector(N);
   delta_vector.fill(0);

   NumericVector vec_y = upper_triangle(y);
   NumericVector vec_y_na = clone(vec_y);

   NumericMatrix zi_ref(N, K);
   NumericVector beta_ite(Iterations);

   List zeta_ite(Iterations);
   List zeta_iteM(K);

   for (int k = 0; k < K; k++)
   {

     NumericMatrix Zk(Iterations, N);
     zeta_iteM(k) = Zk;
   }

   NumericVector range = cseq(0, N - 1, 1); // 19

   for (double ite = 0; ite < Iterations; ite++)
   {

     double prct = (ite + 1) / Iterations;
     printProgress(prct);

     NumericMatrix dist = dist_lcpp(z_i, z_i);
     beta = betaSamplingEn(vec_y, beta, dist, beta_a, beta_b, N, M, prop_sigma_beta, distr_option);
     beta_ite(ite) = beta;

     ///////////////////////////////////

     NumericVector sample_v = cseq(0, N - 1, 1);

     double Nstar = sample_v.length();

     if (ad_option == 0)
     {

       List z_i_aux_obj = zetaSamplingEn(y, beta, z_i, zeta_a, zeta_b, lam_ad_z, mu_mat_z, Sigma_ad_z, count_z, sample_v, N, Nstar, M, K, acc_vector, acc_target_z, ite, zi_ref, distr_option);

       if (ite == 1999)
       {
         zi_ref = clone(z_i);
       }

       // //
       NumericMatrix ziRes = z_i_aux_obj("z_i");
       z_i = clone(ziRes);

       NumericVector aux_vec_lambda = z_i_aux_obj("lam_ad_z");
       lam_ad_z = aux_vec_lambda;

       NumericVector aux_vec_acc = z_i_aux_obj("acc_vector");
       acc_vector = clone(aux_vec_acc);

       Acc_ite(ite, _) = acc_vector;

       NumericMatrix aux_mat_z = z_i_aux_obj("mu_mat_z");
       mu_mat_z = aux_mat_z;

       List aux_list_z = z_i_aux_obj("Sigma_ad_z");
       Sigma_ad_z = aux_list_z;
     }
     else
     {

       List z_i_aux_obj = zetaSamplingEnR(y, beta, z_i, zeta_a, zeta_b, count_z, sample_v, N, Nstar, M, K, delta_vector, acc_vector, acc_target_z, Acc_ite, ite, zi_ref, distr_option);

       NumericMatrix ziRes = z_i_aux_obj("z_i");
       z_i = clone(ziRes);

       NumericVector aux_vec_acc = z_i_aux_obj("acc_vector");
       acc_vector = clone(aux_vec_acc);

       Acc_ite(ite, _) = acc_vector;

       NumericVector aux_delta = z_i_aux_obj("delta_vector");
       delta_vector = clone(aux_delta);
     }

     //
     for (int k = 0; k < K; k++)
     {
       NumericVector zeta_demean = z_i(_, k);
       zeta_demean = zeta_demean - mean(zeta_demean);
       z_i(_, k) = clone(zeta_demean);
     }

     zeta_ite(ite) = clone(z_i);

     for (int k = 0; k < K; k++)
     {

       NumericVector z_i_aux = z_i(_, k);
       NumericMatrix Zk_aux = zeta_iteM(k);
       Zk_aux(ite, _) = clone(z_i_aux);
       zeta_iteM(k) = Zk_aux;
     }
   }

   List RESULT = List::create(Named("beta_ite") = beta_ite, Named("zeta_ite") = zeta_ite, Named("Acc_ite") = Acc_ite);
   return RESULT;
 }
