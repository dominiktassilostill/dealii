// -----------------------------------------------------------------------------
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception OR LGPL-2.1-or-later
// Copyright (C) 2020 - 2026 by the deal.II authors
//
// This file is part of the deal.II library.
//
// Detailed license information governing the source code and contributions
// can be found in LICENSE.md and CONTRIBUTING.md at the top level directory.
//
// -----------------------------------------------------------------------------


#include <deal.II/base/config.h>

#include <deal.II/base/exception_macros.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/point.h>
#include <deal.II/base/polynomial.h>
#include <deal.II/base/polynomials_wedge.h>
#include <deal.II/base/scalar_polynomials_base.h>
#include <deal.II/base/scalar_polynomials_vandermonde_base.h>
#include <deal.II/base/table.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/utilities.h>

#include <Kokkos_Macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

DEAL_II_NAMESPACE_OPEN



template <int dim>
ScalarLagrangePolynomialWedge<dim>::ScalarLagrangePolynomialWedge(
  const unsigned int             degree,
  const std::vector<Point<dim>> &support_points)
  : ScalarPolynomialsVandermondeBase<dim>(degree, support_points.size())
{
  AssertDimension(dim, 3);

  const unsigned int n_dofs = (degree + 1) * (degree + 1) * (degree + 2) / 2;
  AssertDimension(n_dofs, support_points.size());

  this->reinit(support_points);
}



template <int dim>
ScalarLagrangePolynomialWedge<dim>::ScalarLagrangePolynomialWedge(
  const unsigned int degree)
  : ScalarLagrangePolynomialWedge<dim>(degree,
                                       internal::get_wedge_support_points<dim>(
                                         degree))
{
  AssertThrow(
    degree == 1 || degree == 2,
    ExcNotImplemented(
      "This constructor only works for linear and quadratic elements."));
}



template <int dim>
double
ScalarLagrangePolynomialWedge<
  dim>::evaluate_orthogonal_basis_function_by_degree(const unsigned int i,
                                                     const unsigned int j,
                                                     const unsigned int k,
                                                     const Point<dim>  &p) const
{
  AssertIndexRange(i + j, this->degree() + 1);
  AssertIndexRange(k, this->degree() + 1);

  const double x = p[0];
  const double y = p[1];
  const double z = p[2];

  // the basis function looks like
  // P_i^{0,0}(2x/(1-y)-1) * (1-y)^i * P_j^{2*i+1,0}(2*y-1) * P_k^{0,0}(2z-1)
  // separate it into
  // P_i^{0,0}(2x/(1-y)-1) * (1-y)^i
  // P_j^{2*i+1,0}(2*y-1)
  // P_k^{0,0}(2z-1)
  // the first is a homogenized Jacobi polynomial
  // define s = 1 - y so the first term can be written as
  // Q_i^{0,0}(x,s) = P_i^{0,0}(2 * x/s - 1) * s^i
  const double s = 1 - y;

  const double Qi =
    Polynomials::jacobi_polynomial_homogenized_value<double>(i, 0, 0, x, s);

  const double Pj =
    Polynomials::jacobi_polynomial_value<double>(j, 2 * i + 1, 0, y, true);

  const double Pk =
    Polynomials::jacobi_polynomial_value<double>(k, 0, 0, z, true);

  const double phi = Qi * Pj * Pk;

  if (std::fabs(phi) < 1e-14)
    return 0.0;

  return phi;
}



template <int dim>
double
ScalarLagrangePolynomialWedge<dim>::evaluate_orthogonal_basis_function(
  const unsigned int i,
  const Point<dim>  &p) const
{
  AssertIndexRange(i, this->n());

  // find corresponding entry to i
  // it holds 0 <= j + k <= degree
  // 0 <= l <= degree
  for (unsigned int j = 0, counter = 0; j < this->degree() + 1; ++j)
    for (unsigned int k = 0; k < this->degree() + 1 - j; ++k)
      for (unsigned int l = 0; l < this->degree() + 1; ++l, ++counter)
        if (counter == i)
          return evaluate_orthogonal_basis_function_by_degree(j, k, l, p);

  DEAL_II_ASSERT_UNREACHABLE();
  return 0;
}



template <int dim>
Tensor<1, dim>
ScalarLagrangePolynomialWedge<dim>::
  evaluate_orthogonal_basis_derivative_by_degree(const unsigned int i,
                                                 const unsigned int j,
                                                 const unsigned int k,
                                                 const Point<dim>  &p) const
{
  AssertIndexRange(i + j, this->degree() + 1);
  AssertIndexRange(k, this->degree() + 1);

  Tensor<1, dim> grad;

  const double x = p[0];
  const double y = p[1];
  const double z = p[2];

  // the basis function looks like
  // P_i^{0,0}(2x/(1-y)-1) * (1-y)^i * P_j^{2*i+1,0}(2*y-1) * P_k^{0,0}(2z-1)
  // separate it into
  // P_i^{0,0}(2x/(1-y)-1) * (1-y)^i
  // and
  // P_j^{2*i+1,0}(2*y-1)
  // the first is a homogenized Jacobi polynomial
  // define s = 1 - y so the first term can be written as
  // Q_i^{0,0}(x,s) = P_i^{0,0}(2 * x/s - 1) * s^i

  // to get the derivatives just use the product rule with all terms
  const double s     = 1 - y;
  const double ds_dy = -1.0;

  const double Qi =
    Polynomials::jacobi_polynomial_homogenized_value<double>(i, 0, 0, x, s);
  const double Pj =
    Polynomials::jacobi_polynomial_value<double>(j, 2 * i + 1, 0, y, true);
  const double Pk =
    Polynomials::jacobi_polynomial_value<double>(k, 0, 0, z, true);

  const auto dQi_dx =
    Polynomials::jacobi_polynomial_homogenized_derivative<double>(
      1, 0, i, 0, 0, x, s);

  const auto dQi_ds =
    Polynomials::jacobi_polynomial_homogenized_derivative<double>(
      0, 1, i, 0, 0, x, s);

  const auto dPj_dy =
    Polynomials::jacobi_polynomial_derivative<double>(j, 2 * i + 1, 0, y, true);

  const double dPk_dz =
    Polynomials::jacobi_polynomial_derivative<double>(k, 0, 0, z, true);

  grad[0] = dQi_dx * Pj * Pk;
  if constexpr (dim > 1)
    grad[1] = dQi_ds * ds_dy * Pj * Pk + Qi * dPj_dy * Pk;
  if constexpr (dim > 2)
    grad[2] = Qi * Pj * dPk_dz;

  for (unsigned int d = 0; d < dim; ++d)
    if (std::fabs(grad[d]) < 1e-14)
      grad[d] = 0.0;

  return grad;
}



template <int dim>
Tensor<1, dim>
ScalarLagrangePolynomialWedge<dim>::evaluate_orthogonal_basis_derivative(
  const unsigned int i,
  const Point<dim>  &p) const
{
  AssertIndexRange(i, this->n());

  // find corresponding entry to i
  // it holds 0 <= j + k <= degree
  // 0 <= l <= degree
  for (unsigned int j = 0, counter = 0; j < this->degree() + 1; ++j)
    for (unsigned int k = 0; k < this->degree() + 1 - j; ++k)
      for (unsigned int l = 0; l < this->degree() + 1; ++l, ++counter)
        if (counter == i)
          if (counter == i)
            return evaluate_orthogonal_basis_derivative_by_degree(j, k, l, p);

  DEAL_II_ASSERT_UNREACHABLE();
  return Tensor<1, dim>();
}


template <int dim>
Tensor<2, dim>
ScalarLagrangePolynomialWedge<dim>::
  evaluate_orthogonal_basis_2nd_derivative_by_degree(const unsigned int i,
                                                     const unsigned int j,
                                                     const unsigned int k,
                                                     const Point<dim>  &p) const
{
  AssertIndexRange(i + j, this->degree() + 1);
  AssertIndexRange(k, this->degree() + 1);

  if constexpr (dim == 3)
    {
      // nothing to assert
    }
  else
    DEAL_II_ASSERT_UNREACHABLE();

  Tensor<2, dim> deriv;

  const double x = p[0];
  const double y = dim > 1 ? p[1] : 0.0;
  const double z = dim > 2 ? p[2] : 0.0;

  // define t = 1 - y
  // P_i^{0,0}(2x/t-1) * t^i
  // P_j^{2*i+1,0}(2*y-1)
  // P_k^{0,0}(2 z - 1)
  // =
  // Q_i^{0,0}(x,t)
  // Q_j^{2*i+1,0}(y,1)
  // P_k^{0,0}(2 z - 1)

  // get the second derivatives over the product rule
  const double t     = 1 - y;
  const double dt_dy = -1.0;

  // get the values of each polynomial
  const double Qi =
    Polynomials::jacobi_polynomial_homogenized_value<double>(i, 0, 0, x, t);
  const double Qj =
    Polynomials::jacobi_polynomial_value<double>(j, 2 * i + 1, 0, y, true);
  const double Pk =
    Polynomials::jacobi_polynomial_value<double>(k, 0, 0, z, true);

  // get the first derivatives of Qi
  const double dQi_dx =
    Polynomials::jacobi_polynomial_homogenized_derivative<double>(
      1, 0, i, 0, 0, x, t);
  const double dQi_dt =
    Polynomials::jacobi_polynomial_homogenized_derivative<double>(
      0, 1, i, 0, 0, x, t);

  // get the first derivatives of Qj
  const double dQj_dy =
    Polynomials::jacobi_polynomial_derivative<double>(j, 2 * i + 1, 0, y, true);

  // get the first derivative of Pk
  const double dPk_dz =
    Polynomials::jacobi_polynomial_derivative<double>(k, 0, 0, z, true);

  // get the second and mixed derivatives of Qi
  const double dQi_dx_dx =
    Polynomials::jacobi_polynomial_homogenized_derivative<double>(
      2, 0, i, 0, 0, x, t);
  const double dQi_dx_dt =
    Polynomials::jacobi_polynomial_homogenized_derivative<double>(
      1, 1, i, 0, 0, x, t);
  const double dQi_dt_dt =
    Polynomials::jacobi_polynomial_homogenized_derivative<double>(
      0, 2, i, 0, 0, x, t);

  // get the second and mixed derivatives of Qj
  const double dQj_dy_dy =
    Polynomials::jacobi_polynomial_kth_derivative<double>(
      2, j, 2 * i + 1, 0, y, true);

  // get the second derivative of Pk
  const double dPk_dz_dz =
    Polynomials::jacobi_polynomial_kth_derivative<double>(2, k, 0, 0, z, true);

  // now compute the entries
  deriv[0][0] = dQi_dx_dx * Qj * Pk;

  if constexpr (dim > 1)
    {
      deriv[0][1] = dQi_dx_dt * dt_dy * Qj * Pk + dQi_dx * dQj_dy * Pk;
      deriv[1][0] = deriv[0][1];
      deriv[1][1] = dQi_dt_dt * Qj * Pk + dQi_dt * dt_dy * dQj_dy * Pk * 2.0 +
                    Qi * dQj_dy_dy * Pk;

      if constexpr (dim > 2)
        {
          deriv[1][2] = dQi_dt * dt_dy * Qj * dPk_dz + Qi * dQj_dy * dPk_dz;
          deriv[2][1] = deriv[1][2];

          deriv[2][0] = dQi_dx * Qj * dPk_dz;
          deriv[0][2] = deriv[2][0];

          deriv[2][2] = Qi * Qj * dPk_dz_dz;
        }
    }

  for (unsigned int d = 0; d < dim; ++d)
    for (unsigned int e = 0; e < dim; ++e)
      if (std::fabs(deriv[d][e]) < 1e-14)
        deriv[d][e] = 0.0;

  return deriv;
}



template <int dim>
Tensor<2, dim>
ScalarLagrangePolynomialWedge<dim>::evaluate_orthogonal_basis_2nd_derivative(
  const unsigned int i,
  const Point<dim>  &p) const
{
  AssertIndexRange(i, this->n());

  if constexpr (dim == 3)
    {
      // find corresponding entrance to i
      // it holds 0 <= j + k <= degree, 0 <= l <= degree
      for (unsigned int j = 0, counter = 0; j < this->degree() + 1; ++j)
        for (unsigned int k = 0; k < this->degree() + 1 - j; ++k)
          for (unsigned int l = 0; l < this->degree() + 1; ++l, ++counter)
            if (counter == i)
              return evaluate_orthogonal_basis_2nd_derivative_by_degree(j,
                                                                        k,
                                                                        l,
                                                                        p);
    }

  DEAL_II_ASSERT_UNREACHABLE();
  return Tensor<2, dim>();
}



template <int dim>
std::string
ScalarLagrangePolynomialWedge<dim>::name() const
{
  return "ScalarLagrangePolynomialWedge";
}



template <int dim>
std::unique_ptr<ScalarPolynomialsBase<dim>>
ScalarLagrangePolynomialWedge<dim>::clone() const
{
  return std::make_unique<ScalarLagrangePolynomialWedge<dim>>(*this);
}



template <int dim>
void
ScalarLagrangePolynomialWedge<dim>::evaluate(
  const Point<dim>            &unit_point,
  std::vector<double>         &values,
  std::vector<Tensor<1, dim>> &grads,
  std::vector<Tensor<2, dim>> &grad_grads,
  std::vector<Tensor<3, dim>> &third_derivatives,
  std::vector<Tensor<4, dim>> &fourth_derivatives) const
{
  (void)third_derivatives;
  (void)fourth_derivatives;

  if (values.size() == this->n())
    for (unsigned int i = 0; i < this->n(); ++i)
      values[i] = this->compute_value(i, unit_point);

  if (grads.size() == this->n())
    for (unsigned int i = 0; i < this->n(); ++i)
      grads[i] = this->compute_grad(i, unit_point);

  if (grad_grads.size() == this->n())
    for (unsigned int i = 0; i < this->n(); ++i)
      grad_grads[i] = this->compute_grad_grad(i, unit_point);
}



template class ScalarLagrangePolynomialWedge<1>;
template class ScalarLagrangePolynomialWedge<2>;
template class ScalarLagrangePolynomialWedge<3>;

DEAL_II_NAMESPACE_CLOSE
