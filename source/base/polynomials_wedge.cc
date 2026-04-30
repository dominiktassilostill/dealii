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
#include <deal.II/base/polynomials_barycentric.h>
#include <deal.II/base/polynomials_wedge.h>
#include <deal.II/base/scalar_polynomials_base.h>
#include <deal.II/base/scalar_polynomials_vandermonde_base.h>
#include <deal.II/base/table.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/utilities.h>

#include <deal.II/grid/reference_cell.h>

#include <Kokkos_Macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

DEAL_II_NAMESPACE_OPEN

namespace
{
  unsigned int
  compute_n_polynomials_wedge(const unsigned int dim, const unsigned int degree)
  {
    if (dim == 3)
      {
        if (degree == 1)
          return 6;
        if (degree == 2)
          return 18;
      }

    DEAL_II_NOT_IMPLEMENTED();

    return 0;
  }
} // namespace



template <int dim>
ScalarLagrangePolynomialWedge<dim>::ScalarLagrangePolynomialWedge(
  const unsigned int degree)
  : ScalarPolynomialsBase<dim>(degree, compute_n_polynomials_wedge(dim, degree))
  , poly_tri(BarycentricPolynomials<2>::get_fe_p_basis(degree))
  , poly_line(BarycentricPolynomials<1>::get_fe_p_basis(degree))
{}



template <int dim>
double
ScalarLagrangePolynomialWedge<dim>::compute_value(const unsigned int i,
                                                  const Point<dim>  &p) const
{
  const auto pair = this->degree() == 1 ? internal::wedge_table_1[i] :
                                          internal::wedge_table_2[i];

  const Point<2> p_tri(p[0], p[1]);
  const auto     v_tri = poly_tri.compute_value(pair[0], p_tri);

  const Point<1> p_line(p[2]);
  const auto     v_line = poly_line.compute_value(pair[1], p_line);

  return v_tri * v_line;
}



template <int dim>
Tensor<1, dim>
ScalarLagrangePolynomialWedge<dim>::compute_grad(const unsigned int i,
                                                 const Point<dim>  &p) const
{
  const auto pair = this->degree() == 1 ? internal::wedge_table_1[i] :
                                          internal::wedge_table_2[i];

  const Point<2> p_tri(p[0], p[1]);
  const auto     v_tri = poly_tri.compute_value(pair[0], p_tri);
  const auto     g_tri = poly_tri.compute_grad(pair[0], p_tri);

  const Point<1> p_line(p[2]);
  const auto     v_line = poly_line.compute_value(pair[1], p_line);
  const auto     g_line = poly_line.compute_grad(pair[1], p_line);

  Tensor<1, dim> grad;
  grad[0] = g_tri[0] * v_line;
  grad[1] = g_tri[1] * v_line;
  grad[2] = v_tri * g_line[0];

  return grad;
}



template <int dim>
Tensor<2, dim>
ScalarLagrangePolynomialWedge<dim>::compute_grad_grad(const unsigned int i,
                                                      const Point<dim> &p) const
{
  (void)i;
  (void)p;

  DEAL_II_NOT_IMPLEMENTED();
  return Tensor<2, dim>();
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
  (void)grads;
  (void)grad_grads;
  (void)third_derivatives;
  (void)fourth_derivatives;

  if (values.size() == this->n())
    for (unsigned int i = 0; i < this->n(); ++i)
      values[i] = compute_value(i, unit_point);

  if (grads.size() == this->n())
    for (unsigned int i = 0; i < this->n(); ++i)
      grads[i] = compute_grad(i, unit_point);
}



template <int dim>
Tensor<1, dim>
ScalarLagrangePolynomialWedge<dim>::compute_1st_derivative(
  const unsigned int i,
  const Point<dim>  &p) const
{
  return compute_grad(i, p);
}



template <int dim>
Tensor<2, dim>
ScalarLagrangePolynomialWedge<dim>::compute_2nd_derivative(
  const unsigned int i,
  const Point<dim>  &p) const
{
  (void)i;
  (void)p;

  DEAL_II_NOT_IMPLEMENTED();

  return {};
}



template <int dim>
Tensor<3, dim>
ScalarLagrangePolynomialWedge<dim>::compute_3rd_derivative(
  const unsigned int i,
  const Point<dim>  &p) const
{
  (void)i;
  (void)p;

  DEAL_II_NOT_IMPLEMENTED();

  return {};
}



template <int dim>
Tensor<4, dim>
ScalarLagrangePolynomialWedge<dim>::compute_4th_derivative(
  const unsigned int i,
  const Point<dim>  &p) const
{
  (void)i;
  (void)p;

  DEAL_II_NOT_IMPLEMENTED();

  return {};
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



template class ScalarLagrangePolynomialWedge<1>;
template class ScalarLagrangePolynomialWedge<2>;
template class ScalarLagrangePolynomialWedge<3>;



template <int dim>
ScalarNodalPolynomialWedge<dim>::ScalarNodalPolynomialWedge(
  const unsigned int             degree,
  const unsigned int             n_dofs,
  const std::vector<Point<dim>> &support_points)
  : ScalarPolynomialsVandermondeBase<dim>(degree, n_dofs)
{
  AssertDimension(dim, 3);
  this->reinit(support_points);
}



template <int dim>
double
ScalarNodalPolynomialWedge<dim>::evaluate_orthogonal_basis_function_by_degree(
  const unsigned int i,
  const unsigned int j,
  const unsigned int k,
  const Point<dim>  &p) const
{
  AssertIndexRange(i + j, this->degree() + 1);
  AssertIndexRange(k, this->degree() + 1);

  const double x = p[0];
  const double y = p[1];
  const double z = p[2];

  const double factor = std::abs(1.0 - y) < 1e-14 ? 1.0 : 1.0 / (1.0 - y);

  const double x_contribution = i == 0 ?
                                  1.0 :
                                  Polynomials::jacobi_polynomial_value<double>(
                                    i, 0, 0, 2.0 * x * factor - 1.0, false) *
                                    std::pow(1.0 - y, i);

  const double y_contribution =
    Polynomials::jacobi_polynomial_value<double>(j, 2 * i + 1, 0, y, true);

  const double z_contribution =
    Polynomials::jacobi_polynomial_value<double>(k, 0, 0, z, true);

  const double phi = x_contribution * y_contribution * z_contribution;

  if (std::fabs(phi) < 1e-14)
    return 0.0;

  return phi;
}



template <int dim>
double
ScalarNodalPolynomialWedge<dim>::evaluate_orthogonal_basis_function(
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
ScalarNodalPolynomialWedge<dim>::evaluate_orthogonal_basis_derivative_by_degree(
  const unsigned int i,
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

  const double factor = std::abs(1.0 - y) < 1e-14 ? 0.0 : 1.0 / (1.0 - y);

  const double x_contribution = i == 0 ?
                                  1.0 :
                                  Polynomials::jacobi_polynomial_value<double>(
                                    i, 0, 0, 2.0 * x * factor - 1.0, false) *
                                    std::pow(1.0 - y, i);

  const double y_contribution =
    Polynomials::jacobi_polynomial_value<double>(j, 2 * i + 1, 0, y, true);

  const double z_contribution =
    Polynomials::jacobi_polynomial_value<double>(k, 0, 0, z, true);


  const double x_derivative =
    i == 1 ? 2.0 :
             Polynomials::jacobi_polynomial_derivative<double>(
               i, 0, 0, 2.0 * x * factor - 1.0, false) *
               std::pow(1.0 - y, std::max(1U, i)) * 2.0 * factor;

  grad[0] = x_derivative * y_contribution * z_contribution;

  const double y_derivative_x1 =
    i == 1 ? 1.0 :
             Polynomials::jacobi_polynomial_derivative<double>(
               i, 0, 0, 2.0 * x * factor - 1.0, false) *
               2.0 * x * factor * factor * std::pow(1.0 - y, std::max(1U, i));

  const double y_derivative_x2 =
    i == 1 ? 0.0 :
             Polynomials::jacobi_polynomial_value<double>(
               i, 0, 0, 2.0 * x * factor - 1.0, false) *
               i * (-1.0) * std::pow(1.0 - y, std::max(i - 1, 1U));

  const double y_derivative = Polynomials::jacobi_polynomial_derivative<double>(
                                j, 2 * i + 1, 0, 2.0 * y - 1.0, false) *
                              2.0;

  if constexpr (dim > 1)
    grad[1] = y_derivative_x1 * y_contribution * z_contribution +
              y_derivative_x2 * y_contribution * z_contribution +
              x_contribution * y_derivative * z_contribution;


  const double z_derivative = Polynomials::jacobi_polynomial_derivative<double>(
                                k, 0, 0, 2.0 * z - 1.0, false) *
                              2.0;

  if constexpr (dim > 2)
    grad[2] = x_contribution * y_contribution * z_derivative;

  for (unsigned int d = 0; d < dim; ++d)
    if (std::fabs(grad[d]) < 1e-14)
      grad[d] = 0.0;

  return grad;
}



template <int dim>
Tensor<1, dim>
ScalarNodalPolynomialWedge<dim>::evaluate_orthogonal_basis_derivative(
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
std::string
ScalarNodalPolynomialWedge<dim>::name() const
{
  return "ScalarNodalPolynomialWedge";
}



template <int dim>
std::unique_ptr<ScalarPolynomialsBase<dim>>
ScalarNodalPolynomialWedge<dim>::clone() const
{
  return std::make_unique<ScalarNodalPolynomialWedge<dim>>(*this);
}



template class ScalarNodalPolynomialWedge<1>;
template class ScalarNodalPolynomialWedge<2>;
template class ScalarNodalPolynomialWedge<3>;

DEAL_II_NAMESPACE_CLOSE
