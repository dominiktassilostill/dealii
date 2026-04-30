// -----------------------------------------------------------------------------
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception OR LGPL-2.1-or-later
// Copyright (C) 2021 - 2025 by the deal.II authors
//
// This file is part of the deal.II library.
//
// Detailed license information governing the source code and contributions
// can be found in LICENSE.md and CONTRIBUTING.md at the top level directory.
//
// -----------------------------------------------------------------------------


#ifndef dealii_base_polynomials_wedge_h
#define dealii_base_polynomials_wedge_h

#include <deal.II/base/config.h>

#include <deal.II/base/ndarray.h>
#include <deal.II/base/point.h>
#include <deal.II/base/polynomials_barycentric.h>
#include <deal.II/base/scalar_polynomials_base.h>
#include <deal.II/base/scalar_polynomials_vandermonde_base.h>
#include <deal.II/base/tensor.h>

#include <deal.II/lac/full_matrix.h>

DEAL_II_NAMESPACE_OPEN


namespace internal
{
  /**
   * Decompose the shape-function index of a linear wedge into an index
   * to access the right shape function within the triangle and within
   * the line.
   */
  constexpr dealii::ndarray<unsigned int, 6, 2> wedge_table_1{
    {{{0, 0}}, {{1, 0}}, {{2, 0}}, {{0, 1}}, {{1, 1}}, {{2, 1}}}};

  /**
   * Decompose the shape-function index of a quadratic wedge into an index
   * to access the right shape function within the triangle and within
   * the line.
   */
  constexpr dealii::ndarray<unsigned int, 18, 2> wedge_table_2{{{{0, 0}},
                                                                {{1, 0}},
                                                                {{2, 0}},
                                                                {{0, 1}},
                                                                {{1, 1}},
                                                                {{2, 1}},
                                                                {{3, 0}},
                                                                {{4, 0}},
                                                                {{5, 0}},
                                                                {{3, 1}},
                                                                {{4, 1}},
                                                                {{5, 1}},
                                                                {{0, 2}},
                                                                {{1, 2}},
                                                                {{2, 2}},
                                                                {{3, 2}},
                                                                {{4, 2}},
                                                                {{5, 2}}}};
} // namespace internal


/**
 * Polynomials defined on wedge entities. This class is basis of
 * FE_WedgeP.
 *
 * The polynomials are created via a tensor product of a
 * BarycentricPolynomials<2>::get_fe_p_basis(degree) and a
 * BarycentricPolynomials<1>::get_fe_p_basis(degree), however, are
 * re-numerated to better match the definition of FiniteElement.
 */
template <int dim>
class ScalarLagrangePolynomialWedge : public ScalarPolynomialsBase<dim>
{
public:
  /**
   * Make the dimension available to the outside.
   */
  static constexpr unsigned int dimension = dim;

  /*
   * Constructor taking the polynomial @p degree as input.
   *
   * @note Currently, only linear (degree=1) and quadratic polynomials
   *   (degree=2) are implemented.
   */
  ScalarLagrangePolynomialWedge(const unsigned int degree);

  /**
   * @copydoc ScalarPolynomialsBase::evaluate()
   *
   * @note Currently, only the vectors @p values and @p grads are filled.
   */
  void
  evaluate(const Point<dim>            &unit_point,
           std::vector<double>         &values,
           std::vector<Tensor<1, dim>> &grads,
           std::vector<Tensor<2, dim>> &grad_grads,
           std::vector<Tensor<3, dim>> &third_derivatives,
           std::vector<Tensor<4, dim>> &fourth_derivatives) const override;

  double
  compute_value(const unsigned int i, const Point<dim> &p) const override;

  /**
   * @copydoc ScalarPolynomialsBase::compute_derivative()
   *
   * @note Currently, only implemented for first derivative.
   */
  template <int order>
  Tensor<order, dim>
  compute_derivative(const unsigned int i, const Point<dim> &p) const;

  Tensor<1, dim>
  compute_1st_derivative(const unsigned int i,
                         const Point<dim>  &p) const override;

  /**
   * @copydoc ScalarPolynomialsBase::compute_2nd_derivative()
   *
   * @note Not implemented yet.
   */
  Tensor<2, dim>
  compute_2nd_derivative(const unsigned int i,
                         const Point<dim>  &p) const override;

  /**
   * @copydoc ScalarPolynomialsBase::compute_3rd_derivative()
   *
   * @note Not implemented yet.
   */
  Tensor<3, dim>
  compute_3rd_derivative(const unsigned int i,
                         const Point<dim>  &p) const override;

  /**
   * @copydoc ScalarPolynomialsBase::compute_4th_derivative()
   *
   * @note Not implemented yet.
   */
  Tensor<4, dim>
  compute_4th_derivative(const unsigned int i,
                         const Point<dim>  &p) const override;

  /**
   * @copydoc ScalarPolynomialsBase::compute_grad()
   *
   * @note Not implemented yet.
   */
  Tensor<1, dim>
  compute_grad(const unsigned int i, const Point<dim> &p) const override;

  /**
   * @copydoc ScalarPolynomialsBase::compute_grad_grad()
   *
   * @note Not implemented yet.
   */
  Tensor<2, dim>
  compute_grad_grad(const unsigned int i, const Point<dim> &p) const override;

  std::string
  name() const override;

  virtual std::unique_ptr<ScalarPolynomialsBase<dim>>
  clone() const override;

private:
  /**
   * Scalar polynomials defined on a triangle.
   */
  const BarycentricPolynomials<2> poly_tri;

  /**
   * Scalar polynomials defined on a line.
   */
  const BarycentricPolynomials<1> poly_line;
};



template <int dim>
template <int order>
Tensor<order, dim>
ScalarLagrangePolynomialWedge<dim>::compute_derivative(
  const unsigned int i,
  const Point<dim>  &p) const
{
  Tensor<order, dim> der;

  AssertDimension(order, 1);
  const auto grad = compute_grad(i, p);

  for (unsigned int i = 0; i < dim; ++i)
    der[i] = grad[i];

  return der;
}



/**
 * Polynomials defined on wedge entities. This class can be a basis of
 * FE_WedgeP.
 * We first use Jacobi polynomials to construct a modal basis. The polynomials
 * are based on the implementation of triangles (see
 * ScalarLagrangePolynomialSimplex) multiplied by a one dimensional one to
 * accout for the z-direction. With the modal basis a Vandermonde matrix is
 * calculated which leads to a nodal basis. For computing the values of the
 * nodal basis the Vandermonde matrix is multiplied with the modal basis vector
 * evaluated at the evaluation point.
 */
template <int dim>
class ScalarNodalPolynomialWedge : public ScalarPolynomialsVandermondeBase<dim>
{
public:
  /**
   * Make the dimension available to the outside.
   */
  static constexpr unsigned int dimension = dim;

  /*
   * Constructor taking the polynomial @p degree, the number of polynomials
   * @p n_dofs and the support points as input.
   */
  ScalarNodalPolynomialWedge(const unsigned int             degree,
                             const unsigned int             n_dofs,
                             const std::vector<Point<dim>> &support_points);

  std::string
  name() const override;

  virtual std::unique_ptr<ScalarPolynomialsBase<dim>>
  clone() const override;

private:
  /**
   * Evaluate the orthogonal basis at point @p p. The indices @p i, @p j
   * and @p k corresponde to the polynomial degrees of the Jacobi polynomials.
   */
  double
  evaluate_orthogonal_basis_function_by_degree(
    const unsigned int i,
    const unsigned int j,
    const unsigned int k,
    const Point<dim>  &p) const override;

  /**
   * Evaluate the orthogonal basis function @p i at point @p p.
   * This function determines the corresponding indices for the Jacobi
   * polynomials and calls the function taking all indices as arguments.
   */
  double
  evaluate_orthogonal_basis_function(const unsigned int i,
                                     const Point<dim>  &p) const override;

  /**
   * Evaluate the derivative of the orthogonal basis at point @p p.
   * The indices @p i, @p j and @p k corresponde to the polynomial degrees of
   * the Jacobi polynomials.
   */
  Tensor<1, dim>
  evaluate_orthogonal_basis_derivative_by_degree(
    const unsigned int i,
    const unsigned int j,
    const unsigned int k,
    const Point<dim>  &p) const override;

  /**
   * Evaluate the derivative of the orthogonal basis function @p i at point
   * @p p. This function determines the corresponding indices for the Jacobi
   * polynomials and calls the function taking all indices as arguments.
   */
  Tensor<1, dim>
  evaluate_orthogonal_basis_derivative(const unsigned int i,
                                       const Point<dim>  &p) const override;
};

DEAL_II_NAMESPACE_CLOSE

#endif
