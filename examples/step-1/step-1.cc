// ------------------------------------------------------------------------
//
// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2020 - 2026 by the deal.II authors
//
// This file is part of the deal.II library.
//
// Part of the source code is dual licensed under Apache-2.0 WITH
// LLVM-exception OR LGPL-2.1-or-later. Detailed license information
// governing the source code and code contributions can be found in
// LICENSE.md and CONTRIBUTING.md at the top level directory of deal.II.
//
// ------------------------------------------------------------------------


// Test PolynomialsPyramid by comparing it to a implementation for linear
// polynomials


#include <deal.II/base/polynomials_pyramid.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/fe/fe_pyramid_p.h>


using namespace dealii;

template <int dim>
double compute_value(const unsigned int i, const Point<dim> &p)
{
  const double Q14 = 0.25;
  double       ration;

  const double r = p[0];
  const double s = p[1];
  const double t = p[2];

  if (std::fabs(t - 1.0) > 1.0e-14)
    {
      ration = (r * s * t) / (1.0 - t);
    }
  else
    {
      ration = 0.0;
    }

  if (i == 0)
    return Q14 * ((1.0 - r) * (1.0 - s) - t + ration);
  if (i == 1)
    return Q14 * ((1.0 + r) * (1.0 - s) - t - ration);
  if (i == 2)
    return Q14 * ((1.0 - r) * (1.0 + s) - t - ration);
  if (i == 3)
    return Q14 * ((1.0 + r) * (1.0 + s) - t + ration);
  else
    return t;
}


template <int dim>
double compute_value_quadratic(const unsigned int i, const Point<dim> &p)
{
  const double x = p[0];
  const double y = p[1];
  const double z = p[2];

  switch (i)
    {
      case 0:
        return ((z - 1) * (z - 1) *
                  (0.0833333333333334 * x * x +
                   x * (0.333333333333333 * z - 0.0555555555555556) -
                   0.0277777777777778 * x + 0.0833333333333333 * y * y +
                   y * (0.333333333333333 * z - 0.0555555555555555) -
                   0.0277777777777778 * y + 0.222222222222222 * z * z -
                   0.194444444444445 * z - 0.0277777777777778) +
                (z - 1) *
                  (0.0833333333333333 * x * y * (6 * z - 1) -
                   0.166666666666667 * x * y +
                   0.0833333333333333 * x * (3 * y * y - z * z + 2 * z - 1) +
                   0.0833333333333333 * y * (3 * x * x - z * z + 2 * z - 1)) +
                0.0277777777777778 * (3 * x * x - z * z + 2 * z - 1) *
                  (3 * y * y - z * z + 2 * z - 1)) /
               (z - 1) * (z - 1);
      case 1:
        return ((z - 1) * (z - 1) *
                  (0.0833333333333334 * x * x -
                   x * (0.333333333333333 * z - 0.0555555555555556) +
                   0.0277777777777778 * x + 0.0833333333333333 * y * y +
                   y * (0.333333333333333 * z - 0.0555555555555556) -
                   0.0277777777777778 * y + 0.222222222222222 * z * z -
                   0.194444444444444 * z - 0.0277777777777778) -
                (z - 1) *
                  (1.0 * x * y * (0.5 * z - 0.0833333333333333) -
                   0.166666666666667 * x * y +
                   0.0833333333333333 * x * (3 * y * y - z * z + 2 * z - 1) -
                   0.0833333333333333 * y * (3 * x * x - z * z + 2 * z - 1)) +
                0.0277777777777778 * (3 * x * x - z * z + 2 * z - 1) *
                  (3 * y * y - z * z + 2 * z - 1)) /
               (z - 1) * (z - 1);
      case 2:
        return ((z - 1) * (z - 1) *
                  (0.0833333333333333 * x * x +
                   x * (0.333333333333333 * z - 0.0555555555555556) -
                   0.0277777777777778 * x + 0.0833333333333333 * y * y -
                   y * (0.333333333333333 * z - 0.0555555555555556) +
                   0.0277777777777778 * y + 0.222222222222222 * z * z -
                   0.194444444444445 * z - 0.0277777777777778) -
                (z - 1) *
                  (1.0 * x * y * (0.5 * z - 0.0833333333333333) -
                   0.166666666666667 * x * y -
                   0.0833333333333333 * x * (3 * y * y - z * z + 2 * z - 1) +
                   0.0833333333333333 * y * (3 * x * x - z * z + 2 * z - 1)) +
                0.0277777777777778 * (3 * x * x - z * z + 2 * z - 1) *
                  (3 * y * y - z * z + 2 * z - 1)) /
               (z - 1) * (z - 1);
      case 3:
        return ((z - 1) * (z - 1) *
                  (0.0833333333333333 * x * x -
                   x * (0.333333333333333 * z - 0.0555555555555556) +
                   0.0277777777777778 * x + 0.0833333333333333 * y * y -
                   y * (0.333333333333333 * z - 0.0555555555555556) +
                   0.0277777777777778 * y + 0.222222222222222 * z * z -
                   0.194444444444444 * z - 0.0277777777777778) -
                (z - 1) *
                  (-0.0833333333333333 * x * y * (6 * z - 1) +
                   0.166666666666667 * x * y +
                   0.0833333333333333 * x * (3 * y * y - z * z + 2 * z - 1) +
                   0.0833333333333333 * y * (3 * x * x - z * z + 2 * z - 1)) +
                0.0277777777777778 * (3 * x * x - z * z + 2 * z - 1) *
                  (3 * y * y - z * z + 2 * z - 1)) /
               (z - 1) * (z - 1);

      case 4:
        return 2.0 * z * z - z;

      case 5:
        return (-x * (z - 1) *
                  (0.5 * y * y +
                   1.0 * y *
                     (1.38777878078145e-17 * 0.0 * z -
                      2.31296463463574e-18 * 0.0) +
                   2.31296463463574e-18 * 0.0 * y - 0.166666666666667 * z * z +
                   0.333333333333333 * z - 0.166666666666667) +
                (z - 1) * (z - 1) *
                  (0.333333333333333 * x * x +
                   x * (0.333333333333333 * z - 0.0555555555555555) -
                   0.277777777777778 * x - 0.166666666666667 * y * y +
                   y * (1.38777878078145e-17 * 0.0 * z -
                        2.31296463463574e-18 * 0.0) +
                   2.31296463463574e-18 * 0.0 * y + 0.0555555555555556 * z * z -
                   0.111111111111111 * z + 0.0555555555555556) -
                0.0555555555555556 * (3 * x * x - z * z + 2 * z - 1) *
                  (3 * y * y - z * z + 2 * z - 1)) /
               (z - 1) * (z - 1);

      case 6:
        return (x * (z - 1) *
                  (0.5 * y * y + 2.31296463463574e-18 * 0.0 * y * (6 * z - 1) +
                   2.31296463463574e-18 * 0.0 * y - 0.166666666666667 * z * z +
                   0.333333333333333 * z - 0.166666666666667) +
                (z - 1) * (z - 1) *
                  (0.333333333333333 * x * x -
                   x * (0.333333333333333 * z - 0.0555555555555555) +
                   0.277777777777778 * x - 0.166666666666667 * y * y -
                   y * (1.38777878078145e-17 * 0.0 * z -
                        2.31296463463574e-18 * 0.0) -
                   2.31296463463574e-18 * 0.0 * y + 0.0555555555555556 * z * z -
                   0.111111111111111 * z + 0.0555555555555556) -
                0.0555555555555556 * (3 * x * x - z * z + 2 * z - 1) *
                  (3 * y * y - z * z + 2 * z - 1)) /
               (z - 1) * (z - 1);

      case 7:
        return (-0.166666666666667 * y * (z - 1) *
                  (3 * x * x - z * z + 2 * z - 1) +
                (z - 1) * (z - 1) *
                  (-0.166666666666667 * x * x + 0.333333333333333 * y * y +
                   y * (0.333333333333333 * z - 0.0555555555555555) -
                   0.277777777777778 * y + 0.0555555555555555 * z * z -
                   0.111111111111111 * z + 0.0555555555555555) -
                0.0555555555555556 * (3 * x * x - z * z + 2 * z - 1) *
                  (3 * y * y - z * z + 2 * z - 1)) /
               (z - 1) * (z - 1);

      case 8:
        return (0.166666666666667 * y * (z - 1) *
                  (3 * x * x - z * z + 2 * z - 1) +
                (z - 1) * (z - 1) *
                  (-0.166666666666667 * x * x + 0.333333333333333 * y * y -
                   y * (0.333333333333333 * z - 0.0555555555555555) +
                   0.277777777777778 * y + 0.0555555555555555 * z * z -
                   0.111111111111111 * z + 0.0555555555555556) -
                0.0555555555555556 * (3 * x * x - z * z + 2 * z - 1) *
                  (3 * y * y - z * z + 2 * z - 1)) /
               (z - 1) * (z - 1);

      case 9:
        return (-1.0 * x * y * z - 1.0 * x * z * z + 1.0 * x * z -
                1.0 * y * z * z + 1.0 * y * z - 1.0 * z * z * z + 2.0 * z * z -
                1.0 * z - 1.38777878078145e-17 * 0.0) /
               (z - 1);

      case 10:
        return (1.0 * x * y * z + 1.0 * x * z * z - 1.0 * x * z -
                1.0 * y * z * z + 1.0 * y * z - 1.0 * z * z * z + 2.0 * z * z -
                1.0 * z - 2.77555756156289e-17 * 0.0) /
               (z - 1);

      case 11:
        return (1.0 * x * y * z - 1.0 * x * z * z + 1.0 * x * z +
                1.0 * y * z * z - 1.0 * y * z - 1.0 * z * z * z + 2.0 * z * z -
                1.0 * z + 2.77555756156289e-17 * 0.0) /
               (z - 1);

      case 12:
        return 1.0 * z *
               (-1.0 * x * y + 1.0 * x * z - 1.0 * x + 1.0 * y * z - 1.0 * y -
                1.0 * z * z + 2.0 * z - 1.0) /
               (z - 1);

      case 13:
        return ((z - 1) * (z - 1) *
                  (-0.666666666666667 * x * x - 0.666666666666667 * y * y +
                   0.888888888888889 * z * z - 1.77777777777778 * z +
                   0.888888888888889) +
                0.111111111111111 * (3 * x * x - z * z + 2 * z - 1) *
                  (3 * y * y - z * z + 2 * z - 1)) /
               (z - 1) * (z - 1);
    }
  return -1.0;
}

template <int dim>
Tensor<1, dim> compute_grad(const unsigned int i, const Point<dim> &p)
{
  Tensor<1, dim> grad;

  const double Q14 = 0.25;

  const double r = p[0];
  const double s = p[1];
  const double t = p[2];

  double rationdr;
  double rationds;
  double rationdt;

  if (std::fabs(t - 1.0) > 1.0e-14)
    {
      rationdr = s * t / (1.0 - t);
      rationds = r * t / (1.0 - t);
      rationdt = r * s / ((1.0 - t) * (1.0 - t));
    }
  else
    {
      rationdr = 1.0;
      rationds = 1.0;
      rationdt = 1.0;
    }


  if (i == 0)
    {
      grad[0] = Q14 * (-1.0 * (1.0 - s) + rationdr);
      grad[1] = Q14 * (-1.0 * (1.0 - r) + rationds);
      grad[2] = Q14 * (rationdt - 1.0);
    }
  else if (i == 1)
    {
      grad[0] = Q14 * (1.0 * (1.0 - s) - rationdr);
      grad[1] = Q14 * (-1.0 * (1.0 + r) - rationds);
      grad[2] = Q14 * (-1.0 * rationdt - 1.0);
    }
  else if (i == 2)
    {
      grad[0] = Q14 * (-1.0 * (1.0 + s) - rationdr);
      grad[1] = Q14 * (1.0 * (1.0 - r) - rationds);
      grad[2] = Q14 * (-1.0 * rationdt - 1.0);
    }
  else if (i == 3)
    {
      grad[0] = Q14 * (1.0 * (1.0 + s) + rationdr);
      grad[1] = Q14 * (1.0 * (1.0 + r) + rationds);
      grad[2] = Q14 * (rationdt - 1.0);
    }
  else if (i == 4)
    {
      grad[0] = 0.0;
      grad[1] = 0.0;
      grad[2] = 1.0;
    }
  else
    {
      DEAL_II_NOT_IMPLEMENTED();
    }

  return grad;
}


template <int dim>
Tensor<1, dim> compute_grad_quadratic(const unsigned int i, const Point<dim> &p)
{
  Tensor<1, dim> grad;

  const double x = p[0];
  const double y = p[1];
  const double z = p[2];
  if constexpr (dim == 3)
    switch (i)
      {
        case 0:
          {
            grad[0] =
              (0.166666666666667 * x * (3 * y * y - z * z + 2 * z - 1) +
               (z - 1) * (z - 1) *
                 (0.166666666666667 * x + 0.333333333333333 * z -
                  0.0833333333333334) +
               (z - 1) * (0.5 * x * y + 0.25 * y * y +
                          0.0833333333333333 * y * (6 * z - 1) -
                          0.166666666666667 * y - 0.0833333333333333 * z * z +
                          0.166666666666667 * z - 0.0833333333333333)) /
              (z - 1) * (z - 1);
            grad[1] =
              (0.166666666666667 * y * (3 * x * x - z * z + 2 * z - 1) +
               (z - 1) * (z - 1) *
                 (0.166666666666667 * y + 0.333333333333333 * z -
                  0.0833333333333334) +
               (z - 1) * (0.25 * x * x + 0.5 * x * y +
                          0.0833333333333333 * x * (6 * z - 1) -
                          0.166666666666667 * x - 0.0833333333333333 * z * z +
                          0.166666666666667 * z - 0.0833333333333333)) /
              (z - 1) * (z - 1);
            grad[2] = (-0.5 * x * x * y * y - 0.25 * x * x * y * z +
                       0.25 * x * x * y - 0.25 * x * y * y * z +
                       0.25 * x * y * y - 0.25 * x * y * z + 0.25 * x * y +
                       0.25 * x * z * z * z - 0.75 * x * z * z + 0.75 * x * z -
                       0.25 * x + 0.25 * y * z * z * z - 0.75 * y * z * z +
                       0.75 * y * z - 0.25 * y + 0.5 * z * z * z * z -
                       1.75 * z * z * z + 2.25 * z * z - 1.25 * z + 0.25) /
                      (1.0 * z * z * z - 3.0 * z * z + 3.0 * z - 1.0);
            return grad;
          }

        case 1:
          {
            grad[0] =
              (0.166666666666667 * x * (3 * y * y - z * z + 2 * z - 1) +
               (z - 1) * (z - 1) *
                 (0.166666666666667 * x - 0.333333333333333 * z +
                  0.0833333333333334) +
               (z - 1) * (0.5 * x * y - 0.25 * y * y -
                          1.0 * y * (0.5 * z - 0.0833333333333333) +
                          0.166666666666667 * y + 0.0833333333333333 * z * z -
                          0.166666666666667 * z + 0.0833333333333333)) /
              (z - 1) * (z - 1);
            grad[1] =
              (0.166666666666667 * y * (3 * x * x - z * z + 2 * z - 1) +
               (z - 1) * (z - 1) *
                 (0.166666666666667 * y + 0.333333333333333 * z -
                  0.0833333333333334) -
               (z - 1) * (-0.25 * x * x + 0.5 * x * y +
                          1.0 * x * (0.5 * z - 0.0833333333333333) -
                          0.166666666666667 * x + 0.0833333333333333 * z * z -
                          0.166666666666667 * z + 0.0833333333333333)) /
              (z - 1) * (z - 1);
            grad[2] = (-0.5 * x * x * y * y - 0.25 * x * x * y * z +
                       0.25 * x * x * y + 0.25 * x * y * y * z -
                       0.25 * x * y * y + 0.25 * x * y * z - 0.25 * x * y -
                       0.25 * x * z * z * z + 0.75 * x * z * z - 0.75 * x * z +
                       0.25 * x + 0.25 * y * z * z * z - 0.75 * y * z * z +
                       0.75 * y * z - 0.25 * y + 0.5 * z * z * z * z -
                       1.75 * z * z * z + 2.25 * z * z - 1.25 * z + 0.25) /
                      (1.0 * z * z * z - 3.0 * z * z + 3.0 * z - 1.0);
            return grad;
          }

        case 2:
          {
            grad[0] =
              (0.166666666666667 * x * (3 * y * y - z * z + 2 * z - 1) +
               (z - 1) * (z - 1) *
                 (0.166666666666667 * x + 0.333333333333333 * z -
                  0.0833333333333333) -
               (z - 1) * (0.5 * x * y - 0.25 * y * y +
                          1.0 * y * (0.5 * z - 0.0833333333333333) -
                          0.166666666666667 * y + 0.0833333333333333 * z * z -
                          0.166666666666667 * z + 0.0833333333333333)) /
              (z - 1) * (z - 1);
            grad[1] =
              (0.166666666666667 * y * (3 * x * x - z * z + 2 * z - 1) +
               (z - 1) * (z - 1) *
                 (0.166666666666667 * y - 0.333333333333333 * z +
                  0.0833333333333334) +
               (z - 1) * (-0.25 * x * x + 0.5 * x * y -
                          1.0 * x * (0.5 * z - 0.0833333333333333) +
                          0.166666666666667 * x + 0.0833333333333333 * z * z -
                          0.166666666666667 * z + 0.0833333333333333)) /
              (z - 1) * (z - 1);
            grad[2] = (-0.5 * x * x * y * y + 0.25 * x * x * y * z -
                       0.25 * x * x * y - 0.25 * x * y * y * z +
                       0.25 * x * y * y + 0.25 * x * y * z - 0.25 * x * y +
                       0.25 * x * z * z * z - 0.75 * x * z * z + 0.75 * x * z -
                       0.25 * x - 0.25 * y * z * z * z + 0.75 * y * z * z -
                       0.75 * y * z + 0.25 * y + 0.5 * z * z * z * z -
                       1.75 * z * z * z + 2.25 * z * z - 1.25 * z + 0.25) /
                      (1.0 * z * z * z - 3.0 * z * z + 3.0 * z - 1.0);
            return grad;
          }

        case 3:
          {
            grad[0] =
              (0.166666666666667 * x * (3 * y * y - z * z + 2 * z - 1) +
               (z - 1) * (z - 1) *
                 (0.166666666666667 * x - 0.333333333333333 * z +
                  0.0833333333333333) -
               (z - 1) * (0.5 * x * y + 0.25 * y * y -
                          0.0833333333333333 * y * (6 * z - 1) +
                          0.166666666666667 * y - 0.0833333333333333 * z * z +
                          0.166666666666667 * z - 0.0833333333333333)) /
              (z - 1) * (z - 1);
            grad[1] =
              (0.166666666666667 * y * (3 * x * x - z * z + 2 * z - 1) +
               (z - 1) * (z - 1) *
                 (0.166666666666667 * y - 0.333333333333333 * z +
                  0.0833333333333334) -
               (z - 1) * (0.25 * x * x + 0.5 * x * y -
                          0.0833333333333333 * x * (6 * z - 1) +
                          0.166666666666667 * x - 0.0833333333333333 * z * z +
                          0.166666666666667 * z - 0.0833333333333333)) /
              (z - 1) * (z - 1);
            grad[2] = (-0.5 * x * x * y * y + 0.25 * x * x * y * z -
                       0.25 * x * x * y + 0.25 * x * y * y * z -
                       0.25 * x * y * y - 0.25 * x * y * z + 0.25 * x * y -
                       0.25 * x * z * z * z + 0.75 * x * z * z - 0.75 * x * z +
                       0.25 * x - 0.25 * y * z * z * z + 0.75 * y * z * z -
                       0.75 * y * z + 0.25 * y + 0.5 * z * z * z * z -
                       1.75 * z * z * z + 2.25 * z * z - 1.25 * z + 0.25) /
                      (1.0 * z * z * z - 3.0 * z * z + 3.0 * z - 1.0);
            return grad;
          }

        case 4:
          {
            grad[0] = 8.32667268468868e-18 * 0.0 * z * (-y + z - 1) / (z - 1);
            grad[1] = 8.32667268468868e-18 * 0.0 * z * (-x + z - 1) / (z - 1);
            grad[2] = (8.32667268468868e-18 * 0.0 * x * y +
                       8.32667268468868e-18 * 0.0 * x * z * z -
                       1.66533453693774e-17 * 0.0 * x * z +
                       8.32667268468868e-18 * 0.0 * x +
                       8.32667268468868e-18 * 0.0 * y * z * z -
                       1.66533453693774e-17 * 0.0 * y * z +
                       8.32667268468868e-18 * 0.0 * y + 4.0 * z * z * z -
                       9.0 * z * z + 6.0 * z - 1.0) /
                      (1.0 * z * z - 2.0 * z + 1.0);
            return grad;
          }

        case 5:
          {
            grad[0] =
              (-0.333333333333333 * x * (3 * y * y - z * z + 2 * z - 1) +
               (z - 1) * (z - 1) *
                 (0.666666666666667 * x + 0.333333333333333 * z -
                  0.333333333333333) -
               (z - 1) *
                 (0.5 * y * y +
                  1.0 * y *
                    (1.38777878078145e-17 * 0.0 * z -
                     2.31296463463574e-18 * 0.0) +
                  2.31296463463574e-18 * 0.0 * y - 0.166666666666667 * z * z +
                  0.333333333333333 * z - 0.166666666666667)) /
              (z - 1) * (z - 1);
            grad[1] =
              (-x * (1.0 * y + 1.38777878078145e-17 * 0.0 * z) * (z - 1) -
               0.333333333333333 * y * (3 * x * x - z * z + 2 * z - 1) +
               (-0.333333333333333 * y + 1.38777878078145e-17 * 0.0 * z) *
                 (z - 1) * (z - 1)) /
              (z - 1) * (z - 1);
            grad[2] =
              (1.0 * x * x * y * y +
               5.55111512312578e-17 * 0.0 * x * x * z * z -
               1.11022302462516e-16 * 0.0 * x * x * z +
               5.55111512312578e-17 * 0.0 * x * x + 0.5 * x * y * y * z -
               0.5 * x * y * y + 1.38777878078145e-17 * 0.0 * x * y * z -
               1.38777878078145e-17 * 0.0 * x * y + 0.5 * x * z * z * z -
               1.5 * x * z * z + 1.5 * x * z - 0.5 * x +
               1.38777878078145e-17 * 0.0 * y * z * z * z -
               4.16333634234434e-17 * 0.0 * y * z * z +
               4.16333634234434e-17 * 0.0 * y * z -
               1.38777878078145e-17 * 0.0 * y +
               5.55111512312578e-17 * 0.0 * z * z * z * z -
               1.11022302462516e-16 * 0.0 * z * z * z +
               1.11022302462516e-16 * 0.0 * z * z -
               1.11022302462516e-16 * 0.0 * z) /
              (1.0 * z * z * z - 3.0 * z * z + 3.0 * z - 1.0);
            return grad;
          }

        case 6:
          {
            grad[0] =
              (-0.333333333333333 * x * (3 * y * y - z * z + 2 * z - 1) +
               (z - 1) * (z - 1) *
                 (0.666666666666667 * x - 0.333333333333333 * z +
                  0.333333333333333) +
               (z - 1) *
                 (0.5 * y * y + 2.31296463463574e-18 * 0.0 * y * (6 * z - 1) +
                  2.31296463463574e-18 * 0.0 * y - 0.166666666666667 * z * z +
                  0.333333333333333 * z - 0.166666666666667)) /
              (z - 1) * (z - 1);
            grad[1] =
              (x * (1.0 * y + 1.38777878078145e-17 * 0.0 * z) * (z - 1) -
               0.333333333333333 * y * (3 * x * x - z * z + 2 * z - 1) -
               (0.333333333333333 * y + 1.38777878078145e-17 * 0.0 * z) *
                 (z - 1) * (z - 1)) /
              (z - 1) * (z - 1);
            grad[2] =
              (1.0 * x * x * y * y +
               5.55111512312578e-17 * 0.0 * x * x * z * z -
               1.11022302462516e-16 * 0.0 * x * x * z +
               5.55111512312578e-17 * 0.0 * x * x - 0.5 * x * y * y * z +
               0.5 * x * y * y - 1.38777878078145e-17 * 0.0 * x * y * z +
               1.38777878078145e-17 * 0.0 * x * y - 0.5 * x * z * z * z +
               1.5 * x * z * z - 1.5 * x * z + 0.5 * x -
               1.38777878078145e-17 * 0.0 * y * z * z * z +
               4.16333634234434e-17 * 0.0 * y * z * z -
               4.16333634234434e-17 * 0.0 * y * z +
               1.38777878078145e-17 * 0.0 * y +
               5.55111512312578e-17 * 0.0 * z * z * z * z -
               1.11022302462516e-16 * 0.0 * z * z * z +
               1.11022302462516e-16 * 0.0 * z * z -
               1.11022302462516e-16 * 0.0 * z) /
              (1.0 * z * z * z - 3.0 * z * z + 3.0 * z - 1.0);
            return grad;
          }

        case 7:
          {
            grad[0] =
              x *
              (-1.0 * y * y - 1.0 * y * (z - 1) + 0.333333333333333 * z * z -
               0.666666666666667 * z - 0.333333333333333 * (z - 1) * (z - 1) +
               0.333333333333333) /
              (z - 1) * (z - 1);
            grad[1] =
              (-0.333333333333333 * y * (3 * x * x - z * z + 2 * z - 1) +
               (z - 1) * (z - 1) *
                 (0.666666666666667 * y + 0.333333333333333 * z -
                  0.333333333333333) -
               0.166666666666667 * (z - 1) * (3 * x * x - z * z + 2 * z - 1)) /
              (z - 1) * (z - 1);
            grad[2] =
              (1.0 * x * x * y * y + 0.5 * x * x * y * z - 0.5 * x * x * y +
               5.55111512312578e-17 * 0.0 * y * y * z * z -
               1.11022302462516e-16 * 0.0 * y * y * z +
               5.55111512312578e-17 * 0.0 * y * y + 0.5 * y * z * z * z -
               1.5 * y * z * z + 1.5 * y * z - 0.5 * y -
               1.94289029309402e-16 * 0.0 * z * z * z * z +
               6.66133814775094e-16 * 0.0 * z * z * z -
               8.88178419700125e-16 * 0.0 * z * z +
               4.44089209850063e-16 * 0.0 * z - 1.11022302462516e-16 * 0.0) /
              (1.0 * z * z * z - 3.0 * z * z + 3.0 * z - 1.0);
            return grad;
          }

        case 8:
          {
            grad[0] =
              x *
              (-1.0 * y * y + 1.0 * y * (z - 1) + 0.333333333333333 * z * z -
               0.666666666666667 * z - 0.333333333333333 * (z - 1) * (z - 1) +
               0.333333333333333) /
              (z - 1) * (z - 1);
            grad[1] =
              (-0.333333333333333 * y * (3 * x * x - z * z + 2 * z - 1) +
               (z - 1) * (z - 1) *
                 (0.666666666666667 * y - 0.333333333333333 * z +
                  0.333333333333333) +
               0.166666666666667 * (z - 1) * (3 * x * x - z * z + 2 * z - 1)) /
              (z - 1) * (z - 1);
            grad[2] =
              (1.0 * x * x * y * y - 0.5 * x * x * y * z + 0.5 * x * x * y +
               5.55111512312578e-17 * 0.0 * y * y * z * z -
               1.11022302462516e-16 * 0.0 * y * y * z +
               5.55111512312578e-17 * 0.0 * y * y - 0.5 * y * z * z * z +
               1.5 * y * z * z - 1.5 * y * z + 0.5 * y -
               1.38777878078145e-17 * 0.0 * z * z * z * z -
               1.11022302462516e-16 * 0.0 * z * z +
               5.55111512312578e-17 * 0.0 * z + 1.38777878078145e-17 * 0.0) /
              (1.0 * z * z * z - 3.0 * z * z + 3.0 * z - 1.0);
            return grad;
          }

        case 9:
          {
            grad[0] = 1.0 * z * (-y - z + 1) / (z - 1);
            grad[1] = 1.0 * z * (-x - z + 1) / (z - 1);
            grad[2] = (1.0 * x * y - 1.0 * x * z * z + 2.0 * x * z - 1.0 * x -
                       1.0 * y * z * z + 2.0 * y * z - 1.0 * y -
                       2.0 * z * z * z + 5.0 * z * z - 4.0 * z + 1.0) /
                      (1.0 * z * z - 2.0 * z + 1.0);
            return grad;
          }

        case 10:
          {
            grad[0] = 1.0 * z * (y + z - 1) / (z - 1);
            grad[1] = 1.0 * z * (x - z + 1) / (z - 1);
            grad[2] = (-1.0 * x * y + 1.0 * x * z * z - 2.0 * x * z + 1.0 * x -
                       1.0 * y * z * z + 2.0 * y * z - 1.0 * y -
                       2.0 * z * z * z + 5.0 * z * z - 4.0 * z + 1.0) /
                      (1.0 * z * z - 2.0 * z + 1.0);
            return grad;
          }

        case 11:
          {
            grad[0] = 1.0 * z * (y - z + 1) / (z - 1);
            grad[1] = 1.0 * z * (x + z - 1) / (z - 1);
            grad[2] = (-1.0 * x * y - 1.0 * x * z * z + 2.0 * x * z - 1.0 * x +
                       1.0 * y * z * z - 2.0 * y * z + 1.0 * y -
                       2.0 * z * z * z + 5.0 * z * z - 4.0 * z + 1.0) /
                      (1.0 * z * z - 2.0 * z + 1.0);
            return grad;
          }

        case 12:
          {
            grad[0] = 1.0 * z * (-y + z - 1) / (z - 1);
            grad[1] = 1.0 * z * (-x + z - 1) / (z - 1);
            grad[2] = (1.0 * x * y + 1.0 * x * z * z - 2.0 * x * z + 1.0 * x +
                       1.0 * y * z * z - 2.0 * y * z + 1.0 * y -
                       2.0 * z * z * z + 5.0 * z * z - 4.0 * z + 1.0) /
                      (1.0 * z * z - 2.0 * z + 1.0);
            return grad;
          }

        case 13:
          {
            grad[0] = x * (2.0 * y * y - 2.0 * z * z + 4.0 * z - 2.0) /
                      (1.0 * z * z - 2.0 * z + 1.0);
            grad[1] = y * (2.0 * x * x - 2.0 * z * z + 4.0 * z - 2.0) /
                      (1.0 * z * z - 2.0 * z + 1.0);
            grad[2] =
              (-2.0 * x * x * y * y -
               1.11022302462516e-16 * 0.0 * x * x * z * z +
               2.22044604925031e-16 * 0.0 * x * x * z -
               1.11022302462516e-16 * 0.0 * x * x -
               1.11022302462516e-16 * 0.0 * y * y * z * z +
               2.22044604925031e-16 * 0.0 * y * y * z -
               1.11022302462516e-16 * 0.0 * y * y + 2.0 * z * z * z * z -
               8.0 * z * z * z + 12.0 * z * z - 8.0 * z + 2.0) /
              (1.0 * z * z * z - 3.0 * z * z + 3.0 * z - 1.0);
            return grad;
          }
      }
  return grad;
}


template <int dim>
void test()
{
  FE_PyramidP<dim> fe_pyramid(1);
  const auto       support_points = fe_pyramid.get_unit_support_points();
  const auto       poly =
    ScalarLagrangePolynomialPyramid(1,
                                    fe_pyramid.n_dofs_per_cell(),
                                    support_points);

  const auto poly_linear = ScalarLagrangePolynomialPyramid<dim>(1);

  QGaussPyramid<dim> quad(2);

  for (unsigned int i = 0; i < quad.size(); ++i)
    for (unsigned int j = 0; j < 5; ++j)
      {
        const auto v1 = poly.compute_value(j, quad.point(i));
        const auto v2 = poly_linear.compute_value(j, quad.point(i));
        const auto v3 = compute_value(j, quad.point(i));

        if (std::abs(v3 - v1) < 1e-12)
          std::cout << "ok"
                    << " ";
        else
          std::cout << "Failure!!!";
        if (std::abs(v3 - v2) < 1e-12)
          std::cout << "ok"
                    << " ";
        else
          std::cout << "Failure!!!";
        std::cout << std::endl;

        const auto g1 = poly.compute_grad(j, quad.point(i));
        const auto g2 = poly_linear.compute_grad(j, quad.point(i));
        const auto g3 = compute_grad(j, quad.point(i));

        for (unsigned int d = 0; d < dim; ++d)
          {
            if (std::abs((g3 - g1)[d]) < 1e-12)
              std::cout << "ok "
                        << " ";
            else
              std::cout << "Failure!!!";
            if (std::abs((g3 - g2)[d]) < 1e-12)
              std::cout << "ok"
                        << " ";
            else
              std::cout << "Failure!!!";
          }
        std::cout << std::endl;
      }
}


template <int dim>
void test_quadratic()
{
  FE_PyramidP<dim> fe_pyramid(2);
  auto             support_points = fe_pyramid.get_unit_support_points();
  std::cout << "N support points quadratic: " << support_points.size()
            << std::endl;
  if (support_points.size() != 14)
    std::cout << "Failure support points size!!!!" << std::endl;
  const auto poly =
    ScalarLagrangePolynomialPyramid(2,
                                    fe_pyramid.n_dofs_per_cell(),
                                    support_points);

  // support_points.clear();
  // QGaussPyramid<dim> quad(2);
  // for (unsigned int i = 0; i < quad.size(); ++i)
  //   support_points.emplace_back(quad.point(i));
  // for (unsigned int i = 0; i < quad.size(); ++i)
  for (unsigned int i = 0; i < support_points.size(); ++i)
    for (unsigned int j = 0; j < support_points.size(); ++j)
      {
        // const auto v1 = poly.compute_value(j, quad.point(i));
        // const auto v2 = compute_value_quadratic(j, quad.point(i));
        const auto v1 = poly.compute_value(j, support_points[i]);
        const auto v2 = compute_value_quadratic(j, support_points[i]);

        // if (std::abs(v2 - v1) < 1e-12)
        //   std::cout  << "ok"
        //           << " ";
        // else
        //   std::cout  << "Failure!!! " << v2 << " " << v1 << " ";
        if (i == j)
          {
            // has to be 1
            if (std::abs(v1 - 1.0) < 1e-12)
              std::cout << i << " " << j << " ok v1"
                        << " " << v1 << " ";
            else
              std::cout << "Failure for shape function " << j
                        << " on support point " << i << std::endl;

            // has to be 1
            if (std::abs(v2 - 1.0) < 1e-12)
              std::cout << i << " " << j << " ok v2"
                        << " " << v2 << " ";
            else
              std::cout << "Failure for shape function " << j
                        << " on support point " << i << " for python function"
                        << std::endl;
          }
        else
          {
            // has to be 0
            if (std::abs(v1) < 1e-12)
              std::cout << i << " " << j << " ok v1"
                        << " " << v1 << " ";
            else
              std::cout << "Failure for shape function " << j
                        << " on support point " << i << std::endl;

            // has to be 1
            if (std::abs(v2) < 1e-12)
              std::cout << i << " " << j << " ok v2"
                        << " " << v2 << " ";
            else
              std::cout << "Failure for shape function " << j
                        << " on support point " << i << " for python function"
                        << std::endl;
          }

        // const auto g1 = poly.compute_grad(j, quad.point(i));
        // const auto g2 = compute_grad_quadratic(j, quad.point(i));
        const auto g1 = poly.compute_grad(j, support_points[i]);
        const auto g2 = compute_grad_quadratic(j, support_points[i]);
        if (false)
          for (unsigned int d = 0; d < dim; ++d)
            {
              if (std::abs((g2 - g1)[d]) < 1e-12)
                std::cout << "ok "
                          << " ";
              else
                std::cout << "Failure!!! " << (g2 - g1)[d] << " ";
            }
        std::cout << std::endl;
      }
}

template <int dim>
void test_polynomial_space(const unsigned int degree)
{
  const unsigned int n_dofs =
    (degree + 1) * (degree + 2) * (2 * degree + 3) / 6;

  std::vector<Point<dim>> support_points(n_dofs);
  unsigned int            counter = 0;
  for (auto &p : support_points)
    {
      p[0] = counter / (5.0 * n_dofs) - 1.0;
      p[1] = counter / (4.0 * n_dofs) - 1.0;
      p[2] = counter / (3.0 * n_dofs);
      ++counter;
    }

  const auto poly =
    ScalarLagrangePolynomialPyramid(degree, n_dofs, support_points);

  std::vector<Point<dim>> points;
  // points.emplace_back(Point<dim>(0.546,-0.32468,0.21687));
  // points.emplace_back(Point<dim>(0,0,0.9999));
  points.emplace_back(Point<dim>(0, 0, 1.0 - 1e-12));

  if (false)
    for (const auto &p : points)
      for (unsigned int i = 0; i < degree + 1; ++i)
        for (unsigned int j = 0; j < degree + 1; ++j)
          for (unsigned int k = 0; k < degree + 1 - std::max(i, j); ++k)
            std::cout << i << " " << j << " " << k << ": "
                      << poly.compute_polynomial_space(i, j, k, p) << std::endl;

  if (false)
    for (const auto &p : points)
      for (unsigned int i = 0; i < n_dofs; ++i)
        std::cout << i << " " << poly.compute_jacobi_basis(i, p) << std::endl;


  for (const auto &p : points)
    for (unsigned int i = 0; i < degree + 1; ++i)
      for (unsigned int j = 0; j < degree + 1; ++j)
        for (unsigned int k = 0; k < degree + 1 - std::max(i, j); ++k)
          std::cout << i << " " << j << " " << k << ": "
                    << poly.compute_polynomial_space_derivative(i, j, k, p)
                    << std::endl;

  std::cout << std::endl;
}

int main()
{
  {
    // test_quadratic<3>();
    for (unsigned int i = 1; i < 4; ++i)
      test_polynomial_space<3>(i);
  }
}
