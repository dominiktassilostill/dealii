// -----------------------------------------------------------------------------
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception OR LGPL-2.1-or-later
// Copyright (C) 2021 - 2026 by the deal.II authors
//
// This file is part of the deal.II library.
//
// Detailed license information governing the source code and contributions
// can be found in LICENSE.md and CONTRIBUTING.md at the top level directory.
//
// -----------------------------------------------------------------------------

#include <deal.II/base/config.h>

#include <deal.II/base/polynomials_barycentric.h>
#include <deal.II/base/qprojector.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_pyramid_p.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_simplex_p_bubbles.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/fe_wedge_p.h>

#include <deal.II/lac/householder.h>

DEAL_II_NAMESPACE_OPEN

namespace
{
  /**
   * Helper function to set up the dpo vector of FE_PyramidP for a given @p degree.
   */
  unsigned int
  compute_n_dofs(const unsigned int dim, const unsigned int degree)
  {
    AssertDimension(dim, 3);
    return (degree + 1) * (degree + 2) * (2 * degree + 3) / 6;
  }


  /**
   * Helper function to get equidistant support points on the pyramid
   */
  template <int dim>
  std::vector<Point<dim>>
  equidistant_support_points_fe_pyramid_p(const unsigned int degree)
  {
    Assert(degree > 0, ExcNotImplemented());

    if constexpr (dim == 3)
      {
        std::vector<Point<dim>> unit_points;

        const auto reference_cell = ReferenceCells::Pyramid;


        const FE_Q<1> fe_line(QIterated<1>(QTrapezoid<1>(), degree));
        // second argument is use_equidistant_support_points
        const FE_SimplexP<2> fe_triangle(degree, true);
        const FE_Q<2>        fe_quad(QIterated<1>(QTrapezoid<1>(), degree));

        // start with the vertices
        for (const unsigned int v : reference_cell.vertex_indices())
          unit_points.push_back(reference_cell.vertex(v));

        // lines
        for (const unsigned int l : reference_cell.line_indices())
          {
            const Point<dim> v0 =
              reference_cell.vertex(reference_cell.line_to_cell_vertices(l, 0));
            const Point<dim> v1 =
              reference_cell.vertex(reference_cell.line_to_cell_vertices(l, 1));

            for (unsigned int i = 0; i < degree - 1; ++i)
              {
                // shift the point on the line such that the support points on
                // the edges are compatible
                const double distance = fe_line.unit_support_point(
                  fe_line.get_first_line_index() + i)[0];
                unit_points.push_back(v0 + distance * (v1 - v0));
              }
          }

        // faces
        for (const unsigned int f : reference_cell.face_indices())
          {
            const auto face_reference_cell =
              reference_cell.face_reference_cell(f);

            const bool is_triangular_face =
              reference_cell.face_reference_cell(f).is_simplex();

            const unsigned int n_dofs_per_quad =
              is_triangular_face ? fe_triangle.n_dofs_per_quad() :
                                   fe_quad.n_dofs_per_quad();

            const std::vector<Point<2>> face_support_points =
              is_triangular_face ? fe_triangle.get_unit_support_points() :
                                   fe_quad.get_unit_support_points();

            const unsigned int first_quad_index =
              is_triangular_face ? fe_triangle.get_first_quad_index() :
                                   fe_quad.get_first_quad_index();

            // go over all DoFs on the face
            for (unsigned int i = 0; i < n_dofs_per_quad; ++i)
              {
                Point<dim> p(0.0, 0.0, 0.0);
                // linear interpolate the point form the vertices of the face
                // looking at the triangle it is the same as using barycentric
                // coordinates as the linear shape functions are: 1-x-y, x, y
                for (unsigned int v = 0; v < face_reference_cell.n_vertices();
                     ++v)
                  {
                    const auto vertex = reference_cell.vertex(
                      reference_cell.face_to_cell_vertices(
                        f, v, numbers::default_geometric_orientation));

                    p += face_reference_cell.d_linear_shape_function(
                           face_support_points[first_quad_index + i], v) *
                         vertex;
                  }
                unit_points.push_back(p);
              }
          }

        // interior, this is just the tensor product of the interior nodes of
        // the quad with the line but scaled
        for (unsigned int i = 0; i < degree - 1; ++i)
          {
            FE_Q<2> fe(QIterated<1>(QTrapezoid<1>(), degree - i - 1));
            for (unsigned int j = 0; j < fe.n_dofs_per_quad(); ++j)
              {
                const double z = fe_line.unit_support_point(
                  fe_line.get_first_line_index() + i)[0];

                const Point<2> x_y =
                  fe.unit_support_point(fe.get_first_quad_index() + j);

                unit_points.push_back(Point<dim>((1 - z) * (2.0 * x_y[0] - 1.0),
                                                 (1 - z) * (2.0 * x_y[1] - 1.0),
                                                 z));
              }
          }
        return unit_points;
      }
    else
      DEAL_II_ASSERT_UNREACHABLE();
    return {};
  }

  template <int dim>
  std::vector<Point<dim>>
  blend_and_warp_support_points_fe_pyramid_p(const unsigned int degree)
  {
    if constexpr (dim == 3)
      {
        const auto reference_cell = ReferenceCells::Pyramid;

        // the idea of the algorithm is to construct support points compatible
        // with triangles and quads on the faces, then take the boundary support
        // points and compute the displacement to the equidistant support points
        // on the faces, the last step is then to interpolate the displacement
        // to the interior nodes of the pyramid

        // get the equidistant support points
        const auto equidistant_points =
          equidistant_support_points_fe_pyramid_p<dim>(degree);


        const unsigned int n_dofs_per_line = degree - 1;
        const unsigned int n_dofs_per_tri =
          degree < 3 ? 0 : (degree - 1) * (degree - 2) / 2;

        const unsigned int n_boundary_nodes = 3 * degree * degree + 2;

        // basis function of boundary entities adopted from Chan and Warburton
        // returns the value of the hierarchical basis function i, which has
        // support on the vertices, edges and faces
        auto boundary_basis = [&](const unsigned int i, const Point<dim> &p) {
          // there are 3 * degree^2 + 2 shape functions on the vertices, edges
          // and faces
          Assert(i < n_boundary_nodes, ExcInternalError());

          double phi = 0.0;
          if (i < reference_cell.n_vertices())
            {
              // first are the vertices, use the linear shape functions
              phi = reference_cell.d_linear_shape_function(p, i);
            }
          else if (i < reference_cell.n_vertices() +
                         reference_cell.n_lines() * n_dofs_per_line)
            {
              // on the edge the basis functions are the linear basis functions
              // multiplied by 1D modal functions

              // get the line number and the index on the line
              const unsigned int line_index =
                (i - reference_cell.n_vertices()) / n_dofs_per_line;
              const unsigned int index_on_line =
                (i - reference_cell.n_vertices()) % n_dofs_per_line;

              // get the vertex indices determining the line
              const unsigned int v0 =
                reference_cell.line_to_cell_vertices(line_index, 0);
              const unsigned int v1 =
                reference_cell.line_to_cell_vertices(line_index, 1);

              const double l0 = reference_cell.d_linear_shape_function(p, v0);
              const double l1 = reference_cell.d_linear_shape_function(p, v1);

              phi = l0 * l1 *
                    dealii::Polynomials::jacobi_polynomial_value<double>(
                      index_on_line, 1, 1, l0 - l1, false);
            }
          else if (i < reference_cell.n_vertices() +
                         reference_cell.n_lines() * n_dofs_per_line +
                         n_dofs_per_line * n_dofs_per_line)
            {
              // on the quad face
              // get the index on the face
              const unsigned int index_on_quad =
                i - (reference_cell.n_vertices() +
                     reference_cell.n_lines() * n_dofs_per_line);

              // the quad face is made up of vertices 0, 1, 2, 3
              const double l0 = reference_cell.d_linear_shape_function(p, 0);
              const double l1 = reference_cell.d_linear_shape_function(p, 1);
              const double l2 = reference_cell.d_linear_shape_function(p, 2);
              const double l3 = reference_cell.d_linear_shape_function(p, 3);

              // from the index on the quad get the degrees of the jacobi
              // polynomials
              const unsigned int degree_x = index_on_quad / n_dofs_per_line;
              const unsigned int degree_y = index_on_quad % n_dofs_per_line;

              phi = l0 * l1 * l2 * l3 *
                    dealii::Polynomials::jacobi_polynomial_value<double>(
                      degree_x, 1, 1, p[0], false) *
                    dealii::Polynomials::jacobi_polynomial_value<double>(
                      degree_y, 1, 1, p[1], false);
            }
          else if (i < reference_cell.n_vertices() +
                         reference_cell.n_lines() * n_dofs_per_line +
                         n_dofs_per_line * n_dofs_per_line + 4 * n_dofs_per_tri)
            {
              // on a triangular face
              // get the face number and the index on the face
              const unsigned int offset =
                reference_cell.n_vertices() +
                reference_cell.n_lines() * n_dofs_per_line +
                n_dofs_per_line * n_dofs_per_line;
              const unsigned int face_index = (i - offset) / n_dofs_per_tri + 1;
              const unsigned int index_on_tri = (i - offset) % n_dofs_per_tri;

              // get the vertex indices for the face
              const unsigned int v0 = reference_cell.face_to_cell_vertices(
                face_index, 0, numbers::default_geometric_orientation);
              const unsigned int v1 = reference_cell.face_to_cell_vertices(
                face_index, 1, numbers::default_geometric_orientation);
              const unsigned int v2 = reference_cell.face_to_cell_vertices(
                face_index, 2, numbers::default_geometric_orientation);

              const double l0 = reference_cell.d_linear_shape_function(p, v0);
              const double l1 = reference_cell.d_linear_shape_function(p, v1);
              const double l2 = reference_cell.d_linear_shape_function(p, v2);

              // get the degrees of the basis function from the index on the
              // face
              unsigned int jacobi_poly_degree_i = numbers::invalid_unsigned_int;
              unsigned int jacobi_poly_degree_j = numbers::invalid_unsigned_int;
              for (unsigned int a = 0, counter = 0; a < degree - 2; ++a)
                for (unsigned int b = 0; b < degree - a - 2; ++b, ++counter)
                  if (index_on_tri == counter)
                    {
                      jacobi_poly_degree_i = a;
                      jacobi_poly_degree_j = b;
                    }

              // check if we found a valid index
              Assert(jacobi_poly_degree_i != numbers::invalid_unsigned_int,
                     ExcInternalError());
              Assert(jacobi_poly_degree_j != numbers::invalid_unsigned_int,
                     ExcInternalError());

              // now transform l0, l1, l2 to the local coordinates x,y on
              // the triangle, normally x,y = l1,l2 gives the correct results
              // the problem is l0 + l1 + l2 = 1 does not hold
              // but l0 + l1 + l2 = S_f
              // so normalize the barycentric coordinates by subtracting the
              // additional contributions as L_i = l_i + (1-S_f)/3 thus
              // L0 + L1 + L2 = 1 and setting x,y = L_1, L_2
              const double x = 1. / 3. * (2.0 * l1 - l0 - l2 + 1.0);
              const double y = 1. / 3. * (2.0 * l2 - l1 - l0 + 1.0);

              const double x_contribution =
                Polynomials::jacobi_polynomial_homogenized_value<double>(
                  jacobi_poly_degree_i, 0, 0, x, 1 - y);

              const double y_contribution =
                dealii::Polynomials::jacobi_polynomial_value<double>(
                  jacobi_poly_degree_j,
                  2 * jacobi_poly_degree_i + 1,
                  0,
                  y,
                  true);

              phi = l0 * l1 * l2 * x_contribution * y_contribution;
            }
          else
            DEAL_II_ASSERT_UNREACHABLE();

          return phi;
        };

        // start by constructing the vertices, edges and faces for the
        // compatible support points, use GL points and warp and blend nodes
        const FE_Q<1>        fe_line(degree);
        const FE_Q<2>        fe_quad(degree);
        const FE_SimplexP<2> fe_triangle(
          degree, /*use_equidistant_support_points*/ false);

        std::vector<Point<dim>> gl_points;
        // start with the vertices
        for (const unsigned int v : reference_cell.vertex_indices())
          gl_points.push_back(reference_cell.vertex(v));

        // lines
        for (const unsigned int l : reference_cell.line_indices())
          {
            const Point<dim> v0 =
              reference_cell.vertex(reference_cell.line_to_cell_vertices(l, 0));
            const Point<dim> v1 =
              reference_cell.vertex(reference_cell.line_to_cell_vertices(l, 1));

            const auto direction = v1 - v0;

            for (unsigned int i = 0; i < degree - 1; ++i)
              {
                // shift the point on the line such that the support points on
                // the edges are compatible
                const double distance = fe_line.unit_support_point(
                  fe_line.get_first_line_index() + i)[0];
                gl_points.push_back(v0 + distance * direction);
              }
          }

        // faces
        for (const unsigned int f : reference_cell.face_indices())
          {
            const auto face_reference_cell =
              reference_cell.face_reference_cell(f);

            const bool is_triangular_face =
              reference_cell.face_reference_cell(f).is_simplex();

            const unsigned int n_dofs_per_quad =
              is_triangular_face ? fe_triangle.n_dofs_per_quad() :
                                   fe_quad.n_dofs_per_quad();

            const std::vector<Point<2>> &face_support_points =
              is_triangular_face ? fe_triangle.get_unit_support_points() :
                                   fe_quad.get_unit_support_points();

            const unsigned int first_quad_index =
              is_triangular_face ? fe_triangle.get_first_quad_index() :
                                   fe_quad.get_first_quad_index();

            // go over all DoFs on the face
            for (unsigned int i = 0; i < n_dofs_per_quad; ++i)
              {
                const auto face_support_point =
                  face_support_points[first_quad_index + i];

                Point<dim> p(0.0, 0.0, 0.0);
                // linear interpolate the point form the vertices of the face
                // looking at the triangle it is the same as using barycentric
                // coordinates as the linear shape functions are: 1-x-y, x, y
                for (unsigned int v = 0; v < face_reference_cell.n_vertices();
                     ++v)
                  {
                    const unsigned int vertex_index =
                      reference_cell.face_to_cell_vertices(
                        f, v, numbers::default_geometric_orientation);
                    const auto vertex = reference_cell.vertex(vertex_index);

                    p += face_reference_cell.d_linear_shape_function(
                           face_support_point, v) *
                         vertex;
                  }
                gl_points.push_back(p);
              }
          }
        // needs to contain all nodes on the boundary
        Assert(gl_points.size() == n_boundary_nodes, ExcInternalError());

        // get the displacements between the blend and warp nodes and the
        // equidistant points on the boundary
        Vector<double> nodal_displacements_x(n_boundary_nodes);
        Vector<double> nodal_displacements_y(n_boundary_nodes);
        Vector<double> nodal_displacements_z(n_boundary_nodes);
        for (unsigned int i = 0; i < gl_points.size(); ++i)
          {
            const auto displacement_vector =
              gl_points[i] - equidistant_points[i];

            nodal_displacements_x[i] = displacement_vector[0];
            nodal_displacements_y[i] = displacement_vector[1];
            nodal_displacements_z[i] = displacement_vector[2];
          }

        // build the Vandermonde matrix to transform the boundary basis to a
        // nodal basis
        FullMatrix<double> VandermondeMatrix(n_boundary_nodes,
                                             n_boundary_nodes);
        for (unsigned int i = 0; i < VandermondeMatrix.m(); ++i)
          for (unsigned int j = 0; j < VandermondeMatrix.n(); ++j)
            VandermondeMatrix[i][j] = boundary_basis(j, equidistant_points[i]);

        // solve Vandermondematrix * displacements = nodal_displacements to get
        // the displacements expressed in the boundary basis
        Vector<double> displacements_x(n_boundary_nodes);
        Vector<double> displacements_y(n_boundary_nodes);
        Vector<double> displacements_z(n_boundary_nodes);

        Householder<double> householder(VandermondeMatrix);
        householder.least_squares(displacements_x, nodal_displacements_x);
        householder.least_squares(displacements_y, nodal_displacements_y);
        householder.least_squares(displacements_z, nodal_displacements_z);

        // interpolate the displacement to the interor nodes
        for (unsigned int i = n_boundary_nodes; i < equidistant_points.size();
             ++i)
          {
            const auto eq_point = equidistant_points[i];

            Point<dim> displacement(0.0, 0.0, 0.0);

            for (unsigned int j = 0; j < displacements_x.size(); ++j)
              {
                const double basis_value = boundary_basis(j, eq_point);
                displacement[0] += basis_value * displacements_x[j];
                displacement[1] += basis_value * displacements_y[j];
                displacement[2] += basis_value * displacements_z[j];
              }

            gl_points.emplace_back(eq_point + displacement);
          }

        for (unsigned int i = 0; i < gl_points.size(); ++i)
          for (unsigned int d = 0; d < dim; ++d)
            if (std::abs(gl_points[i][d]) < 1e-12)
              gl_points[i][d] = 0.0;

        return gl_points;
      }
    else
      DEAL_II_ASSERT_UNREACHABLE();
    return {};
  }



  /**
   * Helper function to set up the dpo vector of FE_PyramidP for a given @p degree.
   */
  template <int dim>
  std::vector<Point<dim>>
  support_points_fe_pyramid_p(const unsigned int degree,
                              const bool         use_equidistant_support_points)
  {
    if (use_equidistant_support_points || degree < 3)
      return equidistant_support_points_fe_pyramid_p<dim>(degree);
    return blend_and_warp_support_points_fe_pyramid_p<dim>(degree);
  }



  /**
   * Helper function to set up the dpo vector of FE_PyramidP and FE_PyramidDGP for a given @p degree.
   */
  template <int dim>
  internal::GenericDoFsPerObject
  get_dpo(const unsigned int                                degree,
          const typename FiniteElementData<dim>::Conformity conformity)
  {
    AssertDimension(dim, 3);
    internal::GenericDoFsPerObject dpo;

    if (conformity == FiniteElementData<dim>::L2)
      {
        dpo = internal::expand<3>({{0, 0, 0, compute_n_dofs(dim, degree)}},
                                  ReferenceCells::Pyramid);
      }
    else if (conformity == FiniteElementData<dim>::H1)
      {
        // the support points on the 8 lines excluding the vertices
        const unsigned int n_dofs_per_line = degree - 1;

        // support points on the bottom quad face and on the 4 triangular faces,
        // on the triangular faces the number of points is the sum from 1 to
        // (degree - 2) so 4*0.5*(degree - 2)*(degree - 1)
        const unsigned int n_dofs_per_quad = n_dofs_per_line * n_dofs_per_line;
        const unsigned int n_dofs_per_tri  = (degree - 2) * (degree - 1) / 2;
        const unsigned int total_dofs_faces =
          n_dofs_per_quad + 4 * n_dofs_per_tri;
        // total number of DoFs on a tri
        const unsigned int n_dofs_per_tri_inclusive =
          (degree + 1) * (degree + 2) / 2;

        const unsigned int n_dofs_total = compute_n_dofs(dim, degree);

        dpo.dofs_per_object_exclusive = {
          {1, 1, 1, 1, 1},
          {n_dofs_per_line,
           n_dofs_per_line,
           n_dofs_per_line,
           n_dofs_per_line,
           n_dofs_per_line,
           n_dofs_per_line,
           n_dofs_per_line,
           n_dofs_per_line},
          {n_dofs_per_quad,
           n_dofs_per_tri,
           n_dofs_per_tri,
           n_dofs_per_tri,
           n_dofs_per_tri},
          {n_dofs_total - 5 - 8 * n_dofs_per_line - total_dofs_faces}};

        dpo.dofs_per_object_inclusive = {{1, 1, 1, 1, 1},
                                         {
                                           degree + 1,
                                           degree + 1,
                                           degree + 1,
                                           degree + 1,
                                           degree + 1,
                                           degree + 1,
                                           degree + 1,
                                           degree + 1,
                                         },
                                         {(degree + 1) * (degree + 1),
                                          n_dofs_per_tri_inclusive,
                                          n_dofs_per_tri_inclusive,
                                          n_dofs_per_tri_inclusive,
                                          n_dofs_per_tri_inclusive},
                                         {n_dofs_total}};

        dpo.object_index = {
          {0, 1, 2, 3, 4},
          {5,
           5 + 1 * n_dofs_per_line,
           5 + 2 * n_dofs_per_line,
           5 + 3 * n_dofs_per_line,
           5 + 4 * n_dofs_per_line,
           5 + 5 * n_dofs_per_line,
           5 + 6 * n_dofs_per_line,
           5 + 7 * n_dofs_per_line},
          {5 + 8 * n_dofs_per_line,
           5 + 8 * n_dofs_per_line + n_dofs_per_quad,
           5 + 8 * n_dofs_per_line + n_dofs_per_quad + n_dofs_per_tri,
           5 + 8 * n_dofs_per_line + n_dofs_per_quad + 2 * n_dofs_per_tri,
           5 + 8 * n_dofs_per_line + n_dofs_per_quad + 3 * n_dofs_per_tri},
          {5 + 8 * n_dofs_per_line + total_dofs_faces}};

        dpo.first_object_index_on_face = {{0, 0, 0, 0, 0},
                                          {4, 3, 3, 3, 3},
                                          {4 + 4 * n_dofs_per_line,
                                           3 + 3 * n_dofs_per_line,
                                           3 + 3 * n_dofs_per_line,
                                           3 + 3 * n_dofs_per_line,
                                           3 + 3 * n_dofs_per_line}};
      }
    return dpo;
  }
} // namespace


template <int dim, int spacedim>
FE_PyramidPoly<dim, spacedim>::FE_PyramidPoly(
  const unsigned int                                degree,
  const internal::GenericDoFsPerObject              dpos,
  const std::vector<Point<dim>>                     support_points,
  const bool                                        prolongation_is_additive,
  const typename FiniteElementData<dim>::Conformity conformity)
  : dealii::FE_Poly<dim, spacedim>(
      ScalarLagrangePolynomialPyramid<dim>(degree,
                                           compute_n_dofs(dim, degree),
                                           support_points),
      FiniteElementData<dim>(dpos,
                             reinterpret_cast<const ReferenceCell<dim> &>(
                               ReferenceCells::Pyramid),
                             1,
                             degree,
                             conformity),
      std::vector<bool>(
        FiniteElementData<dim>(dpos,
                               reinterpret_cast<const ReferenceCell<dim> &>(
                                 ReferenceCells::Pyramid),
                               1,
                               degree)
          .dofs_per_cell,
        prolongation_is_additive),
      std::vector<ComponentMask>(
        FiniteElementData<dim>(dpos,
                               reinterpret_cast<const ReferenceCell<dim> &>(
                                 ReferenceCells::Pyramid),
                               1,
                               degree)
          .dofs_per_cell,
        ComponentMask(std::vector<bool>(1, true))))
{
  AssertDimension(dim, 3);

  for (auto &support_point : support_points)
    this->unit_support_points.emplace_back(support_point);
}



template <int dim, int spacedim>
void
FE_PyramidPoly<dim, spacedim>::
  convert_generalized_support_point_values_to_dof_values(
    const std::vector<Vector<double>> &support_point_values,
    std::vector<double>               &nodal_values) const
{
  AssertDimension(support_point_values.size(),
                  this->get_unit_support_points().size());
  AssertDimension(support_point_values.size(), nodal_values.size());
  AssertDimension(this->dofs_per_cell, nodal_values.size());

  for (unsigned int i = 0; i < this->dofs_per_cell; ++i)
    {
      AssertDimension(support_point_values[i].size(), 1);

      nodal_values[i] = support_point_values[i](0);
    }
}



template <int dim, int spacedim>
FE_PyramidP<dim, spacedim>::FE_PyramidP(
  const unsigned int degree,
  const bool         use_equidistant_support_points)
  : FE_PyramidPoly<dim, spacedim>(
      degree,
      get_dpo<dim>(degree, FiniteElementData<dim>::H1),
      support_points_fe_pyramid_p<dim>(degree, use_equidistant_support_points),
      false,
      FiniteElementData<dim>::H1)
{
  // face support points
  this->unit_face_support_points.resize(this->reference_cell().n_faces());

  for (const auto f : this->reference_cell().face_indices())
    {
      const auto face_reference_cell =
        this->reference_cell().face_reference_cell(f);

      if (face_reference_cell == ReferenceCells::Quadrilateral)
        {
          FE_Q<2> fe_face = use_equidistant_support_points ?
                              FE_Q<2>(QIterated<1>(QTrapezoid<1>(), degree)) :
                              FE_Q<2>(degree);

          for (const auto &face_support_point :
               fe_face.get_unit_support_points())
            {
              Point<dim - 1> p;
              for (unsigned int d = 0; d < dim - 1; ++d)
                p[d] = face_support_point[d];
              this->unit_face_support_points[f].emplace_back(p);
            }
        }
      else if (face_reference_cell == ReferenceCells::Triangle)
        {
          FE_SimplexP<2> fe_face(degree, use_equidistant_support_points);
          for (const auto &face_support_point :
               fe_face.get_unit_support_points())
            {
              Point<dim - 1> p;
              for (unsigned int d = 0; d < dim - 1; ++d)
                p[d] = face_support_point[d];
              this->unit_face_support_points[f].emplace_back(p);
            }
        }
    }

  // adjust line and face indices
  if (degree > 2)
    {
      // adjust DoFs on lines
      for (unsigned int i = 0; i < this->n_dofs_per_line(); ++i)
        this->adjust_line_dof_index_for_line_orientation_table[i] =
          this->n_dofs_per_line() - 1 - i - i;

      // adjust DoFs on faces
      FETools::adjust_quad_dof_index_for_face_orientation(
        *this, this->adjust_quad_dof_index_for_face_orientation_table);
    }
}



template <int dim, int spacedim>
std::unique_ptr<FiniteElement<dim, spacedim>>
FE_PyramidP<dim, spacedim>::clone() const
{
  return std::make_unique<FE_PyramidP<dim, spacedim>>(*this);
}



template <int dim, int spacedim>
std::string
FE_PyramidP<dim, spacedim>::get_name() const
{
  std::ostringstream namebuf;
  namebuf << "FE_PyramidP<" << Utilities::dim_string(dim, spacedim) << ">("
          << this->degree << ")";

  return namebuf.str();
}



template <int dim, int spacedim>
FiniteElementDomination::Domination
FE_PyramidP<dim, spacedim>::compare_for_domination(
  const FiniteElement<dim, spacedim> &fe_other,
  const unsigned int                  codim) const
{
  Assert(codim <= dim, ExcImpossibleInDim(dim));

  // vertex/line/face domination
  // (if fe_other is derived from FE_SimplexDGP)
  // ------------------------------------
  if (codim > 0)
    if (dynamic_cast<const FE_SimplexDGP<dim, spacedim> *>(&fe_other) !=
        nullptr)
      // there are no requirements between continuous and discontinuous
      // elements
      return FiniteElementDomination::no_requirements;

  // vertex/line/face domination
  // (if fe_other is not derived from FE_SimplexDGP)
  // & cell domination
  // ----------------------------------------
  if (const FE_PyramidP<dim, spacedim> *fe_pp_other =
        dynamic_cast<const FE_PyramidP<dim, spacedim> *>(&fe_other))
    {
      if (this->degree < fe_pp_other->degree)
        return FiniteElementDomination::this_element_dominates;
      else if (this->degree == fe_pp_other->degree)
        return FiniteElementDomination::either_element_can_dominate;
      else
        return FiniteElementDomination::other_element_dominates;
    }
  else if (const FE_SimplexP<dim, spacedim> *fe_p_other =
             dynamic_cast<const FE_SimplexP<dim, spacedim> *>(&fe_other))
    {
      if (this->degree < fe_p_other->degree)
        return FiniteElementDomination::this_element_dominates;
      else if (this->degree == fe_p_other->degree)
        return FiniteElementDomination::either_element_can_dominate;
      else
        return FiniteElementDomination::other_element_dominates;
    }
  else if (const FE_Q<dim, spacedim> *fe_q_other =
             dynamic_cast<const FE_Q<dim, spacedim> *>(&fe_other))
    {
      if (this->degree < fe_q_other->degree)
        return FiniteElementDomination::this_element_dominates;
      else if (this->degree == fe_q_other->degree)
        return FiniteElementDomination::either_element_can_dominate;
      else
        return FiniteElementDomination::other_element_dominates;
    }
  else if (const FE_WedgeP<dim, spacedim> *fe_pp_other =
             dynamic_cast<const FE_WedgeP<dim, spacedim> *>(&fe_other))
    {
      if (this->degree < fe_pp_other->degree)
        return FiniteElementDomination::this_element_dominates;
      else if (this->degree == fe_pp_other->degree)
        return FiniteElementDomination::either_element_can_dominate;
      else
        return FiniteElementDomination::other_element_dominates;
    }
  else if (const FE_Nothing<dim, spacedim> *fe_nothing =
             dynamic_cast<const FE_Nothing<dim, spacedim> *>(&fe_other))
    {
      if (fe_nothing->is_dominating())
        return FiniteElementDomination::other_element_dominates;
      else
        // the FE_Nothing has no degrees of freedom and it is typically used
        // in a context where we don't require any continuity along the
        // interface
        return FiniteElementDomination::no_requirements;
    }

  DEAL_II_NOT_IMPLEMENTED();
  return FiniteElementDomination::neither_element_dominates;
}



template <int dim, int spacedim>
std::vector<std::pair<unsigned int, unsigned int>>
FE_PyramidP<dim, spacedim>::hp_vertex_dof_identities(
  const FiniteElement<dim, spacedim> &fe_other) const
{
  (void)fe_other;

  Assert((dynamic_cast<const FE_SimplexP<dim, spacedim> *>(&fe_other)) ||
           (dynamic_cast<const FE_Q<dim, spacedim> *>(&fe_other)) ||
           (dynamic_cast<const FE_PyramidP<dim, spacedim> *>(&fe_other)) ||
           (dynamic_cast<const FE_WedgeP<dim, spacedim> *>(&fe_other)),
         ExcNotImplemented());

  return {{0, 0}};
}



template <int dim, int spacedim>
std::vector<std::pair<unsigned int, unsigned int>>
FE_PyramidP<dim, spacedim>::hp_line_dof_identities(
  const FiniteElement<dim, spacedim> &fe_other) const
{
  Assert((dynamic_cast<const FE_SimplexP<dim, spacedim> *>(&fe_other)) ||
           (dynamic_cast<const FE_Q<dim, spacedim> *>(&fe_other)) ||
           (dynamic_cast<const FE_PyramidP<dim, spacedim> *>(&fe_other)) ||
           (dynamic_cast<const FE_WedgeP<dim, spacedim> *>(&fe_other)),
         ExcNotImplemented());

  std::vector<std::pair<unsigned int, unsigned int>> identities;
  // check if the support points are the same location on the line
  // the pyramid base is defined by the vertices [-1,-1,0], [1,-1,0],
  // [-1,1,0], [1,1,0], to avoid rescaling use the support points on the faces
  const auto &face_support_points = this->get_unit_face_support_points(0);
  const auto &face_support_points_other =
    fe_other.get_unit_face_support_points(0);

  // now just compare the DoFs on the line going from [0,0] to [1,0]
  // for a triangular face that is the first line
  // for a quad face that is the third line
  // adjust the offsets accordingly
  // face number 0 of the pyramid is a quad
  const unsigned int offset =
    this->reference_cell().face_reference_cell(0).n_vertices() +
    2 * this->n_dofs_per_line();

  const unsigned int offset_other =
    fe_other.reference_cell().face_reference_cell(0).is_hyper_cube() ?
      fe_other.reference_cell().face_reference_cell(0).n_vertices() +
        2 * fe_other.n_dofs_per_line() :
      fe_other.reference_cell().face_reference_cell(0).n_vertices();

  // now get the identities
  for (unsigned int i = 0; i < this->n_dofs_per_line(); ++i)
    for (unsigned int j = 0; j < fe_other.n_dofs_per_line(); ++j)
      if (face_support_points[i + offset].distance(
            face_support_points_other[j + offset_other]) < 1e-14)
        identities.emplace_back(i, j);

  return identities;
}



template <int dim, int spacedim>
std::vector<std::pair<unsigned int, unsigned int>>
FE_PyramidP<dim, spacedim>::hp_quad_dof_identities(
  const FiniteElement<dim, spacedim> &fe_other,
  const unsigned int                  face_no) const
{
  AssertIndexRange(face_no, 5);
  std::vector<std::pair<unsigned int, unsigned int>> identities;

  unsigned int face_no_neighbor;

  if (face_no == 0)
    {
      // on a quad, neighbor can be a hex, a pyramid or a wedge
      Assert((dynamic_cast<const FE_Q<dim, spacedim> *>(&fe_other)) ||
               (dynamic_cast<const FE_PyramidP<dim, spacedim> *>(&fe_other)) ||
               (dynamic_cast<const FE_WedgeP<dim, spacedim> *>(&fe_other)),
             ExcNotImplemented());
      // for the wedge the first quad face is face no. 2
      if (dynamic_cast<const FE_WedgeP<dim, spacedim> *>(&fe_other))
        face_no_neighbor = 2;
      else
        face_no_neighbor = 0;
    }
  else
    {
      Assert((dynamic_cast<const FE_SimplexP<dim, spacedim> *>(&fe_other)) ||
               (dynamic_cast<const FE_PyramidP<dim, spacedim> *>(&fe_other)) ||
               (dynamic_cast<const FE_WedgeP<dim, spacedim> *>(&fe_other)),
             ExcNotImplemented());
      // on tri, neighbor can be a tet, a pyramid or a wedge
      if (dynamic_cast<const FE_PyramidP<dim, spacedim> *>(&fe_other))
        face_no_neighbor = 1;
      else
        face_no_neighbor = 0;
    }

  // compare the face support points
  const auto &face_support_points = this->get_unit_face_support_points(face_no);
  const auto &face_support_points_other =
    fe_other.get_unit_face_support_points(face_no_neighbor);

  // get the offsets to only compare the DoFs within the face as the vertices
  // and lines were done before
  const auto face_reference_cell =
    this->reference_cell().face_reference_cell(face_no);

  Assert(face_reference_cell ==
           fe_other.reference_cell().face_reference_cell(face_no_neighbor),
         ExcInternalError());

  const unsigned int offset =
    face_reference_cell.n_vertices() +
    face_reference_cell.n_lines() * this->n_dofs_per_line();

  const unsigned int offset_other =
    face_reference_cell.n_vertices() +
    face_reference_cell.n_lines() * fe_other.n_dofs_per_line();

  // do the comparison
  for (unsigned int i = 0; i < this->n_dofs_per_quad(face_no); ++i)
    for (unsigned int j = 0; j < fe_other.n_dofs_per_quad(face_no_neighbor);
         ++j)
      if (face_support_points[i + offset].distance(
            face_support_points_other[j + offset_other]) < 1e-14)
        identities.emplace_back(i, j);

  return identities;
}



template <int dim, int spacedim>
FE_PyramidDGP<dim, spacedim>::FE_PyramidDGP(
  const unsigned int degree,
  const bool         use_equidistant_support_points)
  : FE_PyramidPoly<dim, spacedim>(
      degree,
      get_dpo<dim>(degree, FiniteElementData<dim>::L2),
      support_points_fe_pyramid_p<dim>(degree, use_equidistant_support_points),
      true,
      FiniteElementData<dim>::L2)
{}



template <int dim, int spacedim>
std::unique_ptr<FiniteElement<dim, spacedim>>
FE_PyramidDGP<dim, spacedim>::clone() const
{
  return std::make_unique<FE_PyramidDGP<dim, spacedim>>(*this);
}



template <int dim, int spacedim>
std::string
FE_PyramidDGP<dim, spacedim>::get_name() const
{
  std::ostringstream namebuf;
  namebuf << "FE_PyramidDGP<" << Utilities::dim_string(dim, spacedim) << ">("
          << this->degree << ")";

  return namebuf.str();
}

// explicit instantiations
#include "fe/fe_pyramid_p.inst"

DEAL_II_NAMESPACE_CLOSE
