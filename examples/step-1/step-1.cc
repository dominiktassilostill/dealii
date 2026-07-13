#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_pyramid_p.h>
#include <deal.II/fe/fe_wedge_p.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/polynomials_pyramid.h>
#include <deal.II/base/polynomials_simplex.h>
#include <deal.II/base/polynomial.h>

#include <deal.II/lac/lapack_full_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/lac/householder.h>

// This is needed for C++ output:
#include <iostream>
#include <fstream>
// And this for the declarations of the `std::sqrt` and `std::fabs` functions:
#include <cmath>
using namespace dealii;

std::vector<unsigned int> get_dpo_vector_fe_p(const unsigned int dim,
                                              const unsigned int degree)
{
  Assert(degree != 0, ExcNotImplemented());

  switch (dim)
    {
      case 1:
        return {1, degree - 1};
      case 2:
        // the number of support points on the face is
        // \sum_{i=1}^{degree - 2} i = (degree-2)*(degree-1)/2
        return {1, degree - 1, (degree - 2) * (degree - 1) / 2};
      case 3:
        // the number of support points in the volume are that of a tet
        // with a lower degree (degree-4)
        return {1,
                degree - 1,
                (degree - 2) * (degree - 1) / 2,
                (degree - 3) * (degree - 2) * (degree - 1) / 6};
    }

  DEAL_II_ASSERT_UNREACHABLE();
  return {};
}



/**
 * Set up a vector that contains the unit (reference) cell support points
 * for FE_SimplexPoly and sufficiently similar elements.
 */
template <int dim>
std::vector<Point<dim>>
equidistant_support_points_fe_p(const unsigned int degree)
{
  Assert(dim != 0, ExcInternalError());
  std::vector<Point<dim>> unit_points;
  const auto              reference_cell = ReferenceCells::get_simplex<dim>();

  // Piecewise constants are a special case: use a support point at the
  // centroid and only the centroid
  if (degree == 0)
    {
      unit_points.emplace_back(reference_cell.barycenter());
      return unit_points;
    }

  // otherwise write everything as linear combinations of vertices
  const auto dpo = get_dpo_vector_fe_p(dim, degree);
  Assert(dpo.size() == dim + 1, ExcInternalError());
  Assert(dpo[0] == 1, ExcNotImplemented());

  // vertices:
  for (const unsigned int d : reference_cell.vertex_indices())
    unit_points.push_back(reference_cell.vertex(d));

  // lines:
  for (const unsigned int l : reference_cell.line_indices())
    {
      const Point<dim> p0 =
        unit_points[reference_cell.line_to_cell_vertices(l, 0)];
      const Point<dim> p1 =
        unit_points[reference_cell.line_to_cell_vertices(l, 1)];
      for (unsigned int p = 0; p < dpo[1]; ++p)
        unit_points.push_back((double(dpo[1] - p) / (dpo[1] + 1)) * p0 +
                              (double(p + 1) / (dpo[1] + 1)) * p1);
    }

  // faces:
  if constexpr (dim == 2)
    {
      unsigned int counter = 0;
      for (unsigned int i = 1; i < degree; ++i)
        for (unsigned int j = 1; j < degree - i; ++j, ++counter)
          {
            const double x = static_cast<double>(j) / degree;
            const double y = static_cast<double>(i) / degree;

            unit_points.push_back(Point<dim>(x, y));
          }
      Assert(counter == dpo[2], ExcInternalError());
    }

  if constexpr (dim == 3)
    for (const unsigned int f : reference_cell.face_indices())
      {
        const Point<dim> p0 = unit_points[reference_cell.face_to_cell_vertices(
          f, 0, numbers::default_geometric_orientation)];
        const Point<dim> p1 = unit_points[reference_cell.face_to_cell_vertices(
          f, 1, numbers::default_geometric_orientation)];
        const Point<dim> p2 = unit_points[reference_cell.face_to_cell_vertices(
          f, 2, numbers::default_geometric_orientation)];

        unsigned int counter = 0;
        for (unsigned int i = 1; i < degree; ++i)
          for (unsigned int j = 1; j < degree - i; ++j, ++counter)
            {
              const double a = static_cast<double>(j) / degree;
              const double b = static_cast<double>(i) / degree;
              const double c = 1.0 - a - b;
              unit_points.push_back(c * p0 + a * p1 + b * p2);
            }
        Assert(counter == dpo[2], ExcInternalError());
      }

  // interior
  if constexpr (dim == 3)
    {
      unsigned int counter = 0;
      for (unsigned int i = 1; i < degree; ++i)
        for (unsigned int j = 1; j < degree - i; ++j)
          for (unsigned int k = 1; k < degree - i - j; ++k, ++counter)
            {
              const double x = static_cast<double>(i) / degree;
              const double y = static_cast<double>(j) / degree;
              const double z = static_cast<double>(k) / degree;

              unit_points.push_back(Point<dim>(x, y, z));
            }
      Assert(counter == dpo[3], ExcInternalError());
    }

  return unit_points;
}

/**
 * Set up a vector that contains the electrostatic support points
 * for FE_SimplexPoly and sufficiently similar elements.
 * The points are constructed by the blend and warp alogrithm described
 * by Hesthaven and Warburton.
 */
template <int dim>
std::vector<Point<dim>>
electrostatic_support_points_fe_p(const unsigned int degree)
{
  Assert(degree > 0, ExcNotImplemented());

  if constexpr (dim == 1)
    {
      const FE_Q<dim> feq(degree);
      return feq.get_unit_support_points();
    }

  constexpr double tol = 1e-12;

  // get equidistant nodes
  const std::vector<Point<dim>> equidistant_nodes =
    equidistant_support_points_fe_p<dim>(degree);

  // reserve space for electrostatic nodes
  std::vector<Point<dim>> electrostatic_nodes;
  electrostatic_nodes.reserve(equidistant_nodes.size());

  // equidistant feq
  const FE_Q<1> feq_equi(QIterated<1>(QTrapezoid<1>(), degree));
  // feq Gauss-Lobatto
  const FE_Q<1> feq_gl(degree);

  // compute the shift in support points
  std::vector<double> support_points_shift(feq_equi.n_dofs_per_cell());
  for (unsigned int i = 0; i < feq_equi.n_dofs_per_cell(); ++i)
    support_points_shift[i] =
      feq_gl.unit_support_point(i)[0] - feq_equi.unit_support_point(i)[0];

  // warp function
  // interpolates between GL nodes and equidistant nodes
  auto warpfactor = [&support_points_shift, &feq_equi, tol](const double x) {
    // if the node is one of the vertices, i.e. it is at +-1
    // then there is no shift
    if (std::abs(1.0 - std::abs(x)) < tol)
      return 0.0;

    const double scaling = 2.0 / (1.0 - x * x);

    // rescale from [-1,1] to interval [0,1]
    const Point<1> p(0.5 * x + 0.5);

    double warp = 0.0;
    for (unsigned int i = 0; i < feq_equi.n_dofs_per_cell(); ++i)
      warp += support_points_shift[i] * feq_equi.shape_value(i, p);

    return scaling * warp;
  };

  // compute the warp in one face, use barycentric coordinates l0, l1, l2
  auto face_warp = [&warpfactor](const double l0,
                                 const double l1,
                                 const double l2,
                                 const double alpha) {
    const double warp0 =
      2.0 * l0 * l1 * warpfactor(l1 - l0) * (1.0 + alpha * alpha * l2 * l2);
    const double warp1 =
      2.0 * l1 * l2 * warpfactor(l2 - l1) * (1.0 + alpha * alpha * l0 * l0);
    const double warp2 =
      2.0 * l2 * l0 * warpfactor(l0 - l2) * (1.0 + alpha * alpha * l1 * l1);

    return std::array<double, 3>{{warp0, warp1, warp2}};
  };

  if constexpr (dim == 2)
    {
      // optimized alpha values
      const std::array<double, 15> alpha_opt = {{0.0000,
                                                 0.0000,
                                                 1.4152,
                                                 0.1001,
                                                 0.2751,
                                                 0.9800,
                                                 1.0999,
                                                 1.2832,
                                                 1.3648,
                                                 1.4773,
                                                 1.4959,
                                                 1.5743,
                                                 1.5770,
                                                 1.6223,
                                                 1.6258}};

      const double alpha =
        degree <= alpha_opt.size() ? alpha_opt[degree - 1] : 5.0 / 3.0;

      // go over all equidistant points and adjust
      for (const auto &p : equidistant_nodes)
        {
          const double x = p[0];
          const double y = p[1];
          const double l = 1.0 - x - y;

          // get combined blend and warp
          const std::array<double, 3> warp = face_warp(l, x, y, alpha);

          // accumulate deformation
          const double x_electrostatic = x + warp[0] - warp[1];
          const double y_electrostatic = y + warp[1] - warp[2];

          electrostatic_nodes.emplace_back(x_electrostatic, y_electrostatic);
        }
    }
  else if constexpr (dim == 3)
    {
      const auto reference_cell = ReferenceCells::Tetrahedron;

      // optimized alpha values for tetrahedra
      const std::array<double, 15> alpha_opt = {{0.0000,
                                                 0.0000,
                                                 0.0000,
                                                 0.1002,
                                                 1.1332,
                                                 1.5608,
                                                 1.3413,
                                                 1.2577,
                                                 1.1603,
                                                 1.10153,
                                                 0.6080,
                                                 0.4523,
                                                 0.8856,
                                                 0.8717,
                                                 0.9655}};

      const double alpha =
        degree <= alpha_opt.size() ? alpha_opt[degree - 1] : 1.0;

      // go over all equidistant points and adjust
      for (const auto &p : equidistant_nodes)
        {
          const double x = p[0];
          const double y = p[1];
          const double z = p[2];

          // write in barycentric coordinates
          const std::array<double, 4> l = {{1.0 - x - y - z, x, y, z}};

          // reserve space for the shift
          std::array<double, 4> dl = {{0.0, 0.0, 0.0, 0.0}};

          // check if we are on a vertex, edge, face or volume
          unsigned int n_pos = 0;
          for (const auto barycentric_coordinate : l)
            if (std::abs(barycentric_coordinate) > tol)
              ++n_pos;

          // on the vertex
          if (n_pos < 2)
            {
              // nothing to do
            }
          // on the edge apply the warp exactly once
          else if (n_pos == 2)
            {
              // get the two positive coordinates
              std::array<unsigned int, 2> idx;

              unsigned int j = 0;
              for (unsigned int i = 0; i < l.size(); ++i)
                if (std::abs(l[i]) > tol)
                  idx[j++] = i;

              const double l0 = l[idx[0]];
              const double l1 = l[idx[1]];

              // get the warp
              const std::array<double, 3> warp = face_warp(l0, l1, 0.0, alpha);

              // apply to the edge
              dl[idx[0]] = -warp[0];
              dl[idx[1]] = warp[0];
            }
          else
            // in the other cases loop over all faces and accumulate the
            // contributions
            for (const auto f : reference_cell.face_indices())
              {
                // get the vertex ids for the barycentric coordinates
                std::array<unsigned int, 4> idx;

                // the first entry is the vertex opposite the face
                idx[0] = 3 - f;

                // get the vertices in the face
                for (unsigned int i = 0; i < idx.size() - 1; ++i)
                  idx[i + 1] = reference_cell.face_to_cell_vertices(
                    f, i, numbers::default_geometric_orientation);

                // get coordinates of face
                const double l0 = l[idx[0]];
                const double l1 = l[idx[1]];
                const double l2 = l[idx[2]];
                const double l3 = l[idx[3]];

                // get face warp
                const std::array<double, 3> warp = face_warp(l1, l2, l3, alpha);

                // volume blend
                const double blend_linear =
                  (l1 + 0.5 * l0) * (l2 + 0.5 * l0) * (l3 + 0.5 * l0);

                const double blend = (blend_linear > tol) ?
                                       (1.0 + alpha * alpha * l0 * l0) * l1 *
                                         l2 * l3 / blend_linear :
                                       0.0;

                dl[idx[1]] += blend * (warp[2] - warp[0]);
                dl[idx[2]] += blend * (warp[0] - warp[1]);
                dl[idx[3]] += blend * (warp[1] - warp[2]);
              }

          electrostatic_nodes.emplace_back(x + dl[1], y + dl[2], z + dl[3]);
        }
    }
  else
    DEAL_II_ASSERT_UNREACHABLE();

  return electrostatic_nodes;
}



template <int dim>
std::vector<Point<dim>>
get_support_points_fe_pyramid_p(const unsigned int degree)
{
  AssertDimension(dim, 3);
  Assert(degree > 0, ExcInternalError("Degree must be larger than 0."));


  std::vector<Point<dim>> support_points;
  const unsigned int      n_dofs =
    (degree + 1) * (degree + 2) * (2 * degree + 3) / 6;
  support_points.resize(n_dofs);

  const double z_equidistance = 1.0 / degree;

  // the support points on the 8 lines excluding the vertices
  const unsigned int n_dofs_per_line = degree - 1;

  // support points on the bottom quad face and on the 4 triangular faces,
  // on the triangular faces the number of points is the sum from 1 to
  // (degree - 2) so 4*0.5*(degree - 2)*(degree - 1)
  const unsigned int n_dofs_per_quad = n_dofs_per_line * n_dofs_per_line;
  const unsigned int total_dofs_faces =
    n_dofs_per_quad + 2 * (degree - 2) * (degree - 1);

  // starting indices for lines 4 - 7
  std::vector<unsigned int> start_lines(4);
  // line 4 starts after the DoFs at the vertices and the DoFs on the lines of
  // the bottom quad
  start_lines[0] = 5 + 4 * n_dofs_per_line;
  // the rest increments with the number of DoFs on the edges 4 - 7
  for (unsigned int i = 1; i < 4; ++i)
    start_lines[i] = start_lines[i - 1] + n_dofs_per_line;

  // same applies to the triangular faces 1 - 4
  std::vector<unsigned int> start_faces(4);
  start_faces[0] = 5 + 8 * n_dofs_per_line + n_dofs_per_quad;

  for (unsigned int i = 1; i < 4; ++i)
    start_faces[i] = start_faces[i - 1] + (degree - 2) * (degree - 1) / 2;

  unsigned int start_hex = 5 + 8 * n_dofs_per_line + total_dofs_faces;

  auto lift_point =
    [](const Point<2> &p2d, const double scale, const double z) {
      return Point<dim>(scale * (2.0 * p2d[0] - 1.0),
                        scale * (2.0 * p2d[1] - 1.0),
                        z);
    };
  {
    // this gives all info on the vertices, the first 4 edges and the
    // first face
    // switch to FE_Q when simplex supports electrostatic points
    // FE_Q<2> fe_q(degree);
    FE_Q<2> fe_q(QIterated<1>(QTrapezoid<1>(), degree));

    // vertices
    for (unsigned int v = 0; v < fe_q.reference_cell().n_vertices(); ++v)
      {
        support_points[v] =
          lift_point(fe_q.get_unit_support_points()[v], 1.0, 0.0);
      }
    // lines
    for (unsigned int l = 0;
         l < fe_q.reference_cell().n_lines() * fe_q.n_dofs_per_line();
         ++l)
      {
        support_points[5 + l] = lift_point(
          fe_q
            .get_unit_support_points()[fe_q.reference_cell().n_vertices() + l],
          1.0,
          0.0);
      }
    // quad
    for (unsigned int q = 0; q < fe_q.n_dofs_per_quad(); ++q)
      {
        support_points[5 + 8 * n_dofs_per_line + q] = lift_point(
          fe_q.get_unit_support_points()[fe_q.reference_cell().n_vertices() +
                                         fe_q.reference_cell().n_lines() *
                                           fe_q.n_dofs_per_line() +
                                         q],
          1.0,
          0.0);
      }
  }
  // now add the other layers
  for (unsigned int current_degree = degree - 1; current_degree > 0;
       --current_degree)
    {
      // switch to FE_Q when simplex supports electrostatic points
      // FE_Q<2> fe_q(current_degree);
      FE_Q<2> fe_q(QIterated<1>(QTrapezoid<1>(), current_degree));


      const auto  &points = fe_q.get_unit_support_points();
      unsigned int p      = 0;

      const double z     = (degree - current_degree) * z_equidistance;
      const double scale = current_degree * z_equidistance;

      // vertices are on lines
      for (unsigned int line = 0; line < fe_q.reference_cell().n_vertices();
           ++line)
        {
          support_points[start_lines[line]++] =
            lift_point(points[p++], scale, z);
        }
      // lines are on face
      for (unsigned int face = 0; face < fe_q.reference_cell().n_lines();
           ++face)
        {
          for (unsigned int n_dof = 0; n_dof < fe_q.n_dofs_per_line(); ++n_dof)
            support_points[start_faces[face]++] =
              lift_point(points[p++], scale, z);
        }
      // faces are on hex
      for (unsigned int hex = 0; hex < fe_q.n_dofs_per_quad(); ++hex)
        {
          support_points[start_hex++] = lift_point(points[p++], scale, z);
        }
    }
  Point<dim> tip;
  for (unsigned int d = 0; d < dim; ++d)
    {
      if (d == 2)
        tip[d] = 1.0;
      else
        tip[d] = 0.0;
    }
  support_points[4] = tip;

  return support_points;
}



template <int dim>
std::vector<Point<dim>>
equi_unit_support_points_fe_pyramid_p(const unsigned int degree)
{
  Assert(degree > 0, ExcNotImplemented());

  if constexpr (dim == 3)
    {
      std::vector<Point<dim>> unit_points;

      const auto reference_cell = ReferenceCells::Pyramid;


      const FE_Q<1> fe_line(QIterated<1>(QTrapezoid<1>(), degree));
      // const FE_SimplexP<2> fe_triangle(degree);
      const auto triangle_support_points =
        equidistant_support_points_fe_p<2>(degree);
      const FE_Q<2> fe_quad(QIterated<1>(QTrapezoid<1>(), degree));

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
            is_triangular_face ? (degree - 2) * (degree - 1) / 2 :
                                 fe_quad.n_dofs_per_quad();

          const std::vector<Point<2>> face_support_points =
            is_triangular_face ? triangle_support_points :
                                 fe_quad.get_unit_support_points();

          const unsigned int first_quad_index =
            is_triangular_face ? 3 + 3 * (degree - 1) :
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
                  const auto vertex =
                    reference_cell.vertex(reference_cell.face_to_cell_vertices(
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
get_blend_and_warp_support_points(const unsigned int degree)
{
  if constexpr (dim == 3)
    {
      const auto reference_cell = ReferenceCells::Pyramid;

      const unsigned int n_dofs_per_line = degree - 1;
      const unsigned int n_dofs_per_tri =
        degree == 1 ? 0 : (degree - 1) * (degree - 2) / 2;
      const unsigned int n_boundary_nodes = 3 * degree * degree + 2;

      // the idea of the algorithm is to construct support points compatible
      // with triangles and quads on the faces then take the boundary support
      // points and compute the displacement to the equidistant support points
      // on the faces in the last step interpolate the displacement to the
      // interior nodes

      // get the equidistant support points
      const auto equidistant_points =
        equi_unit_support_points_fe_pyramid_p<dim>(degree);

      // helper to evaluate the basis functions for the boundary elements
      // the basis is defined with equidistant support points
      const FE_Q<1> fe_equi(QIterated<1>(QTrapezoid<1>(), degree));

      // const FE_SimplexP<2> fe_triangle(degree);
      const auto support_points_triangle_equi =
        equidistant_support_points_fe_p<2>(degree);

      const auto poly_triangle_equi =
        ScalarLagrangePolynomialSimplex(degree, support_points_triangle_equi);

      const FE_Q<2> fe_quad_equi(QIterated<1>(QTrapezoid<1>(), degree));

      // basis function of boundary entities adopted from Chan and Warburton
      auto boundary_basis = [&](const unsigned int i, const Point<dim> &p) {
        // there are 3 * degree^2 + 2 shape functions on the vertices, edges and
        // faces
        Assert(i < n_boundary_nodes, ExcInternalError());

        double phi = 0.0;
        // first 5 are for the vertices, so just the linear shape functions
        if (i < reference_cell.n_vertices())
          {
            phi = reference_cell.d_linear_shape_function(p, i);
          }
        // now are the edge shape functions, there are degree - 1 dofs on each
        // edge
        else if (i < reference_cell.n_vertices() +
                       reference_cell.n_lines() * n_dofs_per_line)
          {
            // here the basis functions are just the linear basis functions
            // multiplied by the 1D line basis function get the line index
            const unsigned int line_index =
              (i - reference_cell.n_vertices()) / n_dofs_per_line;
            const unsigned int index_on_line =
              (i - reference_cell.n_vertices()) % n_dofs_per_line;

            // get the vertex  indices determining the line
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
        // on the quad face
        else if (i < reference_cell.n_vertices() +
                       reference_cell.n_lines() * n_dofs_per_line +
                       n_dofs_per_line * n_dofs_per_line)
          {
            // get the index on the face
            const unsigned int index_on_quad =
              i - (reference_cell.n_vertices() +
                   reference_cell.n_lines() * n_dofs_per_line);

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
        // on a triangular face
        else if (i < reference_cell.n_vertices() +
                       reference_cell.n_lines() * n_dofs_per_line +
                       n_dofs_per_line * n_dofs_per_line + 4 * n_dofs_per_tri)
          {
            // get the face index
            const unsigned int face_index =
              (i - (reference_cell.n_vertices() +
                    reference_cell.n_lines() * n_dofs_per_line +
                    n_dofs_per_line * n_dofs_per_line)) /
                n_dofs_per_tri +
              1;
            const unsigned int index_on_tri =
              (i - (reference_cell.n_vertices() +
                    reference_cell.n_lines() * n_dofs_per_line +
                    n_dofs_per_line * n_dofs_per_line)) %
              n_dofs_per_tri;

            const unsigned int v0 = reference_cell.face_to_cell_vertices(
              face_index, 0, numbers::default_geometric_orientation);
            const unsigned int v1 = reference_cell.face_to_cell_vertices(
              face_index, 1, numbers::default_geometric_orientation);
            const unsigned int v2 = reference_cell.face_to_cell_vertices(
              face_index, 2, numbers::default_geometric_orientation);

            const double l0 = reference_cell.d_linear_shape_function(p, v0);
            const double l1 = reference_cell.d_linear_shape_function(p, v1);
            const double l2 = reference_cell.d_linear_shape_function(p, v2);

            // from the index on the tri get the degrees of the jacobi
            // polynomials
            unsigned int jacobi_poly_degree_i = numbers::invalid_unsigned_int;
            unsigned int jacobi_poly_degree_j = numbers::invalid_unsigned_int;
            for (unsigned int a = 0, counter = 0; a < degree - 2; ++a)
              for (unsigned int b = 0; b < degree - a - 2; ++b, ++counter)
                if (index_on_tri == counter)
                  {
                    jacobi_poly_degree_i = a;
                    jacobi_poly_degree_j = b;
                  }

            Assert(jacobi_poly_degree_i != numbers::invalid_unsigned_int,
                   ExcInternalError());
            Assert(jacobi_poly_degree_j != numbers::invalid_unsigned_int,
                   ExcInternalError());

            // transforming from l0, l1, l2 to the local coordinates x,y on
            // the triangle the problem is l0 + l1 + l2 = 1 does not hold
            // here, as we had to sum up all pyramid shape functions to reach
            // unity so this normalizes the values
            const double x = 1. / 3. * (2.0 * l1 - l0 - l2 + 1.0);
            const double y = 1. / 3. * (2.0 * l2 - l1 - l0 + 1.0);

            const double x_contribution =
              Polynomials::jacobi_polynomial_homogenized_value<double>(
                jacobi_poly_degree_i, 0, 0, x, 1 - y);

            const double y_contribution =
              dealii::Polynomials::jacobi_polynomial_value<double>(
                jacobi_poly_degree_j, 2 * jacobi_poly_degree_i + 1, 0, y, true);

            phi = l0 * l1 * l2 * x_contribution * y_contribution;
          }
        else
          DEAL_II_ASSERT_UNREACHABLE();

        return phi;
      };

      // start by constructing the vertices, edges and faces
      // use GL points and warp and blend nodes
      const FE_Q<1> fe_line(degree);
      const FE_Q<2> fe_quad(degree);

      const auto support_points_triangle =
        electrostatic_support_points_fe_p<2>(degree);

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
            is_triangular_face ?
              n_dofs_per_tri : // fe_triangle.n_dofs_per_quad() :
              fe_quad.n_dofs_per_quad();

          const std::vector<Point<2>> &face_support_points =
            is_triangular_face ?
              support_points_triangle : // fe_triangle.get_unit_support_points()
                                        // :
                                        fe_quad.get_unit_support_points();

          const unsigned int first_quad_index =
            is_triangular_face ?
              3 + 3 * n_dofs_per_line : // fe_triangle.get_first_quad_index() :
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
                  const auto vertex =
                    reference_cell.vertex(reference_cell.face_to_cell_vertices(
                      f, v, numbers::default_geometric_orientation));

                  p += face_reference_cell.d_linear_shape_function(
                         face_support_points[first_quad_index + i], v) *
                       vertex;
                }
              gl_points.push_back(p);
            }
        }

      // needs to contain all nodes on the boundary
      Assert(gl_points.size() == n_boundary_nodes, ExcInternalError());

      // get the displacements between the electrostatic and the equidistant
      // points
      Vector<double> nodal_displacements_x(n_boundary_nodes);
      Vector<double> nodal_displacements_y(n_boundary_nodes);
      Vector<double> nodal_displacements_z(n_boundary_nodes);
      for (unsigned int i = 0; i < n_boundary_nodes; ++i)
        {
          const auto displacement_vector = gl_points[i] - equidistant_points[i];

          nodal_displacements_x[i] = displacement_vector[0];
          nodal_displacements_y[i] = displacement_vector[1];
          nodal_displacements_z[i] = displacement_vector[2];
        }

      // build the transformation matrix
      FullMatrix<double> VandermondeMatrix(n_boundary_nodes, n_boundary_nodes);
      for (unsigned int i = 0; i < VandermondeMatrix.m(); ++i)
        for (unsigned int j = 0; j < VandermondeMatrix.n(); ++j)
          VandermondeMatrix[i][j] = boundary_basis(j, equidistant_points[i]);

      // solve Vandermondematrix * displacements = nodal_displacement
      Vector<double> displacements_x(n_boundary_nodes);
      Vector<double> displacements_y(n_boundary_nodes);
      Vector<double> displacements_z(n_boundary_nodes);

      Householder<double> householder(VandermondeMatrix);
      householder.least_squares(displacements_x, nodal_displacements_x);
      householder.least_squares(displacements_y, nodal_displacements_y);
      householder.least_squares(displacements_z, nodal_displacements_z);

      // to get the interior nodes interpolate the difference between the
      // boundary nodes to the interor ones
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

template <int dim>
double
compute_VDM_condition_number(const unsigned int             degree,
                             const std::vector<Point<dim>> &support_points)
{
  ScalarLagrangePolynomialPyramid poly(degree,
                                       support_points.size(),
                                       support_points);

  LAPACKFullMatrix<double> VDM;
  VDM.copy_from(poly.vandermonde_matrix);

  VDM.compute_lu_factorization();
  double determinant = VDM.determinant();

  VDM.copy_from(poly.vandermonde_matrix);
  VDM.compute_svd();

  double min = std::numeric_limits<double>::max();
  double max = std::numeric_limits<double>::lowest();

  for (unsigned int i = 0; i < support_points.size(); ++i)
    {
      min = std::min(min, VDM.singular_value(i));
      max = std::max(max, VDM.singular_value(i));
    }

  // std::cout << "eigenvalue (min/max), determinant " << min << " " << max << "
  std::cout << "determinant " << determinant << std::endl;

  const double condition_number = max / min;

  return condition_number;
}

template <int dim>
void print_points(const unsigned int degree)
{
  const auto reference_cell = ReferenceCells::Pyramid;

  std::cout << "Degree " << degree << std::endl;
  const auto points_1 = get_blend_and_warp_support_points<dim>(degree);
  std::cout << std::endl;
  std::cout << "Blend and Warp support points" << std::endl;

  bool all_inside = true;
  for (unsigned int i = 0; i < points_1.size(); ++i)
    {
      if (reference_cell.contains_point(points_1[i], 1e-12))
        {
          // std::cout << points_1[i] << std::endl;
        }
      else
        {
          all_inside = false;
          std::cout << "Point " << points_1[i] << " is outside" << std::endl;
        }
    }
  std::cout << std::endl;
  if (all_inside)
    std::cout << "all points are inside" << std::endl;
  else
    std::cout << "not all point are inside" << std::endl;
  std::cout << std::endl;
}

std::vector<Point<3>> reference_points_p4 = {
  Point<3>(-1, -1, 0),
  Point<3>(-1, -0.6546536707079773, 0),
  Point<3>(-1, -5.551115123125783e-17, 0),
  Point<3>(-1, 0.6546536707079772, 0),
  Point<3>(-1, 1, 0),
  Point<3>(-0.6546536707079773, -1, 0),
  Point<3>(-0.6546536707079771, -0.6546536707079773, 0),
  Point<3>(-0.6546536707079771, 2.775557561562891e-17, 0),
  Point<3>(-0.6546536707079771, 0.6546536707079772, 0),
  Point<3>(-0.6546536707079772, 1, 0),
  Point<3>(-5.551115123125783e-17, -1, 0),
  Point<3>(-5.551115123125783e-17, -0.6546536707079771, 0),
  Point<3>(-3.165870341226791e-17, -3.165870338194613e-17, 0),
  Point<3>(-2.775557561562891e-17, 0.6546536707079773, 0),
  Point<3>(-5.551115123125783e-17, 0.9999999999999999, 0),
  Point<3>(0.6546536707079773, -1, 0),
  Point<3>(0.6546536707079771, -0.6546536707079771, 0),
  Point<3>(0.6546536707079774, 2.775557561562891e-17, 0),
  Point<3>(0.6546536707079771, 0.6546536707079771, 0),
  Point<3>(0.6546536707079772, 0.9999999999999999, 0),
  Point<3>(1, -1, 0),
  Point<3>(1, -0.6546536707079773, 0),
  Point<3>(1, -5.551115123125783e-17, 0),
  Point<3>(1, 0.6546536707079772, 0),
  Point<3>(1, 1, 0),
  Point<3>(-0.8273268353539887, -0.8273268353539888, 0.1726731646460113),
  Point<3>(-0.7757917860454968, -0.3273753581364906, 0.2242082139545032),
  Point<3>(-0.7757917860454968, 0.32737535813649, 0.2242082139545032),
  Point<3>(-0.8273268353539887, 0.8273268353539885, 0.1726731646460113),
  Point<3>(-0.3273753581364905, -0.775791786045497, 0.2242082139545032),
  Point<3>(-0.2978566582945021, -0.2978566582945021, 0.2528675511466786),
  Point<3>(-0.297856658294502, 0.297856658294502, 0.2528675511466785),
  Point<3>(-0.3273753581364902, 0.7757917860454967, 0.2242082139545032),
  Point<3>(0.3273753581364903, -0.7757917860454968, 0.2242082139545032),
  Point<3>(0.297856658294502, -0.2978566582945021, 0.2528675511466785),
  Point<3>(0.297856658294502, 0.297856658294502, 0.2528675511466786),
  Point<3>(0.3273753581364902, 0.7757917860454968, 0.2242082139545032),
  Point<3>(0.8273268353539889, -0.8273268353539887, 0.1726731646460113),
  Point<3>(0.7757917860454968, -0.3273753581364905, 0.2242082139545032),
  Point<3>(0.7757917860454966, 0.3273753581364902, 0.2242082139545032),
  Point<3>(0.8273268353539885, 0.8273268353539887, 0.1726731646460113),
  Point<3>(-0.5, -0.5, 0.4999999999999999),
  Point<3>(-0.4484164279090065, 2.775557561562891e-17, 0.5515835720909934),
  Point<3>(-0.4999999999999999, 0.5, 0.5),
  Point<3>(2.775557561562891e-17, -0.4484164279090065, 0.5515835720909934),
  Point<3>(-6.267885512499976e-18, 4.343337876593664e-17, 0.5773753581376365),
  Point<3>(-8.326672684688674e-17, 0.4484164279090064, 0.5515835720909934),
  Point<3>(0.5, -0.5, 0.5),
  Point<3>(0.4484164279090064, 2.775557561562891e-17, 0.5515835720909936),
  Point<3>(0.5000000000000001, 0.5, 0.4999999999999999),
  Point<3>(-0.1726731646460114, -0.1726731646460114, 0.8273268353539885),
  Point<3>(-0.1726731646460113, 0.1726731646460115, 0.8273268353539887),
  Point<3>(0.1726731646460113, -0.1726731646460114, 0.8273268353539887),
  Point<3>(0.1726731646460114, 0.1726731646460113, 0.8273268353539885),
  Point<3>(-5.551115123125783e-17, -5.551115123125783e-17, 1)};


template <int dim>
void compare_points(const unsigned int degree)
{
  if (degree == 4)
    {
      double       min_distance       = 1000000.0;
      unsigned int min_distance_index = 0;

      const auto points_reference = reference_points_p4;
      const auto points_blend_and_warp =
        get_blend_and_warp_support_points<dim>(degree);

      for (unsigned int i = 0; i < points_blend_and_warp.size(); ++i)
        {
          bool found_point = false;
          for (unsigned int j = 0; j < points_reference.size(); ++j)
            {
              const double distance =
                points_reference[j].distance(points_blend_and_warp[i]);
              if (distance < 1e-6)
                {
                  found_point = true;
                }
              else
                {
                  if (distance < min_distance)
                    {
                      min_distance       = distance;
                      min_distance_index = j;
                    }
                }
            }
          if (found_point == false)
            {
              std::cout << "Did not find point " << i << " "
                        << points_blend_and_warp[i] << std::endl;
              std::cout << "nearest points was "
                        << points_reference[min_distance_index]
                        << " at distance " << min_distance << std::endl;
            }
        }
    }
  else
    std::cout << "no data at degree " << degree << std::endl;
}

int main()
{
  constexpr int dim = 3;

  if (false)
    for (unsigned int degree = 3; degree < 5; ++degree)
      {
        std::cout << "Degree " << degree << " with " << 3 * degree * degree + 2
                  << " boundary nodes" << std::endl;
        const auto p = equi_unit_support_points_fe_pyramid_p<3>(degree);
        const auto points_eqi = get_support_points_fe_pyramid_p<dim>(degree);
        std::cout << "Size " << p.size() << " " << points_eqi.size()
                  << std::endl;

        for (unsigned int i = 0; i < points_eqi.size(); ++i)
          if (p[i].distance(points_eqi[i]) > 1e-12)
            {
              std::cout << "Point new " << p[i] << std::endl;
              std::cout << "Point old " << points_eqi[i] << std::endl;
              std::cout << "Diff " << p[i] - points_eqi[i] << " at index " << i
                        << std::endl;
            }
        std::cout << std::endl;
      }
  // return 1;

  for (unsigned int degree = 4; degree < 9; ++degree)
    {
      compare_points<3>(degree);
      // print_points<3>(degree);

      const auto points_blend_and_warp =
        get_blend_and_warp_support_points<dim>(degree);
      const auto points_eqi =
        equi_unit_support_points_fe_pyramid_p<dim>(degree);

      // if(true)
      // for(unsigned int i = 0; i < points_blend_and_warp.size(); ++i)
      //  std::cout <<  points_eqi[i] << std::endl; //points_blend_and_warp[i] -
      // return 0;

      const double condition_number_blend_and_warp =
        compute_VDM_condition_number(degree, points_blend_and_warp);
      const double condition_number_equi =
        compute_VDM_condition_number(degree, points_eqi);

      std::cout << "Degree " << degree << " gives condition numbers of "
                << condition_number_blend_and_warp
                << " for blend and warp points and " << condition_number_equi
                << " for equidistant points" << std::endl;
    }

  return 0;
}
