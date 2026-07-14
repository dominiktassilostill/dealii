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

  const double alpha = degree <= alpha_opt.size() ? alpha_opt[degree - 1] : 1.0;

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
      // const std::array<double, 15> alpha_opt = {{0.0000,
      //                                            0.0000,
      //                                            1.4152,
      //                                            0.1001,
      //                                            0.2751,
      //                                            0.9800,
      //                                            1.0999,
      //                                            1.2832,
      //                                            1.3648,
      //                                            1.4773,
      //                                            1.4959,
      //                                            1.5743,
      //                                            1.5770,
      //                                            1.6223,
      //                                            1.6258}};

      // const double alpha =
      //   degree <= alpha_opt.size() ? alpha_opt[degree - 1] : 5.0 / 3.0;

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

            const double
              x_contribution = // std::pow(0.5, jacobi_poly_degree_i) *
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
  // std::cout << "determinant " << determinant << std::endl;

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

std::vector<Point<3>> reference_points_p3 = {
  Point<3>(-1, -1, 0),
  Point<3>(-1, -0.4472135954999579, 0),
  Point<3>(-1, 0.4472135954999579, 0),
  Point<3>(-1, 1, 0),
  Point<3>(-0.4472135954999579, -1, 0),
  Point<3>(-0.4472135954999578, -0.447213595499958, 0),
  Point<3>(-0.4472135954999579, 0.447213595499958, 0),
  Point<3>(-0.4472135954999579, 1, 0),
  Point<3>(0.4472135954999579, -1, 0),
  Point<3>(0.4472135954999578, -0.4472135954999578, 0),
  Point<3>(0.447213595499958, 0.4472135954999579, 0),
  Point<3>(0.4472135954999579, 1, 0),
  Point<3>(1, -1, 0),
  Point<3>(1, -0.4472135954999579, 0),
  Point<3>(1, 0.4472135954999579, 0),
  Point<3>(1, 1, 0),
  Point<3>(-0.7236067977499792, -0.7236067977499789, 0.276393202250021),
  Point<3>(-0.6666666666666665, 0, 0.3333333333333333),
  Point<3>(-0.7236067977499789, 0.723606797749979, 0.2763932022500209),
  Point<3>(-5.551115123125783e-17, -0.6666666666666666, 0.3333333333333334),
  Point<3>(7.407421373908719e-17, 2.571090709245841e-17, 0.3618033988749895),
  Point<3>(1.110223024625157e-16, 0.6666666666666667, 0.3333333333333333),
  Point<3>(0.7236067977499789, -0.7236067977499792, 0.2763932022500209),
  Point<3>(0.6666666666666667, 0, 0.3333333333333333),
  Point<3>(0.723606797749979, 0.7236067977499789, 0.2763932022500209),
  Point<3>(-0.2763932022500211, -0.2763932022500212, 0.7236067977499789),
  Point<3>(-0.2763932022500211, 0.2763932022500213, 0.7236067977499789),
  Point<3>(0.2763932022500212, -0.2763932022500213, 0.7236067977499789),
  Point<3>(0.2763932022500212, 0.2763932022500211, 0.7236067977499789),
  Point<3>(-5.551115123125783e-17, -5.551115123125783e-17, 1)};
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
std::vector<Point<3>> reference_points_p5 = {
  Point<3>(-0.9999999999999999, -0.9999999999999999, 0),
  Point<3>(-1, -0.765055323929465, 0),
  Point<3>(-1, -0.2852315164806453, 0),
  Point<3>(-1, 0.2852315164806452, 0),
  Point<3>(-1, 0.7650553239294646, 0),
  Point<3>(-1, 1, 0),
  Point<3>(-0.7650553239294648, -1, 0),
  Point<3>(-0.7650553239294646, -0.7650553239294651, 0),
  Point<3>(-0.7650553239294652, -0.2852315164806452, 0),
  Point<3>(-0.7650553239294646, 0.2852315164806449, 0),
  Point<3>(-0.7650553239294648, 0.765055323929465, 0),
  Point<3>(-0.7650553239294648, 1, 0),
  Point<3>(-0.2852315164806453, -1, 0),
  Point<3>(-0.2852315164806453, -0.765055323929465, 0),
  Point<3>(-0.2852315164806452, -0.2852315164806452, 0),
  Point<3>(-0.2852315164806451, 0.2852315164806449, 0),
  Point<3>(-0.2852315164806453, 0.7650553239294645, 0),
  Point<3>(-0.2852315164806451, 1, 0),
  Point<3>(0.2852315164806453, -1, 0),
  Point<3>(0.2852315164806452, -0.7650553239294648, 0),
  Point<3>(0.285231516480645, -0.2852315164806452, 0),
  Point<3>(0.2852315164806451, 0.2852315164806452, 0),
  Point<3>(0.285231516480645, 0.7650553239294646, 0),
  Point<3>(0.285231516480645, 1, 0),
  Point<3>(0.7650553239294647, -1, 0),
  Point<3>(0.7650553239294647, -0.7650553239294647, 0),
  Point<3>(0.7650553239294646, -0.2852315164806452, 0),
  Point<3>(0.7650553239294646, 0.285231516480645, 0),
  Point<3>(0.7650553239294647, 0.7650553239294645, 0),
  Point<3>(0.7650553239294647, 1, 0),
  Point<3>(1, -1, 0),
  Point<3>(1, -0.7650553239294648, 0),
  Point<3>(1, -0.2852315164806452, 0),
  Point<3>(1, 0.2852315164806452, 0),
  Point<3>(1, 0.7650553239294646, 0),
  Point<3>(0.9999999999999999, 0.9999999999999999, 0),
  Point<3>(-0.8825276619647323, -0.8825276619647323, 0.1174723380352677),
  Point<3>(-0.8442717323249693, -0.5328151969749098, 0.1557282676750302),
  Point<3>(-0.8342478063197913, -5.551115123125783e-17, 0.1657521936802088),
  Point<3>(-0.8442717323249693, 0.5328151969749095, 0.1557282676750302),
  Point<3>(-0.8825276619647323, 0.8825276619647324, 0.1174723380352677),
  Point<3>(-0.5328151969749093, -0.8442717323249695, 0.1557282676750302),
  Point<3>(-0.4886413979740196, -0.4886413979740198, 0.1834443153684131),
  Point<3>(-0.4762638457110937, -5.551115123125783e-17, 0.1917860960269541),
  Point<3>(-0.4886413979740193, 0.4886413979740195, 0.1834443153684132),
  Point<3>(-0.5328151969749094, 0.8442717323249697, 0.1557282676750302),
  Point<3>(5.551115123125783e-17, -0.8342478063197916, 0.1657521936802088),
  Point<3>(-1.387778780781446e-16, -0.4762638457110939, 0.191786096026954),
  Point<3>(-1.894032310888269e-17, -5.54632317359226e-17, 0.2000000000018478),
  Point<3>(0, 0.4762638457110936, 0.1917860960269541),
  Point<3>(5.551115123125783e-17, 0.8342478063197913, 0.1657521936802088),
  Point<3>(0.5328151969749094, -0.8442717323249695, 0.1557282676750302),
  Point<3>(0.4886413979740195, -0.4886413979740198, 0.1834443153684131),
  Point<3>(0.4762638457110937, 2.775557561562891e-17, 0.1917860960269541),
  Point<3>(0.4886413979740197, 0.4886413979740196, 0.1834443153684131),
  Point<3>(0.5328151969749095, 0.8442717323249695, 0.1557282676750302),
  Point<3>(0.8825276619647325, -0.8825276619647324, 0.1174723380352677),
  Point<3>(0.8442717323249697, -0.5328151969749093, 0.1557282676750301),
  Point<3>(0.8342478063197916, 0, 0.1657521936802088),
  Point<3>(0.8442717323249695, 0.5328151969749098, 0.1557282676750301),
  Point<3>(0.8825276619647322, 0.8825276619647323, 0.1174723380352677),
  Point<3>(-0.6426157582403226, -0.6426157582403225, 0.3573842417596774),
  Point<3>(-0.5828760968401043, -0.2513717094796867, 0.4171239031598956),
  Point<3>(-0.5828760968401041, 0.2513717094796866, 0.4171239031598958),
  Point<3>(-0.6426157582403225, 0.6426157582403227, 0.3573842417596773),
  Point<3>(-0.2513717094796868, -0.5828760968401041, 0.4171239031598957),
  Point<3>(-0.2126857383417056, -0.2126857383417057, 0.4505406480785156),
  Point<3>(-0.2126857383417057, 0.2126857383417057, 0.4505406480785158),
  Point<3>(-0.2513717094796866, 0.5828760968401043, 0.4171239031598958),
  Point<3>(0.2513717094796866, -0.5828760968401042, 0.4171239031598957),
  Point<3>(0.2126857383417056, -0.2126857383417058, 0.4505406480785157),
  Point<3>(0.2126857383417057, 0.2126857383417057, 0.4505406480785156),
  Point<3>(0.2513717094796869, 0.5828760968401042, 0.4171239031598956),
  Point<3>(0.6426157582403226, -0.6426157582403226, 0.3573842417596774),
  Point<3>(0.5828760968401042, -0.2513717094796868, 0.4171239031598957),
  Point<3>(0.5828760968401039, 0.2513717094796866, 0.4171239031598957),
  Point<3>(0.6426157582403225, 0.6426157582403226, 0.3573842417596774),
  Point<3>(-0.3573842417596774, -0.3573842417596775, 0.6426157582403225),
  Point<3>(-0.3114565353500605, -2.775557561562891e-17, 0.6885434646499398),
  Point<3>(-0.3573842417596775, 0.3573842417596775, 0.6426157582403225),
  Point<3>(0, -0.3114565353500603, 0.6885434646499398),
  Point<3>(-9.00225915927224e-18, 3.384940449928192e-17, 0.708805828288983),
  Point<3>(8.326672684688674e-17, 0.3114565353500607, 0.6885434646499397),
  Point<3>(0.3573842417596775, -0.3573842417596774, 0.6426157582403225),
  Point<3>(0.3114565353500605, -5.551115123125783e-17, 0.6885434646499395),
  Point<3>(0.3573842417596775, 0.3573842417596775, 0.6426157582403225),
  Point<3>(-0.1174723380352676, -0.1174723380352677, 0.8825276619647323),
  Point<3>(-0.1174723380352675, 0.1174723380352675, 0.8825276619647325),
  Point<3>(0.1174723380352675, -0.1174723380352675, 0.8825276619647324),
  Point<3>(0.1174723380352675, 0.1174723380352676, 0.8825276619647325),
  Point<3>(-5.551115123125783e-17, -5.551115123125783e-17, 1)};
std::vector<Point<3>> reference_points_p6 = {
  Point<3>(-1, -1, 0),
  Point<3>(-1, -0.8302238962785674, 0),
  Point<3>(-1, -0.4688487934707141, 0),
  Point<3>(-0.9999999999999999, 4.402416801590545e-17, 0),
  Point<3>(-1, 0.4688487934707142, 0),
  Point<3>(-1, 0.830223896278567, 0),
  Point<3>(-1, 1, 0),
  Point<3>(-0.8302238962785673, -1, 0),
  Point<3>(-0.8302238962785672, -0.830223896278567, 0),
  Point<3>(-0.8302238962785674, -0.4688487934707142, 0),
  Point<3>(-0.8302238962785677, 4.402416801463692e-17, 0),
  Point<3>(-0.8302238962785673, 0.4688487934707142, 0),
  Point<3>(-0.8302238962785673, 0.8302238962785667, 0),
  Point<3>(-0.8302238962785673, 1, 0),
  Point<3>(-0.4688487934707142, -1, 0),
  Point<3>(-0.4688487934707142, -0.8302238962785672, 0),
  Point<3>(-0.4688487934707142, -0.4688487934707142, 0),
  Point<3>(-0.4688487934707141, 4.402416801957358e-17, 0),
  Point<3>(-0.4688487934707138, 0.4688487934707142, 0),
  Point<3>(-0.4688487934707141, 0.8302238962785669, 0),
  Point<3>(-0.4688487934707142, 0.9999999999999999, 0),
  Point<3>(4.402416793915466e-17, -1, 0),
  Point<3>(4.402416794702978e-17, -0.8302238962785675, 0),
  Point<3>(4.402416795874236e-17, -0.4688487934707142, 0),
  Point<3>(4.402416796261431e-17, 4.402416802023405e-17, 0),
  Point<3>(4.402416797551042e-17, 0.4688487934707144, 0),
  Point<3>(4.402416798798356e-17, 0.8302238962785669, 0),
  Point<3>(4.40241679923761e-17, 1, 0),
  Point<3>(0.4688487934707143, -1, 0),
  Point<3>(0.4688487934707143, -0.8302238962785672, 0),
  Point<3>(0.4688487934707142, -0.4688487934707141, 0),
  Point<3>(0.4688487934707144, 4.402416802311761e-17, 0),
  Point<3>(0.4688487934707143, 0.4688487934707142, 0),
  Point<3>(0.4688487934707142, 0.830223896278567, 0),
  Point<3>(0.4688487934707142, 0.9999999999999998, 0),
  Point<3>(0.8302238962785669, -1, 0),
  Point<3>(0.8302238962785665, -0.8302238962785672, 0),
  Point<3>(0.8302238962785672, -0.4688487934707143, 0),
  Point<3>(0.830223896278567, 1.626859240656768e-17, 0),
  Point<3>(0.830223896278567, 0.4688487934707143, 0),
  Point<3>(0.8302238962785669, 0.830223896278567, 0),
  Point<3>(0.8302238962785669, 1, 0),
  Point<3>(1, -0.9999999999999999, 0),
  Point<3>(1, -0.8302238962785675, 0),
  Point<3>(1, -0.4688487934707141, 0),
  Point<3>(1, 4.402416802391735e-17, 0),
  Point<3>(1, 0.4688487934707142, 0),
  Point<3>(1, 0.830223896278567, 0),
  Point<3>(1, 0.9999999999999999, 0),
  Point<3>(-0.9151119481392833, -0.9151119481392833, 0.0848880518607166),
  Point<3>(-0.8868114611011423, -0.6604343833034265, 0.1131885388988578),
  Point<3>(-0.8793655429845175, -0.2399326648200861, 0.1206344570154826),
  Point<3>(-0.8793655429845175, 0.2399326648200862, 0.1206344570154826),
  Point<3>(-0.8868114611011422, 0.6604343833034265, 0.1131885388988579),
  Point<3>(-0.9151119481392833, 0.9151119481392835, 0.08488805186071662),
  Point<3>(-0.6604343833034267, -0.8868114611011424, 0.1131885388988578),
  Point<3>(-0.6172342580787266, -0.6172342580787268, 0.1365681470723158),
  Point<3>(-0.6029049843953312, -0.2212259467297556, 0.1455335427375846),
  Point<3>(-0.6029049843953315, 0.2212259467297556, 0.1455335427375846),
  Point<3>(-0.6172342580787264, 0.6172342580787261, 0.1365681470723156),
  Point<3>(-0.6604343833034264, 0.8868114611011423, 0.1131885388988579),
  Point<3>(-0.2399326648200862, -0.8793655429845179, 0.1206344570154826),
  Point<3>(-0.2212259467297555, -0.6029049843953311, 0.1455335427375846),
  Point<3>(-0.2150649795408018, -0.2150649795408018, 0.156711160470383),
  Point<3>(-0.2150649795408018, 0.2150649795408018, 0.156711160470383),
  Point<3>(-0.2212259467297555, 0.6029049843953315, 0.1455335427375846),
  Point<3>(-0.2399326648200862, 0.8793655429845176, 0.1206344570154826),
  Point<3>(0.2399326648200862, -0.8793655429845174, 0.1206344570154826),
  Point<3>(0.2212259467297555, -0.602904984395331, 0.1455335427375846),
  Point<3>(0.2150649795408018, -0.2150649795408018, 0.1567111604703829),
  Point<3>(0.2150649795408016, 0.2150649795408018, 0.156711160470383),
  Point<3>(0.2212259467297556, 0.6029049843953315, 0.1455335427375847),
  Point<3>(0.2399326648200861, 0.8793655429845174, 0.1206344570154826),
  Point<3>(0.6604343833034266, -0.8868114611011421, 0.113188538898858),
  Point<3>(0.617234258078726, -0.6172342580787264, 0.1365681470723156),
  Point<3>(0.6029049843953312, -0.2212259467297556, 0.1455335427375846),
  Point<3>(0.6029049843953314, 0.2212259467297555, 0.1455335427375847),
  Point<3>(0.6172342580787264, 0.6172342580787263, 0.1365681470723157),
  Point<3>(0.660434383303427, 0.8868114611011424, 0.1131885388988578),
  Point<3>(0.9151119481392834, -0.9151119481392833, 0.08488805186071656),
  Point<3>(0.8868114611011423, -0.6604343833034269, 0.1131885388988578),
  Point<3>(0.8793655429845181, -0.2399326648200862, 0.1206344570154825),
  Point<3>(0.8793655429845173, 0.2399326648200862, 0.1206344570154826),
  Point<3>(0.8868114611011422, 0.6604343833034267, 0.113188538898858),
  Point<3>(0.9151119481392833, 0.9151119481392834, 0.08488805186071664),
  Point<3>(-0.7344243967353572, -0.7344243967353571, 0.2655756032646429),
  Point<3>(-0.6802835609177846, -0.4390146468868194, 0.3197164390822156),
  Point<3>(-0.6666666666666669, -1.231665479505076e-25, 0.3333333333333334),
  Point<3>(-0.6802835609177845, 0.4390146468868193, 0.3197164390822156),
  Point<3>(-0.7344243967353572, 0.7344243967353572, 0.2655756032646427),
  Point<3>(-0.4390146468868192, -0.6802835609177845, 0.3197164390822158),
  Point<3>(-0.3815389324385921, -0.3815389324385922, 0.3549719515591285),
  Point<3>(-0.3655089425397798, 1.978125535833819e-17, 0.3652740073392002),
  Point<3>(-0.3815389324385923, 0.3815389324385925, 0.3549719515591286),
  Point<3>(-0.4390146468868192, 0.6802835609177849, 0.3197164390822156),
  Point<3>(-2.391283765337351e-17, -0.6666666666666664, 0.3333333333333332),
  Point<3>(-1.454862487362876e-17, -0.3655089425397797, 0.3652740073392002),
  Point<3>(-3.492057981239544e-17, 2.635976826998651e-17, 0.3751766441628566),
  Point<3>(4.447413695080421e-17, 0.3655089425397797, 0.3652740073392003),
  Point<3>(5.166841296117172e-17, 0.6666666666666665, 0.3333333333333334),
  Point<3>(0.4390146468868192, -0.6802835609177845, 0.3197164390822156),
  Point<3>(0.3815389324385924, -0.3815389324385925, 0.3549719515591287),
  Point<3>(0.3655089425397796, 1.69811430760623e-17, 0.3652740073392002),
  Point<3>(0.3815389324385924, 0.3815389324385923, 0.3549719515591284),
  Point<3>(0.439014646886819, 0.6802835609177847, 0.3197164390822156),
  Point<3>(0.7344243967353571, -0.7344243967353574, 0.2655756032646428),
  Point<3>(0.6802835609177849, -0.4390146468868192, 0.3197164390822156),
  Point<3>(0.6666666666666666, 1.387778769277438e-17, 0.3333333333333333),
  Point<3>(0.6802835609177846, 0.4390146468868192, 0.3197164390822157),
  Point<3>(0.7344243967353573, 0.7344243967353571, 0.2655756032646428),
  Point<3>(-0.5, -0.5, 0.5000000000000001),
  Point<3>(-0.4403508960976985, -0.1990819820667331, 0.5596491039023017),
  Point<3>(-0.4403508960976981, 0.1990819820667334, 0.5596491039023019),
  Point<3>(-0.5000000000000001, 0.4999999999999999, 0.5),
  Point<3>(-0.1990819820667332, -0.4403508960976983, 0.5596491039023017),
  Point<3>(-0.1601586431510112, -0.1601586431510112, 0.5895985823200259),
  Point<3>(-0.1601586431510113, 0.1601586431510114, 0.5895985823200258),
  Point<3>(-0.1990819820667333, 0.4403508960976984, 0.5596491039023017),
  Point<3>(0.1990819820667334, -0.4403508960976983, 0.559649103902302),
  Point<3>(0.1601586431510114, -0.1601586431510113, 0.5895985823200257),
  Point<3>(0.1601586431510114, 0.1601586431510114, 0.5895985823200259),
  Point<3>(0.1990819820667332, 0.4403508960976983, 0.5596491039023022),
  Point<3>(0.4999999999999999, -0.4999999999999999, 0.4999999999999999),
  Point<3>(0.4403508960976985, -0.1990819820667331, 0.5596491039023022),
  Point<3>(0.4403508960976982, 0.1990819820667333, 0.5596491039023017),
  Point<3>(0.5, 0.4999999999999998, 0.5),
  Point<3>(-0.2655756032646429, -0.265575603264643, 0.7344243967353571),
  Point<3>(-0.2263770777977158, -1.327575109456281e-16, 0.7736229222022839),
  Point<3>(-0.2655756032646427, 0.2655756032646432, 0.7344243967353571),
  Point<3>(-8.333789217293923e-17, -0.2263770777977159, 0.7736229222022839),
  Point<3>(-2.919272286031984e-17, -1.245323103545052e-16, 0.7889435783466515),
  Point<3>(7.216768552112533e-17, 0.2263770777977158, 0.7736229222022838),
  Point<3>(0.2655756032646431, -0.2655756032646432, 0.7344243967353571),
  Point<3>(0.2263770777977158, -1.272957718321147e-16, 0.7736229222022839),
  Point<3>(0.2655756032646432, 0.2655756032646429, 0.7344243967353569),
  Point<3>(-0.08488805186071639, -0.08488805186071638, 0.9151119481392832),
  Point<3>(-0.08488805186071631, 0.08488805186071637, 0.9151119481392832),
  Point<3>(0.0848880518607163, -0.08488805186071638, 0.9151119481392835),
  Point<3>(0.08488805186071638, 0.08488805186071628, 0.9151119481392834),
  Point<3>(-5.551115123125783e-17, -5.551115123125783e-17, 1)};
std::vector<Point<3>> reference_points_p7 = {
  Point<3>(-1, -1, 0),
  Point<3>(-1, -0.8717401485096069, 0),
  Point<3>(-1, -0.5917001814331424, 0),
  Point<3>(-1, -0.2092992179024788, 0),
  Point<3>(-1, 0.2092992179024791, 0),
  Point<3>(-1, 0.5917001814331421, 0),
  Point<3>(-0.9999999999999998, 0.8717401485096066, 0),
  Point<3>(-1, 1, 0),
  Point<3>(-0.8717401485096066, -1, 0),
  Point<3>(-0.8717401485096069, -0.8717401485096066, 0),
  Point<3>(-0.8717401485096066, -0.5917001814331422, 0),
  Point<3>(-0.8717401485096065, -0.2092992179024785, 0),
  Point<3>(-0.8717401485096065, 0.2092992179024793, 0),
  Point<3>(-0.8717401485096069, 0.5917001814331422, 0),
  Point<3>(-0.871740148509607, 0.8717401485096068, 0),
  Point<3>(-0.8717401485096066, 1, 0),
  Point<3>(-0.5917001814331424, -1, 0),
  Point<3>(-0.5917001814331425, -0.8717401485096065, 0),
  Point<3>(-0.5917001814331422, -0.5917001814331423, 0),
  Point<3>(-0.5917001814331422, -0.209299217902479, 0),
  Point<3>(-0.5917001814331423, 0.209299217902479, 0),
  Point<3>(-0.5917001814331422, 0.591700181433142, 0),
  Point<3>(-0.5917001814331422, 0.8717401485096066, 0),
  Point<3>(-0.5917001814331423, 1, 0),
  Point<3>(-0.2092992179024787, -0.9999999999999999, 0),
  Point<3>(-0.2092992179024788, -0.8717401485096066, 0),
  Point<3>(-0.2092992179024788, -0.5917001814331422, 0),
  Point<3>(-0.2092992179024788, -0.2092992179024787, 0),
  Point<3>(-0.2092992179024787, 0.2092992179024791, 0),
  Point<3>(-0.2092992179024789, 0.5917001814331422, 0),
  Point<3>(-0.2092992179024788, 0.8717401485096065, 0),
  Point<3>(-0.2092992179024789, 1, 0),
  Point<3>(0.2092992179024792, -1, 0),
  Point<3>(0.209299217902479, -0.8717401485096063, 0),
  Point<3>(0.209299217902479, -0.5917001814331424, 0),
  Point<3>(0.2092992179024793, -0.2092992179024788, 0),
  Point<3>(0.2092992179024791, 0.2092992179024791, 0),
  Point<3>(0.2092992179024792, 0.5917001814331421, 0),
  Point<3>(0.2092992179024792, 0.8717401485096067, 0),
  Point<3>(0.2092992179024791, 1, 0),
  Point<3>(0.591700181433142, -1, 0),
  Point<3>(0.591700181433142, -0.8717401485096066, 0),
  Point<3>(0.5917001814331423, -0.5917001814331421, 0),
  Point<3>(0.5917001814331421, -0.2092992179024787, 0),
  Point<3>(0.5917001814331422, 0.209299217902479, 0),
  Point<3>(0.5917001814331417, 0.5917001814331421, 0),
  Point<3>(0.5917001814331418, 0.8717401485096068, 0),
  Point<3>(0.5917001814331422, 1, 0),
  Point<3>(0.8717401485096066, -1, 0),
  Point<3>(0.8717401485096065, -0.8717401485096062, 0),
  Point<3>(0.8717401485096069, -0.5917001814331423, 0),
  Point<3>(0.8717401485096065, -0.2092992179024788, 0),
  Point<3>(0.8717401485096066, 0.2092992179024791, 0),
  Point<3>(0.8717401485096065, 0.591700181433142, 0),
  Point<3>(0.8717401485096066, 0.8717401485096065, 0),
  Point<3>(0.8717401485096066, 1, 0),
  Point<3>(1, -1, 0),
  Point<3>(0.9999999999999998, -0.8717401485096066, 0),
  Point<3>(1, -0.5917001814331423, 0),
  Point<3>(1, -0.2092992179024789, 0),
  Point<3>(1, 0.2092992179024791, 0),
  Point<3>(1, 0.5917001814331422, 0),
  Point<3>(1, 0.8717401485096068, 0),
  Point<3>(1, 1, 0),
  Point<3>(-0.9358700742548036, -0.9358700742548035, 0.06412992574519666),
  Point<3>(-0.9126419083510326, -0.7379257250530984, 0.08735809164896696),
  Point<3>(-0.9033496527273077, -0.4055610230884225, 0.0966503472726928),
  Point<3>(-0.901041703665217, -1.110223024625157e-16, 0.09895829633478326),
  Point<3>(-0.9033496527273071, 0.4055610230884228, 0.09665034727269284),
  Point<3>(-0.9126419083510329, 0.7379257250530995, 0.08735809164896713),
  Point<3>(-0.935870074254803, 0.9358700742548031, 0.06412992574519673),
  Point<3>(-0.7379257250530979, -0.9126419083510332, 0.08735809164896696),
  Point<3>(-0.6993443102316483, -0.6993443102316488, 0.1062645981600723),
  Point<3>(-0.6829884106255608, -0.3813913955667642, 0.1154087878461704),
  Point<3>(-0.6788556804717862, 5.551115123125783e-17, 0.1180853065831378),
  Point<3>(-0.6829884106255608, 0.3813913955667638, 0.1154087878461705),
  Point<3>(-0.6993443102316483, 0.6993443102316487, 0.1062645981600723),
  Point<3>(-0.7379257250530991, 0.9126419083510328, 0.08735809164896713),
  Point<3>(-0.4055610230884219, -0.9033496527273075, 0.09665034727269277),
  Point<3>(-0.3813913955667642, -0.6829884106255613, 0.1154087878461704),
  Point<3>(-0.3722727723025122, -0.3722727723025125, 0.1255890328023157),
  Point<3>(-0.3703069113454807, 1.387778780781446e-16, 0.1288018969492089),
  Point<3>(-0.3722727723025121, 0.3722727723025122, 0.1255890328023157),
  Point<3>(-0.3813913955667637, 0.682988410625561, 0.1154087878461706),
  Point<3>(-0.4055610230884226, 0.9033496527273075, 0.0966503472726928),
  Point<3>(-2.775557561562891e-16, -0.9010417036652167, 0.09895829633478326),
  Point<3>(-2.220446049250313e-16, -0.6788556804717862, 0.1180853065831378),
  Point<3>(0, -0.3703069113454808, 0.128801896949209),
  Point<3>(1.127818256588307e-16, 7.051049700922512e-17, 0.1322468375110974),
  Point<3>(0, 0.3703069113454807, 0.128801896949209),
  Point<3>(5.551115123125783e-17, 0.678855680471786, 0.1180853065831379),
  Point<3>(3.33066907387547e-16, 0.9010417036652169, 0.09895829633478327),
  Point<3>(0.4055610230884222, -0.9033496527273077, 0.09665034727269281),
  Point<3>(0.3813913955667637, -0.6829884106255608, 0.1154087878461705),
  Point<3>(0.3722727723025121, -0.3722727723025125, 0.1255890328023157),
  Point<3>(0.3703069113454806, 1.665334536937735e-16, 0.1288018969492089),
  Point<3>(0.3722727723025122, 0.3722727723025122, 0.1255890328023157),
  Point<3>(0.3813913955667637, 0.6829884106255608, 0.1154087878461705),
  Point<3>(0.4055610230884229, 0.9033496527273074, 0.09665034727269277),
  Point<3>(0.7379257250530986, -0.9126419083510329, 0.0873580916489671),
  Point<3>(0.6993443102316483, -0.6993443102316482, 0.1062645981600722),
  Point<3>(0.6829884106255613, -0.3813913955667639, 0.1154087878461704),
  Point<3>(0.678855680471786, -1.110223024625157e-16, 0.1180853065831377),
  Point<3>(0.6829884106255606, 0.3813913955667639, 0.1154087878461705),
  Point<3>(0.6993443102316482, 0.6993443102316487, 0.1062645981600723),
  Point<3>(0.7379257250530987, 0.9126419083510329, 0.08735809164896698),
  Point<3>(0.9358700742548031, -0.9358700742548031, 0.06412992574519673),
  Point<3>(0.9126419083510324, -0.7379257250530989, 0.087358091648967),
  Point<3>(0.9033496527273074, -0.4055610230884225, 0.09665034727269271),
  Point<3>(0.901041703665217, -1.665334536937735e-16, 0.09895829633478326),
  Point<3>(0.9033496527273069, 0.4055610230884224, 0.09665034727269278),
  Point<3>(0.9126419083510334, 0.7379257250530987, 0.08735809164896714),
  Point<3>(0.9358700742548033, 0.9358700742548031, 0.06412992574519676),
  Point<3>(-0.7958500907165708, -0.7958500907165712, 0.2041499092834289),
  Point<3>(-0.751105685180558, -0.5578049906351725, 0.2488943148194422),
  Point<3>(-0.7333357225659864, -0.2000071676979593, 0.2666642774340134),
  Point<3>(-0.7333357225659864, 0.2000071676979593, 0.2666642774340137),
  Point<3>(-0.7511056851805573, 0.5578049906351722, 0.2488943148194421),
  Point<3>(-0.7958500907165711, 0.7958500907165711, 0.204149909283429),
  Point<3>(-0.5578049906351717, -0.7511056851805579, 0.2488943148194422),
  Point<3>(-0.5032964458763435, -0.5032964458763435, 0.2804401797237104),
  Point<3>(-0.4809834863938812, -0.1781398820330555, 0.2943475772754903),
  Point<3>(-0.4809834863938816, 0.1781398820330557, 0.2943475772754904),
  Point<3>(-0.5032964458763433, 0.5032964458763436, 0.2804401797237103),
  Point<3>(-0.5578049906351722, 0.751105685180558, 0.2488943148194421),
  Point<3>(-0.2000071676979597, -0.7333357225659867, 0.2666642774340134),
  Point<3>(-0.1781398820330554, -0.4809834863938814, 0.2943475772754904),
  Point<3>(-0.1697767281334269, -0.1697767281334271, 0.3073163042159497),
  Point<3>(-0.1697767281334269, 0.1697767281334269, 0.30731630421595),
  Point<3>(-0.1781398820330556, 0.4809834863938817, 0.2943475772754904),
  Point<3>(-0.2000071676979593, 0.7333357225659867, 0.2666642774340135),
  Point<3>(0.2000071676979594, -0.7333357225659867, 0.2666642774340136),
  Point<3>(0.1781398820330558, -0.4809834863938818, 0.2943475772754906),
  Point<3>(0.169776728133427, -0.1697767281334271, 0.3073163042159498),
  Point<3>(0.1697767281334269, 0.1697767281334271, 0.3073163042159497),
  Point<3>(0.1781398820330556, 0.4809834863938816, 0.2943475772754904),
  Point<3>(0.2000071676979596, 0.7333357225659869, 0.2666642774340134),
  Point<3>(0.5578049906351722, -0.751105685180558, 0.2488943148194422),
  Point<3>(0.5032964458763438, -0.5032964458763437, 0.2804401797237104),
  Point<3>(0.4809834863938814, -0.1781398820330559, 0.2943475772754904),
  Point<3>(0.4809834863938814, 0.1781398820330554, 0.2943475772754904),
  Point<3>(0.5032964458763438, 0.5032964458763435, 0.2804401797237103),
  Point<3>(0.5578049906351721, 0.7511056851805578, 0.2488943148194422),
  Point<3>(0.7958500907165709, -0.7958500907165712, 0.204149909283429),
  Point<3>(0.751105685180558, -0.5578049906351721, 0.2488943148194422),
  Point<3>(0.7333357225659863, -0.2000071676979593, 0.2666642774340136),
  Point<3>(0.7333357225659864, 0.2000071676979591, 0.2666642774340135),
  Point<3>(0.7511056851805583, 0.5578049906351721, 0.2488943148194422),
  Point<3>(0.7958500907165714, 0.7958500907165712, 0.2041499092834289),
  Point<3>(-0.6046496089512392, -0.6046496089512393, 0.3953503910487605),
  Point<3>(-0.5494791481673916, -0.3515625554978249, 0.4505208518326083),
  Point<3>(-0.5333285548680272, 0, 0.466671445131973),
  Point<3>(-0.5494791481673917, 0.3515625554978252, 0.4505208518326081),
  Point<3>(-0.6046496089512392, 0.6046496089512393, 0.3953503910487604),
  Point<3>(-0.3515625554978249, -0.5494791481673917, 0.4505208518326083),
  Point<3>(-0.3008517773395495, -0.3008517773395493, 0.4836245036373458),
  Point<3>(-0.2856522555597624, 1.387778780781446e-16, 0.4939314307552738),
  Point<3>(-0.3008517773395493, 0.3008517773395493, 0.4836245036373459),
  Point<3>(-0.3515625554978253, 0.5494791481673914, 0.4505208518326083),
  Point<3>(1.110223024625157e-16, -0.5333285548680271, 0.466671445131973),
  Point<3>(0, -0.2856522555597625, 0.4939314307552738),
  Point<3>(2.19677860114494e-16, 1.048439344160309e-16, 0.5026398065476319),
  Point<3>(1.387778780781446e-16, 0.2856522555597625, 0.4939314307552739),
  Point<3>(0, 0.5333285548680273, 0.4666714451319729),
  Point<3>(0.3515625554978253, -0.5494791481673919, 0.4505208518326081),
  Point<3>(0.3008517773395491, -0.3008517773395493, 0.4836245036373458),
  Point<3>(0.2856522555597627, 5.551115123125783e-17, 0.493931430755274),
  Point<3>(0.3008517773395493, 0.3008517773395495, 0.4836245036373458),
  Point<3>(0.3515625554978249, 0.5494791481673915, 0.4505208518326082),
  Point<3>(0.6046496089512391, -0.6046496089512394, 0.3953503910487604),
  Point<3>(0.5494791481673912, -0.3515625554978253, 0.4505208518326084),
  Point<3>(0.5333285548680274, 1.110223024625157e-16, 0.4666714451319732),
  Point<3>(0.5494791481673914, 0.3515625554978252, 0.4505208518326083),
  Point<3>(0.6046496089512394, 0.604649608951239, 0.3953503910487604),
  Point<3>(-0.3953503910487606, -0.3953503910487605, 0.6046496089512393),
  Point<3>(-0.345544662092135, -0.1522439675467492, 0.6544553379078651),
  Point<3>(-0.3455446620921351, 0.1522439675467497, 0.6544553379078651),
  Point<3>(-0.3953503910487604, 0.3953503910487606, 0.6046496089512394),
  Point<3>(-0.1522439675467495, -0.345544662092135, 0.6544553379078649),
  Point<3>(-0.123327761888504, -0.1233277618885039, 0.6788016280453851),
  Point<3>(-0.1233277618885039, 0.1233277618885042, 0.678801628045385),
  Point<3>(-0.1522439675467495, 0.3455446620921351, 0.654455337907865),
  Point<3>(0.1522439675467495, -0.3455446620921351, 0.654455337907865),
  Point<3>(0.1233277618885041, -0.1233277618885039, 0.6788016280453847),
  Point<3>(0.1233277618885042, 0.1233277618885041, 0.6788016280453852),
  Point<3>(0.1522439675467495, 0.3455446620921351, 0.6544553379078648),
  Point<3>(0.3953503910487605, -0.3953503910487607, 0.6046496089512394),
  Point<3>(0.3455446620921351, -0.1522439675467494, 0.6544553379078651),
  Point<3>(0.345544662092135, 0.1522439675467494, 0.6544553379078656),
  Point<3>(0.3953503910487605, 0.3953503910487602, 0.6046496089512392),
  Point<3>(-0.2041499092834289, -0.2041499092834288, 0.7958500907165712),
  Point<3>(-0.1747161832979342, 0, 0.8252838167020656),
  Point<3>(-0.2041499092834289, 0.2041499092834289, 0.7958500907165713),
  Point<3>(2.775557561562891e-17, -0.1747161832979342, 0.8252838167020656),
  Point<3>(-1.732409977668446e-16, -8.150452206525956e-17, 0.8372614358827385),
  Point<3>(8.326672684688674e-17, 0.1747161832979343, 0.8252838167020657),
  Point<3>(0.2041499092834288, -0.2041499092834288, 0.7958500907165713),
  Point<3>(0.1747161832979342, 5.551115123125783e-17, 0.8252838167020659),
  Point<3>(0.204149909283429, 0.2041499092834289, 0.7958500907165713),
  Point<3>(-0.06412992574519684, -0.06412992574519681, 0.9358700742548032),
  Point<3>(-0.06412992574519658, 0.06412992574519681, 0.9358700742548031),
  Point<3>(0.06412992574519664, -0.0641299257451968, 0.935870074254803),
  Point<3>(0.06412992574519683, 0.0641299257451966, 0.9358700742548033),
  Point<3>(-5.551115123125783e-17, -5.551115123125783e-17, 1)};
std::vector<Point<3>> reference_points_p8 = {
  Point<3>(-1, -1, 0),
  Point<3>(-1, -0.8997579954114606, 0),
  Point<3>(-1, -0.6771862795107377, 0),
  Point<3>(-1, -0.3631174638261784, 0),
  Point<3>(-1, 0, 0),
  Point<3>(-0.9999999999999999, 0.3631174638261784, 0),
  Point<3>(-1, 0.6771862795107378, 0),
  Point<3>(-1, 0.8997579954114598, 0),
  Point<3>(-1, 1, 0),
  Point<3>(-0.8997579954114604, -1, 0),
  Point<3>(-0.899757995411461, -0.8997579954114603, 0),
  Point<3>(-0.8997579954114603, -0.6771862795107378, 0),
  Point<3>(-0.8997579954114605, -0.3631174638261785, 0),
  Point<3>(-0.8997579954114601, 1.665334536937735e-16, 0),
  Point<3>(-0.8997579954114603, 0.3631174638261782, 0),
  Point<3>(-0.8997579954114605, 0.6771862795107378, 0),
  Point<3>(-0.8997579954114605, 0.8997579954114602, 0),
  Point<3>(-0.8997579954114605, 1, 0),
  Point<3>(-0.6771862795107375, -1, 0),
  Point<3>(-0.6771862795107376, -0.8997579954114603, 0),
  Point<3>(-0.6771862795107372, -0.677186279510738, 0),
  Point<3>(-0.6771862795107372, -0.3631174638261783, 0),
  Point<3>(-0.6771862795107376, 2.775557561562891e-17, 0),
  Point<3>(-0.6771862795107376, 0.3631174638261781, 0),
  Point<3>(-0.6771862795107374, 0.6771862795107376, 0),
  Point<3>(-0.6771862795107381, 0.8997579954114598, 0),
  Point<3>(-0.6771862795107377, 1, 0),
  Point<3>(-0.3631174638261783, -1, 0),
  Point<3>(-0.363117463826178, -0.8997579954114601, 0),
  Point<3>(-0.3631174638261782, -0.6771862795107376, 0),
  Point<3>(-0.3631174638261784, -0.3631174638261782, 0),
  Point<3>(-0.3631174638261782, -4.163336342344337e-17, 0),
  Point<3>(-0.3631174638261781, 0.3631174638261782, 0),
  Point<3>(-0.3631174638261781, 0.6771862795107377, 0),
  Point<3>(-0.3631174638261783, 0.8997579954114601, 0),
  Point<3>(-0.3631174638261782, 1, 0),
  Point<3>(0, -1, 0),
  Point<3>(0, -0.8997579954114603, 0),
  Point<3>(0, -0.6771862795107377, 0),
  Point<3>(5.551115123125783e-17, -0.3631174638261783, 0),
  Point<3>(-3.005758984146654e-17, -3.005759060892718e-17, 0),
  Point<3>(-8.326672684688674e-17, 0.3631174638261783, 0),
  Point<3>(-5.551115123125783e-17, 0.677186279510738, 0),
  Point<3>(-1.665334536937735e-16, 0.8997579954114602, 0),
  Point<3>(-5.551115123125783e-17, 1, 0),
  Point<3>(0.3631174638261784, -1, 0),
  Point<3>(0.3631174638261783, -0.8997579954114603, 0),
  Point<3>(0.3631174638261783, -0.6771862795107381, 0),
  Point<3>(0.363117463826178, -0.363117463826178, 0),
  Point<3>(0.3631174638261784, -6.938893903907228e-17, 0),
  Point<3>(0.3631174638261785, 0.3631174638261784, 0),
  Point<3>(0.3631174638261784, 0.677186279510738, 0),
  Point<3>(0.3631174638261777, 0.8997579954114598, 0),
  Point<3>(0.3631174638261783, 1, 0),
  Point<3>(0.6771862795107377, -1, 0),
  Point<3>(0.6771862795107376, -0.8997579954114604, 0),
  Point<3>(0.6771862795107373, -0.677186279510738, 0),
  Point<3>(0.6771862795107377, -0.3631174638261783, 0),
  Point<3>(0.6771862795107375, 1.387778780781446e-16, 0),
  Point<3>(0.6771862795107375, 0.3631174638261784, 0),
  Point<3>(0.6771862795107376, 0.677186279510738, 0),
  Point<3>(0.6771862795107378, 0.8997579954114598, 0),
  Point<3>(0.6771862795107376, 1, 0),
  Point<3>(0.8997579954114602, -1, 0),
  Point<3>(0.8997579954114603, -0.8997579954114602, 0),
  Point<3>(0.8997579954114601, -0.6771862795107378, 0),
  Point<3>(0.8997579954114601, -0.363117463826178, 0),
  Point<3>(0.8997579954114598, 5.551115123125783e-17, 0),
  Point<3>(0.8997579954114603, 0.3631174638261786, 0),
  Point<3>(0.8997579954114598, 0.6771862795107377, 0),
  Point<3>(0.8997579954114602, 0.89975799541146, 0),
  Point<3>(0.8997579954114601, 1, 0),
  Point<3>(1, -1, 0),
  Point<3>(1, -0.8997579954114605, 0),
  Point<3>(1, -0.6771862795107375, 0),
  Point<3>(1, -0.3631174638261783, 0),
  Point<3>(1, -5.551115123125783e-17, 0),
  Point<3>(0.9999999999999998, 0.3631174638261784, 0),
  Point<3>(1, 0.6771862795107377, 0),
  Point<3>(1, 0.8997579954114601, 0),
  Point<3>(1, 1, 0),
  Point<3>(-0.9498789977057303, -0.9498789977057301, 0.05012100229426996),
  Point<3>(-0.9308547472470863, -0.7925642417412598, 0.06914525275291336),
  Point<3>(-0.9216963743236731, -0.5243182604103258, 0.07830362567632733),
  Point<3>(-0.9181963392969246, -0.1832405464268355, 0.0818036607030754),
  Point<3>(-0.9181963392969248, 0.1832405464268349, 0.08180366070307538),
  Point<3>(-0.9216963743236726, 0.5243182604103254, 0.07830362567632725),
  Point<3>(-0.9308547472470867, 0.7925642417412599, 0.0691452527529135),
  Point<3>(-0.9498789977057299, 0.94987899770573, 0.05012100229426997),
  Point<3>(-0.7925642417412596, -0.9308547472470867, 0.06914525275291328),
  Point<3>(-0.7586818371383302, -0.7586818371383295, 0.08482863399717222),
  Point<3>(-0.7431049126454792, -0.4980233765981967, 0.09342164208100781),
  Point<3>(-0.7363917768681747, -0.1738588468150395, 0.09715095897814469),
  Point<3>(-0.7363917768681747, 0.1738588468150397, 0.09715095897814466),
  Point<3>(-0.7431049126454794, 0.4980233765981965, 0.09342164208100756),
  Point<3>(-0.7586818371383288, 0.7586818371383294, 0.084828633997172),
  Point<3>(-0.7925642417412601, 0.9308547472470869, 0.06914525275291343),
  Point<3>(-0.5243182604103258, -0.9216963743236732, 0.07830362567632738),
  Point<3>(-0.4980233765981968, -0.7431049126454793, 0.09342164208100771),
  Point<3>(-0.4876763629105314, -0.4876763629105311, 0.1025155806895429),
  Point<3>(-0.4834199208483784, -0.17066720984335, 0.1067641920558217),
  Point<3>(-0.4834199208483785, 0.1706672098433495, 0.1067641920558216),
  Point<3>(-0.4876763629105308, 0.4876763629105306, 0.1025155806895429),
  Point<3>(-0.4980233765981966, 0.7431049126454787, 0.09342164208100755),
  Point<3>(-0.5243182604103253, 0.9216963743236721, 0.07830362567632732),
  Point<3>(-0.1832405464268356, -0.9181963392969246, 0.08180366070307547),
  Point<3>(-0.1738588468150399, -0.7363917768681745, 0.09715095897814462),
  Point<3>(-0.1706672098433498, -0.4834199208483784, 0.1067641920558216),
  Point<3>(-0.1694445408116378, -0.1694445408116377, 0.1113841671048484),
  Point<3>(-0.1694445408116378, 0.1694445408116377, 0.1113841671048484),
  Point<3>(-0.1706672098433499, 0.4834199208483782, 0.1067641920558217),
  Point<3>(-0.1738588468150396, 0.7363917768681743, 0.09715095897814466),
  Point<3>(-0.1832405464268348, 0.9181963392969246, 0.08180366070307546),
  Point<3>(0.1832405464268351, -0.9181963392969243, 0.08180366070307542),
  Point<3>(0.1738588468150398, -0.7363917768681749, 0.09715095897814462),
  Point<3>(0.1706672098433497, -0.4834199208483786, 0.1067641920558217),
  Point<3>(0.1694445408116378, -0.1694445408116379, 0.1113841671048485),
  Point<3>(0.1694445408116377, 0.1694445408116379, 0.1113841671048485),
  Point<3>(0.17066720984335, 0.4834199208483786, 0.1067641920558216),
  Point<3>(0.1738588468150398, 0.7363917768681745, 0.09715095897814463),
  Point<3>(0.1832405464268353, 0.9181963392969246, 0.08180366070307543),
  Point<3>(0.5243182604103254, -0.9216963743236728, 0.07830362567632732),
  Point<3>(0.4980233765981961, -0.743104912645479, 0.09342164208100756),
  Point<3>(0.4876763629105307, -0.4876763629105313, 0.1025155806895429),
  Point<3>(0.4834199208483784, -0.17066720984335, 0.1067641920558216),
  Point<3>(0.4834199208483783, 0.1706672098433501, 0.1067641920558216),
  Point<3>(0.4876763629105312, 0.4876763629105312, 0.1025155806895429),
  Point<3>(0.498023376598197, 0.7431049126454794, 0.09342164208100763),
  Point<3>(0.5243182604103257, 0.9216963743236735, 0.0783036256763274),
  Point<3>(0.7925642417412598, -0.9308547472470869, 0.06914525275291342),
  Point<3>(0.7586818371383293, -0.7586818371383296, 0.08482863399717214),
  Point<3>(0.743104912645479, -0.4980233765981967, 0.0934216420810077),
  Point<3>(0.7363917768681745, -0.1738588468150396, 0.09715095897814464),
  Point<3>(0.7363917768681743, 0.1738588468150401, 0.09715095897814474),
  Point<3>(0.7431049126454797, 0.4980233765981965, 0.0934216420810077),
  Point<3>(0.7586818371383297, 0.7586818371383295, 0.08482863399717216),
  Point<3>(0.7925642417412595, 0.9308547472470875, 0.06914525275291329),
  Point<3>(0.9498789977057298, -0.9498789977057301, 0.05012100229426997),
  Point<3>(0.9308547472470871, -0.7925642417412595, 0.06914525275291339),
  Point<3>(0.921696374323673, -0.5243182604103261, 0.07830362567632737),
  Point<3>(0.9181963392969247, -0.1832405464268354, 0.08180366070307539),
  Point<3>(0.9181963392969248, 0.1832405464268356, 0.08180366070307546),
  Point<3>(0.9216963743236729, 0.5243182604103249, 0.0783036256763273),
  Point<3>(0.9308547472470862, 0.7925642417412605, 0.06914525275291344),
  Point<3>(0.9498789977057301, 0.9498789977057298, 0.05012100229426999),
  Point<3>(-0.8385931397553695, -0.8385931397553691, 0.1614068602446311),
  Point<3>(-0.8013109430433266, -0.6447036916906723, 0.1986890569566735),
  Point<3>(-0.783146827939339, -0.3494404838180171, 0.2168531720606612),
  Point<3>(-0.7778043774698973, 5.551115123125783e-17, 0.2221956225301027),
  Point<3>(-0.7831468279393388, 0.3494404838180168, 0.2168531720606611),
  Point<3>(-0.8013109430433265, 0.6447036916906718, 0.1986890569566734),
  Point<3>(-0.8385931397553691, 0.8385931397553692, 0.1614068602446312),
  Point<3>(-0.6447036916906723, -0.8013109430433271, 0.1986890569566735),
  Point<3>(-0.5947821825588359, -0.5947821825588357, 0.226476113783121),
  Point<3>(-0.5704809685540351, -0.3185285214890294, 0.2411639500183478),
  Point<3>(-0.5629343851578482, 1.110223024625157e-16, 0.2457254539316421),
  Point<3>(-0.5704809685540353, 0.3185285214890292, 0.2411639500183475),
  Point<3>(-0.5947821825588354, 0.5947821825588357, 0.2264761137831209),
  Point<3>(-0.6447036916906719, 0.8013109430433266, 0.1986890569566737),
  Point<3>(-0.349440483818017, -0.783146827939339, 0.216853172060661),
  Point<3>(-0.3185285214890294, -0.5704809685540352, 0.2411639500183477),
  Point<3>(-0.3048206228332161, -0.3048206228332163, 0.2547715416651276),
  Point<3>(-0.3006436599545749, -2.775557561562891e-16, 0.2591431843592851),
  Point<3>(-0.304820622833216, 0.304820622833216, 0.2547715416651274),
  Point<3>(-0.3185285214890291, 0.5704809685540349, 0.2411639500183476),
  Point<3>(-0.349440483818017, 0.783146827939339, 0.216853172060661),
  Point<3>(-5.551115123125783e-17, -0.7778043774698976, 0.2221956225301026),
  Point<3>(-2.220446049250313e-16, -0.5629343851578477, 0.2457254539316422),
  Point<3>(1.387778780781446e-16, -0.3006436599545749, 0.2591431843592851),
  Point<3>(-5.901102734544855e-17, 3.031823427704703e-17, 0.2634985686228368),
  Point<3>(-1.110223024625157e-16, 0.3006436599545749, 0.2591431843592851),
  Point<3>(5.551115123125783e-17, 0.5629343851578479, 0.2457254539316422),
  Point<3>(5.551115123125783e-17, 0.7778043774698973, 0.2221956225301026),
  Point<3>(0.349440483818017, -0.783146827939339, 0.216853172060661),
  Point<3>(0.3185285214890294, -0.5704809685540354, 0.2411639500183478),
  Point<3>(0.3048206228332163, -0.3048206228332161, 0.2547715416651275),
  Point<3>(0.3006436599545748, 1.387778780781446e-16, 0.2591431843592851),
  Point<3>(0.3048206228332159, 0.3048206228332165, 0.2547715416651276),
  Point<3>(0.3185285214890293, 0.5704809685540357, 0.2411639500183479),
  Point<3>(0.349440483818017, 0.7831468279393392, 0.2168531720606611),
  Point<3>(0.6447036916906714, -0.8013109430433263, 0.1986890569566737),
  Point<3>(0.5947821825588352, -0.5947821825588352, 0.2264761137831211),
  Point<3>(0.5704809685540355, -0.3185285214890291, 0.2411639500183476),
  Point<3>(0.5629343851578474, 3.33066907387547e-16, 0.2457254539316421),
  Point<3>(0.5704809685540355, 0.3185285214890297, 0.2411639500183478),
  Point<3>(0.5947821825588361, 0.5947821825588354, 0.226476113783121),
  Point<3>(0.6447036916906723, 0.8013109430433265, 0.1986890569566734),
  Point<3>(0.8385931397553688, -0.8385931397553693, 0.1614068602446312),
  Point<3>(0.8013109430433262, -0.644703691690672, 0.1986890569566734),
  Point<3>(0.7831468279393394, -0.3494404838180168, 0.216853172060661),
  Point<3>(0.7778043774698972, 1.110223024625157e-16, 0.2221956225301026),
  Point<3>(0.7831468279393392, 0.3494404838180171, 0.2168531720606611),
  Point<3>(0.8013109430433267, 0.6447036916906721, 0.1986890569566736),
  Point<3>(0.8385931397553693, 0.8385931397553691, 0.1614068602446312),
  Point<3>(-0.6815587319130891, -0.6815587319130894, 0.3184412680869109),
  Point<3>(-0.6325221035649551, -0.4689147821588043, 0.3674778964350445),
  Point<3>(-0.6110978112650508, -0.1667065662048461, 0.3889021887349489),
  Point<3>(-0.6110978112650509, 0.1667065662048462, 0.3889021887349488),
  Point<3>(-0.6325221035649552, 0.4689147821588044, 0.3674778964350447),
  Point<3>(-0.6815587319130892, 0.681558731913089, 0.3184412680869109),
  Point<3>(-0.4689147821588043, -0.6325221035649552, 0.3674778964350446),
  Point<3>(-0.416088264686501, -0.416088264686501, 0.4001871994585379),
  Point<3>(-0.3926769007494837, -0.1457794368654244, 0.4152891775355415),
  Point<3>(-0.3926769007494836, 0.1457794368654244, 0.4152891775355414),
  Point<3>(-0.4160882646865011, 0.4160882646865008, 0.4001871994585378),
  Point<3>(-0.4689147821588044, 0.6325221035649551, 0.3674778964350448),
  Point<3>(-0.1667065662048461, -0.6110978112650511, 0.3889021887349488),
  Point<3>(-0.1457794368654244, -0.3926769007494835, 0.4152891775355416),
  Point<3>(-0.1369379844821431, -0.1369379844821433, 0.4278902013579856),
  Point<3>(-0.1369379844821433, 0.1369379844821432, 0.4278902013579853),
  Point<3>(-0.1457794368654245, 0.3926769007494836, 0.4152891775355412),
  Point<3>(-0.1667065662048461, 0.6110978112650509, 0.3889021887349489),
  Point<3>(0.1667065662048462, -0.6110978112650506, 0.3889021887349487),
  Point<3>(0.1457794368654244, -0.3926769007494835, 0.4152891775355417),
  Point<3>(0.1369379844821431, -0.1369379844821431, 0.4278902013579852),
  Point<3>(0.1369379844821432, 0.1369379844821432, 0.4278902013579855),
  Point<3>(0.1457794368654245, 0.3926769007494836, 0.4152891775355415),
  Point<3>(0.166706566204846, 0.6110978112650512, 0.3889021887349489),
  Point<3>(0.4689147821588043, -0.6325221035649551, 0.3674778964350449),
  Point<3>(0.4160882646865009, -0.4160882646865006, 0.4001871994585378),
  Point<3>(0.3926769007494837, -0.1457794368654243, 0.4152891775355413),
  Point<3>(0.3926769007494835, 0.1457794368654245, 0.4152891775355414),
  Point<3>(0.4160882646865006, 0.4160882646865011, 0.4001871994585377),
  Point<3>(0.4689147821588048, 0.6325221035649551, 0.3674778964350446),
  Point<3>(0.6815587319130894, -0.6815587319130892, 0.3184412680869109),
  Point<3>(0.6325221035649552, -0.4689147821588046, 0.3674778964350446),
  Point<3>(0.6110978112650511, -0.1667065662048459, 0.3889021887349489),
  Point<3>(0.6110978112650507, 0.1667065662048462, 0.3889021887349488),
  Point<3>(0.632522103564955, 0.4689147821588043, 0.3674778964350447),
  Point<3>(0.6815587319130891, 0.6815587319130891, 0.3184412680869109),
  Point<3>(-0.4999999999999999, -0.5, 0.5000000000000001),
  Point<3>(-0.4492815571381205, -0.2856742357319694, 0.5507184428618799),
  Point<3>(-0.4337063441213221, 2.775557561562891e-17, 0.5662936558786782),
  Point<3>(-0.4492815571381206, 0.2856742357319691, 0.5507184428618792),
  Point<3>(-0.4999999999999999, 0.4999999999999999, 0.4999999999999998),
  Point<3>(-0.2856742357319693, -0.4492815571381205, 0.5507184428618798),
  Point<3>(-0.2426976526347396, -0.2426976526347396, 0.5801715706838365),
  Point<3>(-0.2293056629440544, 8.326672684688674e-17, 0.5895562484038153),
  Point<3>(-0.2426976526347396, 0.2426976526347396, 0.5801715706838365),
  Point<3>(-0.2856742357319694, 0.4492815571381202, 0.5507184428618799),
  Point<3>(2.775557561562891e-17, -0.4337063441213224, 0.566293655878678),
  Point<3>(1.387778780781446e-16, -0.2293056629440545, 0.589556248403815),
  Point<3>(6.004468785972654e-17, 3.764162102524116e-17, 0.5970835126301588),
  Point<3>(-1.387778780781446e-17, 0.2293056629440544, 0.5895562484038147),
  Point<3>(0, 0.4337063441213221, 0.5662936558786779),
  Point<3>(0.285674235731969, -0.4492815571381202, 0.5507184428618794),
  Point<3>(0.2426976526347397, -0.2426976526347395, 0.580171570683837),
  Point<3>(0.2293056629440545, 1.249000902703301e-16, 0.5895562484038149),
  Point<3>(0.2426976526347396, 0.2426976526347395, 0.580171570683837),
  Point<3>(0.2856742357319692, 0.4492815571381205, 0.5507184428618794),
  Point<3>(0.5, -0.4999999999999999, 0.4999999999999999),
  Point<3>(0.4492815571381205, -0.2856742357319693, 0.5507184428618799),
  Point<3>(0.4337063441213221, 8.326672684688674e-17, 0.566293655878678),
  Point<3>(0.4492815571381202, 0.2856742357319693, 0.5507184428618798),
  Point<3>(0.4999999999999999, 0.4999999999999999, 0.5),
  Point<3>(-0.3184412680869109, -0.318441268086911, 0.6815587319130889),
  Point<3>(-0.2769926826330008, -0.1203854312803461, 0.7230073173669992),
  Point<3>(-0.276992682633001, 0.1203854312803463, 0.723007317366999),
  Point<3>(-0.3184412680869108, 0.3184412680869109, 0.681558731913089),
  Point<3>(-0.1203854312803461, -0.276992682633001, 0.723007317366999),
  Point<3>(-0.0978200272398307, -0.0978200272398308, 0.7430614404325226),
  Point<3>(-0.09782002723983098, 0.09782002723983108, 0.743061440432523),
  Point<3>(-0.1203854312803463, 0.2769926826330009, 0.7230073173669989),
  Point<3>(0.1203854312803462, -0.276992682633001, 0.7230073173669986),
  Point<3>(0.09782002723983124, -0.09782002723983087, 0.7430614404325229),
  Point<3>(0.09782002723983077, 0.09782002723983109, 0.7430614404325225),
  Point<3>(0.120385431280346, 0.2769926826330011, 0.7230073173669987),
  Point<3>(0.3184412680869109, -0.318441268086911, 0.681558731913089),
  Point<3>(0.2769926826330013, -0.120385431280346, 0.7230073173669992),
  Point<3>(0.2769926826330009, 0.1203854312803463, 0.7230073173669994),
  Point<3>(0.3184412680869109, 0.3184412680869109, 0.6815587319130891),
  Point<3>(-0.1614068602446313, -0.1614068602446313, 0.8385931397553688),
  Point<3>(-0.1382905055058268, -2.775557561562891e-17, 0.8617094944941728),
  Point<3>(-0.1614068602446313, 0.1614068602446313, 0.8385931397553688),
  Point<3>(-1.387778780781446e-17, -0.1382905055058268, 0.861709494494173),
  Point<3>(-3.712112430034454e-17, -1.737130915552265e-16, 0.8712808531832384),
  Point<3>(1.387778780781446e-17, 0.1382905055058268, 0.8617094944941731),
  Point<3>(0.1614068602446313, -0.1614068602446312, 0.8385931397553688),
  Point<3>(0.1382905055058269, -4.163336342344337e-17, 0.8617094944941729),
  Point<3>(0.1614068602446313, 0.1614068602446312, 0.8385931397553688),
  Point<3>(-0.05012100229427003, -0.05012100229427004, 0.9498789977057303),
  Point<3>(-0.05012100229426981, 0.05012100229427004, 0.9498789977057303),
  Point<3>(0.05012100229426983, -0.05012100229427001, 0.9498789977057303),
  Point<3>(0.05012100229427003, 0.05012100229426982, 0.9498789977057303),
  Point<3>(0, -1.110223024625157e-16, 0.9999999999999999)};
std::vector<Point<3>> reference_points_p9 = {
  Point<3>(-1, -1, 0),
  Point<3>(-1, -0.9195339081664586, 0),
  Point<3>(-0.9999999999999999, -0.7387738651055047, 0),
  Point<3>(-0.9999999999999998, -0.4779249498104445, 0),
  Point<3>(-1, -0.1652789576663869, 0),
  Point<3>(-1, 0.1652789576663868, 0),
  Point<3>(-0.9999999999999998, 0.4779249498104444, 0),
  Point<3>(-1, 0.7387738651055047, 0),
  Point<3>(-1, 0.9195339081664589, 0),
  Point<3>(-1, 1, 0),
  Point<3>(-0.9195339081664585, -1, 0),
  Point<3>(-0.9195339081664577, -0.9195339081664584, 0),
  Point<3>(-0.9195339081664586, -0.7387738651055048, 0),
  Point<3>(-0.9195339081664586, -0.4779249498104445, 0),
  Point<3>(-0.9195339081664582, -0.1652789576663871, 0),
  Point<3>(-0.9195339081664583, 0.165278957666387, 0),
  Point<3>(-0.9195339081664589, 0.4779249498104443, 0),
  Point<3>(-0.9195339081664587, 0.7387738651055049, 0),
  Point<3>(-0.9195339081664584, 0.9195339081664587, 0),
  Point<3>(-0.9195339081664586, 1, 0),
  Point<3>(-0.7387738651055047, -1, 0),
  Point<3>(-0.7387738651055046, -0.9195339081664591, 0),
  Point<3>(-0.7387738651055048, -0.738773865105504, 0),
  Point<3>(-0.7387738651055047, -0.4779249498104442, 0),
  Point<3>(-0.7387738651055047, -0.1652789576663871, 0),
  Point<3>(-0.7387738651055049, 0.1652789576663868, 0),
  Point<3>(-0.7387738651055047, 0.4779249498104445, 0),
  Point<3>(-0.7387738651055051, 0.738773865105505, 0),
  Point<3>(-0.7387738651055049, 0.9195339081664589, 0),
  Point<3>(-0.7387738651055047, 1, 0),
  Point<3>(-0.4779249498104447, -0.9999999999999998, 0),
  Point<3>(-0.4779249498104443, -0.9195339081664586, 0),
  Point<3>(-0.4779249498104445, -0.7387738651055046, 0),
  Point<3>(-0.4779249498104446, -0.4779249498104443, 0),
  Point<3>(-0.4779249498104444, -0.1652789576663871, 0),
  Point<3>(-0.4779249498104448, 0.1652789576663868, 0),
  Point<3>(-0.4779249498104446, 0.4779249498104443, 0),
  Point<3>(-0.4779249498104446, 0.7387738651055048, 0),
  Point<3>(-0.4779249498104445, 0.919533908166459, 0),
  Point<3>(-0.4779249498104444, 0.9999999999999998, 0),
  Point<3>(-0.1652789576663869, -1, 0),
  Point<3>(-0.1652789576663869, -0.9195339081664587, 0),
  Point<3>(-0.165278957666387, -0.7387738651055051, 0),
  Point<3>(-0.1652789576663869, -0.4779249498104445, 0),
  Point<3>(-0.1652789576663869, -0.1652789576663871, 0),
  Point<3>(-0.1652789576663871, 0.1652789576663869, 0),
  Point<3>(-0.1652789576663872, 0.4779249498104444, 0),
  Point<3>(-0.1652789576663869, 0.7387738651055049, 0),
  Point<3>(-0.165278957666387, 0.9195339081664586, 0),
  Point<3>(-0.1652789576663871, 1, 0),
  Point<3>(0.1652789576663867, -1, 0),
  Point<3>(0.1652789576663869, -0.9195339081664591, 0),
  Point<3>(0.1652789576663869, -0.7387738651055049, 0),
  Point<3>(0.1652789576663868, -0.4779249498104447, 0),
  Point<3>(0.1652789576663869, -0.165278957666387, 0),
  Point<3>(0.1652789576663869, 0.1652789576663868, 0),
  Point<3>(0.1652789576663869, 0.4779249498104446, 0),
  Point<3>(0.1652789576663869, 0.7387738651055045, 0),
  Point<3>(0.1652789576663867, 0.9195339081664586, 0),
  Point<3>(0.1652789576663869, 1, 0),
  Point<3>(0.4779249498104444, -0.9999999999999998, 0),
  Point<3>(0.4779249498104442, -0.9195339081664583, 0),
  Point<3>(0.4779249498104444, -0.7387738651055046, 0),
  Point<3>(0.4779249498104443, -0.4779249498104445, 0),
  Point<3>(0.4779249498104443, -0.165278957666387, 0),
  Point<3>(0.4779249498104444, 0.1652789576663872, 0),
  Point<3>(0.4779249498104446, 0.4779249498104444, 0),
  Point<3>(0.4779249498104445, 0.7387738651055048, 0),
  Point<3>(0.4779249498104446, 0.919533908166459, 0),
  Point<3>(0.4779249498104445, 0.9999999999999996, 0),
  Point<3>(0.7387738651055047, -0.9999999999999999, 0),
  Point<3>(0.7387738651055049, -0.9195339081664586, 0),
  Point<3>(0.7387738651055049, -0.7387738651055042, 0),
  Point<3>(0.7387738651055045, -0.4779249498104444, 0),
  Point<3>(0.7387738651055054, -0.165278957666387, 0),
  Point<3>(0.7387738651055047, 0.1652789576663868, 0),
  Point<3>(0.7387738651055042, 0.4779249498104445, 0),
  Point<3>(0.7387738651055049, 0.7387738651055042, 0),
  Point<3>(0.7387738651055046, 0.919533908166459, 0),
  Point<3>(0.7387738651055048, 1, 0),
  Point<3>(0.9195339081664589, -1, 0),
  Point<3>(0.9195339081664591, -0.9195339081664586, 0),
  Point<3>(0.9195339081664586, -0.738773865105505, 0),
  Point<3>(0.9195339081664592, -0.4779249498104444, 0),
  Point<3>(0.9195339081664595, -0.1652789576663874, 0),
  Point<3>(0.9195339081664587, 0.1652789576663871, 0),
  Point<3>(0.9195339081664586, 0.4779249498104443, 0),
  Point<3>(0.9195339081664585, 0.738773865105505, 0),
  Point<3>(0.9195339081664595, 0.9195339081664589, 0),
  Point<3>(0.9195339081664587, 1, 0),
  Point<3>(1, -1, 0),
  Point<3>(1, -0.9195339081664586, 0),
  Point<3>(1, -0.7387738651055046, 0),
  Point<3>(0.9999999999999998, -0.4779249498104445, 0),
  Point<3>(1, -0.1652789576663869, 0),
  Point<3>(0.9999999999999998, 0.1652789576663868, 0),
  Point<3>(0.9999999999999998, 0.4779249498104444, 0),
  Point<3>(1, 0.7387738651055048, 0),
  Point<3>(0.9999999999999998, 0.9195339081664587, 0),
  Point<3>(1, 1, 0),
  Point<3>(-0.9597669540832294, -0.9597669540832293, 0.04023304591677062),
  Point<3>(-0.9439106438330909, -0.8317319314992734, 0.05608935616690903),
  Point<3>(-0.9350891476525711, -0.6113517093122273, 0.06491085234742934),
  Point<3>(-0.9308171295525887, -0.3234950528640262, 0.06918287044741156),
  Point<3>(-0.9295652427467893, -1.110223024625157e-16, 0.07043475725321098),
  Point<3>(-0.9308171295525889, 0.3234950528640264, 0.06918287044741156),
  Point<3>(-0.9350891476525702, 0.6113517093122276, 0.06491085234742916),
  Point<3>(-0.9439106438330908, 0.8317319314992726, 0.0560893561669089),
  Point<3>(-0.9597669540832292, 0.9597669540832294, 0.04023304591677062),
  Point<3>(-0.8317319314992725, -0.9439106438330911, 0.05608935616690901),
  Point<3>(-0.8019621627412763, -0.8019621627412757, 0.06931742990966996),
  Point<3>(-0.7867700214740103, -0.5856424923264123, 0.07733252201631252),
  Point<3>(-0.7789497562019718, -0.309052899993445, 0.08155947181549004),
  Point<3>(-0.7762919669902514, 5.551115123125783e-17, 0.08287054719531399),
  Point<3>(-0.778949756201972, 0.3090528999934453, 0.08155947181548998),
  Point<3>(-0.7867700214740097, 0.5856424923264131, 0.07733252201631236),
  Point<3>(-0.8019621627412761, 0.8019621627412759, 0.06931742990966978),
  Point<3>(-0.8317319314992736, 0.9439106438330909, 0.0560893561669089),
  Point<3>(-0.6113517093122275, -0.9350891476525699, 0.06491085234742938),
  Point<3>(-0.5856424923264129, -0.7867700214740092, 0.07733252201631244),
  Point<3>(-0.5747019150337009, -0.5747019150337004, 0.08539134293167328),
  Point<3>(-0.5695116223590666, -0.3036356495537913, 0.08989753330378787),
  Point<3>(-0.5676358983099594, -1.110223024625157e-16, 0.0913443820589063),
  Point<3>(-0.5695116223590673, 0.3036356495537916, 0.08989753330378782),
  Point<3>(-0.5747019150337009, 0.5747019150337008, 0.0853913429316733),
  Point<3>(-0.5856424923264141, 0.7867700214740098, 0.07733252201631247),
  Point<3>(-0.6113517093122269, 0.9350891476525711, 0.0649108523474292),
  Point<3>(-0.3234950528640267, -0.9308171295525886, 0.06918287044741153),
  Point<3>(-0.3090528999934454, -0.7789497562019719, 0.08155947181548998),
  Point<3>(-0.3036356495537913, -0.5695116223590673, 0.08989753330378789),
  Point<3>(-0.3012572079515162, -0.3012572079515162, 0.09469796760930471),
  Point<3>(-0.3003544209886008, 5.551115123125783e-17, 0.09626423155521553),
  Point<3>(-0.3012572079515163, 0.3012572079515159, 0.09469796760930466),
  Point<3>(-0.3036356495537916, 0.569511622359067, 0.0898975333037878),
  Point<3>(-0.3090528999934455, 0.7789497562019723, 0.08155947181549002),
  Point<3>(-0.3234950528640265, 0.9308171295525888, 0.0691828704474116),
  Point<3>(-3.33066907387547e-16, -0.9295652427467893, 0.07043475725321095),
  Point<3>(1.110223024625157e-16, -0.7762919669902509, 0.08287054719531396),
  Point<3>(1.942890293094024e-16, -0.5676358983099598, 0.09134438205890627),
  Point<3>(5.551115123125783e-17, -0.3003544209886007, 0.09626423155521549),
  Point<3>(6.961801655336562e-18, -2.442877494666621e-17, 0.09787668199712703),
  Point<3>(4.163336342344337e-17, 0.3003544209886008, 0.0962642315552155),
  Point<3>(5.551115123125783e-17, 0.5676358983099594, 0.09134438205890621),
  Point<3>(5.551115123125783e-17, 0.7762919669902508, 0.08287054719531378),
  Point<3>(1.110223024625157e-16, 0.9295652427467892, 0.07043475725321095),
  Point<3>(0.3234950528640262, -0.9308171295525883, 0.06918287044741162),
  Point<3>(0.3090528999934453, -0.778949756201972, 0.08155947181549004),
  Point<3>(0.3036356495537916, -0.5695116223590668, 0.08989753330378789),
  Point<3>(0.3012572079515164, -0.301257207951516, 0.09469796760930471),
  Point<3>(0.3003544209886008, 1.249000902703301e-16, 0.09626423155521545),
  Point<3>(0.3012572079515159, 0.3012572079515159, 0.09469796760930468),
  Point<3>(0.3036356495537914, 0.5695116223590669, 0.08989753330378784),
  Point<3>(0.3090528999934448, 0.7789497562019717, 0.08155947181548989),
  Point<3>(0.3234950528640261, 0.9308171295525887, 0.06918287044741157),
  Point<3>(0.611351709312227, -0.9350891476525709, 0.06491085234742924),
  Point<3>(0.5856424923264131, -0.7867700214740094, 0.07733252201631248),
  Point<3>(0.5747019150337006, -0.5747019150337005, 0.08539134293167322),
  Point<3>(0.5695116223590674, -0.3036356495537913, 0.08989753330378777),
  Point<3>(0.5676358983099595, -5.551115123125783e-17, 0.0913443820589063),
  Point<3>(0.5695116223590667, 0.3036356495537913, 0.08989753330378787),
  Point<3>(0.5747019150337013, 0.574701915033701, 0.08539134293167326),
  Point<3>(0.5856424923264132, 0.7867700214740091, 0.07733252201631242),
  Point<3>(0.6113517093122276, 0.93508914765257, 0.06491085234742937),
  Point<3>(0.8317319314992729, -0.9439106438330905, 0.0560893561669089),
  Point<3>(0.8019621627412756, -0.8019621627412764, 0.06931742990966974),
  Point<3>(0.7867700214740088, -0.5856424923264137, 0.07733252201631247),
  Point<3>(0.7789497562019719, -0.3090528999934457, 0.08155947181548999),
  Point<3>(0.7762919669902517, -2.775557561562891e-16, 0.08287054719531388),
  Point<3>(0.7789497562019715, 0.3090528999934453, 0.08155947181548999),
  Point<3>(0.78677002147401, 0.5856424923264122, 0.07733252201631238),
  Point<3>(0.8019621627412769, 0.8019621627412759, 0.06931742990966973),
  Point<3>(0.8317319314992732, 0.9439106438330918, 0.05608935616690904),
  Point<3>(0.9597669540832291, -0.9597669540832294, 0.04023304591677061),
  Point<3>(0.943910643833091, -0.8317319314992729, 0.05608935616690904),
  Point<3>(0.9350891476525699, -0.6113517093122272, 0.06491085234742934),
  Point<3>(0.9308171295525882, -0.3234950528640262, 0.06918287044741156),
  Point<3>(0.9295652427467892, -1.110223024625157e-16, 0.07043475725321094),
  Point<3>(0.9308171295525889, 0.3234950528640262, 0.06918287044741159),
  Point<3>(0.9350891476525706, 0.6113517093122277, 0.06491085234742922),
  Point<3>(0.9439106438330913, 0.8317319314992733, 0.05608935616690887),
  Point<3>(0.9597669540832295, 0.9597669540832291, 0.04023304591677062),
  Point<3>(-0.8693869325527522, -0.8693869325527523, 0.1306130674472477),
  Point<3>(-0.8381312808298285, -0.7083095761349696, 0.1618687191701717),
  Point<3>(-0.8206123917911027, -0.461837175373309, 0.1793876082088969),
  Point<3>(-0.8128654852977197, -0.1602884034118262, 0.1871345147022802),
  Point<3>(-0.8128654852977202, 0.1602884034118259, 0.1871345147022801),
  Point<3>(-0.8206123917911033, 0.4618371753733092, 0.1793876082088971),
  Point<3>(-0.8381312808298279, 0.7083095761349697, 0.1618687191701718),
  Point<3>(-0.8693869325527522, 0.8693869325527522, 0.1306130674472477),
  Point<3>(-0.7083095761349694, -0.8381312808298287, 0.1618687191701718),
  Point<3>(-0.6637161946749366, -0.6637161946749364, 0.1861190621843337),
  Point<3>(-0.639544101575837, -0.42807982111122, 0.2005389272308203),
  Point<3>(-0.6283705083530853, -0.1479196123854722, 0.2072181049435331),
  Point<3>(-0.6283705083530855, 0.1479196123854724, 0.2072181049435331),
  Point<3>(-0.6395441015758373, 0.4280798211112196, 0.2005389272308206),
  Point<3>(-0.6637161946749369, 0.6637161946749363, 0.1861190621843338),
  Point<3>(-0.70830957613497, 0.8381312808298285, 0.1618687191701718),
  Point<3>(-0.4618371753733097, -0.8206123917911026, 0.1793876082088969),
  Point<3>(-0.4280798211112198, -0.6395441015758372, 0.2005389272308205),
  Point<3>(-0.4117367501948075, -0.4117367501948073, 0.2137105591038227),
  Point<3>(-0.4043922174695631, -0.1422860500856745, 0.2200152897850517),
  Point<3>(-0.4043922174695636, 0.1422860500856745, 0.2200152897850518),
  Point<3>(-0.4117367501948076, 0.4117367501948074, 0.2137105591038228),
  Point<3>(-0.42807982111122, 0.6395441015758371, 0.2005389272308207),
  Point<3>(-0.4618371753733091, 0.8206123917911027, 0.179387608208897),
  Point<3>(-0.160288403411826, -0.81286548529772, 0.1871345147022801),
  Point<3>(-0.1479196123854723, -0.6283705083530853, 0.2072181049435331),
  Point<3>(-0.1422860500856745, -0.4043922174695634, 0.2200152897850517),
  Point<3>(-0.1398027699095568, -0.139802769909557, 0.2262354331514883),
  Point<3>(-0.139802769909557, 0.139802769909557, 0.2262354331514884),
  Point<3>(-0.1422860500856746, 0.4043922174695635, 0.2200152897850519),
  Point<3>(-0.1479196123854724, 0.6283705083530856, 0.2072181049435332),
  Point<3>(-0.1602884034118262, 0.8128654852977202, 0.1871345147022801),
  Point<3>(0.1602884034118263, -0.8128654852977202, 0.1871345147022801),
  Point<3>(0.1479196123854724, -0.6283705083530857, 0.2072181049435332),
  Point<3>(0.1422860500856747, -0.4043922174695636, 0.2200152897850518),
  Point<3>(0.139802769909557, -0.1398027699095571, 0.2262354331514884),
  Point<3>(0.1398027699095568, 0.139802769909557, 0.2262354331514884),
  Point<3>(0.1422860500856744, 0.4043922174695637, 0.2200152897850518),
  Point<3>(0.1479196123854722, 0.6283705083530853, 0.2072181049435331),
  Point<3>(0.1602884034118262, 0.81286548529772, 0.18713451470228),
  Point<3>(0.4618371753733093, -0.8206123917911029, 0.179387608208897),
  Point<3>(0.4280798211112203, -0.6395441015758374, 0.2005389272308206),
  Point<3>(0.4117367501948078, -0.4117367501948074, 0.2137105591038228),
  Point<3>(0.4043922174695636, -0.1422860500856745, 0.2200152897850518),
  Point<3>(0.4043922174695637, 0.1422860500856745, 0.2200152897850519),
  Point<3>(0.4117367501948074, 0.4117367501948078, 0.2137105591038228),
  Point<3>(0.42807982111122, 0.6395441015758372, 0.2005389272308207),
  Point<3>(0.4618371753733093, 0.8206123917911028, 0.1793876082088969),
  Point<3>(0.7083095761349704, -0.8381312808298292, 0.1618687191701718),
  Point<3>(0.6637161946749365, -0.6637161946749367, 0.1861190621843338),
  Point<3>(0.6395441015758371, -0.4280798211112197, 0.2005389272308205),
  Point<3>(0.6283705083530855, -0.1479196123854727, 0.2072181049435331),
  Point<3>(0.6283705083530853, 0.1479196123854719, 0.2072181049435332),
  Point<3>(0.6395441015758369, 0.4280798211112199, 0.2005389272308205),
  Point<3>(0.6637161946749367, 0.663716194674936, 0.1861190621843338),
  Point<3>(0.7083095761349696, 0.8381312808298282, 0.1618687191701716),
  Point<3>(0.8693869325527522, -0.8693869325527523, 0.1306130674472477),
  Point<3>(0.8381312808298281, -0.7083095761349696, 0.1618687191701717),
  Point<3>(0.8206123917911032, -0.4618371753733095, 0.179387608208897),
  Point<3>(0.8128654852977202, -0.1602884034118259, 0.1871345147022801),
  Point<3>(0.8128654852977201, 0.1602884034118263, 0.1871345147022801),
  Point<3>(0.820612391791103, 0.4618371753733094, 0.1793876082088971),
  Point<3>(0.8381312808298288, 0.7083095761349697, 0.1618687191701718),
  Point<3>(0.8693869325527526, 0.8693869325527522, 0.1306130674472476),
  Point<3>(-0.7389624749052222, -0.7389624749052222, 0.2610375250947776),
  Point<3>(-0.6963389616557185, -0.5579732207608954, 0.303661038344281),
  Point<3>(-0.6737114590570532, -0.2994424296524927, 0.326288540942947),
  Point<3>(-0.6666666666666663, 5.551115123125783e-17, 0.3333333333333334),
  Point<3>(-0.6737114590570532, 0.2994424296524926, 0.3262885409429472),
  Point<3>(-0.696338961655719, 0.5579732207608964, 0.3036610383442811),
  Point<3>(-0.7389624749052218, 0.7389624749052222, 0.2610375250947776),
  Point<3>(-0.5579732207608957, -0.6963389616557187, 0.303661038344281),
  Point<3>(-0.5078379693244461, -0.5078379693244462, 0.3342029313996063),
  Point<3>(-0.4814079243613797, -0.2687940777441542, 0.351140249518656),
  Point<3>(-0.472974349613441, -5.551115123125783e-17, 0.3565526072678357),
  Point<3>(-0.48140792436138, 0.2687940777441541, 0.3511402495186562),
  Point<3>(-0.5078379693244471, 0.5078379693244466, 0.3342029313996063),
  Point<3>(-0.5579732207608961, 0.6963389616557186, 0.3036610383442813),
  Point<3>(-0.2994424296524927, -0.6737114590570533, 0.3262885409429468),
  Point<3>(-0.2687940777441541, -0.4814079243613799, 0.3511402495186559),
  Point<3>(-0.2536212374841474, -0.2536212374841473, 0.3653615438076029),
  Point<3>(-0.248839112161737, -1.804112415015879e-16, 0.3699864217972002),
  Point<3>(-0.2536212374841474, 0.2536212374841472, 0.3653615438076028),
  Point<3>(-0.268794077744154, 0.4814079243613797, 0.3511402495186561),
  Point<3>(-0.299442429652493, 0.6737114590570531, 0.3262885409429468),
  Point<3>(-5.551115123125783e-17, -0.6666666666666666, 0.3333333333333334),
  Point<3>(-8.326672684688674e-17, -0.4729743496134412, 0.3565526072678357),
  Point<3>(-1.387778780781446e-17, -0.2488391121617372, 0.3699864217972004),
  Point<3>(7.163179520108224e-17, -3.564758173066946e-17, 0.3743811289491724),
  Point<3>(9.71445146547012e-17, 0.2488391121617372, 0.3699864217972001),
  Point<3>(1.387778780781446e-16, 0.4729743496134414, 0.3565526072678357),
  Point<3>(0, 0.6666666666666667, 0.3333333333333335),
  Point<3>(0.2994424296524928, -0.6737114590570535, 0.326288540942947),
  Point<3>(0.2687940777441544, -0.48140792436138, 0.351140249518656),
  Point<3>(0.2536212374841473, -0.2536212374841474, 0.3653615438076029),
  Point<3>(0.2488391121617371, -5.551115123125783e-17, 0.3699864217972003),
  Point<3>(0.2536212374841474, 0.2536212374841476, 0.3653615438076029),
  Point<3>(0.2687940777441538, 0.48140792436138, 0.351140249518656),
  Point<3>(0.2994424296524925, 0.6737114590570535, 0.3262885409429469),
  Point<3>(0.5579732207608959, -0.6963389616557187, 0.3036610383442811),
  Point<3>(0.5078379693244467, -0.5078379693244467, 0.3342029313996063),
  Point<3>(0.4814079243613797, -0.2687940777441541, 0.3511402495186559),
  Point<3>(0.4729743496134412, -8.326672684688674e-17, 0.3565526072678358),
  Point<3>(0.4814079243613795, 0.2687940777441539, 0.351140249518656),
  Point<3>(0.5078379693244469, 0.507837969324446, 0.3342029313996064),
  Point<3>(0.5579732207608961, 0.6963389616557192, 0.303661038344281),
  Point<3>(0.7389624749052219, -0.738962474905222, 0.2610375250947776),
  Point<3>(0.6963389616557194, -0.5579732207608964, 0.3036610383442812),
  Point<3>(0.6737114590570534, -0.2994424296524925, 0.3262885409429469),
  Point<3>(0.6666666666666665, 0, 0.3333333333333333),
  Point<3>(0.6737114590570532, 0.2994424296524929, 0.3262885409429469),
  Point<3>(0.696338961655719, 0.5579732207608958, 0.3036610383442813),
  Point<3>(0.7389624749052223, 0.7389624749052222, 0.2610375250947776),
  Point<3>(-0.5826394788331937, -0.5826394788331936, 0.4173605211668066),
  Point<3>(-0.5352173786266061, -0.3943478641201834, 0.4647826213733946),
  Point<3>(-0.5134230556452268, -0.1391540262406669, 0.4865769443547729),
  Point<3>(-0.5134230556452271, 0.1391540262406671, 0.486576944354773),
  Point<3>(-0.5352173786266053, 0.3943478641201832, 0.4647826213733945),
  Point<3>(-0.5826394788331933, 0.5826394788331931, 0.4173605211668067),
  Point<3>(-0.3943478641201834, -0.5352173786266051, 0.4647826213733945),
  Point<3>(-0.347408637824689, -0.3474086378246891, 0.4956579035173297),
  Point<3>(-0.3257528276758783, -0.1207195775357102, 0.5103201837773572),
  Point<3>(-0.3257528276758789, 0.1207195775357098, 0.510320183777357),
  Point<3>(-0.3474086378246892, 0.3474086378246886, 0.49565790351733),
  Point<3>(-0.3943478641201835, 0.5352173786266057, 0.4647826213733949),
  Point<3>(-0.1391540262406669, -0.5134230556452272, 0.4865769443547728),
  Point<3>(-0.1207195775357098, -0.3257528276758785, 0.5103201837773571),
  Point<3>(-0.1124769562440036, -0.1124769562440039, 0.5218220232063475),
  Point<3>(-0.1124769562440039, 0.1124769562440037, 0.5218220232063475),
  Point<3>(-0.1207195775357102, 0.3257528276758783, 0.5103201837773569),
  Point<3>(-0.1391540262406668, 0.5134230556452267, 0.4865769443547729),
  Point<3>(0.1391540262406668, -0.5134230556452273, 0.4865769443547731),
  Point<3>(0.12071957753571, -0.3257528276758786, 0.5103201837773574),
  Point<3>(0.1124769562440039, -0.1124769562440037, 0.5218220232063474),
  Point<3>(0.1124769562440038, 0.1124769562440041, 0.5218220232063475),
  Point<3>(0.12071957753571, 0.3257528276758787, 0.5103201837773569),
  Point<3>(0.1391540262406669, 0.5134230556452274, 0.4865769443547728),
  Point<3>(0.3943478641201833, -0.5352173786266055, 0.4647826213733944),
  Point<3>(0.3474086378246891, -0.3474086378246891, 0.4956579035173299),
  Point<3>(0.3257528276758786, -0.1207195775357097, 0.5103201837773572),
  Point<3>(0.3257528276758788, 0.1207195775357101, 0.5103201837773566),
  Point<3>(0.3474086378246891, 0.3474086378246891, 0.4956579035173299),
  Point<3>(0.3943478641201839, 0.5352173786266048, 0.4647826213733947),
  Point<3>(0.5826394788331933, -0.5826394788331933, 0.4173605211668067),
  Point<3>(0.5352173786266052, -0.3943478641201837, 0.4647826213733945),
  Point<3>(0.5134230556452268, -0.139154026240667, 0.486576944354773),
  Point<3>(0.5134230556452273, 0.1391540262406669, 0.4865769443547728),
  Point<3>(0.5352173786266056, 0.3943478641201839, 0.4647826213733945),
  Point<3>(0.5826394788331938, 0.5826394788331937, 0.4173605211668066),
  Point<3>(-0.4173605211668065, -0.4173605211668064, 0.5826394788331934),
  Point<3>(-0.3728439087916925, -0.2344781678968695, 0.627156091208307),
  Point<3>(-0.358775216417794, -5.551115123125783e-17, 0.6412247835822065),
  Point<3>(-0.3728439087916928, 0.2344781678968694, 0.627156091208307),
  Point<3>(-0.4173605211668066, 0.4173605211668066, 0.5826394788331936),
  Point<3>(-0.2344781678968693, -0.3728439087916928, 0.6271560912083068),
  Point<3>(-0.199111861538122, -0.1991118615381226, 0.652734466508338),
  Point<3>(-0.1878636523452939, -1.387778780781446e-16, 0.6610097133750662),
  Point<3>(-0.1991118615381226, 0.1991118615381224, 0.6527344665083377),
  Point<3>(-0.2344781678968695, 0.3728439087916925, 0.6271560912083071),
  Point<3>(-5.551115123125783e-17, -0.3587752164177939, 0.6412247835822064),
  Point<3>(2.081668171172169e-16, -0.1878636523452939, 0.6610097133750656),
  Point<3>(-2.765273360119211e-17, 5.910603182628874e-17, 0.667472252409014),
  Point<3>(-1.52655665885959e-16, 0.187863652345294, 0.6610097133750659),
  Point<3>(-2.775557561562891e-17, 0.3587752164177939, 0.6412247835822064),
  Point<3>(0.2344781678968692, -0.3728439087916928, 0.6271560912083068),
  Point<3>(0.1991118615381224, -0.1991118615381223, 0.6527344665083379),
  Point<3>(0.1878636523452939, 2.636779683484747e-16, 0.6610097133750662),
  Point<3>(0.1991118615381222, 0.1991118615381227, 0.6527344665083377),
  Point<3>(0.2344781678968694, 0.3728439087916927, 0.6271560912083074),
  Point<3>(0.4173605211668064, -0.4173605211668066, 0.5826394788331936),
  Point<3>(0.3728439087916927, -0.2344781678968695, 0.627156091208307),
  Point<3>(0.3587752164177941, -1.110223024625157e-16, 0.6412247835822062),
  Point<3>(0.3728439087916923, 0.2344781678968694, 0.6271560912083071),
  Point<3>(0.4173605211668066, 0.4173605211668065, 0.5826394788331936),
  Point<3>(-0.2610375250947778, -0.2610375250947778, 0.7389624749052218),
  Point<3>(-0.2267795715176013, -0.09695786682274236, 0.7732204284823994),
  Point<3>(-0.2267795715176007, 0.09695786682274243, 0.7732204284823987),
  Point<3>(-0.2610375250947778, 0.2610375250947781, 0.7389624749052219),
  Point<3>(-0.09695786682274245, -0.226779571517601, 0.7732204284823987),
  Point<3>(-0.07933985399092289, -0.0793398539909223, 0.7898814182506434),
  Point<3>(-0.07933985399092244, 0.07933985399092268, 0.7898814182506428),
  Point<3>(-0.09695786682274229, 0.226779571517601, 0.7732204284823988),
  Point<3>(0.09695786682274238, -0.2267795715176009, 0.7732204284823992),
  Point<3>(0.07933985399092237, -0.07933985399092247, 0.789881418250643),
  Point<3>(0.07933985399092269, 0.07933985399092243, 0.7898814182506426),
  Point<3>(0.09695786682274232, 0.226779571517601, 0.7732204284823988),
  Point<3>(0.2610375250947778, -0.2610375250947781, 0.7389624749052219),
  Point<3>(0.226779571517601, -0.09695786682274235, 0.7732204284823986),
  Point<3>(0.226779571517601, 0.09695786682274217, 0.7732204284823991),
  Point<3>(0.2610375250947781, 0.2610375250947778, 0.738962474905222),
  Point<3>(-0.1306130674472477, -0.1306130674472478, 0.8693869325527525),
  Point<3>(-0.1121787123338178, 1.942890293094024e-16, 0.8878212876661824),
  Point<3>(-0.1306130674472475, 0.1306130674472477, 0.8693869325527525),
  Point<3>(1.52655665885959e-16, -0.1121787123338178, 0.8878212876661824),
  Point<3>(1.130276386076478e-16, 2.045464272006178e-16, 0.8956384954917328),
  Point<3>(-1.665334536937735e-16, 0.1121787123338179, 0.8878212876661823),
  Point<3>(0.1306130674472475, -0.1306130674472477, 0.8693869325527526),
  Point<3>(0.1121787123338179, 1.665334536937735e-16, 0.887821287666182),
  Point<3>(0.1306130674472477, 0.1306130674472476, 0.8693869325527525),
  Point<3>(-0.04023304591677087, -0.04023304591677065, 0.9597669540832294),
  Point<3>(-0.0402330459167706, 0.04023304591677079, 0.9597669540832293),
  Point<3>(0.04023304591677057, -0.04023304591677083, 0.9597669540832293),
  Point<3>(0.0402330459167708, 0.04023304591677059, 0.9597669540832293),
  Point<3>(0, -1.110223024625157e-16, 0.9999999999999999)};
std::vector<Point<3>> reference_points_p10 = {
  Point<3>(-1, -1, 0),
  Point<3>(-1, -0.9340014304080593, 0),
  Point<3>(-1, -0.7844834736631444, 0),
  Point<3>(-1, -0.565235326996205, 0),
  Point<3>(-1, -0.2957581355869395, 0),
  Point<3>(-1, 2.775557561562891e-16, 0),
  Point<3>(-1, 0.2957581355869393, 0),
  Point<3>(-1, 0.565235326996205, 0),
  Point<3>(-1, 0.7844834736631445, 0),
  Point<3>(-1, 0.9340014304080594, 0),
  Point<3>(-1, 1, 0),
  Point<3>(-0.9340014304080593, -0.9999999999999999, 0),
  Point<3>(-0.9340014304080595, -0.9340014304080595, 0),
  Point<3>(-0.9340014304080595, -0.7844834736631444, 0),
  Point<3>(-0.9340014304080597, -0.5652353269962047, 0),
  Point<3>(-0.9340014304080593, -0.2957581355869399, 0),
  Point<3>(-0.9340014304080593, 3.157951595554514e-16, 0),
  Point<3>(-0.9340014304080592, 0.2957581355869394, 0),
  Point<3>(-0.9340014304080594, 0.5652353269962052, 0),
  Point<3>(-0.9340014304080594, 0.784483473663144, 0),
  Point<3>(-0.9340014304080594, 0.9340014304080594, 0),
  Point<3>(-0.9340014304080595, 0.9999999999999999, 0),
  Point<3>(-0.7844834736631443, -1, 0),
  Point<3>(-0.7844834736631443, -0.9340014304080593, 0),
  Point<3>(-0.7844834736631441, -0.7844834736631445, 0),
  Point<3>(-0.7844834736631437, -0.565235326996205, 0),
  Point<3>(-0.7844834736631443, -0.2957581355869393, 0),
  Point<3>(-0.7844834736631441, 1.720491695955389e-16, 0),
  Point<3>(-0.7844834736631442, 0.2957581355869391, 0),
  Point<3>(-0.784483473663144, 0.5652353269962049, 0),
  Point<3>(-0.7844834736631446, 0.7844834736631442, 0),
  Point<3>(-0.7844834736631442, 0.9340014304080593, 0),
  Point<3>(-0.7844834736631443, 1, 0),
  Point<3>(-0.565235326996205, -1, 0),
  Point<3>(-0.565235326996205, -0.9340014304080595, 0),
  Point<3>(-0.5652353269962047, -0.7844834736631444, 0),
  Point<3>(-0.5652353269962053, -0.565235326996205, 0),
  Point<3>(-0.5652353269962049, -0.2957581355869395, 0),
  Point<3>(-0.5652353269962049, 2.241734117992291e-16, 0),
  Point<3>(-0.5652353269962052, 0.2957581355869391, 0),
  Point<3>(-0.5652353269962049, 0.5652353269962053, 0),
  Point<3>(-0.5652353269962048, 0.7844834736631443, 0),
  Point<3>(-0.5652353269962054, 0.9340014304080596, 0),
  Point<3>(-0.565235326996205, 1, 0),
  Point<3>(-0.2957581355869396, -1, 0),
  Point<3>(-0.2957581355869396, -0.9340014304080594, 0),
  Point<3>(-0.2957581355869394, -0.7844834736631445, 0),
  Point<3>(-0.2957581355869394, -0.5652353269962052, 0),
  Point<3>(-0.2957581355869394, -0.2957581355869395, 0),
  Point<3>(-0.2957581355869394, 1.88407419567537e-16, 0),
  Point<3>(-0.2957581355869393, 0.2957581355869391, 0),
  Point<3>(-0.2957581355869394, 0.565235326996205, 0),
  Point<3>(-0.2957581355869393, 0.7844834736631441, 0),
  Point<3>(-0.2957581355869396, 0.9340014304080593, 0),
  Point<3>(-0.2957581355869395, 0.9999999999999998, 0),
  Point<3>(2.775557561562891e-16, -1, 0),
  Point<3>(3.309742231262461e-16, -0.9340014304080593, 0),
  Point<3>(3.547469628237284e-16, -0.7844834736631443, 0),
  Point<3>(2.35807868407152e-16, -0.565235326996205, 0),
  Point<3>(2.480867203551875e-16, -0.2957581355869395, 0),
  Point<3>(2.483325705233218e-16, 2.48332573508087e-16, 0),
  Point<3>(2.389252345464417e-16, 0.2957581355869392, 0),
  Point<3>(2.679648691847335e-16, 0.5652353269962049, 0),
  Point<3>(2.435908440982067e-16, 0.7844834736631444, 0),
  Point<3>(3.837894678390506e-16, 0.9340014304080594, 0),
  Point<3>(2.775557561562891e-16, 0.9999999999999999, 0),
  Point<3>(0.2957581355869393, -1, 0),
  Point<3>(0.2957581355869393, -0.9340014304080595, 0),
  Point<3>(0.2957581355869391, -0.7844834736631442, 0),
  Point<3>(0.2957581355869392, -0.5652353269962052, 0),
  Point<3>(0.2957581355869391, -0.2957581355869394, 0),
  Point<3>(0.2957581355869392, 1.991878201452633e-16, 0),
  Point<3>(0.2957581355869393, 0.295758135586939, 0),
  Point<3>(0.2957581355869393, 0.5652353269962049, 0),
  Point<3>(0.2957581355869391, 0.7844834736631444, 0),
  Point<3>(0.2957581355869394, 0.9340014304080593, 0),
  Point<3>(0.2957581355869391, 1, 0),
  Point<3>(0.5652353269962049, -1, 0),
  Point<3>(0.5652353269962053, -0.9340014304080595, 0),
  Point<3>(0.5652353269962049, -0.7844834736631443, 0),
  Point<3>(0.5652353269962053, -0.565235326996205, 0),
  Point<3>(0.5652353269962048, -0.2957581355869395, 0),
  Point<3>(0.5652353269962052, 2.078821809730752e-16, 0),
  Point<3>(0.5652353269962049, 0.2957581355869393, 0),
  Point<3>(0.565235326996205, 0.5652353269962053, 0),
  Point<3>(0.5652353269962047, 0.7844834736631445, 0),
  Point<3>(0.5652353269962052, 0.9340014304080596, 0),
  Point<3>(0.5652353269962052, 1, 0),
  Point<3>(0.7844834736631444, -1, 0),
  Point<3>(0.7844834736631446, -0.9340014304080593, 0),
  Point<3>(0.7844834736631444, -0.784483473663144, 0),
  Point<3>(0.7844834736631445, -0.5652353269962047, 0),
  Point<3>(0.7844834736631443, -0.2957581355869394, 0),
  Point<3>(0.7844834736631443, 1.073333028157641e-16, 0),
  Point<3>(0.7844834736631445, 0.2957581355869391, 0),
  Point<3>(0.7844834736631441, 0.5652353269962049, 0),
  Point<3>(0.7844834736631439, 0.7844834736631444, 0),
  Point<3>(0.7844834736631443, 0.9340014304080593, 0),
  Point<3>(0.7844834736631446, 1, 0),
  Point<3>(0.9340014304080592, -1, 0),
  Point<3>(0.9340014304080591, -0.9340014304080594, 0),
  Point<3>(0.9340014304080594, -0.7844834736631441, 0),
  Point<3>(0.9340014304080591, -0.5652353269962052, 0),
  Point<3>(0.9340014304080592, -0.2957581355869399, 0),
  Point<3>(0.9340014304080597, 3.7252496600685e-16, 0),
  Point<3>(0.9340014304080593, 0.2957581355869394, 0),
  Point<3>(0.9340014304080595, 0.5652353269962052, 0),
  Point<3>(0.9340014304080591, 0.7844834736631443, 0),
  Point<3>(0.9340014304080592, 0.9340014304080589, 0),
  Point<3>(0.9340014304080594, 1, 0),
  Point<3>(1, -1, 0),
  Point<3>(1, -0.9340014304080595, 0),
  Point<3>(1, -0.7844834736631444, 0),
  Point<3>(1, -0.565235326996205, 0),
  Point<3>(1, -0.2957581355869395, 0),
  Point<3>(1, 2.775557561562891e-16, 0),
  Point<3>(1, 0.2957581355869391, 0),
  Point<3>(1, 0.565235326996205, 0),
  Point<3>(1, 0.7844834736631444, 0),
  Point<3>(1, 0.9340014304080592, 0),
  Point<3>(1, 1, 0),
  Point<3>(-0.9670007152040293, -0.9670007152040293, 0.03299928479597047),
  Point<3>(-0.9536497634886774, -0.8609492904660306, 0.04635023651132298),
  Point<3>(-0.9455066917551167, -0.6770289966665577, 0.05449330824488349),
  Point<3>(-0.9409847818890326, -0.4321839884663217, 0.05901521811096733),
  Point<3>(-0.9389992392541798, -0.148505921549046, 0.06100076074582058),
  Point<3>(-0.9389992392541797, 0.1485059215490455, 0.06100076074582064),
  Point<3>(-0.940984781889033, 0.4321839884663231, 0.05901521811096727),
  Point<3>(-0.9455066917551167, 0.6770289966665569, 0.05449330824488349),
  Point<3>(-0.9536497634886765, 0.8609492904660305, 0.04635023651132311),
  Point<3>(-0.9670007152040294, 0.9670007152040294, 0.0329992847959705),
  Point<3>(-0.8609492904660311, -0.953649763488677, 0.046350236511323),
  Point<3>(-0.8348192719106967, -0.8348192719106963, 0.05764277468139564),
  Point<3>(-0.8208266229430708, -0.6524236261817274, 0.06497227188761426),
  Point<3>(-0.8125599693683694, -0.4153075357077699, 0.0693075906009863),
  Point<3>(-0.8085964174034062, -0.1424585521153158, 0.07130826550899533),
  Point<3>(-0.808596417403406, 0.1424585521153152, 0.07130826550899536),
  Point<3>(-0.8125599693683694, 0.4153075357077702, 0.06930759060098633),
  Point<3>(-0.82082662294307, 0.6524236261817281, 0.06497227188761402),
  Point<3>(-0.8348192719106952, 0.8348192719106954, 0.0576427746813953),
  Point<3>(-0.8609492904660307, 0.9536497634886765, 0.04635023651132309),
  Point<3>(-0.6770289966665578, -0.945506691755117, 0.05449330824488348),
  Point<3>(-0.6524236261817283, -0.8208266229430711, 0.06497227188761415),
  Point<3>(-0.6416117721420372, -0.641611772142038, 0.07214453264102735),
  Point<3>(-0.6355772546376914, -0.4089591624422896, 0.07659511261964652),
  Point<3>(-0.6326830372027442, -0.1403083731342787, 0.07872035160438193),
  Point<3>(-0.6326830372027442, 0.1403083731342782, 0.07872035160438198),
  Point<3>(-0.6355772546376917, 0.4089591624422892, 0.07659511261964658),
  Point<3>(-0.6416117721420375, 0.6416117721420382, 0.07214453264102735),
  Point<3>(-0.6524236261817274, 0.8208266229430705, 0.06497227188761406),
  Point<3>(-0.6770289966665575, 0.9455066917551164, 0.05449330824488349),
  Point<3>(-0.4321839884663222, -0.9409847818890332, 0.05901521811096733),
  Point<3>(-0.4153075357077704, -0.8125599693683695, 0.06930759060098628),
  Point<3>(-0.4089591624422889, -0.6355772546376923, 0.07659511261964652),
  Point<3>(-0.4056374119381718, -0.4056374119381722, 0.08124616907210058),
  Point<3>(-0.4040551273202588, -0.1392115818349581, 0.08350916804207104),
  Point<3>(-0.4040551273202589, 0.139211581834958, 0.08350916804207111),
  Point<3>(-0.4056374119381718, 0.4056374119381713, 0.08124616907210061),
  Point<3>(-0.4089591624422893, 0.6355772546376921, 0.07659511261964654),
  Point<3>(-0.4153075357077703, 0.8125599693683694, 0.06930759060098629),
  Point<3>(-0.4321839884663232, 0.940984781889033, 0.05901521811096729),
  Point<3>(-0.1485059215490457, -0.9389992392541798, 0.0610007607458206),
  Point<3>(-0.1424585521153159, -0.8085964174034065, 0.07130826550899523),
  Point<3>(-0.1403083731342783, -0.6326830372027442, 0.07872035160438189),
  Point<3>(-0.1392115818349581, -0.4040551273202589, 0.08350916804207106),
  Point<3>(-0.1386898993637149, -0.138689899363715, 0.08585762575389355),
  Point<3>(-0.1386898993637149, 0.1386898993637151, 0.08585762575389358),
  Point<3>(-0.1392115818349579, 0.4040551273202586, 0.08350916804207109),
  Point<3>(-0.1403083731342785, 0.6326830372027443, 0.07872035160438197),
  Point<3>(-0.1424585521153153, 0.8085964174034056, 0.07130826550899531),
  Point<3>(-0.1485059215490454, 0.9389992392541795, 0.06100076074582059),
  Point<3>(0.1485059215490454, -0.9389992392541799, 0.06100076074582052),
  Point<3>(0.142458552115316, -0.808596417403406, 0.07130826550899516),
  Point<3>(0.1403083731342785, -0.6326830372027438, 0.07872035160438183),
  Point<3>(0.1392115818349581, -0.4040551273202588, 0.08350916804207101),
  Point<3>(0.1386898993637152, -0.138689899363715, 0.08585762575389354),
  Point<3>(0.1386898993637153, 0.1386898993637151, 0.08585762575389345),
  Point<3>(0.1392115818349586, 0.4040551273202585, 0.08350916804207105),
  Point<3>(0.1403083731342787, 0.6326830372027441, 0.07872035160438196),
  Point<3>(0.1424585521153161, 0.8085964174034057, 0.07130826550899536),
  Point<3>(0.1485059215490458, 0.9389992392541792, 0.06100076074582061),
  Point<3>(0.4321839884663229, -0.9409847818890332, 0.05901521811096726),
  Point<3>(0.4153075357077706, -0.8125599693683696, 0.06930759060098626),
  Point<3>(0.4089591624422891, -0.6355772546376917, 0.07659511261964647),
  Point<3>(0.4056374119381717, -0.405637411938172, 0.08124616907210053),
  Point<3>(0.4040551273202587, -0.139211581834958, 0.08350916804207105),
  Point<3>(0.4040551273202589, 0.1392115818349581, 0.08350916804207108),
  Point<3>(0.4056374119381715, 0.4056374119381717, 0.08124616907210054),
  Point<3>(0.4089591624422892, 0.6355772546376918, 0.0765951126196466),
  Point<3>(0.4153075357077705, 0.8125599693683685, 0.06930759060098636),
  Point<3>(0.4321839884663223, 0.9409847818890329, 0.05901521811096731),
  Point<3>(0.6770289966665575, -0.9455066917551165, 0.05449330824488347),
  Point<3>(0.6524236261817281, -0.8208266229430708, 0.06497227188761411),
  Point<3>(0.6416117721420382, -0.6416117721420376, 0.07214453264102735),
  Point<3>(0.6355772546376914, -0.4089591624422893, 0.07659511261964651),
  Point<3>(0.6326830372027441, -0.1403083731342786, 0.07872035160438198),
  Point<3>(0.6326830372027441, 0.1403083731342787, 0.07872035160438197),
  Point<3>(0.6355772546376914, 0.4089591624422892, 0.07659511261964659),
  Point<3>(0.6416117721420379, 0.6416117721420381, 0.07214453264102737),
  Point<3>(0.6524236261817278, 0.8208266229430704, 0.06497227188761417),
  Point<3>(0.6770289966665577, 0.9455066917551165, 0.05449330824488351),
  Point<3>(0.8609492904660303, -0.9536497634886762, 0.04635023651132313),
  Point<3>(0.8348192719106957, -0.8348192719106953, 0.05764277468139547),
  Point<3>(0.8208266229430707, -0.6524236261817273, 0.06497227188761419),
  Point<3>(0.8125599693683691, -0.41530753570777, 0.06930759060098636),
  Point<3>(0.8085964174034055, -0.1424585521153156, 0.07130826550899538),
  Point<3>(0.8085964174034053, 0.1424585521153151, 0.07130826550899544),
  Point<3>(0.8125599693683688, 0.4153075357077701, 0.06930759060098643),
  Point<3>(0.82082662294307, 0.6524236261817282, 0.06497227188761413),
  Point<3>(0.8348192719106956, 0.8348192719106956, 0.05764277468139546),
  Point<3>(0.860949290466031, 0.9536497634886766, 0.04635023651132297),
  Point<3>(0.9670007152040294, -0.9670007152040295, 0.03299928479597054),
  Point<3>(0.9536497634886769, -0.8609492904660311, 0.04635023651132294),
  Point<3>(0.9455066917551171, -0.6770289966665575, 0.0544933082448835),
  Point<3>(0.9409847818890321, -0.4321839884663219, 0.05901521811096733),
  Point<3>(0.9389992392541794, -0.1485059215490458, 0.06100076074582052),
  Point<3>(0.9389992392541797, 0.1485059215490454, 0.06100076074582057),
  Point<3>(0.9409847818890323, 0.4321839884663231, 0.05901521811096727),
  Point<3>(0.9455066917551167, 0.6770289966665569, 0.05449330824488346),
  Point<3>(0.9536497634886766, 0.8609492904660306, 0.04635023651132317),
  Point<3>(0.9670007152040293, 0.9670007152040294, 0.03299928479597045),
  Point<3>(-0.892241736831572, -0.8922417368315722, 0.1077582631684278),
  Point<3>(-0.8657611524557207, -0.7567745359659537, 0.1342388475442794),
  Point<3>(-0.8495014227369213, -0.5485042682107631, 0.1504985772630792),
  Point<3>(-0.8408260228743198, -0.2879239295745405, 0.1591739771256801),
  Point<3>(-0.8381161726679026, -1.75806842545452e-16, 0.1618838273320974),
  Point<3>(-0.8408260228743198, 0.287923929574541, 0.1591739771256802),
  Point<3>(-0.8495014227369204, 0.5485042682107637, 0.1504985772630793),
  Point<3>(-0.8657611524557211, 0.756774535965954, 0.1342388475442794),
  Point<3>(-0.8922417368315724, 0.8922417368315723, 0.1077582631684278),
  Point<3>(-0.7567745359659546, -0.8657611524557205, 0.1342388475442796),
  Point<3>(-0.7170778765710051, -0.7170778765710055, 0.1554315406229079),
  Point<3>(-0.6939387666501893, -0.5146219730406245, 0.1690520204229368),
  Point<3>(-0.6811380611452986, -0.2688508273071741, 0.1766092493436362),
  Point<3>(-0.6770159344504941, -8.210120951146482e-17, 0.1790249873255169),
  Point<3>(-0.6811380611452981, 0.2688508273071743, 0.1766092493436363),
  Point<3>(-0.6939387666501892, 0.5146219730406246, 0.169052020422937),
  Point<3>(-0.7170778765710055, 0.7170778765710052, 0.1554315406229081),
  Point<3>(-0.7567745359659538, 0.8657611524557211, 0.1342388475442793),
  Point<3>(-0.5485042682107633, -0.8495014227369215, 0.150498577263079),
  Point<3>(-0.5146219730406246, -0.6939387666501897, 0.1690520204229367),
  Point<3>(-0.4971836887190739, -0.4971836887190743, 0.1814398555732853),
  Point<3>(-0.4878609630568127, -0.2597830903273671, 0.1885233918268277),
  Point<3>(-0.4848872674634355, -7.41594285980085e-17, 0.190826042803139),
  Point<3>(-0.4878609630568123, 0.2597830903273672, 0.1885233918268277),
  Point<3>(-0.4971836887190741, 0.4971836887190744, 0.1814398555732855),
  Point<3>(-0.5146219730406247, 0.6939387666501893, 0.169052020422937),
  Point<3>(-0.5485042682107635, 0.8495014227369212, 0.1504985772630792),
  Point<3>(-0.2879239295745404, -0.8408260228743206, 0.1591739771256802),
  Point<3>(-0.2688508273071742, -0.6811380611452992, 0.1766092493436362),
  Point<3>(-0.2597830903273673, -0.487860963056813, 0.1885233918268275),
  Point<3>(-0.2550685154622723, -0.2550685154622723, 0.1954546995530403),
  Point<3>(-0.2535786007138918, -2.376435636816665e-17, 0.1977288059155367),
  Point<3>(-0.255068515462272, 0.255068515462272, 0.1954546995530405),
  Point<3>(-0.2597830903273671, 0.4878609630568126, 0.1885233918268278),
  Point<3>(-0.2688508273071744, 0.6811380611452984, 0.1766092493436364),
  Point<3>(-0.287923929574541, 0.8408260228743197, 0.1591739771256802),
  Point<3>(-2.22152102294247e-16, -0.8381161726679025, 0.1618838273320974),
  Point<3>(-2.524022657546254e-16, -0.6770159344504943, 0.179024987325517),
  Point<3>(-1.092062638236024e-16, -0.4848872674634356, 0.1908260428031389),
  Point<3>(9.025983085941824e-18, -0.2535786007138918, 0.1977288059155367),
  Point<3>(1.517423079062763e-16, 1.207856169500469e-16, 0.2000000000020537),
  Point<3>(1.868690206914547e-16, 0.2535786007138919, 0.1977288059155367),
  Point<3>(3.536667486647715e-16, 0.484887267463436, 0.1908260428031389),
  Point<3>(4.401047668661784e-16, 0.6770159344504949, 0.1790249873255169),
  Point<3>(2.354857143548386e-16, 0.8381161726679024, 0.1618838273320973),
  Point<3>(0.2879239295745409, -0.8408260228743198, 0.1591739771256802),
  Point<3>(0.2688508273071742, -0.6811380611452983, 0.1766092493436363),
  Point<3>(0.2597830903273669, -0.4878609630568124, 0.1885233918268278),
  Point<3>(0.2550685154622721, -0.2550685154622719, 0.1954546995530405),
  Point<3>(0.2535786007138919, 3.532466203229334e-17, 0.1977288059155368),
  Point<3>(0.2550685154622722, 0.2550685154622721, 0.1954546995530405),
  Point<3>(0.259783090327367, 0.4878609630568126, 0.1885233918268276),
  Point<3>(0.2688508273071743, 0.6811380611452985, 0.1766092493436361),
  Point<3>(0.2879239295745408, 0.8408260228743198, 0.15917397712568),
  Point<3>(0.5485042682107629, -0.8495014227369205, 0.1504985772630792),
  Point<3>(0.5146219730406241, -0.6939387666501886, 0.1690520204229369),
  Point<3>(0.4971836887190739, -0.497183688719074, 0.1814398555732853),
  Point<3>(0.4878609630568125, -0.2597830903273673, 0.1885233918268275),
  Point<3>(0.4848872674634362, 1.125943956126196e-16, 0.190826042803139),
  Point<3>(0.4878609630568126, 0.259783090327367, 0.1885233918268277),
  Point<3>(0.4971836887190741, 0.4971836887190741, 0.1814398555732855),
  Point<3>(0.5146219730406247, 0.6939387666501893, 0.1690520204229368),
  Point<3>(0.5485042682107633, 0.8495014227369213, 0.1504985772630791),
  Point<3>(0.7567745359659538, -0.8657611524557213, 0.1342388475442795),
  Point<3>(0.7170778765710051, -0.717077876571005, 0.155431540622908),
  Point<3>(0.6939387666501889, -0.5146219730406241, 0.1690520204229368),
  Point<3>(0.6811380611452986, -0.2688508273071743, 0.176609249343636),
  Point<3>(0.6770159344504949, -1.566672139241554e-16, 0.1790249873255169),
  Point<3>(0.6811380611452985, 0.268850827307174, 0.1766092493436363),
  Point<3>(0.6939387666501894, 0.5146219730406247, 0.169052020422937),
  Point<3>(0.7170778765710054, 0.7170778765710047, 0.1554315406229081),
  Point<3>(0.7567745359659533, 0.8657611524557206, 0.1342388475442794),
  Point<3>(0.8922417368315722, -0.8922417368315724, 0.107758263168428),
  Point<3>(0.8657611524557209, -0.7567745359659537, 0.1342388475442795),
  Point<3>(0.8495014227369213, -0.5485042682107628, 0.1504985772630791),
  Point<3>(0.8408260228743196, -0.2879239295745408, 0.1591739771256801),
  Point<3>(0.8381161726679026, -3.884705612498885e-16, 0.1618838273320973),
  Point<3>(0.8408260228743197, 0.287923929574541, 0.1591739771256802),
  Point<3>(0.849501422736921, 0.5485042682107637, 0.1504985772630792),
  Point<3>(0.8657611524557203, 0.7567745359659537, 0.1342388475442794),
  Point<3>(0.8922417368315724, 0.8922417368315723, 0.1077582631684278),
  Point<3>(-0.7826176634981021, -0.7826176634981022, 0.2173823365018975),
  Point<3>(-0.7455996032886441, -0.6275691670667103, 0.2544003967113548),
  Point<3>(-0.7235489533501106, -0.4052009990987503, 0.2764510466498895),
  Point<3>(-0.7133236888145204, -0.1399710664435604, 0.2866763111854798),
  Point<3>(-0.7133236888145194, 0.1399710664435603, 0.2866763111854796),
  Point<3>(-0.7235489533501109, 0.4052009990987504, 0.2764510466498895),
  Point<3>(-0.7455996032886454, 0.6275691670667101, 0.2544003967113552),
  Point<3>(-0.7826176634981025, 0.7826176634981021, 0.2173823365018974),
  Point<3>(-0.6275691670667104, -0.7455996032886454, 0.2544003967113548),
  Point<3>(-0.581236118459755, -0.581236118459755, 0.2822962556724097),
  Point<3>(-0.5541360335021736, -0.3706375635197582, 0.2995127272339958),
  Point<3>(-0.5413197341567069, -0.1273440846685832, 0.3077022762814914),
  Point<3>(-0.5413197341567061, 0.1273440846685833, 0.3077022762814918),
  Point<3>(-0.5541360335021736, 0.3706375635197586, 0.2995127272339955),
  Point<3>(-0.5812361184597549, 0.5812361184597554, 0.2822962556724095),
  Point<3>(-0.6275691670667102, 0.7455996032886446, 0.254400396711355),
  Point<3>(-0.4052009990987501, -0.7235489533501112, 0.2764510466498898),
  Point<3>(-0.3706375635197582, -0.5541360335021733, 0.2995127272339957),
  Point<3>(-0.3518700133874091, -0.3518700133874094, 0.3141496558253115),
  Point<3>(-0.3431900532687622, -0.1207474900449994, 0.3212453313494183),
  Point<3>(-0.3431900532687617, 0.1207474900449995, 0.3212453313494184),
  Point<3>(-0.3518700133874094, 0.351870013387409, 0.3141496558253115),
  Point<3>(-0.3706375635197586, 0.5541360335021736, 0.2995127272339955),
  Point<3>(-0.4052009990987505, 0.7235489533501105, 0.2764510466498895),
  Point<3>(-0.1399710664435601, -0.7133236888145205, 0.2866763111854796),
  Point<3>(-0.1273440846685831, -0.541319734156707, 0.3077022762814914),
  Point<3>(-0.1207474900449995, -0.343190053268762, 0.3212453313494183),
  Point<3>(-0.1177377835307758, -0.1177377835307757, 0.3278739991546057),
  Point<3>(-0.1177377835307757, 0.1177377835307757, 0.3278739991546061),
  Point<3>(-0.1207474900449992, 0.3431900532687619, 0.3212453313494185),
  Point<3>(-0.1273440846685834, 0.5413197341567066, 0.3077022762814915),
  Point<3>(-0.1399710664435602, 0.7133236888145202, 0.2866763111854797),
  Point<3>(0.1399710664435602, -0.7133236888145199, 0.2866763111854799),
  Point<3>(0.1273440846685833, -0.5413197341567065, 0.3077022762814915),
  Point<3>(0.1207474900449994, -0.343190053268762, 0.3212453313494184),
  Point<3>(0.1177377835307755, -0.1177377835307758, 0.3278739991546058),
  Point<3>(0.1177377835307755, 0.1177377835307756, 0.3278739991546055),
  Point<3>(0.1207474900449994, 0.3431900532687619, 0.3212453313494183),
  Point<3>(0.1273440846685837, 0.5413197341567066, 0.3077022762814912),
  Point<3>(0.1399710664435603, 0.7133236888145204, 0.2866763111854796),
  Point<3>(0.4052009990987504, -0.7235489533501104, 0.2764510466498895),
  Point<3>(0.3706375635197585, -0.5541360335021736, 0.2995127272339957),
  Point<3>(0.3518700133874089, -0.3518700133874093, 0.3141496558253118),
  Point<3>(0.343190053268762, -0.1207474900449996, 0.3212453313494185),
  Point<3>(0.3431900532687618, 0.1207474900449992, 0.3212453313494181),
  Point<3>(0.3518700133874091, 0.351870013387409, 0.3141496558253115),
  Point<3>(0.3706375635197584, 0.5541360335021732, 0.2995127272339954),
  Point<3>(0.4052009990987498, 0.7235489533501102, 0.2764510466498897),
  Point<3>(0.6275691670667104, -0.7455996032886449, 0.2544003967113551),
  Point<3>(0.581236118459755, -0.581236118459755, 0.2822962556724097),
  Point<3>(0.5541360335021733, -0.3706375635197582, 0.2995127272339957),
  Point<3>(0.5413197341567068, -0.1273440846685832, 0.3077022762814915),
  Point<3>(0.5413197341567068, 0.1273440846685832, 0.3077022762814917),
  Point<3>(0.5541360335021733, 0.3706375635197584, 0.2995127272339955),
  Point<3>(0.5812361184597556, 0.5812361184597546, 0.2822962556724098),
  Point<3>(0.6275691670667107, 0.7455996032886446, 0.2544003967113549),
  Point<3>(0.7826176634981022, -0.7826176634981024, 0.2173823365018974),
  Point<3>(0.7455996032886454, -0.627569167066711, 0.254400396711355),
  Point<3>(0.7235489533501104, -0.4052009990987499, 0.2764510466498895),
  Point<3>(0.7133236888145204, -0.1399710664435604, 0.2866763111854796),
  Point<3>(0.7133236888145205, 0.1399710664435601, 0.2866763111854799),
  Point<3>(0.723548953350111, 0.4052009990987501, 0.2764510466498899),
  Point<3>(0.7455996032886447, 0.6275691670667097, 0.254400396711355),
  Point<3>(0.7826176634981025, 0.7826176634981025, 0.2173823365018975),
  Point<3>(-0.6478790677934695, -0.6478790677934696, 0.3521209322065303),
  Point<3>(-0.604753341147433, -0.482751819655792, 0.3952466588525672),
  Point<3>(-0.5809419136660484, -0.257174259001854, 0.4190580863339515),
  Point<3>(-0.5733526223709596, 5.606081987803522e-17, 0.4266473776290403),
  Point<3>(-0.5809419136660486, 0.2571742590018538, 0.4190580863339515),
  Point<3>(-0.6047533411474332, 0.4827518196557919, 0.3952466588525665),
  Point<3>(-0.6478790677934696, 0.6478790677934696, 0.3521209322065305),
  Point<3>(-0.482751819655792, -0.6047533411474334, 0.3952466588525671),
  Point<3>(-0.4360103435771014, -0.4360103435771012, 0.4255394034597),
  Point<3>(-0.4103065821727676, -0.2289578427800015, 0.4427443854298677),
  Point<3>(-0.4020320197505845, -3.371868756429919e-17, 0.4483163854113367),
  Point<3>(-0.4103065821727678, 0.2289578427800019, 0.4427443854298674),
  Point<3>(-0.4360103435771015, 0.4360103435771019, 0.4255394034596994),
  Point<3>(-0.4827518196557919, 0.6047533411474337, 0.3952466588525667),
  Point<3>(-0.2571742590018538, -0.5809419136660485, 0.4190580863339516),
  Point<3>(-0.2289578427800014, -0.4103065821727676, 0.4427443854298679),
  Point<3>(-0.214091629771642, -0.2140916297716422, 0.4564711793272272),
  Point<3>(-0.2093682132578215, 6.581107186987012e-17, 0.4609656036122247),
  Point<3>(-0.214091629771642, 0.2140916297716421, 0.4564711793272272),
  Point<3>(-0.2289578427800018, 0.4103065821727678, 0.4427443854298674),
  Point<3>(-0.2571742590018538, 0.5809419136660486, 0.4190580863339513),
  Point<3>(1.110223024625157e-16, -0.5733526223709597, 0.4266473776290403),
  Point<3>(1.736891880321778e-16, -0.4020320197505848, 0.4483163854113368),
  Point<3>(-6.223320470066795e-17, -0.2093682132578217, 0.4609656036122247),
  Point<3>(-8.044780119842443e-17, -1.092875789865388e-16, 0.4651231554735576),
  Point<3>(8.630249292984615e-17, 0.2093682132578213, 0.4609656036122246),
  Point<3>(3.220080452281948e-17, 0.4020320197505842, 0.4483163854113364),
  Point<3>(9.508283033726312e-17, 0.5733526223709596, 0.4266473776290401),
  Point<3>(0.2571742590018537, -0.580941913666049, 0.4190580863339515),
  Point<3>(0.2289578427800018, -0.4103065821727678, 0.4427443854298677),
  Point<3>(0.2140916297716419, -0.2140916297716423, 0.4564711793272275),
  Point<3>(0.2093682132578213, -5.876375774871434e-17, 0.4609656036122246),
  Point<3>(0.2140916297716421, 0.2140916297716418, 0.4564711793272271),
  Point<3>(0.228957842780002, 0.4103065821727676, 0.4427443854298674),
  Point<3>(0.2571742590018538, 0.5809419136660483, 0.4190580863339516),
  Point<3>(0.4827518196557922, -0.6047533411474337, 0.3952466588525666),
  Point<3>(0.4360103435771013, -0.4360103435771016, 0.4255394034596999),
  Point<3>(0.4103065821727678, -0.2289578427800017, 0.4427443854298673),
  Point<3>(0.4020320197505841, -5.117434254131581e-17, 0.4483163854113368),
  Point<3>(0.4103065821727678, 0.2289578427800016, 0.4427443854298674),
  Point<3>(0.4360103435771018, 0.4360103435771014, 0.4255394034596997),
  Point<3>(0.4827518196557916, 0.604753341147433, 0.3952466588525672),
  Point<3>(0.6478790677934697, -0.6478790677934695, 0.3521209322065305),
  Point<3>(0.604753341147433, -0.4827518196557913, 0.395246658852567),
  Point<3>(0.5809419136660485, -0.2571742590018535, 0.4190580863339514),
  Point<3>(0.5733526223709594, 5.551115123125783e-17, 0.4266473776290404),
  Point<3>(0.5809419136660485, 0.2571742590018538, 0.4190580863339511),
  Point<3>(0.6047533411474336, 0.4827518196557923, 0.3952466588525667),
  Point<3>(0.6478790677934694, 0.6478790677934696, 0.3521209322065303),
  Point<3>(-0.4999999999999999, -0.5000000000000001, 0.4999999999999998),
  Point<3>(-0.4562474195983874, -0.3342458981067465, 0.5437525804016128),
  Point<3>(-0.4356250237755699, -0.1172770695242098, 0.5643749762244303),
  Point<3>(-0.4356250237755699, 0.1172770695242094, 0.5643749762244304),
  Point<3>(-0.4562474195983878, 0.3342458981067464, 0.5437525804016128),
  Point<3>(-0.5, 0.4999999999999999, 0.4999999999999996),
  Point<3>(-0.3342458981067462, -0.4562474195983875, 0.543752580401613),
  Point<3>(-0.2933584022836794, -0.2933584022836796, 0.5718933298931584),
  Point<3>(-0.2740553574282287, -0.1013783636126716, 0.5854565448214792),
  Point<3>(-0.2740553574282285, 0.1013783636126715, 0.5854565448214787),
  Point<3>(-0.2933584022836798, 0.2933584022836799, 0.5718933298931581),
  Point<3>(-0.3342458981067468, 0.4562474195983879, 0.5437525804016128),
  Point<3>(-0.1172770695242097, -0.43562502377557, 0.5643749762244303),
  Point<3>(-0.1013783636126715, -0.2740553574282291, 0.5854565448214795),
  Point<3>(-0.09404198775449589, -0.09404198775449582, 0.5957539091243973),
  Point<3>(-0.0940419877544954, 0.0940419877544957, 0.595753909124397),
  Point<3>(-0.101378363612671, 0.2740553574282287, 0.5854565448214791),
  Point<3>(-0.1172770695242092, 0.4356250237755698, 0.5643749762244303),
  Point<3>(0.1172770695242092, -0.4356250237755699, 0.5643749762244308),
  Point<3>(0.1013783636126713, -0.2740553574282288, 0.585456544821479),
  Point<3>(0.09404198775449564, -0.09404198775449578, 0.5957539091243971),
  Point<3>(0.09404198775449597, 0.09404198775449582, 0.595753909124397),
  Point<3>(0.1013783636126716, 0.2740553574282287, 0.5854565448214787),
  Point<3>(0.1172770695242096, 0.4356250237755697, 0.56437497622443),
  Point<3>(0.3342458981067465, -0.4562474195983877, 0.5437525804016123),
  Point<3>(0.2933584022836797, -0.2933584022836796, 0.5718933298931579),
  Point<3>(0.2740553574282287, -0.1013783636126714, 0.5854565448214792),
  Point<3>(0.2740553574282289, 0.1013783636126716, 0.5854565448214788),
  Point<3>(0.29335840228368, 0.2933584022836799, 0.5718933298931582),
  Point<3>(0.3342458981067462, 0.4562474195983874, 0.5437525804016131),
  Point<3>(0.5000000000000001, -0.5000000000000001, 0.4999999999999998),
  Point<3>(0.4562474195983875, -0.3342458981067461, 0.5437525804016128),
  Point<3>(0.4356250237755698, -0.1172770695242096, 0.5643749762244301),
  Point<3>(0.43562502377557, 0.1172770695242093, 0.5643749762244299),
  Point<3>(0.4562474195983877, 0.3342458981067465, 0.5437525804016127),
  Point<3>(0.5, 0.5, 0.4999999999999998),
  Point<3>(-0.3521209322065304, -0.3521209322065303, 0.6478790677934696),
  Point<3>(-0.3134156148223226, -0.1953851786003878, 0.6865843851776776),
  Point<3>(-0.3009971545261582, 6.094441581627831e-17, 0.6990028454738418),
  Point<3>(-0.3134156148223222, 0.1953851786003878, 0.6865843851776779),
  Point<3>(-0.3521209322065304, 0.3521209322065303, 0.6478790677934698),
  Point<3>(-0.1953851786003878, -0.3134156148223229, 0.6865843851776775),
  Point<3>(-0.1660249809884463, -0.1660249809884463, 0.7087252243042186),
  Point<3>(-0.1565737102493678, -2.387955284899324e-17, 0.7159477846081037),
  Point<3>(-0.1660249809884456, 0.1660249809884461, 0.7087252243042189),
  Point<3>(-0.1953851786003876, 0.3134156148223224, 0.6865843851776781),
  Point<3>(5.551115123125783e-17, -0.3009971545261584, 0.6990028454738416),
  Point<3>(-5.648693318649478e-17, -0.1565737102493679, 0.7159477846081036),
  Point<3>(-1.409462824231156e-17, 1.055470814914639e-16, 0.7215124106874546),
  Point<3>(2.050226308170089e-16, 0.1565737102493679, 0.7159477846081035),
  Point<3>(-9.855035051856843e-17, 0.3009971545261582, 0.6990028454738414),
  Point<3>(0.1953851786003877, -0.3134156148223225, 0.6865843851776778),
  Point<3>(0.166024980988446, -0.1660249809884461, 0.708725224304219),
  Point<3>(0.1565737102493678, 1.402957611196243e-16, 0.7159477846081035),
  Point<3>(0.1660249809884461, 0.1660249809884463, 0.7087252243042185),
  Point<3>(0.1953851786003876, 0.3134156148223223, 0.6865843851776774),
  Point<3>(0.3521209322065305, -0.3521209322065303, 0.6478790677934698),
  Point<3>(0.3134156148223225, -0.1953851786003879, 0.6865843851776774),
  Point<3>(0.3009971545261583, 1.110223024625157e-16, 0.6990028454738416),
  Point<3>(0.3134156148223224, 0.1953851786003879, 0.6865843851776783),
  Point<3>(0.3521209322065303, 0.3521209322065305, 0.6478790677934697),
  Point<3>(-0.2173823365018974, -0.2173823365018975, 0.7826176634981026),
  Point<3>(-0.1887321557891631, -0.0797455392993958, 0.8112678442108374),
  Point<3>(-0.1887321557891632, 0.07974553929939608, 0.8112678442108372),
  Point<3>(-0.2173823365018976, 0.2173823365018977, 0.7826176634981025),
  Point<3>(-0.07974553929939568, -0.1887321557891631, 0.8112678442108373),
  Point<3>(-0.06561058508385006, -0.06561058508384993, 0.8252749599581521),
  Point<3>(-0.06561058508385059, 0.06561058508385022, 0.8252749599581517),
  Point<3>(-0.079745539299396, 0.188732155789163, 0.8112678442108371),
  Point<3>(0.07974553929939598, -0.1887321557891632, 0.8112678442108369),
  Point<3>(0.06561058508385034, -0.06561058508384998, 0.8252749599581519),
  Point<3>(0.06561058508384998, 0.06561058508385025, 0.8252749599581518),
  Point<3>(0.07974553929939572, 0.1887321557891631, 0.8112678442108371),
  Point<3>(0.2173823365018974, -0.2173823365018976, 0.7826176634981025),
  Point<3>(0.1887321557891629, -0.07974553929939576, 0.8112678442108373),
  Point<3>(0.188732155789163, 0.0797455392993961, 0.811267844210837),
  Point<3>(0.2173823365018976, 0.2173823365018974, 0.7826176634981026),
  Point<3>(-0.1077582631684279, -0.1077582631684279, 0.8922417368315723),
  Point<3>(-0.09270047302264613, 1.19155041248048e-17, 0.9072995269773537),
  Point<3>(-0.1077582631684277, 0.1077582631684278, 0.8922417368315722),
  Point<3>(-1.387778780781446e-17, -0.09270047302264618, 0.9072995269773535),
  Point<3>(3.027634566665771e-16, -2.91325123746855e-16, 0.9137886994350229),
  Point<3>(-2.773582145322703e-18, 0.09270047302264618, 0.9072995269773537),
  Point<3>(0.1077582631684278, -0.1077582631684278, 0.8922417368315726),
  Point<3>(0.09270047302264622, -6.938893903907228e-18, 0.9072995269773536),
  Point<3>(0.1077582631684278, 0.1077582631684278, 0.8922417368315724),
  Point<3>(-0.03299928479597043, -0.03299928479597043, 0.9670007152040296),
  Point<3>(-0.03299928479597029, 0.03299928479597039, 0.9670007152040295),
  Point<3>(0.03299928479597034, -0.03299928479597042, 0.9670007152040294),
  Point<3>(0.03299928479597041, 0.03299928479597029, 0.9670007152040296),
  Point<3>(0, -1.110223024625157e-16, 0.9999999999999999)};


std::vector<Point<3>> reference_points_equi_p3 = {
  Point<3>(-1, -1, 0),
  Point<3>(-1, -0.3333333333333333, 0),
  Point<3>(-1, 0.3333333333333333, 0),
  Point<3>(-1, 1, 0),
  Point<3>(-0.3333333333333333, -1, 0),
  Point<3>(-0.3333333333333333, -0.3333333333333333, 0),
  Point<3>(-0.3333333333333333, 0.3333333333333333, 0),
  Point<3>(-0.3333333333333333, 1, 0),
  Point<3>(0.3333333333333333, -1, 0),
  Point<3>(0.3333333333333333, -0.3333333333333333, 0),
  Point<3>(0.3333333333333333, 0.3333333333333333, 0),
  Point<3>(0.3333333333333333, 1, 0),
  Point<3>(1, -1, 0),
  Point<3>(1, -0.3333333333333333, 0),
  Point<3>(1, 0.3333333333333333, 0),
  Point<3>(1, 1, 0),
  Point<3>(-0.6666666666666667, -0.6666666666666667, 0.3333333333333333),
  Point<3>(-0.6666666666666667, 0, 0.3333333333333333),
  Point<3>(-0.6666666666666667, 0.6666666666666667, 0.3333333333333333),
  Point<3>(0, -0.6666666666666667, 0.3333333333333333),
  Point<3>(0, 0, 0.3333333333333333),
  Point<3>(0, 0.6666666666666667, 0.3333333333333333),
  Point<3>(0.6666666666666667, -0.6666666666666667, 0.3333333333333333),
  Point<3>(0.6666666666666667, 0, 0.3333333333333333),
  Point<3>(0.6666666666666667, 0.6666666666666667, 0.3333333333333333),
  Point<3>(-0.3333333333333334, -0.3333333333333334, 0.6666666666666666),
  Point<3>(-0.3333333333333334, 0.3333333333333334, 0.6666666666666666),
  Point<3>(0.3333333333333334, -0.3333333333333334, 0.6666666666666666),
  Point<3>(0.3333333333333334, 0.3333333333333334, 0.6666666666666666),
  Point<3>(0, 0, 1)};
std::vector<Point<3>> reference_points_equi_p4 = {Point<3>(-1, -1, 0),
                                                  Point<3>(-1, -0.5, 0),
                                                  Point<3>(-1, 0, 0),
                                                  Point<3>(-1, 0.5, 0),
                                                  Point<3>(-1, 1, 0),
                                                  Point<3>(-0.5, -1, 0),
                                                  Point<3>(-0.5, -0.5, 0),
                                                  Point<3>(-0.5, 0, 0),
                                                  Point<3>(-0.5, 0.5, 0),
                                                  Point<3>(-0.5, 1, 0),
                                                  Point<3>(0, -1, 0),
                                                  Point<3>(0, -0.5, 0),
                                                  Point<3>(0, 0, 0),
                                                  Point<3>(0, 0.5, 0),
                                                  Point<3>(0, 1, 0),
                                                  Point<3>(0.5, -1, 0),
                                                  Point<3>(0.5, -0.5, 0),
                                                  Point<3>(0.5, 0, 0),
                                                  Point<3>(0.5, 0.5, 0),
                                                  Point<3>(0.5, 1, 0),
                                                  Point<3>(1, -1, 0),
                                                  Point<3>(1, -0.5, 0),
                                                  Point<3>(1, 0, 0),
                                                  Point<3>(1, 0.5, 0),
                                                  Point<3>(1, 1, 0),
                                                  Point<3>(-0.75, -0.75, 0.25),
                                                  Point<3>(-0.75, -0.25, 0.25),
                                                  Point<3>(-0.75, 0.25, 0.25),
                                                  Point<3>(-0.75, 0.75, 0.25),
                                                  Point<3>(-0.25, -0.75, 0.25),
                                                  Point<3>(-0.25, -0.25, 0.25),
                                                  Point<3>(-0.25, 0.25, 0.25),
                                                  Point<3>(-0.25, 0.75, 0.25),
                                                  Point<3>(0.25, -0.75, 0.25),
                                                  Point<3>(0.25, -0.25, 0.25),
                                                  Point<3>(0.25, 0.25, 0.25),
                                                  Point<3>(0.25, 0.75, 0.25),
                                                  Point<3>(0.75, -0.75, 0.25),
                                                  Point<3>(0.75, -0.25, 0.25),
                                                  Point<3>(0.75, 0.25, 0.25),
                                                  Point<3>(0.75, 0.75, 0.25),
                                                  Point<3>(-0.5, -0.5, 0.5),
                                                  Point<3>(-0.5, 0, 0.5),
                                                  Point<3>(-0.5, 0.5, 0.5),
                                                  Point<3>(0, -0.5, 0.5),
                                                  Point<3>(0, 0, 0.5),
                                                  Point<3>(0, 0.5, 0.5),
                                                  Point<3>(0.5, -0.5, 0.5),
                                                  Point<3>(0.5, 0, 0.5),
                                                  Point<3>(0.5, 0.5, 0.5),
                                                  Point<3>(-0.25, -0.25, 0.75),
                                                  Point<3>(-0.25, 0.25, 0.75),
                                                  Point<3>(0.25, -0.25, 0.75),
                                                  Point<3>(0.25, 0.25, 0.75),
                                                  Point<3>(0, 0, 1)};
std::vector<Point<3>> reference_points_equi_p5 = {
  Point<3>(-1, -1, 0),       Point<3>(-1, -0.6, 0),
  Point<3>(-1, -0.2, 0),     Point<3>(-1, 0.2, 0),
  Point<3>(-1, 0.6, 0),      Point<3>(-1, 1, 0),
  Point<3>(-0.6, -1, 0),     Point<3>(-0.6, -0.6, 0),
  Point<3>(-0.6, -0.2, 0),   Point<3>(-0.6, 0.2, 0),
  Point<3>(-0.6, 0.6, 0),    Point<3>(-0.6, 1, 0),
  Point<3>(-0.2, -1, 0),     Point<3>(-0.2, -0.6, 0),
  Point<3>(-0.2, -0.2, 0),   Point<3>(-0.2, 0.2, 0),
  Point<3>(-0.2, 0.6, 0),    Point<3>(-0.2, 1, 0),
  Point<3>(0.2, -1, 0),      Point<3>(0.2, -0.6, 0),
  Point<3>(0.2, -0.2, 0),    Point<3>(0.2, 0.2, 0),
  Point<3>(0.2, 0.6, 0),     Point<3>(0.2, 1, 0),
  Point<3>(0.6, -1, 0),      Point<3>(0.6, -0.6, 0),
  Point<3>(0.6, -0.2, 0),    Point<3>(0.6, 0.2, 0),
  Point<3>(0.6, 0.6, 0),     Point<3>(0.6, 1, 0),
  Point<3>(1, -1, 0),        Point<3>(1, -0.6, 0),
  Point<3>(1, -0.2, 0),      Point<3>(1, 0.2, 0),
  Point<3>(1, 0.6, 0),       Point<3>(1, 1, 0),
  Point<3>(-0.8, -0.8, 0.2), Point<3>(-0.8, -0.4, 0.2),
  Point<3>(-0.8, 0, 0.2),    Point<3>(-0.8, 0.4, 0.2),
  Point<3>(-0.8, 0.8, 0.2),  Point<3>(-0.4, -0.8, 0.2),
  Point<3>(-0.4, -0.4, 0.2), Point<3>(-0.4, 0, 0.2),
  Point<3>(-0.4, 0.4, 0.2),  Point<3>(-0.4, 0.8, 0.2),
  Point<3>(0, -0.8, 0.2),    Point<3>(0, -0.4, 0.2),
  Point<3>(0, 0, 0.2),       Point<3>(0, 0.4, 0.2),
  Point<3>(0, 0.8, 0.2),     Point<3>(0.4, -0.8, 0.2),
  Point<3>(0.4, -0.4, 0.2),  Point<3>(0.4, 0, 0.2),
  Point<3>(0.4, 0.4, 0.2),   Point<3>(0.4, 0.8, 0.2),
  Point<3>(0.8, -0.8, 0.2),  Point<3>(0.8, -0.4, 0.2),
  Point<3>(0.8, 0, 0.2),     Point<3>(0.8, 0.4, 0.2),
  Point<3>(0.8, 0.8, 0.2),   Point<3>(-0.6, -0.6, 0.4),
  Point<3>(-0.6, -0.2, 0.4), Point<3>(-0.6, 0.2, 0.4),
  Point<3>(-0.6, 0.6, 0.4),  Point<3>(-0.2, -0.6, 0.4),
  Point<3>(-0.2, -0.2, 0.4), Point<3>(-0.2, 0.2, 0.4),
  Point<3>(-0.2, 0.6, 0.4),  Point<3>(0.2, -0.6, 0.4),
  Point<3>(0.2, -0.2, 0.4),  Point<3>(0.2, 0.2, 0.4),
  Point<3>(0.2, 0.6, 0.4),   Point<3>(0.6, -0.6, 0.4),
  Point<3>(0.6, -0.2, 0.4),  Point<3>(0.6, 0.2, 0.4),
  Point<3>(0.6, 0.6, 0.4),   Point<3>(-0.4, -0.4, 0.6),
  Point<3>(-0.4, 0, 0.6),    Point<3>(-0.4, 0.4, 0.6),
  Point<3>(0, -0.4, 0.6),    Point<3>(0, 0, 0.6),
  Point<3>(0, 0.4, 0.6),     Point<3>(0.4, -0.4, 0.6),
  Point<3>(0.4, 0, 0.6),     Point<3>(0.4, 0.4, 0.6),
  Point<3>(-0.2, -0.2, 0.8), Point<3>(-0.2, 0.2, 0.8),
  Point<3>(0.2, -0.2, 0.8),  Point<3>(0.2, 0.2, 0.8),
  Point<3>(0, 0, 1)};
std::vector<Point<3>> reference_points_equi_p6 = {
  Point<3>(-1, -1, 0),
  Point<3>(-1, -0.6666666666666666, 0),
  Point<3>(-1, -0.3333333333333333, 0),
  Point<3>(-1, 0, 0),
  Point<3>(-1, 0.3333333333333333, 0),
  Point<3>(-1, 0.6666666666666666, 0),
  Point<3>(-1, 1, 0),
  Point<3>(-0.6666666666666666, -1, 0),
  Point<3>(-0.6666666666666666, -0.6666666666666666, 0),
  Point<3>(-0.6666666666666666, -0.3333333333333333, 0),
  Point<3>(-0.6666666666666666, 0, 0),
  Point<3>(-0.6666666666666666, 0.3333333333333333, 0),
  Point<3>(-0.6666666666666666, 0.6666666666666666, 0),
  Point<3>(-0.6666666666666666, 1, 0),
  Point<3>(-0.3333333333333333, -1, 0),
  Point<3>(-0.3333333333333333, -0.6666666666666666, 0),
  Point<3>(-0.3333333333333333, -0.3333333333333333, 0),
  Point<3>(-0.3333333333333333, 0, 0),
  Point<3>(-0.3333333333333333, 0.3333333333333333, 0),
  Point<3>(-0.3333333333333333, 0.6666666666666666, 0),
  Point<3>(-0.3333333333333333, 1, 0),
  Point<3>(0, -1, 0),
  Point<3>(0, -0.6666666666666666, 0),
  Point<3>(0, -0.3333333333333333, 0),
  Point<3>(0, 0, 0),
  Point<3>(0, 0.3333333333333333, 0),
  Point<3>(0, 0.6666666666666666, 0),
  Point<3>(0, 1, 0),
  Point<3>(0.3333333333333333, -1, 0),
  Point<3>(0.3333333333333333, -0.6666666666666666, 0),
  Point<3>(0.3333333333333333, -0.3333333333333333, 0),
  Point<3>(0.3333333333333333, 0, 0),
  Point<3>(0.3333333333333333, 0.3333333333333333, 0),
  Point<3>(0.3333333333333333, 0.6666666666666666, 0),
  Point<3>(0.3333333333333333, 1, 0),
  Point<3>(0.6666666666666666, -1, 0),
  Point<3>(0.6666666666666666, -0.6666666666666666, 0),
  Point<3>(0.6666666666666666, -0.3333333333333333, 0),
  Point<3>(0.6666666666666666, 0, 0),
  Point<3>(0.6666666666666666, 0.3333333333333333, 0),
  Point<3>(0.6666666666666666, 0.6666666666666666, 0),
  Point<3>(0.6666666666666666, 1, 0),
  Point<3>(1, -1, 0),
  Point<3>(1, -0.6666666666666666, 0),
  Point<3>(1, -0.3333333333333333, 0),
  Point<3>(1, 0, 0),
  Point<3>(1, 0.3333333333333333, 0),
  Point<3>(1, 0.6666666666666666, 0),
  Point<3>(1, 1, 0),
  Point<3>(-0.8333333333333334, -0.8333333333333334, 0.1666666666666667),
  Point<3>(-0.8333333333333334, -0.5, 0.1666666666666667),
  Point<3>(-0.8333333333333334, -0.1666666666666667, 0.1666666666666667),
  Point<3>(-0.8333333333333334, 0.1666666666666667, 0.1666666666666667),
  Point<3>(-0.8333333333333334, 0.5, 0.1666666666666667),
  Point<3>(-0.8333333333333334, 0.8333333333333334, 0.1666666666666667),
  Point<3>(-0.5, -0.8333333333333334, 0.1666666666666667),
  Point<3>(-0.5, -0.5, 0.1666666666666667),
  Point<3>(-0.5, -0.1666666666666667, 0.1666666666666667),
  Point<3>(-0.5, 0.1666666666666667, 0.1666666666666667),
  Point<3>(-0.5, 0.5, 0.1666666666666667),
  Point<3>(-0.5, 0.8333333333333334, 0.1666666666666667),
  Point<3>(-0.1666666666666667, -0.8333333333333334, 0.1666666666666667),
  Point<3>(-0.1666666666666667, -0.5, 0.1666666666666667),
  Point<3>(-0.1666666666666667, -0.1666666666666667, 0.1666666666666667),
  Point<3>(-0.1666666666666667, 0.1666666666666667, 0.1666666666666667),
  Point<3>(-0.1666666666666667, 0.5, 0.1666666666666667),
  Point<3>(-0.1666666666666667, 0.8333333333333334, 0.1666666666666667),
  Point<3>(0.1666666666666667, -0.8333333333333334, 0.1666666666666667),
  Point<3>(0.1666666666666667, -0.5, 0.1666666666666667),
  Point<3>(0.1666666666666667, -0.1666666666666667, 0.1666666666666667),
  Point<3>(0.1666666666666667, 0.1666666666666667, 0.1666666666666667),
  Point<3>(0.1666666666666667, 0.5, 0.1666666666666667),
  Point<3>(0.1666666666666667, 0.8333333333333334, 0.1666666666666667),
  Point<3>(0.5, -0.8333333333333334, 0.1666666666666667),
  Point<3>(0.5, -0.5, 0.1666666666666667),
  Point<3>(0.5, -0.1666666666666667, 0.1666666666666667),
  Point<3>(0.5, 0.1666666666666667, 0.1666666666666667),
  Point<3>(0.5, 0.5, 0.1666666666666667),
  Point<3>(0.5, 0.8333333333333334, 0.1666666666666667),
  Point<3>(0.8333333333333334, -0.8333333333333334, 0.1666666666666667),
  Point<3>(0.8333333333333334, -0.5, 0.1666666666666667),
  Point<3>(0.8333333333333334, -0.1666666666666667, 0.1666666666666667),
  Point<3>(0.8333333333333334, 0.1666666666666667, 0.1666666666666667),
  Point<3>(0.8333333333333334, 0.5, 0.1666666666666667),
  Point<3>(0.8333333333333334, 0.8333333333333334, 0.1666666666666667),
  Point<3>(-0.6666666666666667, -0.6666666666666667, 0.3333333333333333),
  Point<3>(-0.6666666666666667, -0.3333333333333334, 0.3333333333333333),
  Point<3>(-0.6666666666666667, 0, 0.3333333333333333),
  Point<3>(-0.6666666666666667, 0.3333333333333334, 0.3333333333333333),
  Point<3>(-0.6666666666666667, 0.6666666666666667, 0.3333333333333333),
  Point<3>(-0.3333333333333334, -0.6666666666666667, 0.3333333333333333),
  Point<3>(-0.3333333333333334, -0.3333333333333334, 0.3333333333333333),
  Point<3>(-0.3333333333333334, 0, 0.3333333333333333),
  Point<3>(-0.3333333333333334, 0.3333333333333334, 0.3333333333333333),
  Point<3>(-0.3333333333333334, 0.6666666666666667, 0.3333333333333333),
  Point<3>(0, -0.6666666666666667, 0.3333333333333333),
  Point<3>(0, -0.3333333333333334, 0.3333333333333333),
  Point<3>(0, 0, 0.3333333333333333),
  Point<3>(0, 0.3333333333333334, 0.3333333333333333),
  Point<3>(0, 0.6666666666666667, 0.3333333333333333),
  Point<3>(0.3333333333333334, -0.6666666666666667, 0.3333333333333333),
  Point<3>(0.3333333333333334, -0.3333333333333334, 0.3333333333333333),
  Point<3>(0.3333333333333334, 0, 0.3333333333333333),
  Point<3>(0.3333333333333334, 0.3333333333333334, 0.3333333333333333),
  Point<3>(0.3333333333333334, 0.6666666666666667, 0.3333333333333333),
  Point<3>(0.6666666666666667, -0.6666666666666667, 0.3333333333333333),
  Point<3>(0.6666666666666667, -0.3333333333333334, 0.3333333333333333),
  Point<3>(0.6666666666666667, 0, 0.3333333333333333),
  Point<3>(0.6666666666666667, 0.3333333333333334, 0.3333333333333333),
  Point<3>(0.6666666666666667, 0.6666666666666667, 0.3333333333333333),
  Point<3>(-0.5, -0.5, 0.5),
  Point<3>(-0.5, -0.1666666666666667, 0.5),
  Point<3>(-0.5, 0.1666666666666667, 0.5),
  Point<3>(-0.5, 0.5, 0.5),
  Point<3>(-0.1666666666666667, -0.5, 0.5),
  Point<3>(-0.1666666666666667, -0.1666666666666667, 0.5),
  Point<3>(-0.1666666666666667, 0.1666666666666667, 0.5),
  Point<3>(-0.1666666666666667, 0.5, 0.5),
  Point<3>(0.1666666666666667, -0.5, 0.5),
  Point<3>(0.1666666666666667, -0.1666666666666667, 0.5),
  Point<3>(0.1666666666666667, 0.1666666666666667, 0.5),
  Point<3>(0.1666666666666667, 0.5, 0.5),
  Point<3>(0.5, -0.5, 0.5),
  Point<3>(0.5, -0.1666666666666667, 0.5),
  Point<3>(0.5, 0.1666666666666667, 0.5),
  Point<3>(0.5, 0.5, 0.5),
  Point<3>(-0.3333333333333334, -0.3333333333333334, 0.6666666666666666),
  Point<3>(-0.3333333333333334, 0, 0.6666666666666666),
  Point<3>(-0.3333333333333334, 0.3333333333333334, 0.6666666666666666),
  Point<3>(0, -0.3333333333333334, 0.6666666666666666),
  Point<3>(0, 0, 0.6666666666666666),
  Point<3>(0, 0.3333333333333334, 0.6666666666666666),
  Point<3>(0.3333333333333334, -0.3333333333333334, 0.6666666666666666),
  Point<3>(0.3333333333333334, 0, 0.6666666666666666),
  Point<3>(0.3333333333333334, 0.3333333333333334, 0.6666666666666666),
  Point<3>(-0.1666666666666666, -0.1666666666666666, 0.8333333333333334),
  Point<3>(-0.1666666666666666, 0.1666666666666666, 0.8333333333333334),
  Point<3>(0.1666666666666666, -0.1666666666666666, 0.8333333333333334),
  Point<3>(0.1666666666666666, 0.1666666666666666, 0.8333333333333334),
  Point<3>(0, 0, 1)};
std::vector<Point<3>> reference_points_equi_p7 = {
  Point<3>(-1, -1, 0),
  Point<3>(-1, -0.7142857142857143, 0),
  Point<3>(-1, -0.4285714285714285, 0),
  Point<3>(-1, -0.1428571428571428, 0),
  Point<3>(-1, 0.1428571428571428, 0),
  Point<3>(-1, 0.4285714285714285, 0),
  Point<3>(-1, 0.7142857142857143, 0),
  Point<3>(-1, 1, 0),
  Point<3>(-0.7142857142857143, -1, 0),
  Point<3>(-0.7142857142857143, -0.7142857142857143, 0),
  Point<3>(-0.7142857142857143, -0.4285714285714285, 0),
  Point<3>(-0.7142857142857143, -0.1428571428571428, 0),
  Point<3>(-0.7142857142857143, 0.1428571428571428, 0),
  Point<3>(-0.7142857142857143, 0.4285714285714285, 0),
  Point<3>(-0.7142857142857143, 0.7142857142857143, 0),
  Point<3>(-0.7142857142857143, 1, 0),
  Point<3>(-0.4285714285714285, -1, 0),
  Point<3>(-0.4285714285714285, -0.7142857142857143, 0),
  Point<3>(-0.4285714285714285, -0.4285714285714285, 0),
  Point<3>(-0.4285714285714285, -0.1428571428571428, 0),
  Point<3>(-0.4285714285714285, 0.1428571428571428, 0),
  Point<3>(-0.4285714285714285, 0.4285714285714285, 0),
  Point<3>(-0.4285714285714285, 0.7142857142857143, 0),
  Point<3>(-0.4285714285714285, 1, 0),
  Point<3>(-0.1428571428571428, -1, 0),
  Point<3>(-0.1428571428571428, -0.7142857142857143, 0),
  Point<3>(-0.1428571428571428, -0.4285714285714285, 0),
  Point<3>(-0.1428571428571428, -0.1428571428571428, 0),
  Point<3>(-0.1428571428571428, 0.1428571428571428, 0),
  Point<3>(-0.1428571428571428, 0.4285714285714285, 0),
  Point<3>(-0.1428571428571428, 0.7142857142857143, 0),
  Point<3>(-0.1428571428571428, 1, 0),
  Point<3>(0.1428571428571428, -1, 0),
  Point<3>(0.1428571428571428, -0.7142857142857143, 0),
  Point<3>(0.1428571428571428, -0.4285714285714285, 0),
  Point<3>(0.1428571428571428, -0.1428571428571428, 0),
  Point<3>(0.1428571428571428, 0.1428571428571428, 0),
  Point<3>(0.1428571428571428, 0.4285714285714285, 0),
  Point<3>(0.1428571428571428, 0.7142857142857143, 0),
  Point<3>(0.1428571428571428, 1, 0),
  Point<3>(0.4285714285714285, -1, 0),
  Point<3>(0.4285714285714285, -0.7142857142857143, 0),
  Point<3>(0.4285714285714285, -0.4285714285714285, 0),
  Point<3>(0.4285714285714285, -0.1428571428571428, 0),
  Point<3>(0.4285714285714285, 0.1428571428571428, 0),
  Point<3>(0.4285714285714285, 0.4285714285714285, 0),
  Point<3>(0.4285714285714285, 0.7142857142857143, 0),
  Point<3>(0.4285714285714285, 1, 0),
  Point<3>(0.7142857142857143, -1, 0),
  Point<3>(0.7142857142857143, -0.7142857142857143, 0),
  Point<3>(0.7142857142857143, -0.4285714285714285, 0),
  Point<3>(0.7142857142857143, -0.1428571428571428, 0),
  Point<3>(0.7142857142857143, 0.1428571428571428, 0),
  Point<3>(0.7142857142857143, 0.4285714285714285, 0),
  Point<3>(0.7142857142857143, 0.7142857142857143, 0),
  Point<3>(0.7142857142857143, 1, 0),
  Point<3>(1, -1, 0),
  Point<3>(1, -0.7142857142857143, 0),
  Point<3>(1, -0.4285714285714285, 0),
  Point<3>(1, -0.1428571428571428, 0),
  Point<3>(1, 0.1428571428571428, 0),
  Point<3>(1, 0.4285714285714285, 0),
  Point<3>(1, 0.7142857142857143, 0),
  Point<3>(1, 1, 0),
  Point<3>(-0.8571428571428572, -0.8571428571428572, 0.1428571428571428),
  Point<3>(-0.8571428571428572, -0.5714285714285714, 0.1428571428571428),
  Point<3>(-0.8571428571428572, -0.2857142857142857, 0.1428571428571428),
  Point<3>(-0.8571428571428572, 0, 0.1428571428571428),
  Point<3>(-0.8571428571428572, 0.2857142857142857, 0.1428571428571428),
  Point<3>(-0.8571428571428572, 0.5714285714285714, 0.1428571428571428),
  Point<3>(-0.8571428571428572, 0.8571428571428572, 0.1428571428571428),
  Point<3>(-0.5714285714285714, -0.8571428571428572, 0.1428571428571428),
  Point<3>(-0.5714285714285714, -0.5714285714285714, 0.1428571428571428),
  Point<3>(-0.5714285714285714, -0.2857142857142857, 0.1428571428571428),
  Point<3>(-0.5714285714285714, 0, 0.1428571428571428),
  Point<3>(-0.5714285714285714, 0.2857142857142857, 0.1428571428571428),
  Point<3>(-0.5714285714285714, 0.5714285714285714, 0.1428571428571428),
  Point<3>(-0.5714285714285714, 0.8571428571428572, 0.1428571428571428),
  Point<3>(-0.2857142857142857, -0.8571428571428572, 0.1428571428571428),
  Point<3>(-0.2857142857142857, -0.5714285714285714, 0.1428571428571428),
  Point<3>(-0.2857142857142857, -0.2857142857142857, 0.1428571428571428),
  Point<3>(-0.2857142857142857, 0, 0.1428571428571428),
  Point<3>(-0.2857142857142857, 0.2857142857142857, 0.1428571428571428),
  Point<3>(-0.2857142857142857, 0.5714285714285714, 0.1428571428571428),
  Point<3>(-0.2857142857142857, 0.8571428571428572, 0.1428571428571428),
  Point<3>(0, -0.8571428571428572, 0.1428571428571428),
  Point<3>(0, -0.5714285714285714, 0.1428571428571428),
  Point<3>(0, -0.2857142857142857, 0.1428571428571428),
  Point<3>(0, 0, 0.1428571428571428),
  Point<3>(0, 0.2857142857142857, 0.1428571428571428),
  Point<3>(0, 0.5714285714285714, 0.1428571428571428),
  Point<3>(0, 0.8571428571428572, 0.1428571428571428),
  Point<3>(0.2857142857142857, -0.8571428571428572, 0.1428571428571428),
  Point<3>(0.2857142857142857, -0.5714285714285714, 0.1428571428571428),
  Point<3>(0.2857142857142857, -0.2857142857142857, 0.1428571428571428),
  Point<3>(0.2857142857142857, 0, 0.1428571428571428),
  Point<3>(0.2857142857142857, 0.2857142857142857, 0.1428571428571428),
  Point<3>(0.2857142857142857, 0.5714285714285714, 0.1428571428571428),
  Point<3>(0.2857142857142857, 0.8571428571428572, 0.1428571428571428),
  Point<3>(0.5714285714285714, -0.8571428571428572, 0.1428571428571428),
  Point<3>(0.5714285714285714, -0.5714285714285714, 0.1428571428571428),
  Point<3>(0.5714285714285714, -0.2857142857142857, 0.1428571428571428),
  Point<3>(0.5714285714285714, 0, 0.1428571428571428),
  Point<3>(0.5714285714285714, 0.2857142857142857, 0.1428571428571428),
  Point<3>(0.5714285714285714, 0.5714285714285714, 0.1428571428571428),
  Point<3>(0.5714285714285714, 0.8571428571428572, 0.1428571428571428),
  Point<3>(0.8571428571428572, -0.8571428571428572, 0.1428571428571428),
  Point<3>(0.8571428571428572, -0.5714285714285714, 0.1428571428571428),
  Point<3>(0.8571428571428572, -0.2857142857142857, 0.1428571428571428),
  Point<3>(0.8571428571428572, 0, 0.1428571428571428),
  Point<3>(0.8571428571428572, 0.2857142857142857, 0.1428571428571428),
  Point<3>(0.8571428571428572, 0.5714285714285714, 0.1428571428571428),
  Point<3>(0.8571428571428572, 0.8571428571428572, 0.1428571428571428),
  Point<3>(-0.7142857142857143, -0.7142857142857143, 0.2857142857142857),
  Point<3>(-0.7142857142857143, -0.4285714285714285, 0.2857142857142857),
  Point<3>(-0.7142857142857143, -0.1428571428571429, 0.2857142857142857),
  Point<3>(-0.7142857142857143, 0.1428571428571429, 0.2857142857142857),
  Point<3>(-0.7142857142857143, 0.4285714285714285, 0.2857142857142857),
  Point<3>(-0.7142857142857143, 0.7142857142857143, 0.2857142857142857),
  Point<3>(-0.4285714285714285, -0.7142857142857143, 0.2857142857142857),
  Point<3>(-0.4285714285714285, -0.4285714285714285, 0.2857142857142857),
  Point<3>(-0.4285714285714285, -0.1428571428571429, 0.2857142857142857),
  Point<3>(-0.4285714285714285, 0.1428571428571429, 0.2857142857142857),
  Point<3>(-0.4285714285714285, 0.4285714285714285, 0.2857142857142857),
  Point<3>(-0.4285714285714285, 0.7142857142857143, 0.2857142857142857),
  Point<3>(-0.1428571428571429, -0.7142857142857143, 0.2857142857142857),
  Point<3>(-0.1428571428571429, -0.4285714285714285, 0.2857142857142857),
  Point<3>(-0.1428571428571429, -0.1428571428571429, 0.2857142857142857),
  Point<3>(-0.1428571428571429, 0.1428571428571429, 0.2857142857142857),
  Point<3>(-0.1428571428571429, 0.4285714285714285, 0.2857142857142857),
  Point<3>(-0.1428571428571429, 0.7142857142857143, 0.2857142857142857),
  Point<3>(0.1428571428571429, -0.7142857142857143, 0.2857142857142857),
  Point<3>(0.1428571428571429, -0.4285714285714285, 0.2857142857142857),
  Point<3>(0.1428571428571429, -0.1428571428571429, 0.2857142857142857),
  Point<3>(0.1428571428571429, 0.1428571428571429, 0.2857142857142857),
  Point<3>(0.1428571428571429, 0.4285714285714285, 0.2857142857142857),
  Point<3>(0.1428571428571429, 0.7142857142857143, 0.2857142857142857),
  Point<3>(0.4285714285714285, -0.7142857142857143, 0.2857142857142857),
  Point<3>(0.4285714285714285, -0.4285714285714285, 0.2857142857142857),
  Point<3>(0.4285714285714285, -0.1428571428571429, 0.2857142857142857),
  Point<3>(0.4285714285714285, 0.1428571428571429, 0.2857142857142857),
  Point<3>(0.4285714285714285, 0.4285714285714285, 0.2857142857142857),
  Point<3>(0.4285714285714285, 0.7142857142857143, 0.2857142857142857),
  Point<3>(0.7142857142857143, -0.7142857142857143, 0.2857142857142857),
  Point<3>(0.7142857142857143, -0.4285714285714285, 0.2857142857142857),
  Point<3>(0.7142857142857143, -0.1428571428571429, 0.2857142857142857),
  Point<3>(0.7142857142857143, 0.1428571428571429, 0.2857142857142857),
  Point<3>(0.7142857142857143, 0.4285714285714285, 0.2857142857142857),
  Point<3>(0.7142857142857143, 0.7142857142857143, 0.2857142857142857),
  Point<3>(-0.5714285714285714, -0.5714285714285714, 0.4285714285714285),
  Point<3>(-0.5714285714285714, -0.2857142857142857, 0.4285714285714285),
  Point<3>(-0.5714285714285714, 0, 0.4285714285714285),
  Point<3>(-0.5714285714285714, 0.2857142857142857, 0.4285714285714285),
  Point<3>(-0.5714285714285714, 0.5714285714285714, 0.4285714285714285),
  Point<3>(-0.2857142857142857, -0.5714285714285714, 0.4285714285714285),
  Point<3>(-0.2857142857142857, -0.2857142857142857, 0.4285714285714285),
  Point<3>(-0.2857142857142857, 0, 0.4285714285714285),
  Point<3>(-0.2857142857142857, 0.2857142857142857, 0.4285714285714285),
  Point<3>(-0.2857142857142857, 0.5714285714285714, 0.4285714285714285),
  Point<3>(0, -0.5714285714285714, 0.4285714285714285),
  Point<3>(0, -0.2857142857142857, 0.4285714285714285),
  Point<3>(0, 0, 0.4285714285714285),
  Point<3>(0, 0.2857142857142857, 0.4285714285714285),
  Point<3>(0, 0.5714285714285714, 0.4285714285714285),
  Point<3>(0.2857142857142857, -0.5714285714285714, 0.4285714285714285),
  Point<3>(0.2857142857142857, -0.2857142857142857, 0.4285714285714285),
  Point<3>(0.2857142857142857, 0, 0.4285714285714285),
  Point<3>(0.2857142857142857, 0.2857142857142857, 0.4285714285714285),
  Point<3>(0.2857142857142857, 0.5714285714285714, 0.4285714285714285),
  Point<3>(0.5714285714285714, -0.5714285714285714, 0.4285714285714285),
  Point<3>(0.5714285714285714, -0.2857142857142857, 0.4285714285714285),
  Point<3>(0.5714285714285714, 0, 0.4285714285714285),
  Point<3>(0.5714285714285714, 0.2857142857142857, 0.4285714285714285),
  Point<3>(0.5714285714285714, 0.5714285714285714, 0.4285714285714285),
  Point<3>(-0.4285714285714286, -0.4285714285714286, 0.5714285714285714),
  Point<3>(-0.4285714285714286, -0.1428571428571428, 0.5714285714285714),
  Point<3>(-0.4285714285714286, 0.1428571428571428, 0.5714285714285714),
  Point<3>(-0.4285714285714286, 0.4285714285714286, 0.5714285714285714),
  Point<3>(-0.1428571428571428, -0.4285714285714286, 0.5714285714285714),
  Point<3>(-0.1428571428571428, -0.1428571428571428, 0.5714285714285714),
  Point<3>(-0.1428571428571428, 0.1428571428571428, 0.5714285714285714),
  Point<3>(-0.1428571428571428, 0.4285714285714286, 0.5714285714285714),
  Point<3>(0.1428571428571428, -0.4285714285714286, 0.5714285714285714),
  Point<3>(0.1428571428571428, -0.1428571428571428, 0.5714285714285714),
  Point<3>(0.1428571428571428, 0.1428571428571428, 0.5714285714285714),
  Point<3>(0.1428571428571428, 0.4285714285714286, 0.5714285714285714),
  Point<3>(0.4285714285714286, -0.4285714285714286, 0.5714285714285714),
  Point<3>(0.4285714285714286, -0.1428571428571428, 0.5714285714285714),
  Point<3>(0.4285714285714286, 0.1428571428571428, 0.5714285714285714),
  Point<3>(0.4285714285714286, 0.4285714285714286, 0.5714285714285714),
  Point<3>(-0.2857142857142857, -0.2857142857142857, 0.7142857142857143),
  Point<3>(-0.2857142857142857, 0, 0.7142857142857143),
  Point<3>(-0.2857142857142857, 0.2857142857142857, 0.7142857142857143),
  Point<3>(0, -0.2857142857142857, 0.7142857142857143),
  Point<3>(0, 0, 0.7142857142857143),
  Point<3>(0, 0.2857142857142857, 0.7142857142857143),
  Point<3>(0.2857142857142857, -0.2857142857142857, 0.7142857142857143),
  Point<3>(0.2857142857142857, 0, 0.7142857142857143),
  Point<3>(0.2857142857142857, 0.2857142857142857, 0.7142857142857143),
  Point<3>(-0.1428571428571429, -0.1428571428571429, 0.8571428571428571),
  Point<3>(-0.1428571428571429, 0.1428571428571429, 0.8571428571428571),
  Point<3>(0.1428571428571429, -0.1428571428571429, 0.8571428571428571),
  Point<3>(0.1428571428571429, 0.1428571428571429, 0.8571428571428571),
  Point<3>(0, 0, 1)};
std::vector<Point<3>> reference_points_equi_p8 = {
  Point<3>(-1, -1, 0),
  Point<3>(-1, -0.75, 0),
  Point<3>(-1, -0.5, 0),
  Point<3>(-1, -0.25, 0),
  Point<3>(-1, 0, 0),
  Point<3>(-1, 0.25, 0),
  Point<3>(-1, 0.5, 0),
  Point<3>(-1, 0.75, 0),
  Point<3>(-1, 1, 0),
  Point<3>(-0.75, -1, 0),
  Point<3>(-0.75, -0.75, 0),
  Point<3>(-0.75, -0.5, 0),
  Point<3>(-0.75, -0.25, 0),
  Point<3>(-0.75, 0, 0),
  Point<3>(-0.75, 0.25, 0),
  Point<3>(-0.75, 0.5, 0),
  Point<3>(-0.75, 0.75, 0),
  Point<3>(-0.75, 1, 0),
  Point<3>(-0.5, -1, 0),
  Point<3>(-0.5, -0.75, 0),
  Point<3>(-0.5, -0.5, 0),
  Point<3>(-0.5, -0.25, 0),
  Point<3>(-0.5, 0, 0),
  Point<3>(-0.5, 0.25, 0),
  Point<3>(-0.5, 0.5, 0),
  Point<3>(-0.5, 0.75, 0),
  Point<3>(-0.5, 1, 0),
  Point<3>(-0.25, -1, 0),
  Point<3>(-0.25, -0.75, 0),
  Point<3>(-0.25, -0.5, 0),
  Point<3>(-0.25, -0.25, 0),
  Point<3>(-0.25, 0, 0),
  Point<3>(-0.25, 0.25, 0),
  Point<3>(-0.25, 0.5, 0),
  Point<3>(-0.25, 0.75, 0),
  Point<3>(-0.25, 1, 0),
  Point<3>(0, -1, 0),
  Point<3>(0, -0.75, 0),
  Point<3>(0, -0.5, 0),
  Point<3>(0, -0.25, 0),
  Point<3>(0, 0, 0),
  Point<3>(0, 0.25, 0),
  Point<3>(0, 0.5, 0),
  Point<3>(0, 0.75, 0),
  Point<3>(0, 1, 0),
  Point<3>(0.25, -1, 0),
  Point<3>(0.25, -0.75, 0),
  Point<3>(0.25, -0.5, 0),
  Point<3>(0.25, -0.25, 0),
  Point<3>(0.25, 0, 0),
  Point<3>(0.25, 0.25, 0),
  Point<3>(0.25, 0.5, 0),
  Point<3>(0.25, 0.75, 0),
  Point<3>(0.25, 1, 0),
  Point<3>(0.5, -1, 0),
  Point<3>(0.5, -0.75, 0),
  Point<3>(0.5, -0.5, 0),
  Point<3>(0.5, -0.25, 0),
  Point<3>(0.5, 0, 0),
  Point<3>(0.5, 0.25, 0),
  Point<3>(0.5, 0.5, 0),
  Point<3>(0.5, 0.75, 0),
  Point<3>(0.5, 1, 0),
  Point<3>(0.75, -1, 0),
  Point<3>(0.75, -0.75, 0),
  Point<3>(0.75, -0.5, 0),
  Point<3>(0.75, -0.25, 0),
  Point<3>(0.75, 0, 0),
  Point<3>(0.75, 0.25, 0),
  Point<3>(0.75, 0.5, 0),
  Point<3>(0.75, 0.75, 0),
  Point<3>(0.75, 1, 0),
  Point<3>(1, -1, 0),
  Point<3>(1, -0.75, 0),
  Point<3>(1, -0.5, 0),
  Point<3>(1, -0.25, 0),
  Point<3>(1, 0, 0),
  Point<3>(1, 0.25, 0),
  Point<3>(1, 0.5, 0),
  Point<3>(1, 0.75, 0),
  Point<3>(1, 1, 0),
  Point<3>(-0.875, -0.875, 0.125),
  Point<3>(-0.875, -0.625, 0.125),
  Point<3>(-0.875, -0.375, 0.125),
  Point<3>(-0.875, -0.125, 0.125),
  Point<3>(-0.875, 0.125, 0.125),
  Point<3>(-0.875, 0.375, 0.125),
  Point<3>(-0.875, 0.625, 0.125),
  Point<3>(-0.875, 0.875, 0.125),
  Point<3>(-0.625, -0.875, 0.125),
  Point<3>(-0.625, -0.625, 0.125),
  Point<3>(-0.625, -0.375, 0.125),
  Point<3>(-0.625, -0.125, 0.125),
  Point<3>(-0.625, 0.125, 0.125),
  Point<3>(-0.625, 0.375, 0.125),
  Point<3>(-0.625, 0.625, 0.125),
  Point<3>(-0.625, 0.875, 0.125),
  Point<3>(-0.375, -0.875, 0.125),
  Point<3>(-0.375, -0.625, 0.125),
  Point<3>(-0.375, -0.375, 0.125),
  Point<3>(-0.375, -0.125, 0.125),
  Point<3>(-0.375, 0.125, 0.125),
  Point<3>(-0.375, 0.375, 0.125),
  Point<3>(-0.375, 0.625, 0.125),
  Point<3>(-0.375, 0.875, 0.125),
  Point<3>(-0.125, -0.875, 0.125),
  Point<3>(-0.125, -0.625, 0.125),
  Point<3>(-0.125, -0.375, 0.125),
  Point<3>(-0.125, -0.125, 0.125),
  Point<3>(-0.125, 0.125, 0.125),
  Point<3>(-0.125, 0.375, 0.125),
  Point<3>(-0.125, 0.625, 0.125),
  Point<3>(-0.125, 0.875, 0.125),
  Point<3>(0.125, -0.875, 0.125),
  Point<3>(0.125, -0.625, 0.125),
  Point<3>(0.125, -0.375, 0.125),
  Point<3>(0.125, -0.125, 0.125),
  Point<3>(0.125, 0.125, 0.125),
  Point<3>(0.125, 0.375, 0.125),
  Point<3>(0.125, 0.625, 0.125),
  Point<3>(0.125, 0.875, 0.125),
  Point<3>(0.375, -0.875, 0.125),
  Point<3>(0.375, -0.625, 0.125),
  Point<3>(0.375, -0.375, 0.125),
  Point<3>(0.375, -0.125, 0.125),
  Point<3>(0.375, 0.125, 0.125),
  Point<3>(0.375, 0.375, 0.125),
  Point<3>(0.375, 0.625, 0.125),
  Point<3>(0.375, 0.875, 0.125),
  Point<3>(0.625, -0.875, 0.125),
  Point<3>(0.625, -0.625, 0.125),
  Point<3>(0.625, -0.375, 0.125),
  Point<3>(0.625, -0.125, 0.125),
  Point<3>(0.625, 0.125, 0.125),
  Point<3>(0.625, 0.375, 0.125),
  Point<3>(0.625, 0.625, 0.125),
  Point<3>(0.625, 0.875, 0.125),
  Point<3>(0.875, -0.875, 0.125),
  Point<3>(0.875, -0.625, 0.125),
  Point<3>(0.875, -0.375, 0.125),
  Point<3>(0.875, -0.125, 0.125),
  Point<3>(0.875, 0.125, 0.125),
  Point<3>(0.875, 0.375, 0.125),
  Point<3>(0.875, 0.625, 0.125),
  Point<3>(0.875, 0.875, 0.125),
  Point<3>(-0.75, -0.75, 0.25),
  Point<3>(-0.75, -0.5, 0.25),
  Point<3>(-0.75, -0.25, 0.25),
  Point<3>(-0.75, 0, 0.25),
  Point<3>(-0.75, 0.25, 0.25),
  Point<3>(-0.75, 0.5, 0.25),
  Point<3>(-0.75, 0.75, 0.25),
  Point<3>(-0.5, -0.75, 0.25),
  Point<3>(-0.5, -0.5, 0.25),
  Point<3>(-0.5, -0.25, 0.25),
  Point<3>(-0.5, 0, 0.25),
  Point<3>(-0.5, 0.25, 0.25),
  Point<3>(-0.5, 0.5, 0.25),
  Point<3>(-0.5, 0.75, 0.25),
  Point<3>(-0.25, -0.75, 0.25),
  Point<3>(-0.25, -0.5, 0.25),
  Point<3>(-0.25, -0.25, 0.25),
  Point<3>(-0.25, 0, 0.25),
  Point<3>(-0.25, 0.25, 0.25),
  Point<3>(-0.25, 0.5, 0.25),
  Point<3>(-0.25, 0.75, 0.25),
  Point<3>(0, -0.75, 0.25),
  Point<3>(0, -0.5, 0.25),
  Point<3>(0, -0.25, 0.25),
  Point<3>(0, 0, 0.25),
  Point<3>(0, 0.25, 0.25),
  Point<3>(0, 0.5, 0.25),
  Point<3>(0, 0.75, 0.25),
  Point<3>(0.25, -0.75, 0.25),
  Point<3>(0.25, -0.5, 0.25),
  Point<3>(0.25, -0.25, 0.25),
  Point<3>(0.25, 0, 0.25),
  Point<3>(0.25, 0.25, 0.25),
  Point<3>(0.25, 0.5, 0.25),
  Point<3>(0.25, 0.75, 0.25),
  Point<3>(0.5, -0.75, 0.25),
  Point<3>(0.5, -0.5, 0.25),
  Point<3>(0.5, -0.25, 0.25),
  Point<3>(0.5, 0, 0.25),
  Point<3>(0.5, 0.25, 0.25),
  Point<3>(0.5, 0.5, 0.25),
  Point<3>(0.5, 0.75, 0.25),
  Point<3>(0.75, -0.75, 0.25),
  Point<3>(0.75, -0.5, 0.25),
  Point<3>(0.75, -0.25, 0.25),
  Point<3>(0.75, 0, 0.25),
  Point<3>(0.75, 0.25, 0.25),
  Point<3>(0.75, 0.5, 0.25),
  Point<3>(0.75, 0.75, 0.25),
  Point<3>(-0.625, -0.625, 0.375),
  Point<3>(-0.625, -0.375, 0.375),
  Point<3>(-0.625, -0.125, 0.375),
  Point<3>(-0.625, 0.125, 0.375),
  Point<3>(-0.625, 0.375, 0.375),
  Point<3>(-0.625, 0.625, 0.375),
  Point<3>(-0.375, -0.625, 0.375),
  Point<3>(-0.375, -0.375, 0.375),
  Point<3>(-0.375, -0.125, 0.375),
  Point<3>(-0.375, 0.125, 0.375),
  Point<3>(-0.375, 0.375, 0.375),
  Point<3>(-0.375, 0.625, 0.375),
  Point<3>(-0.125, -0.625, 0.375),
  Point<3>(-0.125, -0.375, 0.375),
  Point<3>(-0.125, -0.125, 0.375),
  Point<3>(-0.125, 0.125, 0.375),
  Point<3>(-0.125, 0.375, 0.375),
  Point<3>(-0.125, 0.625, 0.375),
  Point<3>(0.125, -0.625, 0.375),
  Point<3>(0.125, -0.375, 0.375),
  Point<3>(0.125, -0.125, 0.375),
  Point<3>(0.125, 0.125, 0.375),
  Point<3>(0.125, 0.375, 0.375),
  Point<3>(0.125, 0.625, 0.375),
  Point<3>(0.375, -0.625, 0.375),
  Point<3>(0.375, -0.375, 0.375),
  Point<3>(0.375, -0.125, 0.375),
  Point<3>(0.375, 0.125, 0.375),
  Point<3>(0.375, 0.375, 0.375),
  Point<3>(0.375, 0.625, 0.375),
  Point<3>(0.625, -0.625, 0.375),
  Point<3>(0.625, -0.375, 0.375),
  Point<3>(0.625, -0.125, 0.375),
  Point<3>(0.625, 0.125, 0.375),
  Point<3>(0.625, 0.375, 0.375),
  Point<3>(0.625, 0.625, 0.375),
  Point<3>(-0.5, -0.5, 0.5),
  Point<3>(-0.5, -0.25, 0.5),
  Point<3>(-0.5, 0, 0.5),
  Point<3>(-0.5, 0.25, 0.5),
  Point<3>(-0.5, 0.5, 0.5),
  Point<3>(-0.25, -0.5, 0.5),
  Point<3>(-0.25, -0.25, 0.5),
  Point<3>(-0.25, 0, 0.5),
  Point<3>(-0.25, 0.25, 0.5),
  Point<3>(-0.25, 0.5, 0.5),
  Point<3>(0, -0.5, 0.5),
  Point<3>(0, -0.25, 0.5),
  Point<3>(0, 0, 0.5),
  Point<3>(0, 0.25, 0.5),
  Point<3>(0, 0.5, 0.5),
  Point<3>(0.25, -0.5, 0.5),
  Point<3>(0.25, -0.25, 0.5),
  Point<3>(0.25, 0, 0.5),
  Point<3>(0.25, 0.25, 0.5),
  Point<3>(0.25, 0.5, 0.5),
  Point<3>(0.5, -0.5, 0.5),
  Point<3>(0.5, -0.25, 0.5),
  Point<3>(0.5, 0, 0.5),
  Point<3>(0.5, 0.25, 0.5),
  Point<3>(0.5, 0.5, 0.5),
  Point<3>(-0.375, -0.375, 0.625),
  Point<3>(-0.375, -0.125, 0.625),
  Point<3>(-0.375, 0.125, 0.625),
  Point<3>(-0.375, 0.375, 0.625),
  Point<3>(-0.125, -0.375, 0.625),
  Point<3>(-0.125, -0.125, 0.625),
  Point<3>(-0.125, 0.125, 0.625),
  Point<3>(-0.125, 0.375, 0.625),
  Point<3>(0.125, -0.375, 0.625),
  Point<3>(0.125, -0.125, 0.625),
  Point<3>(0.125, 0.125, 0.625),
  Point<3>(0.125, 0.375, 0.625),
  Point<3>(0.375, -0.375, 0.625),
  Point<3>(0.375, -0.125, 0.625),
  Point<3>(0.375, 0.125, 0.625),
  Point<3>(0.375, 0.375, 0.625),
  Point<3>(-0.25, -0.25, 0.75),
  Point<3>(-0.25, 0, 0.75),
  Point<3>(-0.25, 0.25, 0.75),
  Point<3>(0, -0.25, 0.75),
  Point<3>(0, 0, 0.75),
  Point<3>(0, 0.25, 0.75),
  Point<3>(0.25, -0.25, 0.75),
  Point<3>(0.25, 0, 0.75),
  Point<3>(0.25, 0.25, 0.75),
  Point<3>(-0.125, -0.125, 0.875),
  Point<3>(-0.125, 0.125, 0.875),
  Point<3>(0.125, -0.125, 0.875),
  Point<3>(0.125, 0.125, 0.875),
  Point<3>(0, 0, 1)};
std::vector<Point<3>> reference_points_equi_p9 = {
  Point<3>(-1, -1, 0),
  Point<3>(-1, -0.7777777777777778, 0),
  Point<3>(-1, -0.5555555555555556, 0),
  Point<3>(-1, -0.3333333333333333, 0),
  Point<3>(-1, -0.1111111111111111, 0),
  Point<3>(-1, 0.1111111111111111, 0),
  Point<3>(-1, 0.3333333333333333, 0),
  Point<3>(-1, 0.5555555555555556, 0),
  Point<3>(-1, 0.7777777777777778, 0),
  Point<3>(-1, 1, 0),
  Point<3>(-0.7777777777777778, -1, 0),
  Point<3>(-0.7777777777777778, -0.7777777777777778, 0),
  Point<3>(-0.7777777777777778, -0.5555555555555556, 0),
  Point<3>(-0.7777777777777778, -0.3333333333333333, 0),
  Point<3>(-0.7777777777777778, -0.1111111111111111, 0),
  Point<3>(-0.7777777777777778, 0.1111111111111111, 0),
  Point<3>(-0.7777777777777778, 0.3333333333333333, 0),
  Point<3>(-0.7777777777777778, 0.5555555555555556, 0),
  Point<3>(-0.7777777777777778, 0.7777777777777778, 0),
  Point<3>(-0.7777777777777778, 1, 0),
  Point<3>(-0.5555555555555556, -1, 0),
  Point<3>(-0.5555555555555556, -0.7777777777777778, 0),
  Point<3>(-0.5555555555555556, -0.5555555555555556, 0),
  Point<3>(-0.5555555555555556, -0.3333333333333333, 0),
  Point<3>(-0.5555555555555556, -0.1111111111111111, 0),
  Point<3>(-0.5555555555555556, 0.1111111111111111, 0),
  Point<3>(-0.5555555555555556, 0.3333333333333333, 0),
  Point<3>(-0.5555555555555556, 0.5555555555555556, 0),
  Point<3>(-0.5555555555555556, 0.7777777777777778, 0),
  Point<3>(-0.5555555555555556, 1, 0),
  Point<3>(-0.3333333333333333, -1, 0),
  Point<3>(-0.3333333333333333, -0.7777777777777778, 0),
  Point<3>(-0.3333333333333333, -0.5555555555555556, 0),
  Point<3>(-0.3333333333333333, -0.3333333333333333, 0),
  Point<3>(-0.3333333333333333, -0.1111111111111111, 0),
  Point<3>(-0.3333333333333333, 0.1111111111111111, 0),
  Point<3>(-0.3333333333333333, 0.3333333333333333, 0),
  Point<3>(-0.3333333333333333, 0.5555555555555556, 0),
  Point<3>(-0.3333333333333333, 0.7777777777777778, 0),
  Point<3>(-0.3333333333333333, 1, 0),
  Point<3>(-0.1111111111111111, -1, 0),
  Point<3>(-0.1111111111111111, -0.7777777777777778, 0),
  Point<3>(-0.1111111111111111, -0.5555555555555556, 0),
  Point<3>(-0.1111111111111111, -0.3333333333333333, 0),
  Point<3>(-0.1111111111111111, -0.1111111111111111, 0),
  Point<3>(-0.1111111111111111, 0.1111111111111111, 0),
  Point<3>(-0.1111111111111111, 0.3333333333333333, 0),
  Point<3>(-0.1111111111111111, 0.5555555555555556, 0),
  Point<3>(-0.1111111111111111, 0.7777777777777778, 0),
  Point<3>(-0.1111111111111111, 1, 0),
  Point<3>(0.1111111111111111, -1, 0),
  Point<3>(0.1111111111111111, -0.7777777777777778, 0),
  Point<3>(0.1111111111111111, -0.5555555555555556, 0),
  Point<3>(0.1111111111111111, -0.3333333333333333, 0),
  Point<3>(0.1111111111111111, -0.1111111111111111, 0),
  Point<3>(0.1111111111111111, 0.1111111111111111, 0),
  Point<3>(0.1111111111111111, 0.3333333333333333, 0),
  Point<3>(0.1111111111111111, 0.5555555555555556, 0),
  Point<3>(0.1111111111111111, 0.7777777777777778, 0),
  Point<3>(0.1111111111111111, 1, 0),
  Point<3>(0.3333333333333333, -1, 0),
  Point<3>(0.3333333333333333, -0.7777777777777778, 0),
  Point<3>(0.3333333333333333, -0.5555555555555556, 0),
  Point<3>(0.3333333333333333, -0.3333333333333333, 0),
  Point<3>(0.3333333333333333, -0.1111111111111111, 0),
  Point<3>(0.3333333333333333, 0.1111111111111111, 0),
  Point<3>(0.3333333333333333, 0.3333333333333333, 0),
  Point<3>(0.3333333333333333, 0.5555555555555556, 0),
  Point<3>(0.3333333333333333, 0.7777777777777778, 0),
  Point<3>(0.3333333333333333, 1, 0),
  Point<3>(0.5555555555555556, -1, 0),
  Point<3>(0.5555555555555556, -0.7777777777777778, 0),
  Point<3>(0.5555555555555556, -0.5555555555555556, 0),
  Point<3>(0.5555555555555556, -0.3333333333333333, 0),
  Point<3>(0.5555555555555556, -0.1111111111111111, 0),
  Point<3>(0.5555555555555556, 0.1111111111111111, 0),
  Point<3>(0.5555555555555556, 0.3333333333333333, 0),
  Point<3>(0.5555555555555556, 0.5555555555555556, 0),
  Point<3>(0.5555555555555556, 0.7777777777777778, 0),
  Point<3>(0.5555555555555556, 1, 0),
  Point<3>(0.7777777777777778, -1, 0),
  Point<3>(0.7777777777777778, -0.7777777777777778, 0),
  Point<3>(0.7777777777777778, -0.5555555555555556, 0),
  Point<3>(0.7777777777777778, -0.3333333333333333, 0),
  Point<3>(0.7777777777777778, -0.1111111111111111, 0),
  Point<3>(0.7777777777777778, 0.1111111111111111, 0),
  Point<3>(0.7777777777777778, 0.3333333333333333, 0),
  Point<3>(0.7777777777777778, 0.5555555555555556, 0),
  Point<3>(0.7777777777777778, 0.7777777777777778, 0),
  Point<3>(0.7777777777777778, 1, 0),
  Point<3>(1, -1, 0),
  Point<3>(1, -0.7777777777777778, 0),
  Point<3>(1, -0.5555555555555556, 0),
  Point<3>(1, -0.3333333333333333, 0),
  Point<3>(1, -0.1111111111111111, 0),
  Point<3>(1, 0.1111111111111111, 0),
  Point<3>(1, 0.3333333333333333, 0),
  Point<3>(1, 0.5555555555555556, 0),
  Point<3>(1, 0.7777777777777778, 0),
  Point<3>(1, 1, 0),
  Point<3>(-0.8888888888888888, -0.8888888888888888, 0.1111111111111111),
  Point<3>(-0.8888888888888888, -0.6666666666666666, 0.1111111111111111),
  Point<3>(-0.8888888888888888, -0.4444444444444444, 0.1111111111111111),
  Point<3>(-0.8888888888888888, -0.2222222222222222, 0.1111111111111111),
  Point<3>(-0.8888888888888888, 0, 0.1111111111111111),
  Point<3>(-0.8888888888888888, 0.2222222222222222, 0.1111111111111111),
  Point<3>(-0.8888888888888888, 0.4444444444444444, 0.1111111111111111),
  Point<3>(-0.8888888888888888, 0.6666666666666666, 0.1111111111111111),
  Point<3>(-0.8888888888888888, 0.8888888888888888, 0.1111111111111111),
  Point<3>(-0.6666666666666666, -0.8888888888888888, 0.1111111111111111),
  Point<3>(-0.6666666666666666, -0.6666666666666666, 0.1111111111111111),
  Point<3>(-0.6666666666666666, -0.4444444444444444, 0.1111111111111111),
  Point<3>(-0.6666666666666666, -0.2222222222222222, 0.1111111111111111),
  Point<3>(-0.6666666666666666, 0, 0.1111111111111111),
  Point<3>(-0.6666666666666666, 0.2222222222222222, 0.1111111111111111),
  Point<3>(-0.6666666666666666, 0.4444444444444444, 0.1111111111111111),
  Point<3>(-0.6666666666666666, 0.6666666666666666, 0.1111111111111111),
  Point<3>(-0.6666666666666666, 0.8888888888888888, 0.1111111111111111),
  Point<3>(-0.4444444444444444, -0.8888888888888888, 0.1111111111111111),
  Point<3>(-0.4444444444444444, -0.6666666666666666, 0.1111111111111111),
  Point<3>(-0.4444444444444444, -0.4444444444444444, 0.1111111111111111),
  Point<3>(-0.4444444444444444, -0.2222222222222222, 0.1111111111111111),
  Point<3>(-0.4444444444444444, 0, 0.1111111111111111),
  Point<3>(-0.4444444444444444, 0.2222222222222222, 0.1111111111111111),
  Point<3>(-0.4444444444444444, 0.4444444444444444, 0.1111111111111111),
  Point<3>(-0.4444444444444444, 0.6666666666666666, 0.1111111111111111),
  Point<3>(-0.4444444444444444, 0.8888888888888888, 0.1111111111111111),
  Point<3>(-0.2222222222222222, -0.8888888888888888, 0.1111111111111111),
  Point<3>(-0.2222222222222222, -0.6666666666666666, 0.1111111111111111),
  Point<3>(-0.2222222222222222, -0.4444444444444444, 0.1111111111111111),
  Point<3>(-0.2222222222222222, -0.2222222222222222, 0.1111111111111111),
  Point<3>(-0.2222222222222222, 0, 0.1111111111111111),
  Point<3>(-0.2222222222222222, 0.2222222222222222, 0.1111111111111111),
  Point<3>(-0.2222222222222222, 0.4444444444444444, 0.1111111111111111),
  Point<3>(-0.2222222222222222, 0.6666666666666666, 0.1111111111111111),
  Point<3>(-0.2222222222222222, 0.8888888888888888, 0.1111111111111111),
  Point<3>(0, -0.8888888888888888, 0.1111111111111111),
  Point<3>(0, -0.6666666666666666, 0.1111111111111111),
  Point<3>(0, -0.4444444444444444, 0.1111111111111111),
  Point<3>(0, -0.2222222222222222, 0.1111111111111111),
  Point<3>(0, 0, 0.1111111111111111),
  Point<3>(0, 0.2222222222222222, 0.1111111111111111),
  Point<3>(0, 0.4444444444444444, 0.1111111111111111),
  Point<3>(0, 0.6666666666666666, 0.1111111111111111),
  Point<3>(0, 0.8888888888888888, 0.1111111111111111),
  Point<3>(0.2222222222222222, -0.8888888888888888, 0.1111111111111111),
  Point<3>(0.2222222222222222, -0.6666666666666666, 0.1111111111111111),
  Point<3>(0.2222222222222222, -0.4444444444444444, 0.1111111111111111),
  Point<3>(0.2222222222222222, -0.2222222222222222, 0.1111111111111111),
  Point<3>(0.2222222222222222, 0, 0.1111111111111111),
  Point<3>(0.2222222222222222, 0.2222222222222222, 0.1111111111111111),
  Point<3>(0.2222222222222222, 0.4444444444444444, 0.1111111111111111),
  Point<3>(0.2222222222222222, 0.6666666666666666, 0.1111111111111111),
  Point<3>(0.2222222222222222, 0.8888888888888888, 0.1111111111111111),
  Point<3>(0.4444444444444444, -0.8888888888888888, 0.1111111111111111),
  Point<3>(0.4444444444444444, -0.6666666666666666, 0.1111111111111111),
  Point<3>(0.4444444444444444, -0.4444444444444444, 0.1111111111111111),
  Point<3>(0.4444444444444444, -0.2222222222222222, 0.1111111111111111),
  Point<3>(0.4444444444444444, 0, 0.1111111111111111),
  Point<3>(0.4444444444444444, 0.2222222222222222, 0.1111111111111111),
  Point<3>(0.4444444444444444, 0.4444444444444444, 0.1111111111111111),
  Point<3>(0.4444444444444444, 0.6666666666666666, 0.1111111111111111),
  Point<3>(0.4444444444444444, 0.8888888888888888, 0.1111111111111111),
  Point<3>(0.6666666666666666, -0.8888888888888888, 0.1111111111111111),
  Point<3>(0.6666666666666666, -0.6666666666666666, 0.1111111111111111),
  Point<3>(0.6666666666666666, -0.4444444444444444, 0.1111111111111111),
  Point<3>(0.6666666666666666, -0.2222222222222222, 0.1111111111111111),
  Point<3>(0.6666666666666666, 0, 0.1111111111111111),
  Point<3>(0.6666666666666666, 0.2222222222222222, 0.1111111111111111),
  Point<3>(0.6666666666666666, 0.4444444444444444, 0.1111111111111111),
  Point<3>(0.6666666666666666, 0.6666666666666666, 0.1111111111111111),
  Point<3>(0.6666666666666666, 0.8888888888888888, 0.1111111111111111),
  Point<3>(0.8888888888888888, -0.8888888888888888, 0.1111111111111111),
  Point<3>(0.8888888888888888, -0.6666666666666666, 0.1111111111111111),
  Point<3>(0.8888888888888888, -0.4444444444444444, 0.1111111111111111),
  Point<3>(0.8888888888888888, -0.2222222222222222, 0.1111111111111111),
  Point<3>(0.8888888888888888, 0, 0.1111111111111111),
  Point<3>(0.8888888888888888, 0.2222222222222222, 0.1111111111111111),
  Point<3>(0.8888888888888888, 0.4444444444444444, 0.1111111111111111),
  Point<3>(0.8888888888888888, 0.6666666666666666, 0.1111111111111111),
  Point<3>(0.8888888888888888, 0.8888888888888888, 0.1111111111111111),
  Point<3>(-0.7777777777777778, -0.7777777777777778, 0.2222222222222222),
  Point<3>(-0.7777777777777778, -0.5555555555555556, 0.2222222222222222),
  Point<3>(-0.7777777777777778, -0.3333333333333333, 0.2222222222222222),
  Point<3>(-0.7777777777777778, -0.1111111111111111, 0.2222222222222222),
  Point<3>(-0.7777777777777778, 0.1111111111111111, 0.2222222222222222),
  Point<3>(-0.7777777777777778, 0.3333333333333333, 0.2222222222222222),
  Point<3>(-0.7777777777777778, 0.5555555555555556, 0.2222222222222222),
  Point<3>(-0.7777777777777778, 0.7777777777777778, 0.2222222222222222),
  Point<3>(-0.5555555555555556, -0.7777777777777778, 0.2222222222222222),
  Point<3>(-0.5555555555555556, -0.5555555555555556, 0.2222222222222222),
  Point<3>(-0.5555555555555556, -0.3333333333333333, 0.2222222222222222),
  Point<3>(-0.5555555555555556, -0.1111111111111111, 0.2222222222222222),
  Point<3>(-0.5555555555555556, 0.1111111111111111, 0.2222222222222222),
  Point<3>(-0.5555555555555556, 0.3333333333333333, 0.2222222222222222),
  Point<3>(-0.5555555555555556, 0.5555555555555556, 0.2222222222222222),
  Point<3>(-0.5555555555555556, 0.7777777777777778, 0.2222222222222222),
  Point<3>(-0.3333333333333333, -0.7777777777777778, 0.2222222222222222),
  Point<3>(-0.3333333333333333, -0.5555555555555556, 0.2222222222222222),
  Point<3>(-0.3333333333333333, -0.3333333333333333, 0.2222222222222222),
  Point<3>(-0.3333333333333333, -0.1111111111111111, 0.2222222222222222),
  Point<3>(-0.3333333333333333, 0.1111111111111111, 0.2222222222222222),
  Point<3>(-0.3333333333333333, 0.3333333333333333, 0.2222222222222222),
  Point<3>(-0.3333333333333333, 0.5555555555555556, 0.2222222222222222),
  Point<3>(-0.3333333333333333, 0.7777777777777778, 0.2222222222222222),
  Point<3>(-0.1111111111111111, -0.7777777777777778, 0.2222222222222222),
  Point<3>(-0.1111111111111111, -0.5555555555555556, 0.2222222222222222),
  Point<3>(-0.1111111111111111, -0.3333333333333333, 0.2222222222222222),
  Point<3>(-0.1111111111111111, -0.1111111111111111, 0.2222222222222222),
  Point<3>(-0.1111111111111111, 0.1111111111111111, 0.2222222222222222),
  Point<3>(-0.1111111111111111, 0.3333333333333333, 0.2222222222222222),
  Point<3>(-0.1111111111111111, 0.5555555555555556, 0.2222222222222222),
  Point<3>(-0.1111111111111111, 0.7777777777777778, 0.2222222222222222),
  Point<3>(0.1111111111111111, -0.7777777777777778, 0.2222222222222222),
  Point<3>(0.1111111111111111, -0.5555555555555556, 0.2222222222222222),
  Point<3>(0.1111111111111111, -0.3333333333333333, 0.2222222222222222),
  Point<3>(0.1111111111111111, -0.1111111111111111, 0.2222222222222222),
  Point<3>(0.1111111111111111, 0.1111111111111111, 0.2222222222222222),
  Point<3>(0.1111111111111111, 0.3333333333333333, 0.2222222222222222),
  Point<3>(0.1111111111111111, 0.5555555555555556, 0.2222222222222222),
  Point<3>(0.1111111111111111, 0.7777777777777778, 0.2222222222222222),
  Point<3>(0.3333333333333333, -0.7777777777777778, 0.2222222222222222),
  Point<3>(0.3333333333333333, -0.5555555555555556, 0.2222222222222222),
  Point<3>(0.3333333333333333, -0.3333333333333333, 0.2222222222222222),
  Point<3>(0.3333333333333333, -0.1111111111111111, 0.2222222222222222),
  Point<3>(0.3333333333333333, 0.1111111111111111, 0.2222222222222222),
  Point<3>(0.3333333333333333, 0.3333333333333333, 0.2222222222222222),
  Point<3>(0.3333333333333333, 0.5555555555555556, 0.2222222222222222),
  Point<3>(0.3333333333333333, 0.7777777777777778, 0.2222222222222222),
  Point<3>(0.5555555555555556, -0.7777777777777778, 0.2222222222222222),
  Point<3>(0.5555555555555556, -0.5555555555555556, 0.2222222222222222),
  Point<3>(0.5555555555555556, -0.3333333333333333, 0.2222222222222222),
  Point<3>(0.5555555555555556, -0.1111111111111111, 0.2222222222222222),
  Point<3>(0.5555555555555556, 0.1111111111111111, 0.2222222222222222),
  Point<3>(0.5555555555555556, 0.3333333333333333, 0.2222222222222222),
  Point<3>(0.5555555555555556, 0.5555555555555556, 0.2222222222222222),
  Point<3>(0.5555555555555556, 0.7777777777777778, 0.2222222222222222),
  Point<3>(0.7777777777777778, -0.7777777777777778, 0.2222222222222222),
  Point<3>(0.7777777777777778, -0.5555555555555556, 0.2222222222222222),
  Point<3>(0.7777777777777778, -0.3333333333333333, 0.2222222222222222),
  Point<3>(0.7777777777777778, -0.1111111111111111, 0.2222222222222222),
  Point<3>(0.7777777777777778, 0.1111111111111111, 0.2222222222222222),
  Point<3>(0.7777777777777778, 0.3333333333333333, 0.2222222222222222),
  Point<3>(0.7777777777777778, 0.5555555555555556, 0.2222222222222222),
  Point<3>(0.7777777777777778, 0.7777777777777778, 0.2222222222222222),
  Point<3>(-0.6666666666666667, -0.6666666666666667, 0.3333333333333333),
  Point<3>(-0.6666666666666667, -0.4444444444444445, 0.3333333333333333),
  Point<3>(-0.6666666666666667, -0.2222222222222222, 0.3333333333333333),
  Point<3>(-0.6666666666666667, 0, 0.3333333333333333),
  Point<3>(-0.6666666666666667, 0.2222222222222222, 0.3333333333333333),
  Point<3>(-0.6666666666666667, 0.4444444444444445, 0.3333333333333333),
  Point<3>(-0.6666666666666667, 0.6666666666666667, 0.3333333333333333),
  Point<3>(-0.4444444444444445, -0.6666666666666667, 0.3333333333333333),
  Point<3>(-0.4444444444444445, -0.4444444444444445, 0.3333333333333333),
  Point<3>(-0.4444444444444445, -0.2222222222222222, 0.3333333333333333),
  Point<3>(-0.4444444444444445, 0, 0.3333333333333333),
  Point<3>(-0.4444444444444445, 0.2222222222222222, 0.3333333333333333),
  Point<3>(-0.4444444444444445, 0.4444444444444445, 0.3333333333333333),
  Point<3>(-0.4444444444444445, 0.6666666666666667, 0.3333333333333333),
  Point<3>(-0.2222222222222222, -0.6666666666666667, 0.3333333333333333),
  Point<3>(-0.2222222222222222, -0.4444444444444445, 0.3333333333333333),
  Point<3>(-0.2222222222222222, -0.2222222222222222, 0.3333333333333333),
  Point<3>(-0.2222222222222222, 0, 0.3333333333333333),
  Point<3>(-0.2222222222222222, 0.2222222222222222, 0.3333333333333333),
  Point<3>(-0.2222222222222222, 0.4444444444444445, 0.3333333333333333),
  Point<3>(-0.2222222222222222, 0.6666666666666667, 0.3333333333333333),
  Point<3>(0, -0.6666666666666667, 0.3333333333333333),
  Point<3>(0, -0.4444444444444445, 0.3333333333333333),
  Point<3>(0, -0.2222222222222222, 0.3333333333333333),
  Point<3>(0, 0, 0.3333333333333333),
  Point<3>(0, 0.2222222222222222, 0.3333333333333333),
  Point<3>(0, 0.4444444444444445, 0.3333333333333333),
  Point<3>(0, 0.6666666666666667, 0.3333333333333333),
  Point<3>(0.2222222222222222, -0.6666666666666667, 0.3333333333333333),
  Point<3>(0.2222222222222222, -0.4444444444444445, 0.3333333333333333),
  Point<3>(0.2222222222222222, -0.2222222222222222, 0.3333333333333333),
  Point<3>(0.2222222222222222, 0, 0.3333333333333333),
  Point<3>(0.2222222222222222, 0.2222222222222222, 0.3333333333333333),
  Point<3>(0.2222222222222222, 0.4444444444444445, 0.3333333333333333),
  Point<3>(0.2222222222222222, 0.6666666666666667, 0.3333333333333333),
  Point<3>(0.4444444444444445, -0.6666666666666667, 0.3333333333333333),
  Point<3>(0.4444444444444445, -0.4444444444444445, 0.3333333333333333),
  Point<3>(0.4444444444444445, -0.2222222222222222, 0.3333333333333333),
  Point<3>(0.4444444444444445, 0, 0.3333333333333333),
  Point<3>(0.4444444444444445, 0.2222222222222222, 0.3333333333333333),
  Point<3>(0.4444444444444445, 0.4444444444444445, 0.3333333333333333),
  Point<3>(0.4444444444444445, 0.6666666666666667, 0.3333333333333333),
  Point<3>(0.6666666666666667, -0.6666666666666667, 0.3333333333333333),
  Point<3>(0.6666666666666667, -0.4444444444444445, 0.3333333333333333),
  Point<3>(0.6666666666666667, -0.2222222222222222, 0.3333333333333333),
  Point<3>(0.6666666666666667, 0, 0.3333333333333333),
  Point<3>(0.6666666666666667, 0.2222222222222222, 0.3333333333333333),
  Point<3>(0.6666666666666667, 0.4444444444444445, 0.3333333333333333),
  Point<3>(0.6666666666666667, 0.6666666666666667, 0.3333333333333333),
  Point<3>(-0.5555555555555556, -0.5555555555555556, 0.4444444444444444),
  Point<3>(-0.5555555555555556, -0.3333333333333333, 0.4444444444444444),
  Point<3>(-0.5555555555555556, -0.1111111111111111, 0.4444444444444444),
  Point<3>(-0.5555555555555556, 0.1111111111111111, 0.4444444444444444),
  Point<3>(-0.5555555555555556, 0.3333333333333333, 0.4444444444444444),
  Point<3>(-0.5555555555555556, 0.5555555555555556, 0.4444444444444444),
  Point<3>(-0.3333333333333333, -0.5555555555555556, 0.4444444444444444),
  Point<3>(-0.3333333333333333, -0.3333333333333333, 0.4444444444444444),
  Point<3>(-0.3333333333333333, -0.1111111111111111, 0.4444444444444444),
  Point<3>(-0.3333333333333333, 0.1111111111111111, 0.4444444444444444),
  Point<3>(-0.3333333333333333, 0.3333333333333333, 0.4444444444444444),
  Point<3>(-0.3333333333333333, 0.5555555555555556, 0.4444444444444444),
  Point<3>(-0.1111111111111111, -0.5555555555555556, 0.4444444444444444),
  Point<3>(-0.1111111111111111, -0.3333333333333333, 0.4444444444444444),
  Point<3>(-0.1111111111111111, -0.1111111111111111, 0.4444444444444444),
  Point<3>(-0.1111111111111111, 0.1111111111111111, 0.4444444444444444),
  Point<3>(-0.1111111111111111, 0.3333333333333333, 0.4444444444444444),
  Point<3>(-0.1111111111111111, 0.5555555555555556, 0.4444444444444444),
  Point<3>(0.1111111111111111, -0.5555555555555556, 0.4444444444444444),
  Point<3>(0.1111111111111111, -0.3333333333333333, 0.4444444444444444),
  Point<3>(0.1111111111111111, -0.1111111111111111, 0.4444444444444444),
  Point<3>(0.1111111111111111, 0.1111111111111111, 0.4444444444444444),
  Point<3>(0.1111111111111111, 0.3333333333333333, 0.4444444444444444),
  Point<3>(0.1111111111111111, 0.5555555555555556, 0.4444444444444444),
  Point<3>(0.3333333333333333, -0.5555555555555556, 0.4444444444444444),
  Point<3>(0.3333333333333333, -0.3333333333333333, 0.4444444444444444),
  Point<3>(0.3333333333333333, -0.1111111111111111, 0.4444444444444444),
  Point<3>(0.3333333333333333, 0.1111111111111111, 0.4444444444444444),
  Point<3>(0.3333333333333333, 0.3333333333333333, 0.4444444444444444),
  Point<3>(0.3333333333333333, 0.5555555555555556, 0.4444444444444444),
  Point<3>(0.5555555555555556, -0.5555555555555556, 0.4444444444444444),
  Point<3>(0.5555555555555556, -0.3333333333333333, 0.4444444444444444),
  Point<3>(0.5555555555555556, -0.1111111111111111, 0.4444444444444444),
  Point<3>(0.5555555555555556, 0.1111111111111111, 0.4444444444444444),
  Point<3>(0.5555555555555556, 0.3333333333333333, 0.4444444444444444),
  Point<3>(0.5555555555555556, 0.5555555555555556, 0.4444444444444444),
  Point<3>(-0.4444444444444444, -0.4444444444444444, 0.5555555555555556),
  Point<3>(-0.4444444444444444, -0.2222222222222222, 0.5555555555555556),
  Point<3>(-0.4444444444444444, 0, 0.5555555555555556),
  Point<3>(-0.4444444444444444, 0.2222222222222222, 0.5555555555555556),
  Point<3>(-0.4444444444444444, 0.4444444444444444, 0.5555555555555556),
  Point<3>(-0.2222222222222222, -0.4444444444444444, 0.5555555555555556),
  Point<3>(-0.2222222222222222, -0.2222222222222222, 0.5555555555555556),
  Point<3>(-0.2222222222222222, 0, 0.5555555555555556),
  Point<3>(-0.2222222222222222, 0.2222222222222222, 0.5555555555555556),
  Point<3>(-0.2222222222222222, 0.4444444444444444, 0.5555555555555556),
  Point<3>(0, -0.4444444444444444, 0.5555555555555556),
  Point<3>(0, -0.2222222222222222, 0.5555555555555556),
  Point<3>(0, 0, 0.5555555555555556),
  Point<3>(0, 0.2222222222222222, 0.5555555555555556),
  Point<3>(0, 0.4444444444444444, 0.5555555555555556),
  Point<3>(0.2222222222222222, -0.4444444444444444, 0.5555555555555556),
  Point<3>(0.2222222222222222, -0.2222222222222222, 0.5555555555555556),
  Point<3>(0.2222222222222222, 0, 0.5555555555555556),
  Point<3>(0.2222222222222222, 0.2222222222222222, 0.5555555555555556),
  Point<3>(0.2222222222222222, 0.4444444444444444, 0.5555555555555556),
  Point<3>(0.4444444444444444, -0.4444444444444444, 0.5555555555555556),
  Point<3>(0.4444444444444444, -0.2222222222222222, 0.5555555555555556),
  Point<3>(0.4444444444444444, 0, 0.5555555555555556),
  Point<3>(0.4444444444444444, 0.2222222222222222, 0.5555555555555556),
  Point<3>(0.4444444444444444, 0.4444444444444444, 0.5555555555555556),
  Point<3>(-0.3333333333333334, -0.3333333333333334, 0.6666666666666666),
  Point<3>(-0.3333333333333334, -0.1111111111111111, 0.6666666666666666),
  Point<3>(-0.3333333333333334, 0.1111111111111111, 0.6666666666666666),
  Point<3>(-0.3333333333333334, 0.3333333333333334, 0.6666666666666666),
  Point<3>(-0.1111111111111111, -0.3333333333333334, 0.6666666666666666),
  Point<3>(-0.1111111111111111, -0.1111111111111111, 0.6666666666666666),
  Point<3>(-0.1111111111111111, 0.1111111111111111, 0.6666666666666666),
  Point<3>(-0.1111111111111111, 0.3333333333333334, 0.6666666666666666),
  Point<3>(0.1111111111111111, -0.3333333333333334, 0.6666666666666666),
  Point<3>(0.1111111111111111, -0.1111111111111111, 0.6666666666666666),
  Point<3>(0.1111111111111111, 0.1111111111111111, 0.6666666666666666),
  Point<3>(0.1111111111111111, 0.3333333333333334, 0.6666666666666666),
  Point<3>(0.3333333333333334, -0.3333333333333334, 0.6666666666666666),
  Point<3>(0.3333333333333334, -0.1111111111111111, 0.6666666666666666),
  Point<3>(0.3333333333333334, 0.1111111111111111, 0.6666666666666666),
  Point<3>(0.3333333333333334, 0.3333333333333334, 0.6666666666666666),
  Point<3>(-0.2222222222222222, -0.2222222222222222, 0.7777777777777778),
  Point<3>(-0.2222222222222222, 0, 0.7777777777777778),
  Point<3>(-0.2222222222222222, 0.2222222222222222, 0.7777777777777778),
  Point<3>(0, -0.2222222222222222, 0.7777777777777778),
  Point<3>(0, 0, 0.7777777777777778),
  Point<3>(0, 0.2222222222222222, 0.7777777777777778),
  Point<3>(0.2222222222222222, -0.2222222222222222, 0.7777777777777778),
  Point<3>(0.2222222222222222, 0, 0.7777777777777778),
  Point<3>(0.2222222222222222, 0.2222222222222222, 0.7777777777777778),
  Point<3>(-0.1111111111111112, -0.1111111111111112, 0.8888888888888888),
  Point<3>(-0.1111111111111112, 0.1111111111111112, 0.8888888888888888),
  Point<3>(0.1111111111111112, -0.1111111111111112, 0.8888888888888888),
  Point<3>(0.1111111111111112, 0.1111111111111112, 0.8888888888888888),
  Point<3>(0, 0, 1)};
std::vector<Point<3>> reference_points_equi_p10 = {
  Point<3>(-1, -1, 0),
  Point<3>(-1, -0.8, 0),
  Point<3>(-1, -0.6, 0),
  Point<3>(-1, -0.4, 0),
  Point<3>(-1, -0.2, 0),
  Point<3>(-1, 0, 0),
  Point<3>(-1, 0.2, 0),
  Point<3>(-1, 0.4, 0),
  Point<3>(-1, 0.6, 0),
  Point<3>(-1, 0.8, 0),
  Point<3>(-1, 1, 0),
  Point<3>(-0.8, -1, 0),
  Point<3>(-0.8, -0.8, 0),
  Point<3>(-0.8, -0.6, 0),
  Point<3>(-0.8, -0.4, 0),
  Point<3>(-0.8, -0.2, 0),
  Point<3>(-0.8, 0, 0),
  Point<3>(-0.8, 0.2, 0),
  Point<3>(-0.8, 0.4, 0),
  Point<3>(-0.8, 0.6, 0),
  Point<3>(-0.8, 0.8, 0),
  Point<3>(-0.8, 1, 0),
  Point<3>(-0.6, -1, 0),
  Point<3>(-0.6, -0.8, 0),
  Point<3>(-0.6, -0.6, 0),
  Point<3>(-0.6, -0.4, 0),
  Point<3>(-0.6, -0.2, 0),
  Point<3>(-0.6, 0, 0),
  Point<3>(-0.6, 0.2, 0),
  Point<3>(-0.6, 0.4, 0),
  Point<3>(-0.6, 0.6, 0),
  Point<3>(-0.6, 0.8, 0),
  Point<3>(-0.6, 1, 0),
  Point<3>(-0.4, -1, 0),
  Point<3>(-0.4, -0.8, 0),
  Point<3>(-0.4, -0.6, 0),
  Point<3>(-0.4, -0.4, 0),
  Point<3>(-0.4, -0.2, 0),
  Point<3>(-0.4, 0, 0),
  Point<3>(-0.4, 0.2, 0),
  Point<3>(-0.4, 0.4, 0),
  Point<3>(-0.4, 0.6, 0),
  Point<3>(-0.4, 0.8, 0),
  Point<3>(-0.4, 1, 0),
  Point<3>(-0.2, -1, 0),
  Point<3>(-0.2, -0.8, 0),
  Point<3>(-0.2, -0.6, 0),
  Point<3>(-0.2, -0.4, 0),
  Point<3>(-0.2, -0.2, 0),
  Point<3>(-0.2, 0, 0),
  Point<3>(-0.2, 0.2, 0),
  Point<3>(-0.2, 0.4, 0),
  Point<3>(-0.2, 0.6, 0),
  Point<3>(-0.2, 0.8, 0),
  Point<3>(-0.2, 1, 0),
  Point<3>(0, -1, 0),
  Point<3>(0, -0.8, 0),
  Point<3>(0, -0.6, 0),
  Point<3>(0, -0.4, 0),
  Point<3>(0, -0.2, 0),
  Point<3>(0, 0, 0),
  Point<3>(0, 0.2, 0),
  Point<3>(0, 0.4, 0),
  Point<3>(0, 0.6, 0),
  Point<3>(0, 0.8, 0),
  Point<3>(0, 1, 0),
  Point<3>(0.2, -1, 0),
  Point<3>(0.2, -0.8, 0),
  Point<3>(0.2, -0.6, 0),
  Point<3>(0.2, -0.4, 0),
  Point<3>(0.2, -0.2, 0),
  Point<3>(0.2, 0, 0),
  Point<3>(0.2, 0.2, 0),
  Point<3>(0.2, 0.4, 0),
  Point<3>(0.2, 0.6, 0),
  Point<3>(0.2, 0.8, 0),
  Point<3>(0.2, 1, 0),
  Point<3>(0.4, -1, 0),
  Point<3>(0.4, -0.8, 0),
  Point<3>(0.4, -0.6, 0),
  Point<3>(0.4, -0.4, 0),
  Point<3>(0.4, -0.2, 0),
  Point<3>(0.4, 0, 0),
  Point<3>(0.4, 0.2, 0),
  Point<3>(0.4, 0.4, 0),
  Point<3>(0.4, 0.6, 0),
  Point<3>(0.4, 0.8, 0),
  Point<3>(0.4, 1, 0),
  Point<3>(0.6, -1, 0),
  Point<3>(0.6, -0.8, 0),
  Point<3>(0.6, -0.6, 0),
  Point<3>(0.6, -0.4, 0),
  Point<3>(0.6, -0.2, 0),
  Point<3>(0.6, 0, 0),
  Point<3>(0.6, 0.2, 0),
  Point<3>(0.6, 0.4, 0),
  Point<3>(0.6, 0.6, 0),
  Point<3>(0.6, 0.8, 0),
  Point<3>(0.6, 1, 0),
  Point<3>(0.8, -1, 0),
  Point<3>(0.8, -0.8, 0),
  Point<3>(0.8, -0.6, 0),
  Point<3>(0.8, -0.4, 0),
  Point<3>(0.8, -0.2, 0),
  Point<3>(0.8, 0, 0),
  Point<3>(0.8, 0.2, 0),
  Point<3>(0.8, 0.4, 0),
  Point<3>(0.8, 0.6, 0),
  Point<3>(0.8, 0.8, 0),
  Point<3>(0.8, 1, 0),
  Point<3>(1, -1, 0),
  Point<3>(1, -0.8, 0),
  Point<3>(1, -0.6, 0),
  Point<3>(1, -0.4, 0),
  Point<3>(1, -0.2, 0),
  Point<3>(1, 0, 0),
  Point<3>(1, 0.2, 0),
  Point<3>(1, 0.4, 0),
  Point<3>(1, 0.6, 0),
  Point<3>(1, 0.8, 0),
  Point<3>(1, 1, 0),
  Point<3>(-0.9, -0.9, 0.1),
  Point<3>(-0.9, -0.7000000000000001, 0.1),
  Point<3>(-0.9, -0.5, 0.1),
  Point<3>(-0.9, -0.3, 0.1),
  Point<3>(-0.9, -0.09999999999999999, 0.1),
  Point<3>(-0.9, 0.09999999999999999, 0.1),
  Point<3>(-0.9, 0.3, 0.1),
  Point<3>(-0.9, 0.5, 0.1),
  Point<3>(-0.9, 0.7000000000000001, 0.1),
  Point<3>(-0.9, 0.9, 0.1),
  Point<3>(-0.7000000000000001, -0.9, 0.1),
  Point<3>(-0.7000000000000001, -0.7000000000000001, 0.1),
  Point<3>(-0.7000000000000001, -0.5, 0.1),
  Point<3>(-0.7000000000000001, -0.3, 0.1),
  Point<3>(-0.7000000000000001, -0.09999999999999999, 0.1),
  Point<3>(-0.7000000000000001, 0.09999999999999999, 0.1),
  Point<3>(-0.7000000000000001, 0.3, 0.1),
  Point<3>(-0.7000000000000001, 0.5, 0.1),
  Point<3>(-0.7000000000000001, 0.7000000000000001, 0.1),
  Point<3>(-0.7000000000000001, 0.9, 0.1),
  Point<3>(-0.5, -0.9, 0.1),
  Point<3>(-0.5, -0.7000000000000001, 0.1),
  Point<3>(-0.5, -0.5, 0.1),
  Point<3>(-0.5, -0.3, 0.1),
  Point<3>(-0.5, -0.09999999999999999, 0.1),
  Point<3>(-0.5, 0.09999999999999999, 0.1),
  Point<3>(-0.5, 0.3, 0.1),
  Point<3>(-0.5, 0.5, 0.1),
  Point<3>(-0.5, 0.7000000000000001, 0.1),
  Point<3>(-0.5, 0.9, 0.1),
  Point<3>(-0.3, -0.9, 0.1),
  Point<3>(-0.3, -0.7000000000000001, 0.1),
  Point<3>(-0.3, -0.5, 0.1),
  Point<3>(-0.3, -0.3, 0.1),
  Point<3>(-0.3, -0.09999999999999999, 0.1),
  Point<3>(-0.3, 0.09999999999999999, 0.1),
  Point<3>(-0.3, 0.3, 0.1),
  Point<3>(-0.3, 0.5, 0.1),
  Point<3>(-0.3, 0.7000000000000001, 0.1),
  Point<3>(-0.3, 0.9, 0.1),
  Point<3>(-0.09999999999999999, -0.9, 0.1),
  Point<3>(-0.09999999999999999, -0.7000000000000001, 0.1),
  Point<3>(-0.09999999999999999, -0.5, 0.1),
  Point<3>(-0.09999999999999999, -0.3, 0.1),
  Point<3>(-0.09999999999999999, -0.09999999999999999, 0.1),
  Point<3>(-0.09999999999999999, 0.09999999999999999, 0.1),
  Point<3>(-0.09999999999999999, 0.3, 0.1),
  Point<3>(-0.09999999999999999, 0.5, 0.1),
  Point<3>(-0.09999999999999999, 0.7000000000000001, 0.1),
  Point<3>(-0.09999999999999999, 0.9, 0.1),
  Point<3>(0.09999999999999999, -0.9, 0.1),
  Point<3>(0.09999999999999999, -0.7000000000000001, 0.1),
  Point<3>(0.09999999999999999, -0.5, 0.1),
  Point<3>(0.09999999999999999, -0.3, 0.1),
  Point<3>(0.09999999999999999, -0.09999999999999999, 0.1),
  Point<3>(0.09999999999999999, 0.09999999999999999, 0.1),
  Point<3>(0.09999999999999999, 0.3, 0.1),
  Point<3>(0.09999999999999999, 0.5, 0.1),
  Point<3>(0.09999999999999999, 0.7000000000000001, 0.1),
  Point<3>(0.09999999999999999, 0.9, 0.1),
  Point<3>(0.3, -0.9, 0.1),
  Point<3>(0.3, -0.7000000000000001, 0.1),
  Point<3>(0.3, -0.5, 0.1),
  Point<3>(0.3, -0.3, 0.1),
  Point<3>(0.3, -0.09999999999999999, 0.1),
  Point<3>(0.3, 0.09999999999999999, 0.1),
  Point<3>(0.3, 0.3, 0.1),
  Point<3>(0.3, 0.5, 0.1),
  Point<3>(0.3, 0.7000000000000001, 0.1),
  Point<3>(0.3, 0.9, 0.1),
  Point<3>(0.5, -0.9, 0.1),
  Point<3>(0.5, -0.7000000000000001, 0.1),
  Point<3>(0.5, -0.5, 0.1),
  Point<3>(0.5, -0.3, 0.1),
  Point<3>(0.5, -0.09999999999999999, 0.1),
  Point<3>(0.5, 0.09999999999999999, 0.1),
  Point<3>(0.5, 0.3, 0.1),
  Point<3>(0.5, 0.5, 0.1),
  Point<3>(0.5, 0.7000000000000001, 0.1),
  Point<3>(0.5, 0.9, 0.1),
  Point<3>(0.7000000000000001, -0.9, 0.1),
  Point<3>(0.7000000000000001, -0.7000000000000001, 0.1),
  Point<3>(0.7000000000000001, -0.5, 0.1),
  Point<3>(0.7000000000000001, -0.3, 0.1),
  Point<3>(0.7000000000000001, -0.09999999999999999, 0.1),
  Point<3>(0.7000000000000001, 0.09999999999999999, 0.1),
  Point<3>(0.7000000000000001, 0.3, 0.1),
  Point<3>(0.7000000000000001, 0.5, 0.1),
  Point<3>(0.7000000000000001, 0.7000000000000001, 0.1),
  Point<3>(0.7000000000000001, 0.9, 0.1),
  Point<3>(0.9, -0.9, 0.1),
  Point<3>(0.9, -0.7000000000000001, 0.1),
  Point<3>(0.9, -0.5, 0.1),
  Point<3>(0.9, -0.3, 0.1),
  Point<3>(0.9, -0.09999999999999999, 0.1),
  Point<3>(0.9, 0.09999999999999999, 0.1),
  Point<3>(0.9, 0.3, 0.1),
  Point<3>(0.9, 0.5, 0.1),
  Point<3>(0.9, 0.7000000000000001, 0.1),
  Point<3>(0.9, 0.9, 0.1),
  Point<3>(-0.8, -0.8, 0.2),
  Point<3>(-0.8, -0.6000000000000001, 0.2),
  Point<3>(-0.8, -0.4, 0.2),
  Point<3>(-0.8, -0.2, 0.2),
  Point<3>(-0.8, 0, 0.2),
  Point<3>(-0.8, 0.2, 0.2),
  Point<3>(-0.8, 0.4, 0.2),
  Point<3>(-0.8, 0.6000000000000001, 0.2),
  Point<3>(-0.8, 0.8, 0.2),
  Point<3>(-0.6000000000000001, -0.8, 0.2),
  Point<3>(-0.6000000000000001, -0.6000000000000001, 0.2),
  Point<3>(-0.6000000000000001, -0.4, 0.2),
  Point<3>(-0.6000000000000001, -0.2, 0.2),
  Point<3>(-0.6000000000000001, 0, 0.2),
  Point<3>(-0.6000000000000001, 0.2, 0.2),
  Point<3>(-0.6000000000000001, 0.4, 0.2),
  Point<3>(-0.6000000000000001, 0.6000000000000001, 0.2),
  Point<3>(-0.6000000000000001, 0.8, 0.2),
  Point<3>(-0.4, -0.8, 0.2),
  Point<3>(-0.4, -0.6000000000000001, 0.2),
  Point<3>(-0.4, -0.4, 0.2),
  Point<3>(-0.4, -0.2, 0.2),
  Point<3>(-0.4, 0, 0.2),
  Point<3>(-0.4, 0.2, 0.2),
  Point<3>(-0.4, 0.4, 0.2),
  Point<3>(-0.4, 0.6000000000000001, 0.2),
  Point<3>(-0.4, 0.8, 0.2),
  Point<3>(-0.2, -0.8, 0.2),
  Point<3>(-0.2, -0.6000000000000001, 0.2),
  Point<3>(-0.2, -0.4, 0.2),
  Point<3>(-0.2, -0.2, 0.2),
  Point<3>(-0.2, 0, 0.2),
  Point<3>(-0.2, 0.2, 0.2),
  Point<3>(-0.2, 0.4, 0.2),
  Point<3>(-0.2, 0.6000000000000001, 0.2),
  Point<3>(-0.2, 0.8, 0.2),
  Point<3>(0, -0.8, 0.2),
  Point<3>(0, -0.6000000000000001, 0.2),
  Point<3>(0, -0.4, 0.2),
  Point<3>(0, -0.2, 0.2),
  Point<3>(0, 0, 0.2),
  Point<3>(0, 0.2, 0.2),
  Point<3>(0, 0.4, 0.2),
  Point<3>(0, 0.6000000000000001, 0.2),
  Point<3>(0, 0.8, 0.2),
  Point<3>(0.2, -0.8, 0.2),
  Point<3>(0.2, -0.6000000000000001, 0.2),
  Point<3>(0.2, -0.4, 0.2),
  Point<3>(0.2, -0.2, 0.2),
  Point<3>(0.2, 0, 0.2),
  Point<3>(0.2, 0.2, 0.2),
  Point<3>(0.2, 0.4, 0.2),
  Point<3>(0.2, 0.6000000000000001, 0.2),
  Point<3>(0.2, 0.8, 0.2),
  Point<3>(0.4, -0.8, 0.2),
  Point<3>(0.4, -0.6000000000000001, 0.2),
  Point<3>(0.4, -0.4, 0.2),
  Point<3>(0.4, -0.2, 0.2),
  Point<3>(0.4, 0, 0.2),
  Point<3>(0.4, 0.2, 0.2),
  Point<3>(0.4, 0.4, 0.2),
  Point<3>(0.4, 0.6000000000000001, 0.2),
  Point<3>(0.4, 0.8, 0.2),
  Point<3>(0.6000000000000001, -0.8, 0.2),
  Point<3>(0.6000000000000001, -0.6000000000000001, 0.2),
  Point<3>(0.6000000000000001, -0.4, 0.2),
  Point<3>(0.6000000000000001, -0.2, 0.2),
  Point<3>(0.6000000000000001, 0, 0.2),
  Point<3>(0.6000000000000001, 0.2, 0.2),
  Point<3>(0.6000000000000001, 0.4, 0.2),
  Point<3>(0.6000000000000001, 0.6000000000000001, 0.2),
  Point<3>(0.6000000000000001, 0.8, 0.2),
  Point<3>(0.8, -0.8, 0.2),
  Point<3>(0.8, -0.6000000000000001, 0.2),
  Point<3>(0.8, -0.4, 0.2),
  Point<3>(0.8, -0.2, 0.2),
  Point<3>(0.8, 0, 0.2),
  Point<3>(0.8, 0.2, 0.2),
  Point<3>(0.8, 0.4, 0.2),
  Point<3>(0.8, 0.6000000000000001, 0.2),
  Point<3>(0.8, 0.8, 0.2),
  Point<3>(-0.7, -0.7, 0.3),
  Point<3>(-0.7, -0.5, 0.3),
  Point<3>(-0.7, -0.3, 0.3),
  Point<3>(-0.7, -0.09999999999999999, 0.3),
  Point<3>(-0.7, 0.09999999999999999, 0.3),
  Point<3>(-0.7, 0.3, 0.3),
  Point<3>(-0.7, 0.5, 0.3),
  Point<3>(-0.7, 0.7, 0.3),
  Point<3>(-0.5, -0.7, 0.3),
  Point<3>(-0.5, -0.5, 0.3),
  Point<3>(-0.5, -0.3, 0.3),
  Point<3>(-0.5, -0.09999999999999999, 0.3),
  Point<3>(-0.5, 0.09999999999999999, 0.3),
  Point<3>(-0.5, 0.3, 0.3),
  Point<3>(-0.5, 0.5, 0.3),
  Point<3>(-0.5, 0.7, 0.3),
  Point<3>(-0.3, -0.7, 0.3),
  Point<3>(-0.3, -0.5, 0.3),
  Point<3>(-0.3, -0.3, 0.3),
  Point<3>(-0.3, -0.09999999999999999, 0.3),
  Point<3>(-0.3, 0.09999999999999999, 0.3),
  Point<3>(-0.3, 0.3, 0.3),
  Point<3>(-0.3, 0.5, 0.3),
  Point<3>(-0.3, 0.7, 0.3),
  Point<3>(-0.09999999999999999, -0.7, 0.3),
  Point<3>(-0.09999999999999999, -0.5, 0.3),
  Point<3>(-0.09999999999999999, -0.3, 0.3),
  Point<3>(-0.09999999999999999, -0.09999999999999999, 0.3),
  Point<3>(-0.09999999999999999, 0.09999999999999999, 0.3),
  Point<3>(-0.09999999999999999, 0.3, 0.3),
  Point<3>(-0.09999999999999999, 0.5, 0.3),
  Point<3>(-0.09999999999999999, 0.7, 0.3),
  Point<3>(0.09999999999999999, -0.7, 0.3),
  Point<3>(0.09999999999999999, -0.5, 0.3),
  Point<3>(0.09999999999999999, -0.3, 0.3),
  Point<3>(0.09999999999999999, -0.09999999999999999, 0.3),
  Point<3>(0.09999999999999999, 0.09999999999999999, 0.3),
  Point<3>(0.09999999999999999, 0.3, 0.3),
  Point<3>(0.09999999999999999, 0.5, 0.3),
  Point<3>(0.09999999999999999, 0.7, 0.3),
  Point<3>(0.3, -0.7, 0.3),
  Point<3>(0.3, -0.5, 0.3),
  Point<3>(0.3, -0.3, 0.3),
  Point<3>(0.3, -0.09999999999999999, 0.3),
  Point<3>(0.3, 0.09999999999999999, 0.3),
  Point<3>(0.3, 0.3, 0.3),
  Point<3>(0.3, 0.5, 0.3),
  Point<3>(0.3, 0.7, 0.3),
  Point<3>(0.5, -0.7, 0.3),
  Point<3>(0.5, -0.5, 0.3),
  Point<3>(0.5, -0.3, 0.3),
  Point<3>(0.5, -0.09999999999999999, 0.3),
  Point<3>(0.5, 0.09999999999999999, 0.3),
  Point<3>(0.5, 0.3, 0.3),
  Point<3>(0.5, 0.5, 0.3),
  Point<3>(0.5, 0.7, 0.3),
  Point<3>(0.7, -0.7, 0.3),
  Point<3>(0.7, -0.5, 0.3),
  Point<3>(0.7, -0.3, 0.3),
  Point<3>(0.7, -0.09999999999999999, 0.3),
  Point<3>(0.7, 0.09999999999999999, 0.3),
  Point<3>(0.7, 0.3, 0.3),
  Point<3>(0.7, 0.5, 0.3),
  Point<3>(0.7, 0.7, 0.3),
  Point<3>(-0.6, -0.6, 0.4),
  Point<3>(-0.6, -0.4, 0.4),
  Point<3>(-0.6, -0.2, 0.4),
  Point<3>(-0.6, 0, 0.4),
  Point<3>(-0.6, 0.2, 0.4),
  Point<3>(-0.6, 0.4, 0.4),
  Point<3>(-0.6, 0.6, 0.4),
  Point<3>(-0.4, -0.6, 0.4),
  Point<3>(-0.4, -0.4, 0.4),
  Point<3>(-0.4, -0.2, 0.4),
  Point<3>(-0.4, 0, 0.4),
  Point<3>(-0.4, 0.2, 0.4),
  Point<3>(-0.4, 0.4, 0.4),
  Point<3>(-0.4, 0.6, 0.4),
  Point<3>(-0.2, -0.6, 0.4),
  Point<3>(-0.2, -0.4, 0.4),
  Point<3>(-0.2, -0.2, 0.4),
  Point<3>(-0.2, 0, 0.4),
  Point<3>(-0.2, 0.2, 0.4),
  Point<3>(-0.2, 0.4, 0.4),
  Point<3>(-0.2, 0.6, 0.4),
  Point<3>(0, -0.6, 0.4),
  Point<3>(0, -0.4, 0.4),
  Point<3>(0, -0.2, 0.4),
  Point<3>(0, 0, 0.4),
  Point<3>(0, 0.2, 0.4),
  Point<3>(0, 0.4, 0.4),
  Point<3>(0, 0.6, 0.4),
  Point<3>(0.2, -0.6, 0.4),
  Point<3>(0.2, -0.4, 0.4),
  Point<3>(0.2, -0.2, 0.4),
  Point<3>(0.2, 0, 0.4),
  Point<3>(0.2, 0.2, 0.4),
  Point<3>(0.2, 0.4, 0.4),
  Point<3>(0.2, 0.6, 0.4),
  Point<3>(0.4, -0.6, 0.4),
  Point<3>(0.4, -0.4, 0.4),
  Point<3>(0.4, -0.2, 0.4),
  Point<3>(0.4, 0, 0.4),
  Point<3>(0.4, 0.2, 0.4),
  Point<3>(0.4, 0.4, 0.4),
  Point<3>(0.4, 0.6, 0.4),
  Point<3>(0.6, -0.6, 0.4),
  Point<3>(0.6, -0.4, 0.4),
  Point<3>(0.6, -0.2, 0.4),
  Point<3>(0.6, 0, 0.4),
  Point<3>(0.6, 0.2, 0.4),
  Point<3>(0.6, 0.4, 0.4),
  Point<3>(0.6, 0.6, 0.4),
  Point<3>(-0.5, -0.5, 0.5),
  Point<3>(-0.5, -0.3, 0.5),
  Point<3>(-0.5, -0.1, 0.5),
  Point<3>(-0.5, 0.1, 0.5),
  Point<3>(-0.5, 0.3, 0.5),
  Point<3>(-0.5, 0.5, 0.5),
  Point<3>(-0.3, -0.5, 0.5),
  Point<3>(-0.3, -0.3, 0.5),
  Point<3>(-0.3, -0.1, 0.5),
  Point<3>(-0.3, 0.1, 0.5),
  Point<3>(-0.3, 0.3, 0.5),
  Point<3>(-0.3, 0.5, 0.5),
  Point<3>(-0.1, -0.5, 0.5),
  Point<3>(-0.1, -0.3, 0.5),
  Point<3>(-0.1, -0.1, 0.5),
  Point<3>(-0.1, 0.1, 0.5),
  Point<3>(-0.1, 0.3, 0.5),
  Point<3>(-0.1, 0.5, 0.5),
  Point<3>(0.1, -0.5, 0.5),
  Point<3>(0.1, -0.3, 0.5),
  Point<3>(0.1, -0.1, 0.5),
  Point<3>(0.1, 0.1, 0.5),
  Point<3>(0.1, 0.3, 0.5),
  Point<3>(0.1, 0.5, 0.5),
  Point<3>(0.3, -0.5, 0.5),
  Point<3>(0.3, -0.3, 0.5),
  Point<3>(0.3, -0.1, 0.5),
  Point<3>(0.3, 0.1, 0.5),
  Point<3>(0.3, 0.3, 0.5),
  Point<3>(0.3, 0.5, 0.5),
  Point<3>(0.5, -0.5, 0.5),
  Point<3>(0.5, -0.3, 0.5),
  Point<3>(0.5, -0.1, 0.5),
  Point<3>(0.5, 0.1, 0.5),
  Point<3>(0.5, 0.3, 0.5),
  Point<3>(0.5, 0.5, 0.5),
  Point<3>(-0.4, -0.4, 0.6),
  Point<3>(-0.4, -0.2, 0.6),
  Point<3>(-0.4, 0, 0.6),
  Point<3>(-0.4, 0.2, 0.6),
  Point<3>(-0.4, 0.4, 0.6),
  Point<3>(-0.2, -0.4, 0.6),
  Point<3>(-0.2, -0.2, 0.6),
  Point<3>(-0.2, 0, 0.6),
  Point<3>(-0.2, 0.2, 0.6),
  Point<3>(-0.2, 0.4, 0.6),
  Point<3>(0, -0.4, 0.6),
  Point<3>(0, -0.2, 0.6),
  Point<3>(0, 0, 0.6),
  Point<3>(0, 0.2, 0.6),
  Point<3>(0, 0.4, 0.6),
  Point<3>(0.2, -0.4, 0.6),
  Point<3>(0.2, -0.2, 0.6),
  Point<3>(0.2, 0, 0.6),
  Point<3>(0.2, 0.2, 0.6),
  Point<3>(0.2, 0.4, 0.6),
  Point<3>(0.4, -0.4, 0.6),
  Point<3>(0.4, -0.2, 0.6),
  Point<3>(0.4, 0, 0.6),
  Point<3>(0.4, 0.2, 0.6),
  Point<3>(0.4, 0.4, 0.6),
  Point<3>(-0.3, -0.3, 0.7),
  Point<3>(-0.3, -0.1, 0.7),
  Point<3>(-0.3, 0.1, 0.7),
  Point<3>(-0.3, 0.3, 0.7),
  Point<3>(-0.1, -0.3, 0.7),
  Point<3>(-0.1, -0.1, 0.7),
  Point<3>(-0.1, 0.1, 0.7),
  Point<3>(-0.1, 0.3, 0.7),
  Point<3>(0.1, -0.3, 0.7),
  Point<3>(0.1, -0.1, 0.7),
  Point<3>(0.1, 0.1, 0.7),
  Point<3>(0.1, 0.3, 0.7),
  Point<3>(0.3, -0.3, 0.7),
  Point<3>(0.3, -0.1, 0.7),
  Point<3>(0.3, 0.1, 0.7),
  Point<3>(0.3, 0.3, 0.7),
  Point<3>(-0.2, -0.2, 0.8),
  Point<3>(-0.2, 0, 0.8),
  Point<3>(-0.2, 0.2, 0.8),
  Point<3>(0, -0.2, 0.8),
  Point<3>(0, 0, 0.8),
  Point<3>(0, 0.2, 0.8),
  Point<3>(0.2, -0.2, 0.8),
  Point<3>(0.2, 0, 0.8),
  Point<3>(0.2, 0.2, 0.8),
  Point<3>(-0.09999999999999998, -0.09999999999999998, 0.9),
  Point<3>(-0.09999999999999998, 0.09999999999999998, 0.9),
  Point<3>(0.09999999999999998, -0.09999999999999998, 0.9),
  Point<3>(0.09999999999999998, 0.09999999999999998, 0.9),
  Point<3>(0, 0, 1)};

template <int dim>
void compare_points(const unsigned int degree)
{
  std::vector<std::vector<Point<3>>> reference_points;
  reference_points.emplace_back(reference_points_p3);
  reference_points.emplace_back(reference_points_p4);
  reference_points.emplace_back(reference_points_p5);
  reference_points.emplace_back(reference_points_p6);
  reference_points.emplace_back(reference_points_p7);
  reference_points.emplace_back(reference_points_p8);
  reference_points.emplace_back(reference_points_p9);
  reference_points.emplace_back(reference_points_p10);

  // reference_points.clear();
  // reference_points.emplace_back(reference_points_equi_p3);
  // reference_points.emplace_back(reference_points_equi_p4);
  // reference_points.emplace_back(reference_points_equi_p5);
  // reference_points.emplace_back(reference_points_equi_p6);
  // reference_points.emplace_back(reference_points_equi_p7);
  // reference_points.emplace_back(reference_points_equi_p8);
  // reference_points.emplace_back(reference_points_equi_p9);
  // reference_points.emplace_back(reference_points_equi_p10);



  if (degree > 2 && degree < 11)
    {
      bool         found_all_points     = true;
      double       max_derivation       = 0.0;
      unsigned int max_derivation_index = 100000;

      const std::vector<Point<3>> points_reference =
        reference_points[degree - 3];
      const auto points_blend_and_warp =
        // equi_unit_support_points_fe_pyramid_p<dim>(degree);
        get_blend_and_warp_support_points<dim>(degree);

      for (unsigned int i = 0; i < points_blend_and_warp.size(); ++i)
        {
          double       min_distance       = 1000000.0;
          unsigned int min_distance_index = 1000000;

          bool found_point = false;
          for (unsigned int j = 0; j < points_reference.size(); ++j)
            {
              const double distance = std::abs(
                points_reference[j].distance(points_blend_and_warp[i]));
              if (distance < 1e-9)
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
              found_all_points = false;
              if (max_derivation < min_distance)
                {
                  max_derivation       = min_distance;
                  max_derivation_index = min_distance_index;
                }
              std::cout << "Did not find point " << i << " "
                        << points_blend_and_warp[i] << std::endl;
              std::cout << "nearest points was "
                        << points_reference[min_distance_index]
                        << " at distance " << min_distance << std::endl;
            }
        }

      if (found_all_points == false)
        {
          std::cout << "Did not find all points, the maximum difference was "
                    << max_derivation << " at index " << max_derivation_index
                    << " boundary ends at " << 3 * degree * degree + 2
                    << std::endl;
        }
      else
        std::cout << "all points check out" << std::endl;
    }
  else
    std::cout << "no data at degree " << degree << std::endl;
}

int main()
{
  constexpr int dim = 3;

  for (unsigned int degree = 1; degree < 11; ++degree)
    {
      // print_points<3>(degree);

      const auto points_blend_and_warp =
        get_blend_and_warp_support_points<dim>(degree);
      std::cout << "created points" << std::endl;
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

      compare_points<3>(degree);
    }

  return 0;
}
