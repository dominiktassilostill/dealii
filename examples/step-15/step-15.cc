// -----------------------------------------------------------------------------
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception OR LGPL-2.1-or-later
// Copyright (C) 2002 - 2024 by the deal.II authors
//
// This file is part of the deal.II library.
//
// Detailed license information governing the source code and contributions
// can be found in LICENSE.md and CONTRIBUTING.md at the top level directory.
//
// -----------------------------------------------------------------------------

// like fe/up_and_down test but use simplex elements instead
// Restrict a vector to a coarse grid and prolongate it again,
// repeat the procedure twice and check if the reslting vectors are the same
// i.e. if prolongation and restriction are successful.

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_accessor.h>

#include <deal.II/fe/fe_pyramid_p.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/hp/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/lac/vector.h>

#include <deal.II/numerics/vector_tools.h>


#include <deal.II/base/function_lib.h>

#include <string>
#include <vector>

using namespace dealii;


template <int dim>
void check_constant_prolongation(const Triangulation<dim>    &tr,
                                 const hp::FECollection<dim> &fe)
{
  (void)tr;

  const double value_coarse = 1.5;

  const unsigned int n_dofs_coarse = fe[0].dofs_per_cell;
  Vector<double>     vector_coarse(n_dofs_coarse);
  for (unsigned int i = 0; i < n_dofs_coarse; ++i)
    {
      vector_coarse[i] = value_coarse;
    }

  Vector<double> vector_fine;
  for (unsigned int c = 0; c < fe[0].reference_cell().n_isotropic_children();
       ++c)
    {
      const bool pyramid_child = (c > 3 && c < 8) ? false : true;

      const unsigned int n_dofs_fine =
        pyramid_child ? n_dofs_coarse : fe[1].dofs_per_cell;

      vector_fine.grow_or_shrink(n_dofs_fine);

      const auto prolongation_matrix = fe[0].get_prolongation_matrix(c);

      prolongation_matrix.vmult(vector_fine, vector_coarse);


      for (unsigned int i = 0; i < vector_fine.size(); ++i)
        Assert(std::abs(value_coarse - vector_fine[i]) < 1e-12,
               ExcInternalError());
      // std::cout << std::endl << "Child " << c << ": ";
      // for (unsigned int i = 0; i < vector_fine.size(); ++i)
      //   std::cout << vector_fine[i] << " ";
      // std::cout << std::endl;
    }
}


template <int dim>
void check_basis_function_prolongation(const Triangulation<dim>    &tr,
                                       const hp::FECollection<dim> &fe)
{
  DoFHandler<dim> dof_handler(tr);
  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (cell->reference_cell() == ReferenceCells::Pyramid)
        cell->set_active_fe_index(0);
      else if (cell->reference_cell() == ReferenceCells::Tetrahedron)
        cell->set_active_fe_index(1);
      else
        DEAL_II_NOT_IMPLEMENTED();
    }

  dof_handler.distribute_dofs(fe);
  // const unsigned int n_dofs_fine   = dof_handler.n_dofs();
  unsigned int n_dofs_fine = 0;
  for (const auto &cell : dof_handler.active_cell_iterators())
    n_dofs_fine += cell->get_fe().dofs_per_cell;

  const unsigned int n_dofs_coarse = fe[0].dofs_per_cell;
  Vector<double>     vector_coarse(n_dofs_coarse);
  Vector<double>     vector_fine(n_dofs_fine);
  Vector<double>     vector_nodal(n_dofs_fine);
  Vector<double>     vector_child;

  hp::MappingCollection<dim, dim> mappings;
  mappings.push_back(fe[0].reference_cell().get_default_linear_mapping());
  mappings.push_back(fe[1].reference_cell().get_default_linear_mapping());
  hp::QCollection<dim> quadratures;
  Quadrature<dim>      q0(fe[0].get_unit_support_points());
  Quadrature<dim>      q1(fe[1].get_unit_support_points());
  quadratures.push_back(q0);
  quadratures.push_back(q1);

  hp::FEValues<dim> hp_fe_values(mappings,
                                 fe,
                                 quadratures,
                                 update_quadrature_points);

  for (unsigned e = 0; e < n_dofs_coarse; ++e)
    {
      vector_coarse    = 0.;
      vector_coarse[e] = 1.0;

      unsigned int counter = 0;
      for (unsigned int c = 0;
           c < fe[0].reference_cell().n_isotropic_children();
           ++c)
        {
          const bool pyramid_child = (c > 3 && c < 8) ? false : true;

          const unsigned int n_dofs_child =
            pyramid_child ? n_dofs_coarse : fe[1].dofs_per_cell;

          vector_child.grow_or_shrink(n_dofs_child);

          const auto prolongation_matrix = fe[0].get_prolongation_matrix(c);

          prolongation_matrix.vmult(vector_child, vector_coarse);

          for (unsigned int i = 0; i < vector_child.size(); ++i)
            vector_fine[counter++] = vector_child[i];
        }

      counter = 0;
      for (const auto &cell : dof_handler.active_cell_iterators())
        {
          hp_fe_values.reinit(cell);
          auto      &fe_values        = hp_fe_values.get_present_fe_values();
          const auto quadrature_point = fe_values.get_quadrature_points();

          for (unsigned int i = 0; i < quadrature_point.size(); ++i)
            vector_nodal[counter++] = fe[0].shape_value(e, quadrature_point[i]);
        }

      for (unsigned int i = 0; i < vector_nodal.size(); ++i)
        Assert(std::abs(vector_nodal[i] - vector_fine[i]) < 1e-12,
               ExcInternalError());
    }
}

template <int dim>
void check_polynomials_prolongation(const Triangulation<dim>    &tr,
                                    const hp::FECollection<dim> &fe)
{
  Triangulation<dim> tr_coarse;
  GridGenerator::reference_cell(tr_coarse, ReferenceCells::Pyramid);
  DoFHandler<dim> dof_handler_coarse(tr_coarse);
  for (const auto &cell : dof_handler_coarse.active_cell_iterators())
    cell->set_active_fe_index(0);
  dof_handler_coarse.distribute_dofs(fe);

  DoFHandler<dim> dof_handler(tr);
  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (cell->reference_cell() == ReferenceCells::Pyramid)
        cell->set_active_fe_index(0);
      else if (cell->reference_cell() == ReferenceCells::Tetrahedron)
        cell->set_active_fe_index(1);
      else
        DEAL_II_NOT_IMPLEMENTED();
    }

  dof_handler.distribute_dofs(fe);

  const unsigned int degree        = fe[0].degree;
  const unsigned int n_dofs_coarse = fe[0].dofs_per_cell;
  const unsigned int n_dofs_fine   = dof_handler.n_dofs();

  Vector<double> vector_coarse(n_dofs_coarse);
  Vector<double> vector_coarse_global(n_dofs_coarse);
  Vector<double> vector_fine(n_dofs_fine);

  hp::MappingCollection<dim, dim> mappings;
  mappings.push_back(fe[0].reference_cell().get_default_linear_mapping());
  mappings.push_back(fe[1].reference_cell().get_default_linear_mapping());


  for (unsigned int i = 0; i < degree + 1; ++i)
    for (unsigned int j = 0; j < degree + 1; ++j)
      for (unsigned int k = 0; k < degree + 1; ++k)
        if (i + j + k < degree + 1)
          {
            Tensor<1, dim> exponents;
            exponents[0] = i;
            exponents[1] = j;
            exponents[2] = k;
            const Functions::Monomial function_to_interpolate(exponents, 1);

            VectorTools::interpolate(mappings,
                                     dof_handler_coarse,
                                     function_to_interpolate,
                                     vector_coarse_global);

            const auto coarse_cell = dof_handler_coarse.begin_active();
            coarse_cell->get_dof_values(vector_coarse_global, vector_coarse);

            VectorTools::interpolate(mappings,
                                     dof_handler,
                                     function_to_interpolate,
                                     vector_fine);

            const auto parent = dof_handler.begin(0);
            // for (const auto &cell : dof_handler.active_cell_iterators())
            for (unsigned int c = 0; c < parent->n_children(); ++c)
              {
                const auto &cell = parent->child(c);

                Vector<double> vector_child(cell->get_fe().dofs_per_cell);
                Vector<double> vector_child_interpolated(
                  cell->get_fe().dofs_per_cell);

                cell->get_dof_values(vector_fine, vector_child);

                fe[0].get_prolongation_matrix(c).vmult(
                  vector_child_interpolated, vector_coarse);

                for (unsigned int v = 0; v < vector_child_interpolated.size();
                     ++v)
                  if (std::abs(vector_child_interpolated[v] - vector_child[v]) >
                      1e-12)
                    {
                      Assert(false, ExcInternalError());
                    }
              }
          }
}

template <int dim>
void check_constant_prolongation_transpose(const Triangulation<dim>    &tr,
                                           const hp::FECollection<dim> &fe)
{
  (void)tr;

  const double value_fine = 1.5;

  const unsigned int n_dofs_coarse = fe[0].dofs_per_cell;
  Vector<double>     vector_coarse(n_dofs_coarse);
  Vector<double>     vector_coarse_sum(n_dofs_coarse);

  Vector<double> vector_fine;
  for (unsigned int c = 0; c < fe[0].reference_cell().n_isotropic_children();
       ++c)
    {
      const bool pyramid_child = (c > 3 && c < 8) ? false : true;

      const unsigned int n_dofs_fine =
        pyramid_child ? n_dofs_coarse : fe[1].dofs_per_cell;

      vector_fine.grow_or_shrink(n_dofs_fine);
      for (unsigned int i = 0; i < n_dofs_fine; ++i)
        {
          vector_fine[i] = value_fine;
        }

      const auto prolongation_matrix = fe[0].get_prolongation_matrix(c);

      prolongation_matrix.Tvmult(vector_coarse, vector_fine);
      prolongation_matrix.Tvmult_add(vector_coarse_sum, vector_fine);


      std::cout << "Child " << c << ": ";
      for (unsigned int i = 0; i < vector_coarse.size(); ++i)
        std::cout << vector_coarse[i] << " ";
      std::cout << std::endl;

      // for (unsigned int i = 0; i < vector_coarse.size(); ++i)
      //   Assert(std::abs(value_fine - vector_coarse[i]) < 1e-12,
      //          ExcInternalError());
    }

  std::cout << "Sum of results " << std::endl;
  for (unsigned int i = 0; i < vector_coarse_sum.size(); ++i)
    std::cout << vector_coarse_sum[i] << " ";
  std::cout << std::endl;
}

template <int dim>
void check_constant_restriction(const Triangulation<dim>    &tr,
                                const hp::FECollection<dim> &fe)
{
  DoFHandler<dim> dof_handler(tr);

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (cell->reference_cell() == ReferenceCells::Pyramid)
        cell->set_active_fe_index(0);
      else if (cell->reference_cell() == ReferenceCells::Tetrahedron)
        cell->set_active_fe_index(1);
      else
        DEAL_II_NOT_IMPLEMENTED();
    }

  dof_handler.distribute_dofs(fe);

  const double value_fine = 1.5;

  const unsigned int n_dofs_coarse = fe[0].dofs_per_cell;
  Vector<double>     vector_coarse(n_dofs_coarse);
  Vector<double>     vector_coarse_sum(n_dofs_coarse);
  vector_coarse_sum = 0.;

  Vector<double> vector_fine_global(dof_handler.n_dofs());
  for (auto &v : vector_fine_global)
    v = value_fine;
  Vector<double> vector_fine;
  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      const bool pyramid_child =
        cell->reference_cell() == ReferenceCells::Pyramid;

      const unsigned int n_dofs_fine =
        pyramid_child ? n_dofs_coarse : fe[1].dofs_per_cell;

      vector_fine.grow_or_shrink(n_dofs_fine);
      cell->get_dof_values(vector_fine_global, vector_fine);

      const auto restriction_matrix =
        fe[0].get_restriction_matrix(cell->active_cell_index());


      if (fe[0].n_dofs_per_vertex() == 0)
        restriction_matrix.vmult_add(vector_coarse_sum, vector_fine);
      else
        {
          vector_coarse = 0.;
          restriction_matrix.vmult(vector_coarse, vector_fine);
          for (unsigned int i = 0; i < vector_coarse.size(); ++i)
            if (std::abs(vector_coarse[i]) > 1e-12)
              vector_coarse_sum[i] = vector_coarse[i];
        }

      // std::cout << "Child " << cell->active_cell_index() << ": ";
      // for (const auto &v : vector_coarse)
      //   std::cout << v << " ";
      // std::cout << std::endl;

      // for (unsigned int i = 0; i < vector_coarse.size(); ++i)
      //   Assert(std::abs(value_fine - vector_coarse[i]) < 1e-12,
      //          ExcInternalError());
    }

  // std::cout << "Sum of results " << std::endl;
  // for (const auto &v : vector_coarse_sum)
  //   std::cout << v << " ";
  // std::cout << std::endl;
  // bool wrote_value = false;
  for (const auto &v : vector_coarse_sum)
    Assert(std::abs(value_fine - v) < 1e-8, ExcInternalError());
  //   if (std::abs(value_fine - v) > 1e-12)
  //     {
  //       std::cout << std::abs(value_fine - v) << " ";
  //       wrote_value = true;
  //     }
  // if (wrote_value)
  //   std::cout << std::endl;
}



template <int dim>
void check_basis_function_restriction(const Triangulation<dim>    &tr,
                                      const hp::FECollection<dim> &fe)
{
  DoFHandler<dim> dof_handler(tr);

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (cell->reference_cell() == ReferenceCells::Pyramid)
        cell->set_active_fe_index(0);
      else if (cell->reference_cell() == ReferenceCells::Tetrahedron)
        cell->set_active_fe_index(1);
      else
        DEAL_II_NOT_IMPLEMENTED();
    }

  dof_handler.distribute_dofs(fe);

  const unsigned int n_dofs_coarse = fe[0].dofs_per_cell;
  Vector<double>     vector_coarse(n_dofs_coarse);
  Vector<double>     vector_coarse_sum(n_dofs_coarse);
  Vector<double>     vector_coarse_one_basis(n_dofs_coarse);

  Vector<double> vector_fine_global(dof_handler.n_dofs());
  for (unsigned int i = 0; i < dof_handler.n_dofs(); ++i)
    {
      vector_fine_global      = 0.0;
      vector_fine_global[i]   = 1.0;
      vector_coarse_one_basis = 0.0;

      Vector<double> vector_fine(n_dofs_coarse);
      for (const auto &cell : dof_handler.active_cell_iterators())
        {
          vector_fine = 0.0;
          const bool pyramid_child =
            cell->reference_cell() == ReferenceCells::Pyramid;

          const unsigned int n_dofs_fine =
            pyramid_child ? n_dofs_coarse : fe[1].dofs_per_cell;

          vector_fine.grow_or_shrink(n_dofs_fine);
          cell->get_dof_values(vector_fine_global, vector_fine);

          const auto restriction_matrix =
            fe[0].get_restriction_matrix(cell->active_cell_index());

          if (fe[0].n_dofs_per_vertex() == 0)
            restriction_matrix.vmult_add(vector_coarse_one_basis, vector_fine);
          else
            {
              vector_coarse = 0.;
              restriction_matrix.vmult(vector_coarse, vector_fine);
              for (unsigned int i = 0; i < vector_coarse.size(); ++i)
                if (std::abs(vector_coarse[i]) > 1e-12)
                  vector_coarse_one_basis[i] = vector_coarse[i];
            }
        }
      for (unsigned int i = 0; i < vector_coarse_one_basis.size(); ++i)
        vector_coarse_sum[i] += vector_coarse_one_basis[i];
    }
  for (const auto &v : vector_coarse_sum)
    if (std::abs(1.0 - v) > 1e-8)
      std::cout << std::abs(1.0 - v) << " ";
  std::cout << std::endl;
}

template <int dim>
void check_polynomials_restriction(const Triangulation<dim>    &tr,
                                   const hp::FECollection<dim> &fe)
{
  Triangulation<dim> tr_coarse;
  GridGenerator::reference_cell(tr_coarse, ReferenceCells::Pyramid);
  DoFHandler<dim> dof_handler_coarse(tr_coarse);
  for (const auto &cell : dof_handler_coarse.active_cell_iterators())
    cell->set_active_fe_index(0);
  dof_handler_coarse.distribute_dofs(fe);

  DoFHandler<dim> dof_handler(tr);
  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (cell->reference_cell() == ReferenceCells::Pyramid)
        cell->set_active_fe_index(0);
      else if (cell->reference_cell() == ReferenceCells::Tetrahedron)
        cell->set_active_fe_index(1);
      else
        DEAL_II_NOT_IMPLEMENTED();
    }

  dof_handler.distribute_dofs(fe);

  const unsigned int degree        = fe[0].degree;
  const unsigned int n_dofs_coarse = fe[0].dofs_per_cell;
  const unsigned int n_dofs_fine   = dof_handler.n_dofs();

  Vector<double> vector_coarse(n_dofs_coarse);
  Vector<double> vector_coarse_global(n_dofs_coarse);
  Vector<double> vector_fine_global(n_dofs_fine);
  Vector<double> vector_fine_global_interpolated(n_dofs_fine);

  hp::MappingCollection<dim, dim> mappings;
  mappings.push_back(fe[0].reference_cell().get_default_linear_mapping());
  mappings.push_back(fe[1].reference_cell().get_default_linear_mapping());


  for (unsigned int i = 0; i < degree + 1; ++i)
    for (unsigned int j = 0; j < degree + 1; ++j)
      for (unsigned int k = 0; k < degree + 1; ++k)
        if (i + j + k < degree + 1)
          {
            Tensor<1, dim> exponents;
            exponents[0] = i;
            exponents[1] = j;
            exponents[2] = k;
            const Functions::Monomial function_to_interpolate(exponents, 1);

            VectorTools::interpolate(mappings,
                                     dof_handler_coarse,
                                     function_to_interpolate,
                                     vector_coarse_global);

            VectorTools::interpolate(mappings,
                                     dof_handler,
                                     function_to_interpolate,
                                     vector_fine_global_interpolated);

            const auto coarse_cell = dof_handler_coarse.begin_active();
            coarse_cell->get_dof_values(vector_coarse_global, vector_coarse);

            const auto parent = dof_handler.begin(0);
            for (unsigned int c = 0; c < parent->n_children(); ++c)
              {
                const auto    &cell = parent->child(c);
                Vector<double> vector_child(cell->get_fe().dofs_per_cell);
                fe[0].get_prolongation_matrix(c).vmult(vector_child,
                                                       vector_coarse);
                cell->set_dof_values(vector_child, vector_fine_global);
              }
            {
              // check that interpolated vector and set vector are similar
              bool printed = false;
              for (unsigned int v = 0; v < n_dofs_fine; ++v)
                if (std::abs(vector_fine_global_interpolated[v] -
                             vector_fine_global[v]) > 1e-12)
                  {
                    printed = true;
                    std::cout << std::abs(vector_fine_global_interpolated[v] -
                                          vector_fine_global[v])
                              << " ";
                  }
              if (printed)
                std::cout << "Values above from interpolation" << std::endl;
            }

            // now restrict the function
            vector_coarse = 0.0;
            Vector<double>                       tmp(n_dofs_coarse);
            std::vector<types::global_dof_index> dof_indices(n_dofs_coarse);
            coarse_cell->get_dof_indices(dof_indices);

            for (unsigned int c = 0; c < parent->n_children(); ++c)
              {
                const auto    &cell = parent->child(c);
                Vector<double> vector_child(cell->get_fe().dofs_per_cell);
                cell->get_dof_values(vector_fine_global, vector_child);
                tmp = 0.0;
                fe[0].get_restriction_matrix(c).vmult(tmp, vector_child);

                if (fe[0].n_dofs_per_vertex() == 0)
                  for (unsigned int v = 0; v < dof_indices.size(); ++v)
                    vector_coarse[dof_indices[v]] += tmp[v];
                else
                  for (unsigned int v = 0; v < dof_indices.size(); ++v)
                    if (std::abs(tmp[v]) > 1e-12)
                      vector_coarse[dof_indices[v]] = tmp[v];
              }

            // now check that the coarse and the interpolated coarse vectors are
            // same same
            bool printed = false;
            for (unsigned int v = 0; v < n_dofs_coarse; ++v)
              if (std::abs(vector_coarse[v] - vector_coarse_global[v]) > 1e-8)
                {
                  std::cout << vector_coarse[v] - vector_coarse_global[v]
                            << " ";

                  printed = true;
                }
            if (printed)
              std::cout << std::endl
                        << "Above is for " << i << " " << j << " " << k
                        << std::endl;
          }
}

template <int dim>
void check_element(const Triangulation<dim>    &tr,
                   const hp::FECollection<dim> &fe)
{
  DoFHandler<dim> dof_handler(tr);

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (cell->reference_cell() == ReferenceCells::Pyramid)
        cell->set_active_fe_index(0);
      else if (cell->reference_cell() == ReferenceCells::Tetrahedron)
        cell->set_active_fe_index(1);
      else
        DEAL_II_NOT_IMPLEMENTED();
    }

  dof_handler.distribute_dofs(fe);

  const unsigned int n_dofs_fine = dof_handler.n_dofs();
  // create a mostly arbitrary
  // function plus a trend on this
  // grid
  Vector<double> vector_fine(n_dofs_fine);
  vector_fine = 0.0;

  hp::MappingCollection<dim> mappings;
  hp::QCollection<dim>       quadratures;
  mappings.push_back(fe[0].reference_cell().get_default_linear_mapping());
  mappings.push_back(fe[1].reference_cell().get_default_linear_mapping());
  quadratures.push_back(Quadrature<dim>(fe[0].get_unit_support_points()));
  quadratures.push_back(Quadrature<dim>(fe[1].get_unit_support_points()));

  hp::FEValues<dim> hp_fe_values(mappings,
                                 fe,
                                 quadratures,
                                 update_quadrature_points);
  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      hp_fe_values.reinit(cell);
      auto &fe_values = hp_fe_values.get_present_fe_values();
      std::vector<types::global_dof_index> dof_indices(
        cell->get_fe().dofs_per_cell);
      cell->get_dof_indices(dof_indices);
      for (unsigned int i = 0; i < dof_indices.size(); ++i)
        {
          vector_fine(dof_indices[i]) =
            fe_values.quadrature_point(i)[0] + fe_values.quadrature_point(i)[1];
        }
    }

  // restrict this function to the
  // next coarser level and
  // distribute it again to the
  // higher level
  Vector<double> x(vector_fine.size());
  Vector<double> vector_coarse(fe[0].dofs_per_cell);
  for (typename DoFHandler<dim>::cell_iterator cell = dof_handler.begin();
       cell != dof_handler.end();
       ++cell)
    if (cell->has_children() && cell->child(0)->is_active())
      {
        // first make sure that what
        // we do is reasonable. for
        // this, _all_ children have
        // to be active, not only
        // some of them
        for (unsigned int c = 0;
             c < fe[0].reference_cell().n_isotropic_children();
             ++c)
          AssertThrow(cell->child(c)->is_active(), ExcInternalError());

        // then restrict and prolongate
        cell->get_interpolated_dof_values(vector_fine, vector_coarse, 0);
        cell->set_dof_values_by_interpolation(vector_coarse, x, 0);
      };

  // now x is a function on the fine
  // grid that is representable on
  // the coarse grid. so another
  // cycle should not alter it any
  // more:
  Vector<double> x2(x.size());
  for (typename DoFHandler<dim>::cell_iterator cell = dof_handler.begin();
       cell != dof_handler.end();
       ++cell)
    if (cell->has_children() && cell->child(0)->is_active())
      {
        cell->get_interpolated_dof_values(x, vector_coarse, 0);
        cell->set_dof_values_by_interpolation(vector_coarse, x2, 0);
      };

  // then check that this is so:
  x2 -= x;
  const double relative_residual = (x2.l2_norm() / x.l2_norm());

  const double threshold = 1e-6;
  std::cout << ", dofs_per_cell=" << fe[0].dofs_per_cell
            << "; relative residual: " << relative_residual << " "
            << (relative_residual < threshold ? "ok" : "botched up!")
            << std::endl;

  // TODO:[WB] Why this exception with a value different from above. Output of
  // the error should be sufficient!
  //  Assert (relative_residual < threshold*x.l2_norm(), ExcInternalError());
}


template <int dim>
void test()
{
  // make a coarse triangulation as a
  // hypercube. if in more than 1d,
  // distort it so that it is no more
  // an affine image of the
  // hypercube, to make things more
  // difficult. then refine it twice
  Triangulation<dim> tr;
  GridGenerator::reference_cell(tr, ReferenceCells::Pyramid);

  tr.refine_global(1);

  // now for a list of finite
  // elements, for which we want to
  // test. we happily waste tons of
  // memory here, but who cares...
  std::vector<hp::FECollection<dim>> fe_collections;

  const unsigned int degree_max = 4;

  for (unsigned int fe_degree = 1; fe_degree < degree_max + 1; ++fe_degree)
    {
      const FE_SimplexP<dim> fe_p(fe_degree, false);
      const FE_PyramidP<dim> fe_pyramid(fe_degree, false);
      fe_collections.emplace_back(fe_pyramid, fe_p);
    }

  for (unsigned int fe_degree = 1; fe_degree < degree_max + 1; ++fe_degree)
    {
      const FE_SimplexDGP<dim> fe_p(fe_degree, false);
      const FE_PyramidDGP<dim> fe_pyramid(fe_degree, false);
      fe_collections.emplace_back(fe_pyramid, fe_p);
    }

  for (const auto &fe_collection : fe_collections)
    {
      std::cout << dim << "d, uniform grid, fe ";
      for (unsigned int i = 0; i < fe_collection.size(); ++i)
        std::cout << fe_collection[i].get_name() << ", ";
      std::cout << std::endl;

      if (false)
        {
          const auto pm = fe_collection[0].get_prolongation_matrix(0);
          const auto pm2 =
            fe_collection[1].get_prolongation_matrix(0, RefinementCase<3>(1));

          const auto rm = fe_collection[0].get_restriction_matrix(0);
          const auto rm2 =
            fe_collection[1].get_restriction_matrix(0, RefinementCase<3>(1));

          pm.print(std::cout);
          std::cout << std::endl;
          pm2.print(std::cout);
          std::cout << std::endl;
          rm.print(std::cout);
          std::cout << std::endl;
          rm2.print(std::cout);
        }

      check_element(tr, fe_collection);
      // check_constant_prolongation(tr, fe_collection);
      // check_basis_function_prolongation(tr, fe_collection);
      // check_polynomials_prolongation(tr, fe_collection);
      {
        check_constant_restriction(tr, fe_collection);
        check_basis_function_restriction(tr, fe_collection);
        check_polynomials_restriction(tr, fe_collection);
      }
      // check_constant_prolongation_transpose(tr, fe_collection);
    }
}



int main()
{
  test<3>();

  return 0;
}
