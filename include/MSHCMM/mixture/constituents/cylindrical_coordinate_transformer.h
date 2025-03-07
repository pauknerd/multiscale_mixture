#pragma once

#include <deal.II/base/exceptions.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria.h>

#include <deal.II/physics/transformations.h>

#include <cmath>
#include <fstream>
#include <iostream>


// todo: derive from dealii::Function so that it can be used as prestretch?

namespace Mixture::Constituents
{

  template <int dim, typename Number = double>
  class CylindricalCoordinateTransformer
  {
  public:
    CylindricalCoordinateTransformer(
      const dealii::Manifold<dim>  &manifold,
      const Number                  inner_radius,
      const Number                  outer_radius,
      const Number                  rotation_angle = 0.0, // in radians
      const Number                  eps            = 1e-10,
      const dealii::Tensor<1, dim> &axis = dealii::Tensor<1, dim>({0, 0, 1}))
      : manifold(manifold)
      , inner_radius(inner_radius)
      , outer_radius(outer_radius)
      , rotation_angle(rotation_angle)
      , eps(eps)
      , axis(axis)
    {
      Assert(dim == 3, dealii::ExcMessage("Only works in 3D!"));
      AssertThrow(inner_radius < outer_radius,
                  dealii::ExcMessage(
                    "Inner radius must be smaller than the outer radius!"));
      // temporary axis
      const dealii::Tensor<1, dim> temp_axis({0, 0, 1});
      AssertThrow(
        axis == temp_axis,
        dealii::ExcMessage(
          "Currently only implemented for a cylinder aligned in z-direction"
          "as ist the default for the GridGenerator::cylinder_shell() function!"));
    }

    dealii::Tensor<1, dim>
    get_tangent(const dealii::Point<dim> &point) const
    {
      // origin helper point at same z coordinate
      const dealii::Point<dim> p0(0, 0, point[2]);
      // compute distance of point to origin, i.e. radius
      const Number radius = p0.distance(point);
      // check that radius is between inner and outer radius
      Assert(radius > inner_radius && radius < outer_radius,
             dealii::ExcMessage("Point is outside of the domain!"));

      // create vector to the point from but only in radial direction
      const dealii::Tensor<1, dim> direction = point - p0;

      // compute angle between x-axis and direction
      // if y of point is negative, use -unit_vec_x
      const Number angle = point[1] >= 0.0 ?
                             std::acos(direction * unit_vec_x / radius) :
                             std::acos(direction * -unit_vec_x / radius);

      // if y-coordinate of point is negative remove eps instead of adding
      const Number angle_pos = point[1] >= 0.0 ? angle + eps : angle - eps;

      // now generate new point by varying the angle slightly
      const Number temp_x2 = radius * std::cos(angle_pos);
      const Number temp_y2 = radius * std::sin(angle_pos);

      // now create Point with those coordinates but with same z-coordinate as
      // point
      const dealii::Point<dim> p2(temp_x2, temp_y2, point[2]);

      // use manifold to get tangent vector
      auto tangent = manifold.get_tangent_vector(point, p2);
      // normalize tangent vector
      tangent /= tangent.norm();

      // rotate tangent if rotation_angle != 0.0
      if (rotation_angle != 0.0)
        {
          // rotation axis from direction vector (needs to be normalized to a
          // unit vector)
          const auto rotation_axis =
            static_cast<dealii::Point<dim>>(direction / radius);
          // create 3d rotation matrix around direction vector with
          // given rotation_angle
          const auto rotation_matrix =
            dealii::Physics::Transformations::Rotations::rotation_matrix_3d(
              rotation_axis, rotation_angle);
          // apply rotation matrix to tangent vector
          tangent = rotation_matrix * tangent;
        }

      return tangent;
    }

    // transform a vector from cartesian coordinates to a rotated coordinate
    // system that matches a cylindrical coordinate system. Note that it is
    // assumed that the cylinder is aligned in z-direction!
    dealii::Tensor<1, dim>
    from_cartesian(const dealii::Point<dim>     &p,
                   const dealii::Tensor<1, dim> &vec) const
    {
      // compute angle of rotation around z-axis
      // create tensor in radial direction, i.e., remove z-component
      const dealii::Tensor<1, dim> direction({p[0], p[1], 0.0});

      // check that radius is between inner and outer radius
      Assert(direction.norm() > inner_radius && direction.norm() < outer_radius,
             dealii::ExcMessage("Point is outside of the domain!"));

      // compute angle between x-axis and direction
      // if y of point is negative, use -unit_vec_x
      const Number angle =
        p[1] >= 0.0 ? std::acos(direction * unit_vec_x / direction.norm()) :
                      std::acos(direction * -unit_vec_x / direction.norm());

      // create transformation matrix
      const auto rotation_axis = static_cast<dealii::Point<dim>>(axis);
      const auto rotation_matrix =
        dealii::Physics::Transformations::Rotations::rotation_matrix_3d(
          rotation_axis, angle);

      // std::cout << dealii::transpose(rotation_matrix) << std::endl;
      //  return v' = Q_T * v
      return dealii::transpose(rotation_matrix) * vec;
    }

    dealii::Tensor<2, dim>
    from_cartesian(const dealii::Point<dim>     &p,
                   const dealii::Tensor<2, dim> &mat) const
    {
      // compute angle of rotation around z-axis
      // create tensor in radial direction, i.e., remove z-component
      const dealii::Tensor<1, dim> direction({p[0], p[1], 0.0});

      // check that radius is between inner and outer radius
      Assert(direction.norm() > inner_radius && direction.norm() < outer_radius,
             dealii::ExcMessage("Point is outside of the domain!"));

      // compute angle between x-axis and direction
      // if y of point is negative, use -unit_vec_x
      const Number angle =
        p[1] > 0.0 ? std::acos(direction * unit_vec_x / direction.norm()) :
                     std::acos(direction * -unit_vec_x / direction.norm());

      // create transformation matrix
      const auto rotation_axis = static_cast<dealii::Point<dim>>(axis);
      const auto rotation_matrix =
        dealii::Physics::Transformations::Rotations::rotation_matrix_3d(
          rotation_axis, angle);

      // std::cout << dealii::transpose(rotation_matrix) << std::endl;
      //  return A' = Q_T * A * Q
      return dealii::transpose(rotation_matrix) * mat * rotation_matrix;
    }

    dealii::Tensor<1, dim>
    to_cartesian(const dealii::Point<dim>     &p,
                 const dealii::Tensor<1, dim> &vec) const
    {
      // compute angle of rotation around z-axis
      // create tensor in radial direction, i.e., remove z-component
      const dealii::Tensor<1, dim> direction({p[0], p[1], 0.0});

      // check that radius is between inner and outer radius
      Assert(direction.norm() > inner_radius && direction.norm() < outer_radius,
             dealii::ExcMessage("Point is outside of the domain!"));

      // compute angle between x-axis and direction
      // if y of point is negative, use -unit_vec_x
      const Number angle =
        p[1] >= 0.0 ? std::acos(direction * unit_vec_x / direction.norm()) :
                      std::acos(direction * -unit_vec_x / direction.norm());

      // create transformation matrix
      const auto rotation_axis = static_cast<dealii::Point<dim>>(axis);
      const auto rotation_matrix =
        dealii::Physics::Transformations::Rotations::rotation_matrix_3d(
          rotation_axis, angle);

      // std::cout << dealii::transpose(rotation_matrix) << std::endl;
      //  return v = Q * v'
      return rotation_matrix * vec;
    }

    dealii::Tensor<2, dim>
    to_cartesian(const dealii::Point<dim>     &p,
                 const dealii::Tensor<2, dim> &mat) const
    {
      // compute angle of rotation around z-axis
      // create tensor in radial direction, i.e., remove z-component
      const dealii::Tensor<1, dim> direction({p[0], p[1], 0.0});

      // check that radius is between inner and outer radius
      Assert(direction.norm() > inner_radius && direction.norm() < outer_radius,
             dealii::ExcMessage("Point is outside of the domain!"));

      // compute angle between x-axis and direction
      // if y of point is negative, use -unit_vec_x
      const Number angle =
        p[1] >= 0.0 ? std::acos(direction * unit_vec_x / direction.norm()) :
                      std::acos(direction * -unit_vec_x / direction.norm());

      // create transformation matrix
      const auto rotation_axis = static_cast<dealii::Point<dim>>(axis);
      const auto rotation_matrix =
        dealii::Physics::Transformations::Rotations::rotation_matrix_3d(
          rotation_axis, angle);

      // std::cout << dealii::transpose(rotation_matrix) << std::endl;
      //  return A = Q * A' * Q_T
      return rotation_matrix * mat * dealii::transpose(rotation_matrix);
    }

  private:
    const dealii::Manifold<dim> &manifold;
    Number                       inner_radius;
    Number                       outer_radius;
    Number                       rotation_angle;

    Number eps;

    dealii::Tensor<1, dim> unit_vec_x = dealii::Tensor<1, dim>({1.0, 0.0, 0.0});

    dealii::Tensor<1, dim> axis;
  };
} // namespace Mixture::Constituents
