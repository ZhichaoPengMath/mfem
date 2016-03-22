//                                MFEM Example 15
//
// Compile with: make ex15p
//
// Sample runs:  ex15p
//               ex15p -o 1 -y 0.2
//               ex15p -o 4 -y 0.1
//               ex15p -n 5
//               ex15p -p 1 -n 3
//
//               Other meshes:
//
//               ex15p -m ../data/square-disc-nurbs.mesh
//               ex15p -m ../data/disc-nurbs.mesh
//               ex15p -m ../data/fichera.mesh
//               ex15p -m ../data/ball-nurbs.mesh
//               ex15p -m ../data/mobius-strip.mesh
//               ex15p -m ../data/amr-quad.mesh
//
//               Conforming meshes (no derefinement):
//
//               ex15p -m ../data/square-disc.mesh
//               ex15p -m ../data/escher.mesh -o 1
//               ex15p -m ../data/square-disc-surf.mesh
//
// Description:  Building on Example 6, this example demonstrates dynamic AMR.
//               The mesh is adapted to a time-dependent solution by refinement
//               as well as by derefinement. For simplicity, the solution is
//               prescribed and no time integration is done. However, the error
//               estimation and refinement/derefinement decisions are realistic.
//
//               At each outer iteration the right hand side function is changed
//               to mimic a time dependent problem.  Within each inner iteration
//               the problem is solved on a sequence of meshes which are locally
//               refined according to a simple ZZ error estimator.  At the end
//               of the inner iteration the error estimates are also used to
//               identify any elements which may be over-refined and a single
//               derefinement step is performed.
//
//               The example demonstrates MFEM's capability to refine, derefine
//               nonconforming meshes, in 2D and 3D, and on linear, curved and
//               surface meshes. Interpolation of functions between coarse and
//               fine meshes, persistent GLVis visualization, and saving of
//               time-dependent fields for external visualization with VisIt
//               (visit.llnl.gov) are also illustrated.
//
//               We recommend viewing Examples 1, 6 and 9 before viewing this
//               example.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

// Choices for the problem setup. Affect bdr_func and rhs_func.
int problem;
int nfeatures;

// Prescribed time-dependent boundary and right-hand side functions.
double bdr_func(const Vector &pt, double t);
double rhs_func(const Vector &pt, double t);

// Estimate the solution errors with a simple (ZZ-type) error estimator.
double EstimateErrors(int order, int dim, int sdim, Mesh & mesh,
                      const GridFunction & x, Vector & errors);

// Update the finite element space, interpolate the solution and perform
// parallel load balancing.
void UpdateProblem(Mesh &mesh, FiniteElementSpace &fespace,
                   GridFunction &x, BilinearForm &a, LinearForm &b);


int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   problem = 0;
   nfeatures = 1;
   const char *mesh_file = "../data/star-hilbert.mesh";
   int order = 2;
   double max_elem_error = 5.0e-3;
   double hysteresis = 0.15; // derefinement safety coefficient
   int nc_limit = 3;         // maximum level of hanging nodes
   bool visualization = true;
   bool visit = false;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&problem, "-p", "--problem",
                  "Problem setup to use: 0 = spherical front, 1 = ball.");
   args.AddOption(&nfeatures, "-n", "--nfeatures",
                  "Number of solution features (fronts/balls).");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&max_elem_error, "-e", "--max-err",
                  "Maximum element error");
   args.AddOption(&hysteresis, "-y", "--hysteresis",
                  "Derefinement safety coefficient.");
   args.AddOption(&nc_limit, "-l", "--nc-limit",
                  "Maximum level of hanging nodes.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&visit, "-visit", "--visit-datafiles", "-no-visit",
                  "--no-visit-datafiles",
                  "Save data files for VisIt (visit.llnl.gov) visualization.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   // 2. Read the mesh from the given mesh file on all processors. We can
   //    handle triangular, quadrilateral, tetrahedral, hexahedral, surface and
   //    volume meshes with the same code.
   ifstream imesh(mesh_file);
   if (!imesh)
   {
      cerr << "\nCan not open mesh file: " << mesh_file << '\n' << endl;
      return 2;
   }
   Mesh mesh(imesh, 1, 1);
   imesh.close();
   int dim = mesh.Dimension();
   int sdim = mesh.SpaceDimension();

   // 4. Project a NURBS mesh to a piecewise-quadratic curved mesh. Make sure
   //    that the mesh is non-conforming if it has quads or hexes.
   if (mesh.NURBSext)
   {
      mesh.UniformRefinement();
      mesh.SetCurvature(2);
   }
   mesh.EnsureNCMesh();

   // 7. All boundary attributes will be used for essential (Dirichlet) BC.
   MFEM_VERIFY(mesh.bdr_attributes.Size() > 0,
               "Boundary attributes required in the mesh.");
   Array<int> ess_bdr(mesh.bdr_attributes.Max());
   ess_bdr = 1;

   // 6. Define a finite element space on the mesh. The polynomial order is one
   //    (linear) by default, but this can be changed on the command line.
   H1_FECollection fec(order, dim);
   FiniteElementSpace fespace(&mesh, &fec);

   // 7. As in Example 1p, we set up bilinear and linear forms corresponding to
   //    the Laplace problem -\Delta u = 1. We don't assemble the discrete
   //    problem yet, this will be done in the inner loop.
   BilinearForm a(&fespace);
   LinearForm b(&fespace);

   ConstantCoefficient one(1.0);
   FunctionCoefficient bdr(bdr_func);
   FunctionCoefficient rhs(rhs_func);

   a.AddDomainIntegrator(new DiffusionIntegrator(one));
   b.AddDomainIntegrator(new DomainLFIntegrator(rhs));

   // 8. The solution vector x and the associated finite element grid function
   //    will be maintained over the AMR iterations.
   GridFunction x(&fespace);

   // 9. Connect to GLVis. Prepare for VisIt output.
   char vishost[] = "localhost";
   int  visport   = 19916;

   socketstream sout;
   if (visualization)
   {
      sout.open(vishost, visport);
      if (!sout)
      {
         cout << "Unable to connect to GLVis server at "
              << vishost << ':' << visport << endl;
         cout << "GLVis visualization disabled.\n";
         visualization = false;
      }
      sout.precision(8);
   }

   VisItDataCollection visit_dc("Example15-Parallel", &mesh);
   visit_dc.RegisterField("solution", &x);
   int vis_cycle = 0;

   // 10. The outer time loop. In each iteration we update the right hand side,
   //     solve the problem on the current mesh, visualize the solution,
   //     estimate the error on all elements, refine bad elements and update all
   //     objects to work with the new mesh.  Then we derefine any elements
   //     which have very small errors.
   for (double time = 0.0; time < 1.0 + 1e-10; time += 0.01)
   {
      cout << "\nTime " << time << "\n\nRefinement:" << endl;

      // Set the current time in the coefficients
      bdr.SetTime(time);
      rhs.SetTime(time);

      Vector errors;

      // 11. The inner refinement loop. At the end we want to have the current
      //     time step resolved to the prescribed tolerance in each element.
      for (int ref_it = 1; ; ref_it++)
      {
         cout << "Iteration: " << ref_it << ", number of unknowns: "
              << fespace.GetVSize() << endl;

         // 11a. Recompute the field on the current mesh: assemble the stiffness
         //      matrix and the right-hand side.
         a.Assemble();
         b.Assemble();

         // 11b. Project the exact solution to the essential DOFs.
         x.ProjectBdrCoefficient(bdr, ess_bdr);

         // 11c. Create and solve the linear system.
         Array<int> ess_tdof_list;
         fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

         SparseMatrix A;
         Vector B, X;
         a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

#ifndef MFEM_USE_SUITESPARSE
         GSSmoother M(A);
         PCG(A, M, B, X, 0, 200, 1e-12, 0.0);
#else
         UMFPackSolver umf_solver;
         umf_solver.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
         umf_solver.SetOperator(A);
         umf_solver.Mult(B, X);
#endif

         // 11d. Extract the local solution on each processor.
         a.RecoverFEMSolution(X, b, x);

         // 11e. Send the solution by socket to a GLVis server and optionally
         //      save it in VisIt format.
         if (visualization)
         {
            sout.precision(8);
            sout << "solution\n" << mesh << x << flush;
         }
         if (visit)
         {
            visit_dc.SetCycle(vis_cycle++);
            visit_dc.SetTime(time);
            visit_dc.Save();
         }

         // 11f. Estimate element errors using the Zienkiewicz-Zhu error
         //      estimator. The bilinear form integrator must have the
         //      'ComputeElementFlux' method defined.
         {
            DiffusionIntegrator flux_integrator(one);
            FiniteElementSpace flux_fespace(&mesh, &fec, sdim);
            GridFunction flux(&flux_fespace);
            ZZErrorEstimator(flux_integrator, x, flux, errors);
         }

         // 11g. Refine elements
         if (!mesh.RefineByError(errors, max_elem_error, -1, nc_limit))
         {
            break;
         }

         // 11h. Update the space and interpolate the solution.
         UpdateProblem(mesh, fespace, x, a, b);
      }

      // 12. Use error estimates from the last iterations to check for possible
      //     derefinements.
      if (mesh.Nonconforming())
      {
         double threshold = hysteresis * max_elem_error;
         if (mesh.DerefineByError(errors, threshold, nc_limit))
         {
            cout << "\nDerefined elements." << endl;

            // 12a. Update the space and interpolate the solution.
            UpdateProblem(mesh, fespace, x, a, b);
         }
      }

      a.Update();
      b.Update();
   }

   return 0;
}


void UpdateProblem(Mesh &mesh, FiniteElementSpace &fespace,
                   GridFunction &x, BilinearForm &a, LinearForm &b)
{
   // Update the space: recalculate the number of DOFs and construct a matrix
   // that will adjust any GridFunctions to the new mesh state.
   fespace.Update();

   // Interpolate the solution on the new mesh by applying the transformation
   // matrix computed in the finite element space. Multiple GridFunctions could
   // be updated here.
   x.Update();

   // Free any transformation matrices to save memory.
   fespace.UpdatesFinished();

   // Inform the linear and bilinear forms that the space has changed.
   a.Update();
   b.Update();
}


const double alpha = 0.02;

// Spherical front with a Gaussian cross section and radius t
double front(double x, double y, double z, double t, int)
{
   double r = sqrt(x*x + y*y + z*z);
   return exp(-0.5*pow((r - t)/alpha, 2));
}

double front_laplace(double x, double y, double z, double t, int dim)
{
   double x2 = x*x, y2 = y*y, z2 = z*z, t2 = t*t;
   double r = sqrt(x2 + y2 + z2);
   double a2 = alpha*alpha, a4 = a2*a2;
   return -exp(-0.5*pow((r - t)/alpha, 2)) / a4 *
          (-2*t*(x2 + y2 + z2 - (dim-1)*a2/2)/r + x2 + y2 + z2 + t2 - dim*a2);
}

// Smooth spherical step function with radius t
double ball(double x, double y, double z, double t, int)
{
   double r = sqrt(x*x + y*y + z*z);
   return -atan(2*(r - t)/alpha);
}

double ball_laplace(double x, double y, double z, double t, int dim)
{
   double x2 = x*x, y2 = y*y, z2 = z*z, t2 = 4*t*t;
   double r = sqrt(x2 + y2 + z2);
   double a2 = alpha*alpha;
   double den = pow(-a2 - 4*(x2 + y2 + z2 - 2*r*t) - t2, 2.0);
   return (dim == 2) ? 2*alpha*(a2 + t2 - 4*x2 - 4*y2)/r/den
          /*      */ : 4*alpha*(a2 + t2 - 4*r*t)/r/den;
}

// Composes several features into one function
template<typename F0, typename F1>
double composite_func(const Vector &pt, double t, F0 f0, F1 f1)
{
   int dim = pt.Size();
   double x = pt(0), y = pt(1), z = 0.0;
   if (dim == 3) { z = pt(2); }

   if (problem == 0)
   {
      if (nfeatures <= 1)
      {
         return f0(x, y, z, t, dim);
      }
      else
      {
         double sum = 0.0;
         for (int i = 0; i < nfeatures; i++)
         {
            double x0 = 0.5*cos(2*M_PI * i / nfeatures);
            double y0 = 0.5*sin(2*M_PI * i / nfeatures);
            sum += f0(x - x0, y - y0, z, t, dim);
         }
         return sum;
      }
   }
   else
   {
      double sum = 0.0;
      for (int i = 0; i < nfeatures; i++)
      {
         double x0 = 0.5*cos(2*M_PI * i / nfeatures + M_PI*t);
         double y0 = 0.5*sin(2*M_PI * i / nfeatures + M_PI*t);
         sum += f1(x - x0, y - y0, z, 0.25, dim);
      }
      return sum;
   }
}

// Exact solution, used for the Dirichlet BC.
double bdr_func(const Vector &pt, double t)
{
   return composite_func(pt, t, front, ball);
}

// Laplace of the exact solution, used for the right hand side.
double rhs_func(const Vector &pt, double t)
{
   return composite_func(pt, t, front_laplace, ball_laplace);
}

