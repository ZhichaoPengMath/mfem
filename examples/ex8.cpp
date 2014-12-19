//                                MFEM Example 8
//
// Compile with: make ex8
//
// Sample runs:  ex8 ../data/square-disc.mesh
//               ex8 ../data/star.mesh
//               ex8 ../data/escher.mesh
//               ex8 ../data/fichera.mesh
//
// Description:  This example code demonstrates the use of the Discontinuous
//               Petrov-Galerkin (DPG) method as a simple isoparametric finite
//               element discretization of the Laplace problem -Delta u = f with
//               homogeneous Dirichlet boundary conditions. Specifically, we
//               discretize with the FE space coming from the mesh (linear by
//               default, quadratic for quadratic curvilinear mesh, NURBS for
//               NURBS mesh, etc.)
//
//               The example highlights the use of interfacial (numerical trace)
//               finite elements, interior face boundary integrators and the
//               definition of block operators and preconditioners.
//
//               We recommend viewing examples 1-5 before viewing this example.

#include <fstream>
#include "mfem.hpp"

using namespace std;
using namespace mfem;


SparseMatrix *RAP(const SparseMatrix & Rt, const SparseMatrix & A,
                  const SparseMatrix & P)
{
   SparseMatrix * R = Transpose(Rt);
   SparseMatrix * RA = Mult(*R,A);
   delete R;
   SparseMatrix * out = Mult(*RA, P);
   delete RA;
   return out;
}

class RAPOperator : public Operator
{
public:
   RAPOperator(Operator &Rt_, Operator &A_, Operator &P_)
      : Operator(Rt_.Width(), P_.Width()),
        Rt(Rt_),
        A(A_),
        P(P_),
        Px(P.Height()),
        APx(A.Height())
   {

   }

   void Mult(const Vector & x, Vector & y) const
   {
      P.Mult(x, Px);
      A.Mult(Px, APx);
      Rt.MultTranspose(APx, y);
   }

   void MultTranspose(const Vector & x, Vector & y) const
   {
      Rt.Mult(x, APx);
      A.MultTranspose(APx, Px);
      P.MultTranspose(Px, y);
   }
private:
   Operator & Rt;
   Operator & A;
   Operator & P;
   mutable Vector Px;
   mutable Vector APx;
};

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../data/star.mesh";
   int order = 1;
   bool visualization = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
	              "Finite element order (polynomial degree).");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
	              "--no-visualization",
	              "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   Mesh *mesh;

   // 2. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral or hexahedral elements with the same code.
   ifstream imesh(mesh_file);
   if (!imesh)
   {
      cerr << "\nCan not open mesh file: " << mesh_file << '\n' << endl;
      return 2;
   }
   mesh = new Mesh(imesh, 1, 1);
   imesh.close();
   const int dim = mesh->Dimension();

   // 3. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement. We choose 'ref_levels' to be the
   //    largest number that gives a final mesh with no more than 50,000
   //    elements.
   {
      int ref_levels =
         (int)floor(log(5000./mesh->GetNE())/log(2.)/dim);
      for (int l = 0; l < ref_levels; l++)
         mesh->UniformRefinement();
   }

   // 4. Define the trial, interfacial (numerical trace) and test DPG spaces:
   //    - The trial space, x0_space, contains the non-interfacial unknowns and
   //      has the essential BC.
   //    - The interfacial space, xhat_space, contains the interfacial unknowns
   //      and does not have essential BC.
   //    - The test space, test_space, is an enriched space where the enrichment
   //      degree depends on the spatial dimension of the domain.
   int trial_order = order;
   int nt_order    = trial_order - 1;
   int test_order  = nt_order + dim;
   if (test_order < trial_order)
      cerr << "Warning, test space not enriched enough to handle primal trial space\n";

   FiniteElementCollection *x0_fec   = new H1_FECollection(trial_order, dim);
   FiniteElementCollection *xhat_fec = new NT_FECollection(nt_order, dim);
   FiniteElementCollection *test_fec = new L2_FECollection(test_order, dim);

   FiniteElementSpace *x0_space   = new FiniteElementSpace(mesh, x0_fec);
   FiniteElementSpace *xhat_space = new FiniteElementSpace(mesh, xhat_fec);
   FiniteElementSpace *test_space = new FiniteElementSpace(mesh, test_fec);

   // 5. Define the block structure of the problem, by creating the offset variable.
   // Also allocate two BlockVector objects to store the solution and rhs.

   enum {x0_var, xhat_var, NVAR};

   int s0 = x0_space->GetVSize();
   int s1 = xhat_space->GetVSize();
   int s_test = test_space->GetVSize();

   Array<int> offsets(NVAR+1);
   offsets[0] = 0;
   offsets[1] = s0;
   offsets[2] = s0+s1;

   Array<int> offsets_test(2);
   offsets_test[0] = 0;
   offsets_test[1] = s_test;

   std::cout << "Number of Unknowns: " << s0 << " (trial space) " << s1 << " (interfacial space) " << s_test << " (test space)\n";

   BlockVector x(offsets), b(offsets);

   x = 0.;

   // 6. Set up the linear form F(.) which corresponds to the right-hand side of
   //    the FEM linear system, which in this case is (f,phi_i) where f=1.0 and
   //    phi_i are the basis functions in the finite element fespace.

   ConstantCoefficient one(1.0);
   LinearForm F(test_space);
   F.AddDomainIntegrator(new DomainLFIntegrator(one));
   F.Assemble();

   // 7. Set up the mixed bilinear form for the non interfacial unknowns, B0,
   //    the mixed bilinear form for the interfacial unknowns, Bhat,
   //    the stiffness matrix and its inverse on the discontinuous test space, S and Sinv,
   //    the stiffness matrix on the continuous trial space, S0.

   Array<int> ess_bdr(mesh->bdr_attributes.Max());
   ess_bdr = 1;

   MixedBilinearForm *B0 = new MixedBilinearForm(x0_space,test_space);
   B0->AddDomainIntegrator(new DiffusionIntegrator(one));
   B0->Assemble();
   B0->EliminateTrialDofs(ess_bdr, x.GetBlock(x0_var), F);
   B0->Finalize();


   MixedBilinearForm *Bhat = new MixedBilinearForm(xhat_space,test_space);
   Bhat->AddFaceIntegrator(new NTMassJumpIntegrator());
   Bhat->Assemble();
   Bhat->Finalize();

   BilinearForm *Sinv = new BilinearForm(test_space);
   Sinv->AddDomainIntegrator(new DiffusionIntegrator(one));
   Sinv->AddDomainIntegrator(new MassIntegrator(one));
   Sinv->AssembleDomainInverse();
   Sinv->Finalize();

   BilinearForm *S0 = new BilinearForm(x0_space);
   S0->AddDomainIntegrator(new DiffusionIntegrator(one));
   S0->AddDomainIntegrator(new MassIntegrator(one));
   S0->Assemble();
   S0->EliminateEssentialBC(ess_bdr);
   S0->Finalize();

   SparseMatrix &matB0   = B0->SpMat();
   SparseMatrix &matBhat = Bhat->SpMat();
   SparseMatrix &matSinv = Sinv->SpMat();
   SparseMatrix &matS0 = S0->SpMat();

   // 8. Set up the 1x2 block Least Squares DPG operator, B = [ B0   Bhat ],
   //    the normal equation operator, A = B^t Sinv B,
   //    the normal equation right-hand-size, b = B^t Sinv F.

   BlockOperator B(offsets_test, offsets);
   B.SetBlock(0,0,&matB0);
   B.SetBlock(0,1,&matBhat);

   RAPOperator A(B, matSinv, B);

   {
      Vector SinvF(s_test);
      matSinv.Mult(F,SinvF);
      B.MultTranspose(SinvF, b);
   }

   // 9. Set up a block-diagonal preconditioner for the 2x2 normal equation
   //
   //        [ S0^{-1}     0     ]
   //        [   0     Shat^{-1} ]      Shat = (Bhat^T Sinv Bhat)
   //
   //    corresponding to the primal (x0), interfacial (xhat) unknowns.

   SparseMatrix * Shat = RAP(matBhat, matSinv, matBhat);

#ifdef MFEM_USE_UMFPACK
   Operator * S0inv = new UMFPackSolver(matS0);
   Operator * Shatinv = new UMFPackSolver(*Shat);
#else
   CGSolver * S0inv = new CGSolver;
   S0inv->SetOperator(matS0);
   S0inv->SetPrintLevel(-1);
   S0inv->SetRelTol(1e-12);
   S0inv->SetMaxIter(300);
   CGSolver * Shatinv = new CGSolver;
   Shatinv->SetOperator(*Shat);
   Shatinv->SetPrintLevel(-1);
   Shatinv->SetRelTol(1e-12);
   Shatinv->SetMaxIter(300);
#endif

   BlockDiagonalPreconditioner P(offsets);
   P.SetDiagonalBlock(0, S0inv);
   P.SetDiagonalBlock(1, Shatinv);

   // 10. Solve the normal equation sytem using the PCG iterative solver.
   //     Check the weighted norm of residual for the DPG least square problem
   //     Wrap the primal variable in a GridFunction for visualization purposes.
   PCG(A, P, b, x, 1, 300, 1e-16, 0.);

   Vector LSres(s_test);
   B.Mult(x, LSres);
   LSres.Add(-1., F);
   double res2;
   res2 = matSinv.InnerProduct(LSres, LSres);
   std::cout << " || B0*x0 + Bhat*xhat - F ||_{S^-1} = " << sqrt(res2) << "\n";

   GridFunction x0;
   x0.Update(x0_space, x.GetBlock(x0_var), 0);

   // 11. Save the refined mesh and the solution. This output can be viewed later
   //     using GLVis: "glvis -m refined.mesh -g sol.gf".
   {
      ofstream mesh_ofs("refined.mesh");
      mesh_ofs.precision(8);
      mesh->Print(mesh_ofs);
      ofstream sol_ofs("sol.gf");
      sol_ofs.precision(8);
      x0.Save(sol_ofs);
   }

   // 12. (Optional) Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "solution\n" << *mesh << x0 << flush;
   }

   // 13. Free the used memory.
   delete S0inv;
   delete Shatinv;
   delete Shat;
   delete Bhat;
   delete B0;
   delete S0;
   delete Sinv;
   delete test_space;
   delete test_fec;
   delete xhat_space;
   delete xhat_fec;
   delete x0_space;
   delete x0_fec;
   delete mesh;

   return 0;
}
