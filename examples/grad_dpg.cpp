//	2019/05/30 PZC
//
// Compile with: make grad_dpg
//
// Sample runs:  ./grad_dpg -m ../data/square-disc.mesh
//               ./grad_dpg -m ../data/star.mesh
//               ./grad_dpg -m ../data/star-mixed.mesh
//               ./grad_dpg -m ../data/escher.mesh
//               ./grad_dpg -m ../data/fichera.mesh
//               ./grad_dpg -m ../data/fichera-mixed.mesh
//               ./grad_dpg -m ../data/square-disc-p2.vtk
//               ./grad_dpg -m ../data/square-disc-p3.mesh
//               ./grad_dpg -m ../data/star-surf.mesh -o 2
//               ./grad_dpg -m ../data/mobius-strip.mesh
//
// Description: This code uses DPG method to solve 
//					\grad u = f
//				The variational form
//					- ( u , div(v) )  + \lgl \hat{u}, v\cdot n \rgl  = (f,v)
//               Here, \hat{u} is the trace term of a H^1 function
//
//               The code is written to test the DPG face integretor
//                DGNormalTraceJumpIntegrator
//               

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

void f_exact(const Vector & x, Vector & f);
double u_exact(const Vector & x);

double alpha_pzc = 100.;

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../data/xperiodic.mesh";
   int order = 1;
   bool visualization = 1;
   int ref_levels = -1;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&alpha_pzc, "-alpha", "--alpha",
                  "arctan( alpha * x) as exact solution");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly, -1 for auto.");

   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   // 2. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral, hexahedral, surface and volume meshes with
   //    the same code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   // 3. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement. We choose 'ref_levels' to be the
   //    largest number that gives a final mesh with no more than 10,000
   //    elements.
   {
      if (ref_levels < 0)
      {
         ref_levels = (int)floor(log(50000./mesh->GetNE())/log(2.)/dim);
      }
      for (int l = 0; l < ref_levels; l++)
      {
         mesh->UniformRefinement();
      }
   }

   // 4. Define the trial, interfacial (trace) and test DPG spaces:
   // u0_space:     L2
   // utrace_space: H1_trace
   // test_space:   L2
   unsigned int trial_order = order;
   unsigned int trace_order = order + 1;
   unsigned int test_order  = order + dim; /* reduced order, full order is
                                        (order + dim - 1) */

   FiniteElementCollection *u0_fec, *uhat_fec, *test_fec;

   u0_fec   = new L2_FECollection(trial_order, dim);
   uhat_fec = new H1_Trace_FECollection(trace_order, dim);
   test_fec = new L2_FECollection(test_order, dim);

   FiniteElementSpace *u0_space   = new FiniteElementSpace(mesh, u0_fec);
   FiniteElementSpace *uhat_space = new FiniteElementSpace(mesh, uhat_fec);
   FiniteElementSpace *test_space = new FiniteElementSpace(mesh, test_fec, dim);

   // 5. Define the block structure of the problem, by creating the offset
   //    variables. Also allocate two BlockVector objects to store the solution
   //    and rhs.
   enum {u0_var, uhat_var, NVAR};

   int s0 = u0_space->GetVSize();
   int s1 = uhat_space->GetVSize();
   int s_test = test_space->GetVSize();

   Array<int> offsets(NVAR+1);
   offsets[0] = 0;
   offsets[1] = s0;
   offsets[2] = s0+s1;

   Array<int> offsets_test(2);
   offsets_test[0] = 0;
   offsets_test[1] = s_test;

   std::cout << "\nNumber of Unknowns:\n"
             << " Trial space,     U0   : " << s0
             << " (order " << trial_order << ")\n"
             << " Interface space, Uhat : " << s1
             << " (order " << trace_order << ")\n"
             << " Test space,      Y    : " << s_test
             << " (order " << test_order << ")\n\n";

   std::cout<< " \n order of spaces \n"
	        << " Trial space, U0   : "<< trial_order
	        << " Trace space, Uhat : "<< trace_order
	        << " Test space,  Y    : "<< test_order<<std::endl;

   BlockVector x(offsets), b(offsets);
   x = 0.;

   // 6. Set up the linear form F(.) which corresponds to the right-hand side of
   //    the FEM linear system, which in this case is (f,phi_i) where f=1.0 and
   //    phi_i are the basis functions in the test finite element fespace.
   ConstantCoefficient one(1.0);          /* coefficients */
   VectorFunctionCoefficient f_coeff( dim, f_exact );/* coefficients */
   FunctionCoefficient u_coeff( u_exact );/* coefficients */

   LinearForm F(test_space);
//   F.AddDomainIntegrator(new DomainLFIntegrator(one));
   F.AddDomainIntegrator(new VectorDomainLFIntegrator(f_coeff));
   F.Assemble();

//   // 6. Deal with boundary conditions
//   Array<int> ess_bdr(mesh->bdr_attributes.Max());
//   ess_bdr = 1;
//
//   Array<int> ess_trial_dof_list;/* store the location (index) of  boundary element  */
//   u0_space->GetEssentialTrueDofs(ess_bdr, ess_trial_dof_list);
//
//   cout<<endl<<endl<<"Boundary information: "<<endl;
//   cout<<" boundary attribute size " <<mesh->bdr_attributes.Max() <<endl;
//   cout<<" number of essential true dofs "<<ess_trial_dof_list.Size()<<endl;
//
////   x.GetBlock(x0_var).ProjectBdrCoefficient(u_coeff, ess_bdr);
////   x.GetBlock(x0_var).SetSubVector(ess_trial_dof_list,10.);
   GridFunction x0;
   x0.MakeRef(u0_space, x.GetBlock(u0_var), 0);
//   x0.ProjectCoefficient(u_coeff);

   // 7. Set up the mixed bilinear form for the primal trial unknowns, B0,
   //    the mixed bilinear form for the interfacial unknowns, Bhat,
   //    the inverse stiffness matrix on the discontinuous test space, Sinv,
   //    and the stiffness matrix on the continuous trial space, S0.
   
   /* diffusion integrator (trial,test) */
   MixedBilinearForm *B0 = new MixedBilinearForm(u0_space,test_space);
   B0->AddDomainIntegrator(new TransposeIntegrator
								(new VectorDivergenceIntegrator() )
						  );
   B0->Assemble();
//   B0->EliminateTrialDofs(ess_bdr, x.GetBlock(x0_var), F);
   B0->Finalize();

   /* trace terms */
   MixedBilinearForm *Bhat = new MixedBilinearForm(uhat_space,test_space);
   Bhat->AddTraceFaceIntegrator(new DGNormalTraceJumpIntegrator());
   Bhat->Assemble();
   Bhat->Finalize();

   /* mass matrix corresponding to the test norm */
   BilinearForm *Sinv = new BilinearForm(test_space);
   SumIntegrator *Sum = new SumIntegrator;
   Sum->AddIntegrator(new DiffusionIntegrator(one));
   Sum->AddIntegrator(new MassIntegrator(one));

   Sinv->AddDomainIntegrator(new InverseIntegrator(Sum));
   Sinv->Assemble();
   Sinv->Finalize();

   /* diffusion integrator in trial space */
//   BilinearForm *S0 = new BilinearForm(u0_space);
//   S0->AddDomainIntegrator(new DiffusionIntegrator(one));
//   S0->Assemble();
////   S0->EliminateEssentialBC(ess_bdr);
//   S0->Finalize();

   SparseMatrix &matB0   = B0->SpMat();	
   matB0 *= -1.;
   SparseMatrix &matBhat = Bhat->SpMat();
   SparseMatrix &matSinv = Sinv->SpMat();

//   matSinv.PrintMatlab();

   cout<<endl<<" matrices size:"<<endl
	   <<" B0:   "<< matB0.Width()<<" X "<<matB0.Height()<<endl
	   <<" Bhat: "<< matBhat.Width()<<" X "<<matBhat.Height()<<endl
	   <<" Sinv: "<< matSinv.Width()<< " X "<<matSinv.Height()<<endl
	   <<endl<<endl;
//   SparseMatrix &matS0   = S0->SpMat();
     ofstream myfile ("Kmat.dat");

   // 8. Set up the 1x2 block Least Squares DPG operator, B = [B0  Bhat],
   //    the normal equation operator, A = B^t Sinv B, and
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
   SparseMatrix * matS0 = RAP(matB0, matSinv, matB0);
   SparseMatrix * Shat = RAP(matBhat, matSinv, matBhat);

#ifndef MFEM_USE_SUITESPARSE
   const double prec_rtol = 1e-10;
   const int prec_maxit = 20000;

   CGSolver *S0inv = new CGSolver;
   S0inv->SetOperator(*matS0);
   S0inv->SetPrintLevel(-1);
   S0inv->SetRelTol(prec_rtol);
   S0inv->SetMaxIter(prec_maxit);

   CGSolver *Shatinv = new CGSolver;
   Shatinv->SetOperator(*Shat);
   Shatinv->SetPrintLevel(-1);
   Shatinv->SetRelTol(prec_rtol);
   Shatinv->SetMaxIter(prec_maxit);
   // Disable 'iterative_mode' when using CGSolver (or any IterativeSolver) as
   // a preconditioner:
   S0inv->iterative_mode = false;
   Shatinv->iterative_mode = false;
#else
   Operator *S0inv = new UMFPackSolver(matS0);
   Operator *Shatinv = new UMFPackSolver(*Shat);
#endif

   BlockDiagonalPreconditioner P(offsets);
   P.SetDiagonalBlock(0, S0inv);
   P.SetDiagonalBlock(1, Shatinv);

   // 10. Solve the normal equation system using the PCG iterative solver.
   //     Check the weighted norm of residual for the DPG least square problem.
   //     Wrap the primal variable in a GridFunction for visualization purposes.
//   PCG(A, P, b, x, 1, 200, 1e-12, 0.0);
     GMRES(A, P, b, x, 1, 1000, 1000, 1e-12, 0.0);

   {
      Vector LSres(s_test);
      B.Mult(x, LSres);
      LSres -= F;
      double res = sqrt(matSinv.InnerProduct(LSres, LSres));
      cout << "\n|| B0*x0 + Bhat*xhat - F ||_{S^-1} = " << res << endl;
   }

   // 10b. error 
//   x0.ProjectCoefficient(u_coeff);
   cout<< "\n dimension: "<<dim<<endl;
   cout << "\nelement number of the mesh: "<< mesh->GetNE ()<<endl; 
   cout << "\n|| u_h - u ||_{L^2} = " << x0.ComputeL2Error(u_coeff) << '\n' << endl;

//
   // 11. Save the refined mesh and the solution. This output can be viewed
   //     later using GLVis: "glvis -m refined.mesh -g sol.gf".
   {
      ofstream mesh_ofs("refined.mesh");
      mesh_ofs.precision(8);
      mesh->Print(mesh_ofs);
      ofstream sol_ofs("sol.gf");
      sol_ofs.precision(8);
      x0.Save(sol_ofs);
   }

   // 12. Send the solution by socket to a GLVis server.
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
//   delete S0;
   delete Sinv;
   delete test_space;
   delete test_fec;
   delete uhat_space;
   delete uhat_fec;
   delete u0_space;
   delete u0_fec;
   delete mesh;

   return 0;
}


/* define the source term on the right hand side */
// The right hand side
//  grad u = f
void f_exact(const Vector & x, Vector & f){
	if(x.Size() == 2){
		f(0) = M_PI * cos(M_PI * x(1) );
		f(1) = 0.;

//		f(0) = 2.*M_PI* cos(2.*M_PI*x(0) );
//		f(1) = 2.*M_PI* cos(2.*M_PI*x(1) );
	}
	else{
		f(0) = 0;
	}

}

/* exact solution */
double u_exact(const Vector & x){
	if(x.Size() == 2){
		return  sin( M_PI * x(1) ); /* first index is 0 */
//		return sin(2.*M_PI* x(0) ) + sin(2.* M_PI * x(1) ); /* first index is 0 */
	}
	else{
		return 0;
	}
}

