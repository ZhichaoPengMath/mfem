//                                MFEM Example 8
//
// Compile with: make ultra_weak_dpg
//
// Sample runs:  ./ultra_weak_dpg -m ../data/square-disc.mesh
//               ./ultra_weak_dpg -m ../data/star.mesh
//               ./ultra_weak_dpg -m ../data/star-mixed.mesh
//               ./ultra_weak_dpg -m ../data/escher.mesh
//               ./ultra_weak_dpg -m ../data/fichera.mesh
//               ./ultra_weak_dpg -m ../data/fichera-mixed.mesh
//               ./ultra_weak_dpg -m ../data/square-disc-p2.vtk
//               ./ultra_weak_dpg -m ../data/square-disc-p3.mesh
//               ./ultra_weak_dpg -m ../data/star-surf.mesh -o 2
//               ./ultra_weak_dpg -m ../data/mobius-strip.mesh
//
// Description:  This example code demonstrates the use of the Discontinuous
//               Petrov-Galerkin (DPG) method in its ultra-weak form
//						-\Laplace u = f with Dirichlet boundary condition
//				 Rewrite the equation in its first order form
//						   q  =  \grad u
//					  -div(q) =  f
//				 Variational form:
//						( q, v ) - (u, div(v) ) + \lgl \hat{u}, v\cdot n \rgl = 0
//						( q, \grad \tau) - \lgl \hat{q}, \tau \rgl  = f
//				 here, \hat{q} \approx q\cdot n
//				 Trial space:
//					Interior terms:
//						  q, u \in L^2
//					Trace terms 
//					      \hat{q} \in H^{-1/2} = trace of H(div)
//					      \hat{u} \in H^{1/2}  = trace of H^1
//
//				 Check paper:
//						"AN ANALYSIS OF THE PRACTICAL DPG METHOD", 
//						J. GOPALAKRISHNAN AND W. QIU, 2014
//				 for details
//
//               The example highlights the use of interfacial (trace) finite
//               elements and spaces, trace face integrators and the definition
//               of block operators and preconditioners.
//
//               We recommend viewing examples 1-5 before viewing this example.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

double f_exact(const Vector & x);
double u_exact(const Vector & x);

double alpha_pzc = 100.;

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../data/star.mesh";
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
   //		q = grad u
   //		qhat trace q\codt n, trace H(div),
   //		u
   //		uhat trace u, trace H1
   //
   //		test_space L2
   //    - u0_space: scalar, contains the non-interfacial unknowns
   //    - uhat_space: trace space, contains the interfacial unkowns and the
   //      essential boundary unknowns
   //    - q0_space: vector, contains the non-interfacial unkowns
   //    - qhat_space: trace space, contains the interfacial unkowns
   //
   //    - The test space, test_space, is an enriched space where the enrichment
   //      degree may depend on the spatial dimension of the domain, the type of
   //      the mesh and the trial space order.
   //    - vtest_space for vector test functions
   //    - stest_space for scalar test functions
   
	/* order of polynomial spaces */
   unsigned int trial_order = order;				
   unsigned int h1_trace_order = order + 1;
   unsigned int rt_trace_order = order;
   unsigned int test_order = order + dim;

   FiniteElementCollection * u0_fec, * q0_fec, * uhat_fec, *qhat_fec, * test_fec;

   u0_fec = new L2_FECollection(trial_order,dim);
   q0_fec = new L2_FECollection(trial_order,dim);

   uhat_fec = new H1_Trace_FECollection(h1_trace_order,dim);
   qhat_fec = new RT_Trace_FECollection(rt_trace_order,dim);

   test_fec = new L2_FECollection(test_order,dim);

   FiniteElementSpace * u0_space = new FiniteElementSpace(mesh, u0_fec);
   FiniteElementSpace * q0_space = new FiniteElementSpace(mesh, q0_fec, dim);
   FiniteElementSpace * uhat_space = new FiniteElementSpace(mesh, uhat_fec);
   FiniteElementSpace * qhat_space = new FiniteElementSpace(mesh, qhat_fec);
   
   FiniteElementSpace * vtest_space = new FiniteElementSpace(mesh, test_fec,dim);
   FiniteElementSpace * stest_space = new FiniteElementSpace(mesh, test_fec);
   

   // 5. Define the block structure of the problem, by creating the offset
   //    variables. Also allocate two BlockVector objects to store the solution
   //    and rhs.
   enum {q0_var, u0_var,qhat_var,uhat_var, NVAR};

   int size_u0 = u0_space->GetVSize();
   int size_q0 = q0_space->GetVSize();
   int size_uhat = uhat_space->GetVSize();
   int size_qhat = qhat_space->GetVSize();
   int size_vtest = vtest_space->GetVSize();
   int size_stest = stest_space->GetVSize();

   Array<int> offsets(NVAR+1);
   offsets[0] = 0;
   offsets[1] = size_q0;
   offsets[2] = offsets[1] + size_u0;
   offsets[3] = offsets[2] + size_qhat;
   offsets[4] = offsets[3] + size_uhat;

   Array<int> offsets_test(3);
   offsets_test[0] = 0;
   offsets_test[1] = size_vtest;
   offsets_test[2] = offsets_test[1] + size_stest;

   std::cout << "\nNumber of Unknowns:\n"     << endl
		     << " U0          " <<  size_u0   << endl
		     << " Q0          " <<  size_q0   << endl
			 << " Uhat        " <<  size_uhat << endl
			 << " Qhat        " <<  size_qhat << endl
			 << " Vector-test " << size_vtest << endl
			 << " Scalar-test " << size_stest << endl << endl;


   // 6. Set up the linear form F(.) which corresponds to the right-hand side of
   //    the FEM linear system, which in this case is (f,phi_i) where f=1.0 and
   //    phi_i are the basis functions in the test finite element fespace.

   BlockVector x(offsets), b(offsets);
   x = 0.;

   BlockVector F(offsets_test); /* block vector for the linear form on the right hand side */
   F = 0.;

   ConstantCoefficient one(1.0);          /* coefficients */
   ConstantCoefficient zero(0.);          /* coefficients */
   FunctionCoefficient f_coeff( f_exact );/* coefficients */
   FunctionCoefficient u_coeff( u_exact );/* coefficients */

//   LinearForm * fq(new LinearForm);
//   fq->Update(vtest_space, F.GetBlock(0) ,0);
//   fq->AddDomainIntegrator(new DomainLFIntegrator( zero ) );
//   fq->Assemble();

   LinearForm * fu(new LinearForm);
   fu->Update(stest_space, F.GetBlock(1) ,0);
   fu->AddDomainIntegrator( new DomainLFIntegrator(f_coeff) );
   fu->Assemble();

   // 6. Deal with boundary conditions
   //    Dirichlet boundary condition is imposed throught trace term  \hat{u}
   Array<int> ess_bdr(mesh->bdr_attributes.Max());
   ess_bdr = 1;

   Array<int> ess_trace_dof_list;/* store the location (index) of  boundary element  */
   uhat_space->GetEssentialTrueDofs(ess_bdr, ess_trace_dof_list);

   cout<<endl<<endl<<"Boundary information: "<<endl;
   cout<<" boundary attribute size " <<mesh->bdr_attributes.Max() <<endl;
   cout<<" number of essential true dofs "<<ess_trace_dof_list.Size()<<endl;

   GridFunction uhat;
   uhat.MakeRef(uhat_space, x.GetBlock(uhat_var), 0);
   uhat.ProjectBdrCoefficient(u_coeff,ess_trace_dof_list);

   // 7. Set up the mixed bilinear forms 
   //    B_mass_q: (q,v)
   //    B_u_dot_div: (u, div(v) ) 
   //    B_u_normal_jump:  \lgl \hat{u} , v\cdot n\rgl
   //
   //	 B_q_grad: (q, \grad \tau)
   //	 B_q_jump: \lgl \hat{q}, \tau \rgl
   //
   //    the inverse stiffness matrix on the discontinuous test space, Sinv,
   //    and the stiffness matrix on the continuous trial space, S0.
   
   /* operator (q,v) */
   MixedBilinearForm *B_mass_q = new MixedBilinearForm(q0_space,vtest_space);
   B_mass_q->AddDomainIntegrator(new VectorMassIntegrator() );
   B_mass_q->Assemble();
   B_mass_q->Finalize();

   cout<<endl<< "(q,v) assembled"<<endl;

   /* operator ( u , div(v) ) */
   /* Vector DivergenceIntegrator(): (div(u), v), where u is a vector and v is a scalar*/
   /* here we want (u, div(v) ), so we take its transpose */
   MixedBilinearForm *B_u_dot_div = new MixedBilinearForm(u0_space,vtest_space);
   B_u_dot_div->AddDomainIntegrator(new TransposeIntegrator
											( new VectorDivergenceIntegrator() )
								   );
   B_u_dot_div->Assemble();
   B_u_dot_div->Finalize();
   cout<< "( u, div(v) ) assembled"<<endl;

   /* operator \lgl u, v\cdot n rgl */
   MixedBilinearForm *B_u_normal_jump = new MixedBilinearForm(uhat_space, vtest_space);
   B_u_normal_jump->AddTraceFaceIntegrator( new DGNormalTraceJumpIntegrator() );
   B_u_normal_jump->Assemble();
   B_u_normal_jump->EliminateTrialDofs(ess_bdr, x.GetBlock(uhat_var), F);
   B_u_normal_jump->Finalize();

   cout<<endl<<"< u, v cdot n > assembled"<<endl;

   /* operator  (div q, \tau) */
   MixedBilinearForm * B_q_div = new MixedBilinearForm(q0_space,stest_space);
   B_q_div->AddDomainIntegrator( new VectorDivergenceIntegrator() );
   B_q_div->Assemble();
   B_q_div->Finalize();

   cout<<endl<<"( div(q), v ) assembled"<<endl;

   /* mass matrix corresponding to the test norm */
   BilinearForm *inv_vtest = new BilinearForm(vtest_space);
   SumIntegrator *vsum = new SumIntegrator;
//   vsum->AddIntegrator


   /* size of matrices */
   SparseMatrix &matB_mass_q = B_mass_q->SpMat();
   SparseMatrix &matB_u_dot_div = B_u_dot_div->SpMat();
   SparseMatrix &matB_u_normal_jump = B_u_normal_jump->SpMat();
   SparseMatrix &matB_q_div = B_q_div->SpMat();

   cout<<endl<<endl<<"matrix dimensions: "<<endl
	   <<" mass_q:        "<<matB_mass_q.Width()   <<" X "<<matB_mass_q.Height()<<endl
	   <<" u_dot_div:     "<<matB_u_dot_div.Width()<<" X "<<matB_u_dot_div.Height()<<endl
	   <<" u_normal_jump: "<<matB_u_normal_jump.Width()<<" X "<<matB_u_normal_jump.Height()<<endl
	   <<" q_div:         "<<matB_q_div.Width()<<" X "<<matB_q_div.Height()<<endl;



//   /* mass matrix corresponding to the test norm */
//   BilinearForm *Sinv = new BilinearForm(test_space);
//   SumIntegrator *Sum = new SumIntegrator;
//   Sum->AddIntegrator(new DiffusionIntegrator(one));
//   Sum->AddIntegrator(new MassIntegrator(one));
//   Sinv->AddDomainIntegrator(new InverseIntegrator(Sum));
//   Sinv->Assemble();
//   Sinv->Finalize();
//
//   /* diffusion integrator in trial space */
//   BilinearForm *S0 = new BilinearForm(x0_space);
//   S0->AddDomainIntegrator(new DiffusionIntegrator(one));
//   S0->Assemble();
//   S0->EliminateEssentialBC(ess_bdr);
//   S0->Finalize();
//
//   SparseMatrix &matB0   = B0->SpMat();
//   SparseMatrix &matBhat = Bhat->SpMat();
//   SparseMatrix &matSinv = Sinv->SpMat();
//   SparseMatrix &matS0   = S0->SpMat();
//
//   // 8. Set up the 1x2 block Least Squares DPG operator, B = [B0  Bhat],
//   //    the normal equation operator, A = B^t Sinv B, and
//   //    the normal equation right-hand-size, b = B^t Sinv F.
//   BlockOperator B(offsets_test, offsets);
//   B.SetBlock(0,0,&matB0);
//   B.SetBlock(0,1,&matBhat);
//   RAPOperator A(B, matSinv, B);
//   {
//      Vector SinvF(s_test);
//      matSinv.Mult(F,SinvF);
//      B.MultTranspose(SinvF, b);
//   }
//
//   // 9. Set up a block-diagonal preconditioner for the 2x2 normal equation
//   //
//   //        [ S0^{-1}     0     ]
//   //        [   0     Shat^{-1} ]      Shat = (Bhat^T Sinv Bhat)
//   //
//   //    corresponding to the primal (x0) and interfacial (xhat) unknowns.
//   SparseMatrix * Shat = RAP(matBhat, matSinv, matBhat);
//
//#ifndef MFEM_USE_SUITESPARSE
//   const double prec_rtol = 1e-3;
//   const int prec_maxit = 200;
//   CGSolver *S0inv = new CGSolver;
//   S0inv->SetOperator(matS0);
//   S0inv->SetPrintLevel(-1);
//   S0inv->SetRelTol(prec_rtol);
//   S0inv->SetMaxIter(prec_maxit);
//   CGSolver *Shatinv = new CGSolver;
//   Shatinv->SetOperator(*Shat);
//   Shatinv->SetPrintLevel(-1);
//   Shatinv->SetRelTol(prec_rtol);
//   Shatinv->SetMaxIter(prec_maxit);
//   // Disable 'iterative_mode' when using CGSolver (or any IterativeSolver) as
//   // a preconditioner:
//   S0inv->iterative_mode = false;
//   Shatinv->iterative_mode = false;
//#else
//   Operator *S0inv = new UMFPackSolver(matS0);
//   Operator *Shatinv = new UMFPackSolver(*Shat);
//#endif
//
//   BlockDiagonalPreconditioner P(offsets);
//   P.SetDiagonalBlock(0, S0inv);
//   P.SetDiagonalBlock(1, Shatinv);
//
//   // 10. Solve the normal equation system using the PCG iterative solver.
//   //     Check the weighted norm of residual for the DPG least square problem.
//   //     Wrap the primal variable in a GridFunction for visualization purposes.
//   PCG(A, P, b, x, 1, 200, 1e-12, 0.0);
//
//   {
//      Vector LSres(s_test);
//      B.Mult(x, LSres);
//      LSres -= F;
//      double res = sqrt(matSinv.InnerProduct(LSres, LSres));
//      cout << "\n|| B0*x0 + Bhat*xhat - F ||_{S^-1} = " << res << endl;
//   }
//
//   // 10b. error 
//   cout<< "\n dimension: "<<dim<<endl;
//   cout << "\nelement number of the mesh: "<< mesh->GetNE ()<<endl; 
//   cout << "\n|| u_h - u ||_{L^2} = " << x0.ComputeL2Error(u_coeff) << '\n' << endl;
//
//
//   // 11. Save the refined mesh and the solution. This output can be viewed
//   //     later using GLVis: "glvis -m refined.mesh -g sol.gf".
//   {
//      ofstream mesh_ofs("refined.mesh");
//      mesh_ofs.precision(8);
//      mesh->Print(mesh_ofs);
//      ofstream sol_ofs("sol.gf");
//      sol_ofs.precision(8);
//      x0.Save(sol_ofs);
//   }
//
//   // 12. Send the solution by socket to a GLVis server.
//   if (visualization)
//   {
//      char vishost[] = "localhost";
//      int  visport   = 19916;
//      socketstream sol_sock(vishost, visport);
//      sol_sock.precision(8);
//      sol_sock << "solution\n" << *mesh << x0 << flush;
//   }
//
//   // 13. Free the used memory.
//   delete S0inv;
//   delete Shatinv;
//   delete Shat;
//   delete Bhat;
//   delete B0;
//   delete S0;
//   delete Sinv;
//   delete test_space;
//   delete test_fec;
//   delete xhat_space;
//   delete xhat_fec;
//   delete x0_space;
//   delete x0_fec;
   delete mesh;

   return 0;
}


/* define the source term on the right hand side */
// The right hand side
//  - u'' = f
double f_exact(const Vector & x){
	if(x.Size() == 2){
		return   2*alpha_pzc*alpha_pzc*alpha_pzc*x(1)/
				(1+alpha_pzc*alpha_pzc*x(1)*x(1) )/
				(1+alpha_pzc*alpha_pzc*x(1)*x(1) );

		return M_PI * M_PI * ( sin(M_PI*x(0) ) + sin( M_PI*x(1) ) ); /* first index is 0 */
		return M_PI * M_PI *sin( M_PI*x(1) ); /* first index is 0 */
	}
	else if(x.Size() == 1){
//		return   2*alpha_pzc*alpha_pzc*alpha_pzc*x(0)/
//				(1+alpha_pzc*alpha_pzc*x(0)*x(0) )/
//				(1+alpha_pzc*alpha_pzc*x(0)*x(0) );
		return 4.*M_PI*M_PI*sin( 2.*M_PI* x(0) ) ;
	}
	else{
		return 0;
	}

}

/* exact solution */
double u_exact(const Vector & x){
	if(x.Size() == 2){
//		return  sin( M_PI * x(1) ); /* first index is 0 */
		return atan(alpha_pzc * x(1) );
		return sin(M_PI* x(0) ) + sin( M_PI * x(1) ); /* first index is 0 */
		return 10. +  sin( M_PI * x(1) ); /* first index is 0 */
//		return sin(2.*M_PI* x(0) ) + sin(2.*M_PI * x(1) ); /* first index is 0 */
	}
	else if(x.Size() == 1){
//		return atan(alpha_pzc * x(0) );
		return sin(2. * M_PI* x(0) ) ;
	}
	else{
		return 0;
	}

}

