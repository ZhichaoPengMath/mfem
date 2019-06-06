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
//					  q + \grad u = 0
//					  div(q) =  f
//				 Variational form:
//						 ( q, \tau ) - (u, div(\tau) ) + \lgl \hat{u}, \tau\cdot n \rgl = 0
//						-( q, \grad v) + \lgl \hat{q}, v \rgl  = f
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
void  zero_fun(const Vector & x, Vector & f);

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

   FiniteElementCollection * u0_fec, * q0_fec, * uhat_fec, *qhat_fec, * vtest_fec, * stest_fec;

   u0_fec = new L2_FECollection(trial_order,dim);
   q0_fec = new L2_FECollection(trial_order,dim);

   uhat_fec = new H1_Trace_FECollection(h1_trace_order,dim);
   qhat_fec = new RT_Trace_FECollection(rt_trace_order,dim);

   vtest_fec = new L2_FECollection(test_order,dim); 
   stest_fec = new L2_FECollection(test_order,dim); /* in general the vector test space for \tau
													   and the scalar test space for v can be
													   polynomial space with different order */

   FiniteElementSpace * u0_space = new FiniteElementSpace(mesh, u0_fec);
   FiniteElementSpace * q0_space = new FiniteElementSpace(mesh, q0_fec, dim);
   FiniteElementSpace * uhat_space = new FiniteElementSpace(mesh, uhat_fec);
   FiniteElementSpace * qhat_space = new FiniteElementSpace(mesh, qhat_fec);
   
   FiniteElementSpace * vtest_space = new FiniteElementSpace(mesh, vtest_fec,dim);
   FiniteElementSpace * stest_space = new FiniteElementSpace(mesh, stest_fec);
   

   // 5. Define the block structure of the problem, by creating the offset
   //    variables. Also allocate two BlockVector objects to store the solution
   //    and rhs.
   enum {q0_var, u0_var,qhat_var,uhat_var, NVAR};

   int size_q0 = q0_space->GetVSize();
   int size_u0 = u0_space->GetVSize();
   int size_qhat = qhat_space->GetVSize();
   int size_uhat = uhat_space->GetVSize();
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
   VectorFunctionCoefficient vec_zero(dim, zero_fun);          /* coefficients */
   FunctionCoefficient f_coeff( f_exact );/* coefficients */
   FunctionCoefficient u_coeff( u_exact );/* coefficients */

   /* rhs for (q,\tau) - (u,\div(\tau) ) + \lgl hhat,\tau\cdot n \rgl = 0 */
   LinearForm * f_grad(new LinearForm);
   f_grad->Update(vtest_space, F.GetBlock(0) ,0);
   f_grad->AddDomainIntegrator(new VectorDomainLFIntegrator( vec_zero ) );
   f_grad->Assemble();

   /* rhs for -(q,\grad v) + \lgl qhat, v \rgl = (f,v) */
   LinearForm * f_div(new LinearForm);
   f_div->Update(stest_space, F.GetBlock(1) ,0);
   f_div->AddDomainIntegrator( new DomainLFIntegrator(f_coeff) );
   f_div->Assemble();

   // 6. Deal with boundary conditions
   //    Dirichlet boundary condition is imposed throught trace term  \hat{u}
//   Array<int> ess_bdr(mesh->bdr_attributes.Max());
//   ess_bdr = 1;
//
//   Array<int> ess_trace_dof_list;/* store the location (index) of  boundary element  */
//   uhat_space->GetEssentialTrueDofs(ess_bdr, ess_trace_dof_list);
//
//   cout<<endl<<endl<<"Boundary information: "<<endl;
//   cout<<" boundary attribute size " <<mesh->bdr_attributes.Max() <<endl;
//   cout<<" number of essential true dofs "<<ess_trace_dof_list.Size()<<endl;
//
//   GridFunction uhat;
//   uhat.MakeRef(uhat_space, x.GetBlock(uhat_var), 0);
//   uhat.ProjectBdrCoefficient(u_coeff,ess_trace_dof_list);

   // 7. Set up the mixed bilinear forms 
   //    B_mass_q: (q,\tau)
   //    B_u_dot_div: (u, div(\tau) ) 
   //    B_u_normal_jump:  \lgl \hat{u} , \tau\cdot n\rgl
   //
   //	 B_q_weak_div: -(q, \grad v)
   //	 B_q_jump: \lgl \hat{q}, v \rgl
   //
   //    the inverse energy matrix on the discontinuous test space,
   //    Vinv, Sinv
   //    and the energy matrix on the continuous trial space, S0.
   //    V corresponding to ||\tau||^2 + || div(\tau) ||^2
   //    S corresponding to ||v||^2 + || \grad(v) ||^2
   
   /* operator (q,v) */
   MixedBilinearForm *B_mass_q = new MixedBilinearForm(q0_space,vtest_space);
   B_mass_q->AddDomainIntegrator(new VectorMassIntegrator() );
   B_mass_q->Assemble();
   B_mass_q->Finalize();

   cout<<endl<< "(q,tau) assembled"<<endl;

   /* operator ( u , div(v) ) */
   /* Vector DivergenceIntegrator(): (div(u), v), where u is a vector and v is a scalar*/
   /* here we want (u, div(v) ), so we take its transpose */
   MixedBilinearForm *B_u_dot_div = new MixedBilinearForm(u0_space,vtest_space);
   B_u_dot_div->AddDomainIntegrator(new TransposeIntegrator
											( new VectorDivergenceIntegrator() )
								   );
   B_u_dot_div->Assemble();
   B_u_dot_div->Finalize();
   cout<< "( u, div(tau) ) assembled"<<endl;

   /* operator \lgl u, \tau\cdot n rgl */
   MixedBilinearForm *B_u_normal_jump = new MixedBilinearForm(uhat_space, vtest_space);
   B_u_normal_jump->AddTraceFaceIntegrator( new DGNormalTraceJumpIntegrator() );
   B_u_normal_jump->Assemble();
//   B_u_normal_jump->EliminateTrialDofs(ess_bdr, x.GetBlock(uhat_var), F);
   B_u_normal_jump->Finalize();

   cout<<endl<<"< u, tau cdot n > assembled"<<endl;

   /* operator  -( q, \grad v) */
   MixedBilinearForm * B_q_weak_div = new MixedBilinearForm(q0_space, stest_space);
   B_q_weak_div->AddDomainIntegrator(new DGVectorWeakDivergenceIntegrator( ) );
   B_q_weak_div->Assemble();
   B_q_weak_div->Finalize();

   cout<<endl<<"-(q, grad(v)  ) assembled"<<endl;

   /* operator < u_hat,v> */
   MixedBilinearForm *B_q_jump = new MixedBilinearForm(qhat_space, stest_space);
   B_q_jump->AddTraceFaceIntegrator( new TraceJumpIntegrator() );
   B_q_jump->Assemble();
   B_q_jump->Finalize();

   cout<<endl<<"< q, v > assembled"<<endl;

   /* size of matrices */
   SparseMatrix &matB_mass_q = B_mass_q->SpMat();
   SparseMatrix &matB_u_dot_div = B_u_dot_div->SpMat();
   SparseMatrix &matB_u_normal_jump = B_u_normal_jump->SpMat();
   SparseMatrix &matB_q_weak_div = B_q_weak_div->SpMat();
   SparseMatrix &matB_q_jump = B_q_jump->SpMat();

   /* mass matrix corresponding to the test norm, or the so-called Gram matrix in literature */
   BilinearForm *Vinv = new BilinearForm(vtest_space);
   SumIntegrator *VSum = new SumIntegrator;
   VSum->AddIntegrator(new VectorMassIntegrator() );
   VSum->AddIntegrator(new DGDivDivIntegrator() );
   Vinv->AddDomainIntegrator(new InverseIntegrator(VSum));
   Vinv->Assemble();
   Vinv->Finalize();

   BilinearForm *Sinv = new BilinearForm(stest_space);
   SumIntegrator *SSum = new SumIntegrator;
   SSum->AddIntegrator(new MassIntegrator(one) );
   SSum->AddIntegrator(new DiffusionIntegrator(one));
   Sinv->AddDomainIntegrator(new InverseIntegrator(SSum));
   Sinv->Assemble();
   Sinv->Finalize();

   SparseMatrix &matVinv = Vinv->SpMat();
   SparseMatrix &matSinv = Sinv->SpMat();
	
   cout<<endl<<endl<<"matrix dimensions: "<<endl
	   <<" mass_q:        "<<matB_mass_q.Width()   <<" X "<<matB_mass_q.Height()<<endl
	   <<" u_dot_div:     "<<matB_u_dot_div.Width()<<" X "<<matB_u_dot_div.Height()<<endl
	   <<" u_normal_jump: "<<matB_u_normal_jump.Width()<<" X "<<matB_u_normal_jump.Height()<<endl
	   <<" q_weak_div:    "<<matB_q_weak_div.Width()<<" X "<<matB_q_weak_div.Height()<<endl
	   <<" q_jump:        "<<matB_q_jump.Width()<<" X "<<matB_q_jump.Height()<<endl;
    cout<<endl<<"matrix in test space: "<<endl
	   <<" V_inv:         "<<matVinv.Width()<<" X "<< matVinv.Height()<<endl
	   <<" S_inv:         "<<matSinv.Width()<<" X "<< matSinv.Height()<<endl;

   // 8. Set up the 1x2 block Least Squares DPG operator, 
   //    the normal equation operator, A = B^t InverseGram B, and
   //    the normal equation right-hand-size, b = B^t InverseGram F.
   //
   //    B = mass_q     -u_dot_div 0        u_normal_jump
   //        q_weak_div  0         q_jump   0
   BlockOperator B(offsets_test, offsets);
   B.SetBlock(0,0,&matB_mass_q);
   B.SetBlock(0,1,&matB_u_dot_div);
   B.SetBlock(0,3,&matB_u_normal_jump);

   B.SetBlock(1,0,&matB_q_weak_div);
   B.SetBlock(1,2,&matB_q_jump);

   BlockOperator InverseGram(offsets_test, offsets_test);
   InverseGram.SetBlock(0,0,Vinv);
   InverseGram.SetBlock(1,1,Sinv);

   RAPOperator A(B, InverseGram, B);

   /* calculate right hand side b = B^T InverseGram F */
   {
		Vector IGF(size_vtest + size_stest);
		InverseGram.Mult(F,IGF);
		B.MultTranspose(IGF,b);

   }
   // 9. Set up a block-diagonal preconditioner for the 4x4 normal equation
   //   V0
   //			S0 
   //					Vhat
   //							Shat
   //    corresponding to the primal (x0) and interfacial (xhat) unknowns.
   BilinearForm *V0 = new BilinearForm(q0_space);
   V0->AddDomainIntegrator(new VectorDiffusionIntegrator(one));
   V0->Assemble();
   V0->Finalize();

   BilinearForm *S0 = new BilinearForm(u0_space);
   S0->AddDomainIntegrator(new DiffusionIntegrator(one));
   S0->Assemble();
   S0->Finalize();
   
   SparseMatrix & matV0 = V0->SpMat();
   SparseMatrix & matS0 = S0->SpMat();
   SparseMatrix * Vhat  = RAP(matB_q_jump, matSinv, matB_q_jump);
   SparseMatrix * Shat  = RAP(matB_u_normal_jump, matVinv, matB_u_normal_jump);
#ifndef MFEM_USE_SUITESPARSE
   const double prec_rtol = 1e-3;
   const int prec_maxit = 200;

   CGSolver *V0inv = new CGSolver;
   V0inv->SetOperator(matV0);
   V0inv->SetPrintLevel(-1);
   V0inv->SetRelTol(prec_rtol);
   V0inv->SetMaxIter(prec_maxit);

   CGSolver *S0inv = new CGSolver;
   S0inv->SetOperator(matS0);
   S0inv->SetPrintLevel(-1);
   S0inv->SetRelTol(prec_rtol);
   S0inv->SetMaxIter(prec_maxit);

   CGSolver *Vhatinv = new CGSolver;
   Vhatinv->SetOperator(*Vhat);
   Vhatinv->SetPrintLevel(-1);
   Vhatinv->SetRelTol(prec_rtol);
   Vhatinv->SetMaxIter(prec_maxit);

   CGSolver *Shatinv = new CGSolver;
   Shatinv->SetOperator(*Shat);
   Shatinv->SetPrintLevel(-1);
   Shatinv->SetRelTol(prec_rtol);
   Shatinv->SetMaxIter(prec_maxit);

   // Disable 'iterative_mode' when using CGSolver (or any IterativeSolver) as
   // a preconditioner:
   V0inv->iterative_mode = false;
   S0inv->iterative_mode = false;
   Vhatinv->iterative_mode = false;
   Shatinv->iterative_mode = false;
#else
   Operator *V0inv = new UMFPackSolver(matV0);
   Operator *Vhatinv = new UMFPackSolver(*Vhat);

   Operator *S0inv = new UMFPackSolver(matS0);
   Operator *Shatinv = new UMFPackSolver(*Shat);
#endif
   BlockDiagonalPreconditioner P(offsets);
   P.SetDiagonalBlock(0, V0inv);
   P.SetDiagonalBlock(1, S0inv);
   P.SetDiagonalBlock(2, Vhatinv);
   P.SetDiagonalBlock(3, Shatinv);

//   // 10. Solve the normal equation system using the PCG iterative solver.
//   //     Check the weighted norm of residual for the DPG least square problem.
//   //     Wrap the primal variable in a GridFunction for visualization purposes.
     GMRES(A, P, b, x, 1, 1000, 1000, 1e-12, 0.0);
//   PCG(A, P, b, x, 1, 200, 1e-12, 0.0);

   GridFunction u0;
   u0.MakeRef(u0_space, x.GetBlock(u0_var), 0);
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
   cout<< "\n dimension: "<<dim<<endl;
   cout << "\nelement number of the mesh: "<< mesh->GetNE ()<<endl; 
   cout << "\n|| u_h - u ||_{L^2} = " << u0.ComputeL2Error(u_coeff) << '\n' << endl;
//
//
   // 11. Save the refined mesh and the solution. This output can be viewed
   //     later using GLVis: "glvis -m refined.mesh -g sol.gf".
   {
      ofstream mesh_ofs("refined.mesh");
      mesh_ofs.precision(8);
      mesh->Print(mesh_ofs);
      ofstream sol_ofs("sol.gf");
      sol_ofs.precision(8);
      u0.Save(sol_ofs);
   }

   // 12. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "solution\n" << *mesh << u0 << flush;
   }

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
		return 2*M_PI*M_PI*sin(M_PI*x(0) ) * sin(M_PI*x(1) );
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
		return  sin(M_PI*x(0) ) * sin( M_PI * x(1) ); /* first index is 0 */
	}
	else if(x.Size() == 1){
//		return atan(alpha_pzc * x(0) );
		return sin(2. * M_PI* x(0) ) ;
	}
	else{
		return 0;
	}

}

/* vector 0 */
void zero_fun(const Vector & x, Vector & f){
	f = 0.;
}

