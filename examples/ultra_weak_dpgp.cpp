// Compile with: make ultra_weak_dpg
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
double q_trace_exact(const Vector & x);
void  zero_fun(const Vector & x, Vector & f);
void  q_exact(const Vector & x, Vector & f);

double alpha_pzc = 100.;
int atan_opt = 0;

int main(int argc, char *argv[])
{
   // 1. Initialize MPI. Parse command-line options
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   const char *mesh_file = "../data/inline-quad.mesh";
   int order = 1;
   bool visualization = 1;
   bool q_visual = 0;
   int ref_levels = -1;
   int gradgrad_opt = 0;
   int solver_print_opt = 0;
   int h1_trace_opt = 0;/* use lower order h1_trace term */
   int rt_trace_opt = 0;/* use lower order rt_trace term */

   atan_opt = 0;/* which exact solution to use */

   double c_divdiv = 1.;
   double c_gradgrad = 1.;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&q_visual, "-q_vis", "--visualization for grad term", "-no-vis",
                  "--no-visualization-for-grad-term",
                  "Enable or disable GLVis visualization for grad term.");
   args.AddOption(&alpha_pzc, "-alpha", "--alpha",
                  "arctan( alpha * x) as exact solution");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly, -1 by default.");
//   args.AddOption(&divdiv_opt, "-divdiv", "--divdiv",
//                  "Whether add || ||_{H(div)} in the test norm or not, 1 by default");
   args.AddOption(&gradgrad_opt, "-gradgrad", "--gradgrad",
                  "Whether add ||grad tau || in the test norm or not, tau is a vector, 0 by default");
   args.AddOption(&solver_print_opt, "-solver_print", "--solver_print",
                  "printing option for linear solver, 0 by default");

   args.AddOption(&c_divdiv, "-c_divdiv", "--c_divdiv",
                  "constant to penalize divdiv in the test norm, 1. by default");
   args.AddOption(&c_gradgrad, "-c_gradgrad", "--c_gradgrad",
                  "constant to penalize gradgrad in the test norm, 1. by default");

   args.AddOption(&h1_trace_opt, "-h1_trace", "--h1_trace",
				  " use lower order h1 trace or not, 0 by default");
   args.AddOption(&rt_trace_opt, "-rt_trace", "--rt_trace",
				  " use lower order rt trace or not, 0 by default");

   args.AddOption(&atan_opt, "-atan", "--atan",
				  " which exact solution to use, 0 by default, sin + polynomial by default");


   args.Parse();
   if (!args.Good())
   {
	   if(myid==0){
			args.PrintUsage(cout);
	   }
	  MPI_Finalize();
      return 1;
   }
   if(myid==0){
		args.PrintOptions(cout);
   }

   // 2. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral, hexahedral, surface and volume meshes with
   //    the same code.
   Mesh *serial_mesh = new Mesh(mesh_file, 1, 1);
   int dim = serial_mesh->Dimension();

   // 3. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement. We choose 'ref_levels' to be the
   //    largest number that gives a final mesh with no more than 10,000
   //    elements.
   
   // 3a. Serial Refinement
   if (ref_levels < 0)
   {
      ref_levels = (int)floor(log(50000./serial_mesh->GetNE())/log(2.)/dim);
   }
   int serial_refine_levels = min(ref_levels, 5);
   for (int l = 0; l < serial_refine_levels; l++)
   {
      serial_mesh->UniformRefinement();
   }
   // 3b. Coonstruct Parallel mesh and do the parallel refinement
   int par_ref_levels = ref_levels - serial_refine_levels;
   ParMesh *mesh = new ParMesh(MPI_COMM_WORLD, *serial_mesh);
   delete serial_mesh;
   for(int l=0; l < par_ref_levels; l++){
		mesh->UniformRefinement();
   }
   mesh->ReorientTetMesh();

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
   unsigned int rt_trace_order = order ;
   unsigned int test_order = order + dim;

   if(h1_trace_opt){
	 h1_trace_order --;
   }
   if(rt_trace_opt){
	  rt_trace_order --;
   }

   FiniteElementCollection * u0_fec, * q0_fec, * uhat_fec, *qhat_fec, * vtest_fec, * stest_fec;

   u0_fec = new L2_FECollection(trial_order,dim);
   q0_fec = new L2_FECollection(trial_order,dim);

   uhat_fec = new H1_Trace_FECollection(h1_trace_order,dim);
   qhat_fec = new RT_Trace_FECollection(rt_trace_order,dim);

   vtest_fec = new L2_FECollection(test_order,dim); 
   stest_fec = new L2_FECollection(test_order,dim); /* in general the vector test space for \tau
													   and the scalar test space for v can be
													   polynomial space with different order */

   ParFiniteElementSpace * u0_space   = new ParFiniteElementSpace(mesh, u0_fec);
   ParFiniteElementSpace * q0_space   = new ParFiniteElementSpace(mesh, q0_fec, dim);
   ParFiniteElementSpace * uhat_space = new ParFiniteElementSpace(mesh, uhat_fec);
   ParFiniteElementSpace * qhat_space = new ParFiniteElementSpace(mesh, qhat_fec);
   
   ParFiniteElementSpace * vtest_space = new ParFiniteElementSpace(mesh, vtest_fec,dim);
   ParFiniteElementSpace * stest_space = new ParFiniteElementSpace(mesh, stest_fec);
   

   // 5. Define the block structure of the problem, by creating the offset
   //    variables. Also allocate two BlockVector objects to store the solution
   //    and rhs.
   enum {q0_var, u0_var,qhat_var,uhat_var, NVAR};

   int size_q0 = q0_space->TrueVSize();
   int size_u0 = u0_space->TrueVSize();
   int size_qhat = qhat_space->TrueVSize();
   int size_uhat = uhat_space->TrueVSize();
   int size_vtest = vtest_space->TrueVSize();
   int size_stest = stest_space->TrueVSize();

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

   HYPRE_Int global_size_q0    = q0_space->GlobalTrueVSize();
   HYPRE_Int global_size_u0    = u0_space->GlobalTrueVSize();
   HYPRE_Int global_size_qhat  = qhat_space->GlobalTrueVSize();
   HYPRE_Int global_size_uhat  = uhat_space->GlobalTrueVSize();
   HYPRE_Int global_size_vtest = vtest_space->GlobalTrueVSize();
   HYPRE_Int global_size_stest = stest_space->GlobalTrueVSize();

   if(myid == 0){
	   std::cout << "\nNumber of Unknowns rank "<<myid<<" :\n"<< endl
			     << " U0          " <<  global_size_u0   << endl
			     << " Q0          " <<  global_size_q0   << endl
				 << " Uhat        " <<  global_size_uhat << endl
				 << " Qhat        " <<  global_size_qhat << endl
				 << " Vector-test " << global_size_vtest << endl
				 << " Scalar-test " << global_size_stest << endl 
				 << endl;
	}
   // 6. Set up the linear form F(.) which corresponds to the right-hand side of
   //    the FEM linear system, which in this case is (f,phi_i) where f=1.0 and
   //    phi_i are the basis functions in the test finite element fespace.

   BlockVector x(offsets), b(offsets);
   x = 0.;


   BlockVector F(offsets_test); /* block vector for the linear form on the right hand side */
   F = 0.;

   ConstantCoefficient one(1.0);          /* coefficients */
   VectorFunctionCoefficient vec_zero(dim, zero_fun);          /* coefficients */
   VectorFunctionCoefficient q_coeff(dim, q_exact);          /* coefficients */
   FunctionCoefficient q_trace_coeff( q_trace_exact ); /* coefficients */
   FunctionCoefficient f_coeff( f_exact );/* coefficients */
   FunctionCoefficient u_coeff( u_exact );/* coefficients */


   ParGridFunction u0(u0_space);

   ParGridFunction q0(q0_space);

   ParGridFunction uhat;
   uhat.MakeTRef(uhat_space, x.GetBlock(uhat_var), 0);
   uhat.ProjectCoefficientSkeletonDG(u_coeff);

   /* rhs for -(q,\grad v) + \lgl qhat, v \rgl = (f,v) */
   ParLinearForm * f_div(new ParLinearForm);
   f_div->Update(stest_space, F.GetBlock(1) ,0);
   f_div->AddDomainIntegrator( new DomainLFIntegrator(f_coeff) );
   f_div->Assemble();

   // 6. Deal with boundary conditions
   //    Dirichlet boundary condition is imposed throught trace term  \hat{u}
   Array<int> ess_bdr(mesh->bdr_attributes.Max());
   ess_bdr = 1;

   Array<int> ess_trace_dof_list;/* store the location (index) of  boundary element  */
   uhat_space->GetEssentialVDofs(ess_bdr, ess_trace_dof_list);
//   uhat_space->GetEssentialTrueDofs(ess_bdr, ess_trace_dof_list);

   if(myid == 0){
	  cout<<endl<<endl<<"Boundary information: "<<endl;
 	  cout<<" boundary attribute size " <<mesh->bdr_attributes.Max() <<endl;
 	  cout<<" number of essential true dofs "<<ess_trace_dof_list.Size()<<endl;
   }


   // 7. Set up the mixed bilinear forms 
   //    B_mass_q:    (q,\tau)
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
   ParMixedBilinearForm *B_mass_q = new ParMixedBilinearForm(q0_space,vtest_space);
   B_mass_q->AddDomainIntegrator(new VectorMassIntegrator() );
   B_mass_q->Assemble();
   B_mass_q->Finalize();

   if(myid == 0){
		cout<<endl<< "(q,tau) assembled"<<endl;
	}
   /* operator ( u , div(v) ) */
   /* Vector DivergenceIntegrator(): (div(u), v), where u is a vector and v is a scalar*/
   /* here we want (u, div(v) ), so we take its transpose */
   ParMixedBilinearForm *B_u_dot_div = new ParMixedBilinearForm(u0_space,vtest_space);
   B_u_dot_div->AddDomainIntegrator(new TransposeIntegrator
											( new VectorDivergenceIntegrator() )
								   );
   B_u_dot_div->Assemble();
   B_u_dot_div->Finalize();
   B_u_dot_div->SpMat() *= -1.;
   if(myid == 0){
		cout<< "( u, div(tau) ) assembled"<<endl;
   }

   /* operator \lgl u, \tau\cdot n rgl */
   ParMixedBilinearForm *B_u_normal_jump = new ParMixedBilinearForm(uhat_space, vtest_space);
   B_u_normal_jump->AddTraceFaceIntegrator( new DGNormalTraceJumpIntegrator() );
   B_u_normal_jump->Assemble();
   B_u_normal_jump->EliminateEssentialBCFromTrialDofs(ess_trace_dof_list, uhat, F);
//   B_u_normal_jump->EliminateTrialDofs(ess_bdr, x.GetBlock(uhat_var), F);
   B_u_normal_jump->Finalize();

   if(myid == 0){
		cout<<endl<<"< u, tau cdot n > assembled"<<endl;
   }

   /* operator  -( q, \grad v) */
   ParMixedBilinearForm * B_q_weak_div = new ParMixedBilinearForm(q0_space, stest_space);
   B_q_weak_div->AddDomainIntegrator(new DGVectorWeakDivergenceIntegrator( ) );
   B_q_weak_div->Assemble();
   B_q_weak_div->Finalize();

   if(myid == 0){
		cout<<endl<<"-(q, grad(v)  ) assembled"<<endl;
   }

   /* operator < u_hat,v> */
   ParMixedBilinearForm *B_q_jump = new ParMixedBilinearForm(qhat_space, stest_space);
   B_q_jump->AddTraceFaceIntegrator( new TraceJumpIntegrator() );
   B_q_jump->Assemble();
   B_q_jump->Finalize();

   if(myid == 0){
		cout<<endl<<"< q, v > assembled"<<endl;
   }

   /* get  parallel matrices */
   HypreParMatrix * matB_mass_q = B_mass_q->ParallelAssemble();
   HypreParMatrix * matB_u_dot_div = B_u_dot_div->ParallelAssemble();
   HypreParMatrix * matB_u_normal_jump = B_u_normal_jump->ParallelAssemble();
   HypreParMatrix * matB_q_weak_div = B_q_weak_div->ParallelAssemble();
   HypreParMatrix * matB_q_jump = B_q_jump->ParallelAssemble();

   delete B_mass_q;
   delete B_u_dot_div;
   delete B_u_normal_jump;
   delete B_q_weak_div;
   delete B_q_jump;

   MPI_Barrier(MPI_COMM_WORLD);
   /* mass matrix corresponding to the test norm, or the so-called Gram matrix in literature */
   ParBilinearForm *Vinv = new ParBilinearForm(vtest_space);

   ConstantCoefficient const_divdiv( c_divdiv );          /* coefficients */
   ConstantCoefficient const_gradgrad( c_gradgrad );          /* coefficients */
 
   SumIntegrator *VSum = new SumIntegrator;
   VSum->AddIntegrator(new VectorMassIntegrator() );
   VSum->AddIntegrator(new DGDivDivIntegrator(const_divdiv) );
   if(gradgrad_opt==1){
		VSum->AddIntegrator(new VectorDiffusionIntegrator() );
   }



   Vinv->AddDomainIntegrator(new InverseIntegrator(VSum));
   Vinv->Assemble();
   Vinv->Finalize();

   ParBilinearForm *Sinv = new ParBilinearForm(stest_space);
   SumIntegrator *SSum = new SumIntegrator;
   SSum->AddIntegrator(new MassIntegrator(one) );
   SSum->AddIntegrator(new DiffusionIntegrator(const_gradgrad) );
   Sinv->AddDomainIntegrator(new InverseIntegrator(SSum));
   Sinv->Assemble();
   Sinv->Finalize();

   if(myid == 0){
	   cout<<"test norm: "<<endl
		   <<"|| (tau,v) ||_V^2 = || tau ||^2";
	   if(c_divdiv==1.){
	    	cout<<"+|| div(tau) ||^2";
	   }
	   else{
	    	cout<<"+"<<c_divdiv<<"|| div(tau) ||^2";
	   }
	   if(gradgrad_opt==1){
		   cout<<"+|| grad(tau) ||^2";
	   }
	   cout<<"+||v||^2";
	   if(c_gradgrad ==1.){
		   cout<<"+||grad(v)||^2";
	   }
	   else{
		   cout<<"+"<<c_gradgrad<<"||grad(v)||^2";
	   }
	   cout<<endl<<endl;
   }
   HypreParMatrix *matVinv = Vinv->ParallelAssemble();
   HypreParMatrix *matSinv = Sinv->ParallelAssemble();

   delete Vinv;
   delete Sinv;
   MPI_Barrier(MPI_COMM_WORLD);
	

	/************************************************/

   // 8. Set up the 1x2 block Least Squares DPG operator, 
   //    the normal equation operator, A = B^t InverseGram B, and
   //    the normal equation right-hand-size, b = B^t InverseGram F.
   //
   //    B = mass_q     -u_dot_div 0        u_normal_jump
   //        q_weak_div  0         q_jump   0
	   BlockOperator B(offsets_test, offsets);
	   B.SetBlock(0, q0_var  ,matB_mass_q);
	   B.SetBlock(0, u0_var  ,matB_u_dot_div);
	   B.SetBlock(0, uhat_var,matB_u_normal_jump);
	
	   B.SetBlock(1, q0_var   ,matB_q_weak_div);
	   B.SetBlock(1, qhat_var ,matB_q_jump);
	
	   BlockOperator InverseGram(offsets_test, offsets_test);
	   InverseGram.SetBlock(0,0,matVinv);
	   InverseGram.SetBlock(1,1,matSinv);
	
	   RAPOperator A(B, InverseGram, B);
	
	/**************************************************/
	
	   /* calculate right hand side b = B^T InverseGram F */
	   {
		    BlockVector IGF(offsets_test);
			InverseGram.Mult(F,IGF);
			B.MultTranspose(IGF,b);
	   }
	   // 9. Set up a block-diagonal preconditioner for the 4x4 normal equation
	   //   We use the "Jacobian" preconditionner
	   //
	   //   V0
	   //			S0 
	   //					Vhat
	   //							Shat
	   //    corresponding to the primal (x0) and interfacial (xhat) unknowns.
	   //
	   //  Actually, the exact blocks are
	   //		V0 = B_q_weak_div^T S^-1 B_q_weak_div
	   //		    +Mass_q^T  V^-1  Mass_q
	   //
	   //		S0 = u_dot_div^T  S^-1  u_dot_div
	   //
	   //       Vhat = q_jump^T S^-1 q_jump.
	   //  
	   //       Shat = u_normal_jump^T V^-1 u_normal_jump
	   //
	   //       One interesting fact:
	   //			V0 \approx Mass
	   //			S0 \approx Mass
	   //
	   // We want to approximate them.
	/***************************************************************/
	   ParBilinearForm *S0 = new ParBilinearForm(u0_space);
	   S0->AddDomainIntegrator(new MassIntegrator() );
	   S0->Assemble();
	   S0->Finalize();
	   HypreParMatrix * AmatS0 = S0->ParallelAssemble();
	
		// the exact form of the diagonal block //
	   HypreParMatrix * matV00 = RAP(matB_q_weak_div, matSinv, matB_q_weak_div);
	   HypreParMatrix * matV0  = RAP(matB_mass_q, matVinv, matB_mass_q);
	   matV0->Add(1.,*matV00); delete matV00;
	
	   HypreParMatrix * Vhat   = RAP(matB_q_jump, matSinv, matB_q_jump);
	   HypreParMatrix * Shat   = RAP(matB_u_normal_jump, matVinv, matB_u_normal_jump);


	   HypreBoomerAMG *V0inv = new HypreBoomerAMG( *matV0 );
	   V0inv->SetPrintLevel(0);
	
	   HypreBoomerAMG *S0inv = new HypreBoomerAMG( *AmatS0 );
	   S0inv->SetPrintLevel(0);

   	   HypreSolver *Vhatinv;
   	   if (dim == 2) { Vhatinv = new HypreAMS(*Vhat, qhat_space); }
   	   else          { Vhatinv = new HypreADS(*Vhat, qhat_space); }

//	   HypreBoomerAMG *Shatinv = new HypreBoomerAMG( *Shat );
//	   Shatinv->SetPrintLevel(0);

	   const double prec_rtol = 1e-3;
	   const int prec_maxit = 200;
	   HyprePCG * Shatinv = new HyprePCG( *Shat );
	   Shatinv->SetTol(prec_rtol);
	   Shatinv->SetMaxIter(prec_maxit);
	

	   BlockDiagonalPreconditioner P(offsets);
	   P.SetDiagonalBlock(0, V0inv);
	   P.SetDiagonalBlock(1, S0inv);
	   P.SetDiagonalBlock(2, Vhatinv);
	   P.SetDiagonalBlock(3, Shatinv);

//	// 10. Solve the normal equation system using the PCG iterative solver.
//	//     Check the weighted norm of residual for the DPG least square problem.
//	//     Wrap the primal variable in a GridFunction for visualization purposes.
	   CGSolver pcg(MPI_COMM_WORLD);
	   pcg.SetOperator(A);
	   pcg.SetPreconditioner(P);
	   pcg.SetRelTol(1e-8);
	   pcg.SetMaxIter(150);
	   pcg.SetPrintLevel(solver_print_opt);
	   pcg.Mult(b,x);
	
	   {
	      BlockVector LSres( offsets_test ), tmp( offsets_test );
	      B.Mult(x, LSres);
	      LSres -= F;
	      InverseGram.Mult(LSres, tmp);
		  double res = sqrt(InnerProduct(LSres,tmp) );
		  if(myid == 0){
			cout << "\n|| Bx - F ||_{S^-1} = " << res << endl;
		  }
	   }

	// 10b. error 
	   u0.Distribute( x.GetBlock(u0_var) );
	   q0.Distribute( x.GetBlock(q0_var) );

//	   u0.MakeRef( u0_space, x.GetBlock(u0_var) );
//	   q0.MakeRef( q0_space, x.GetBlock(q0_var) );

	   double u_error = u0.ComputeL2Error(u_coeff);
	   double q_error = q0.ComputeL2Error(q_coeff);

	   int global_ne = mesh->GetGlobalNE();
	   if(myid == 0){
			cout << "\nelement number of the mesh: "<< global_ne<<endl; 
			cout<< "\n dimension: "<<dim<<endl;
			printf("\n|| u_h - u ||_{L^2} = %e \n", u_error );
			printf("\n|| q_h - q ||_{L^2} = %e \n", q_error );
			cout<<endl;
	   }
	
	   // 11. Save the refined mesh and the solution. This output can be viewed
	   //     later using GLVis: "glvis -m refined.mesh -g sol.gf".
	   {
		  ostringstream mesh_name, sol_name;
      	  mesh_name << "mesh." << setfill('0') << setw(6) << myid;
      	  sol_name << "sol." << setfill('0') << setw(6) << myid;

	      ofstream mesh_ofs(mesh_name.str().c_str() );
	      mesh_ofs.precision(8);
	      mesh->Print(mesh_ofs);
	
	      ofstream sol_ofs(sol_name.str().c_str() );
	      sol_ofs.precision(8);
	      u0.Save(sol_ofs);
	
		  if(q_visual){
			 ostringstream q_name;
			 mesh_name<< "q."<<setfill('0')<<setw(6)<<myid;
	         ofstream q_variable_ofs(q_name.str().c_str() );
	         q_variable_ofs.precision(8);
	         q0.Save(q_variable_ofs);
		  }
	   }
	
	   // 12. Send the solution by socket to a GLVis server.
	   if (visualization)
	   {
	      char vishost[] = "localhost";
	      int  visport   = 19916;
	      socketstream sol_sock(vishost, visport);
		  sol_sock << "parallel " << num_procs << " " << myid << "\n";
      	  sol_sock.precision(8);
      	  sol_sock << "solution\n" << *mesh <<  u0 << flush;
	
		  if(q_visual){
	         socketstream q_sock(vishost, visport);
			 q_sock << "parallel " << num_procs << " " << myid << "\n";
      	  	 q_sock.precision(8);
      	  	 q_sock << "solution\n" << *mesh <<  q0 << flush;
		  }
	   }

//   // 13. Free the used memory.
	/* bilinear form */
//   delete Vinv;
//   delete Sinv; 
//
//   delete matV00;
//   delete matV0;
////   delete matS0;
//   delete Vhat;
//   delete Shat;
//
//   /* preconditionner */
   delete V0inv;
   delete S0inv;
   delete Vhatinv;
   delete Shatinv;
   /* finite element collection */
   delete u0_fec;
   delete q0_fec;
   delete uhat_fec;
   delete qhat_fec;
   delete vtest_fec;
   delete stest_fec;

   /* finite element spaces */
   delete u0_space;
   delete q0_space;
   delete uhat_space;
   delete qhat_space;
   delete vtest_space;
   delete stest_space;

   delete mesh;

   MPI_Finalize();

   return 0;
} /* end of main */


/* define the source term on the right hand side */
// The right hand side
//  - u'' = f
double f_exact(const Vector & x){
	if(x.Size() == 2){
		if(atan_opt == 0){
			return 8.*M_PI*M_PI* sin(2.*M_PI*x(0) ) * sin(2.*M_PI*x(1) );/* HDG */
		}
		else if(atan_opt == 1){
			double yy = x(1) - 0.5;
			return   2*alpha_pzc*alpha_pzc*alpha_pzc*yy/
					(1+alpha_pzc*alpha_pzc*yy*yy )/
					(1+alpha_pzc*alpha_pzc*yy*yy );
		}
		else if(atan_opt == 2){
			double yy = x(1) - 0.5;
			double xx = x(0) - 0.5;
			return   2*alpha_pzc*alpha_pzc*alpha_pzc*yy/
					(1+alpha_pzc*alpha_pzc*yy*yy )/
					(1+alpha_pzc*alpha_pzc*yy*yy )
			        +2*alpha_pzc*alpha_pzc*alpha_pzc*xx/
					(1+alpha_pzc*alpha_pzc*xx*xx )/
					(1+alpha_pzc*alpha_pzc*xx*xx );
		}
		else{
			double yy = x(1) - 0.5;
			double yy2 = x(1) - 0.77;
			return   2*alpha_pzc*alpha_pzc*alpha_pzc*yy/
					(1+alpha_pzc*alpha_pzc*yy*yy )/
					(1+alpha_pzc*alpha_pzc*yy*yy )
			        +2*alpha_pzc*alpha_pzc*alpha_pzc*yy2/
					(1+alpha_pzc*alpha_pzc*yy2*yy2 )/
					(1+alpha_pzc*alpha_pzc*yy2*yy2 );
		}
	}
	else if(x.Size() == 3){
		return 12.*M_PI*M_PI* sin(2.*M_PI*x(0) ) * sin(2.*M_PI*x(1) ) * sin(2.*M_PI*x(2) );
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
		if(atan_opt == 0){
//			return  sin(2.*M_PI*x(0) ) * sin(2.* M_PI * x(1) ); /* HDG */
			return  1. + x(0) + sin(2.*M_PI*x(0) ) * sin(2.* M_PI * x(1) ); /* HDG */
		}
		else if(atan_opt == 1){
			return atan(alpha_pzc * (x(1) - 0.5)  );
		}
		else if(atan_opt == 2){
			return atan(alpha_pzc * (x(1) - 0.5)  )
				  +atan(alpha_pzc * (x(0) - 0.5)  );
		}
		else{
			return atan(alpha_pzc * (x(1) - 0.5)  )
				  +atan(alpha_pzc * (x(1) - 0.77)  );
		}
	}
	else if(x.Size() ==3 ){
		return x(0) + sin(2.*M_PI*x(0) ) * sin(2.*M_PI*x(1) ) * sin(2.*M_PI*x(2) ); 
	}
	else if(x.Size() == 1){
		return atan(alpha_pzc * (x(0)-0.5)  );
		return sin(2. * M_PI* x(0) ) ;
	}
	else{
		return 0;
	}

}

/* exact q = -grad u */
void q_exact(const Vector & x,Vector & f){
	if(x.Size() == 2){
		if(atan_opt == 0){
			f(0) = -1. - 2.*M_PI*cos(2.*M_PI*x(0) ) * sin(2.*M_PI*x(1) );/* HDG */
			f(1) =     - 2.*M_PI*sin(2.*M_PI*x(0) ) * cos(2.*M_PI*x(1) );/* HDG */
		}
		else if(atan_opt == 1){
			f(0) = 0.;
			f(1) = -alpha_pzc/( 1. + alpha_pzc*alpha_pzc * (x(1) - 0.5) * (x(1) - 0.5)  );
		}
		else if(atan_opt == 2){
			f(0) = -alpha_pzc/( 1. + alpha_pzc*alpha_pzc * (x(0) - 0.5) * (x(0) - 0.5)  );
			f(1) = -alpha_pzc/( 1. + alpha_pzc*alpha_pzc * (x(1) - 0.5) * (x(1) - 0.5)  );
		}
		else{
			f(0) = 0.; 
			f(1) = -alpha_pzc/( 1. + alpha_pzc*alpha_pzc * (x(1) - 0.5)  * (x(1) - 0.5)  )
			       -alpha_pzc/( 1. + alpha_pzc*alpha_pzc * (x(1) - 0.75) * (x(1) - 0.77)  );
		}
	}
	else if(x.Size() == 3){
		f(0) = -1. - 2.*M_PI* cos(2.*M_PI*x(0) ) * sin(2.*M_PI*x(1) ) * sin(2.*M_PI*x(2) );
		f(1) =     - 2.*M_PI* sin(2.*M_PI*x(0) ) * cos(2.*M_PI*x(1) ) * sin(2.*M_PI*x(2) );
		f(2) =     - 2.*M_PI* sin(2.*M_PI*x(0) ) * sin(2.*M_PI*x(1) ) * cos(2.*M_PI*x(2) );
	}
	else if(x.Size() == 1){
		f(0) = -2.*M_PI * cos(2.*M_PI* x(0) );
	}
	else{
		f  = 0.;
	}
}

/* trace of q on square mesh */
double q_trace_exact(const Vector &x){
	double res = 0.;
	if( (x(0) == 0)||(x(0)==1) ){
//		res =  M_PI*sin(M_PI*x(0) ) * cos(M_PI*x(1) ) ;
//		res = 2.* x(1);
		res = +1.;

		res = 0.;
	}
	else{
//		res = -M_PI*cos(M_PI*x(0) ) * sin(M_PI*x(1) ) ;
//		res = -2.*x(0);
		res = -1.;

		res = 0.;
	}
	return res;
}

/* vector 0 */
void zero_fun(const Vector & x, Vector & f){
	f = 0.;
}

