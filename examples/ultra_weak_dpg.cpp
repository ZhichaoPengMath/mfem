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
double q_trace_exact(const Vector & x);
void  zero_fun(const Vector & x, Vector & f);
void  q_exact(const Vector & x, Vector & f);

double alpha_pzc = 100.;

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../data/inline-quad-pzc.mesh";
   int order = 1;
   bool visualization = 1;
   bool q_visual = 0;
   int ref_levels = -1;
//   int divdiv_opt = 1;
   int gradgrad_opt = 0;
   int solver_print_opt = 0;
   int h1_trace_opt = 0;/* use lower order h1_trace term */
   int rt_trace_opt = 0;/* use lower order rt_trace term */

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
   VectorFunctionCoefficient q_coeff(dim, q_exact);          /* coefficients */
   FunctionCoefficient q_trace_coeff( q_trace_exact ); /* coefficients */
   FunctionCoefficient f_coeff( f_exact );/* coefficients */
   FunctionCoefficient u_coeff( u_exact );/* coefficients */


   GridFunction u0;
   u0.MakeRef(u0_space, x.GetBlock(u0_var), 0);
//   u0.ProjectCoefficient(u_coeff);
//
   GridFunction q0;
   q0.MakeRef(q0_space, x.GetBlock(q0_var), 0);
//   q0.ProjectCoefficient(q_coeff);

   GridFunction uhat;
   uhat.MakeRef(uhat_space, x.GetBlock(uhat_var), 0);
   uhat.ProjectCoefficientSkeletonDG(u_coeff);

   GridFunction qhat;
   qhat.MakeRef(qhat_space, x.GetBlock(qhat_var), 0);
//   qhat.ProjectCoefficientSkeletonDG(q_trace_coeff);


   /* rhs for (q,\tau) - (u,\div(\tau) ) + \lgl hhat,\tau\cdot n \rgl = 0 */
//   LinearForm * f_grad(new LinearForm);
//   f_grad->Update(vtest_space, F.GetBlock(0) ,0);
//   f_grad->AddDomainIntegrator(new VectorDomainLFIntegrator( vec_zero ) );
//   f_grad->Assemble();

   /* rhs for -(q,\grad v) + \lgl qhat, v \rgl = (f,v) */
   LinearForm * f_div(new LinearForm);
   f_div->Update(stest_space, F.GetBlock(1) ,0);
   f_div->AddDomainIntegrator( new DomainLFIntegrator(f_coeff) );
   f_div->Assemble();

   // 6. Deal with boundary conditions
   //    Dirichlet boundary condition is imposed throught trace term  \hat{u}
   Array<int> ess_bdr(mesh->bdr_attributes.Max());
   ess_bdr = 1;

   Array<int> ess_trace_dof_list;/* store the location (index) of  boundary element  */
   uhat_space->GetEssentialTrueDofs(ess_bdr, ess_trace_dof_list);

   cout<<endl<<endl<<"Boundary information: "<<endl;
   cout<<" boundary attribute size " <<mesh->bdr_attributes.Max() <<endl;
   cout<<" number of essential true dofs "<<ess_trace_dof_list.Size()<<endl;


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
   B_u_normal_jump->EliminateTrialDofs(ess_bdr, x.GetBlock(uhat_var), F);
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

   /* get  matrices */
   SparseMatrix &matB_mass_q = B_mass_q->SpMat();
   SparseMatrix &matB_u_dot_div = B_u_dot_div->SpMat();
   matB_u_dot_div *= -1.;
   SparseMatrix &matB_u_normal_jump = B_u_normal_jump->SpMat();
   SparseMatrix &matB_q_weak_div = B_q_weak_div->SpMat();
   SparseMatrix &matB_q_jump = B_q_jump->SpMat();

/*****************************************************************/
	/* debug */
//	Vector tmp1(vtest_space->GetNDofs()*dim );
//	Vector tmp2(vtest_space->GetNDofs()*dim );
//	Vector tmp3(vtest_space->GetNDofs()*dim );
//	Vector  res(vtest_space->GetNDofs()*dim );
//	Vector tmp12(vtest_space->GetNDofs()*dim );
//
//	B_mass_q->Mult(q0,tmp1);
//	B_u_dot_div->Mult(u0,tmp2);
//	B_u_normal_jump->Mult(uhat,tmp3);
//	add(tmp1,tmp2,tmp12);
//	add(tmp3,tmp12,res);
//
//	for(int i=0; i<tmp3.Size(); i++){
//		cout<<i<<": "<<endl
//			<<"tmp1: "<<tmp1(i)<<endl
//			<<"tmp2: "<<tmp2(i)<<endl
//			<<"tmp3: "<<tmp3(i)<<endl
//			<<"tmp1+tmp2="<< tmp12(i) <<endl
//			<<"tmp3 + tmp1 +tmp2 ="<<tmp3(i)+tmp12(i)<<endl
//			<<endl;
//	}
//
//	cout<<endl<<endl<<"Debug, norm of 0, equation 1: "<< res.Norml2() <<endl;
//
//	Vector ppg1(stest_space->GetNDofs() );
//	Vector res_ppg(stest_space->GetNDofs() );
//
//	MixedBilinearForm *blf = new MixedBilinearForm(q0_space,stest_space);
//	blf->AddDomainIntegrator(new VectorDivergenceIntegrator() );
//	blf->Assemble();
//	blf->Finalize();
//
//	blf->Mult(q0,ppg1);
//	subtract(ppg1,F.GetBlock(1) ,res_ppg);
//
//	cout<<endl<<endl<<"Debug, norm of 0, equation 2: "<< res_ppg.Norml2() <<endl;

/*****************************************************************/


   /* mass matrix corresponding to the test norm, or the so-called Gram matrix in literature */
   BilinearForm *Vinv = new BilinearForm(vtest_space);

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

   BilinearForm *Sinv = new BilinearForm(stest_space);
   SumIntegrator *SSum = new SumIntegrator;
   SSum->AddIntegrator(new MassIntegrator(one) );
   SSum->AddIntegrator(new DiffusionIntegrator(const_gradgrad) );
   Sinv->AddDomainIntegrator(new InverseIntegrator(SSum));
   Sinv->Assemble();
   Sinv->Finalize();

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

   SparseMatrix &matVinv = Vinv->SpMat();
   SparseMatrix &matSinv = Sinv->SpMat();
	
   cout<<endl<<endl<<"matrix dimensions: "<<endl
	   <<" mass_q:        "<<matB_mass_q.Height()   <<" X "<<matB_mass_q.Width()<<endl
	   <<" u_dot_div:     "<<matB_u_dot_div.Height()<<" X "<<matB_u_dot_div.Width()<<endl
	   <<" u_normal_jump: "<<matB_u_normal_jump.Height()<<" X "<<matB_u_normal_jump.Width()<<endl
	   <<" q_weak_div:    "<<matB_q_weak_div.Height()<<" X "<<matB_q_weak_div.Width()<<endl
	   <<" q_jump:        "<<matB_q_jump.Height()<<" X "<<matB_q_jump.Width()<<endl;
    cout<<endl<<"matrix in test space: "<<endl
	   <<" V_inv:         "<<matVinv.Height()<<" X "<< matVinv.Width()<<endl
	   <<" S_inv:         "<<matSinv.Height()<<" X "<< matSinv.Width()<<endl;


	/************************************************/

   // 8. Set up the 1x2 block Least Squares DPG operator, 
   //    the normal equation operator, A = B^t InverseGram B, and
   //    the normal equation right-hand-size, b = B^t InverseGram F.
   //
   //    B = mass_q     -u_dot_div 0        u_normal_jump
   //        q_weak_div  0         q_jump   0
   BlockOperator B(offsets_test, offsets);
   B.SetBlock(0, q0_var  ,&matB_mass_q);
   B.SetBlock(0, u0_var  ,&matB_u_dot_div);
   B.SetBlock(0, uhat_var,&matB_u_normal_jump);

   B.SetBlock(1, q0_var   ,&matB_q_weak_div);
   B.SetBlock(1, qhat_var ,&matB_q_jump);

   BlockOperator InverseGram(offsets_test, offsets_test);
   InverseGram.SetBlock(0,0,Vinv);
   InverseGram.SetBlock(1,1,Sinv);

   RAPOperator A(B, InverseGram, B);

/**************************************************/
   /* non-zero pattern is not correct */
   /* debug */
//   Vector y(B.Height() );
//   Vector vy(InverseGram.Height()  );
//   Vector ry(x.Size() );
//
//   Vector z(x.Size() );
//
//
//   B.Mult(x,y);
//   InverseGram.Mult(y,vy);
//
//   for(int i=0;i<vtest_space->GetVSize();i++){
//	 cout<<" y("<<i<<")= "<< y(i)<<endl
//		 <<"vy("<<i<<")= "<<vy(i)<<endl
//		 <<endl;
//   }
//   cout<<endl;
//
//   for(int i=vtest_space->GetVSize();i<vy.Size();i++){
//	 cout<<" y("<<i<<")= "<< y(i)<<endl
//		 <<"vy("<<i<<")= "<<vy(i)<<endl
//		 <<endl;
//   }
//   cout<<endl;
//
//   B.MultTranspose(vy,ry);
//   A.Mult(x,z);
//
////   for(int i=0;i<q0.Size();i++){
//   for(int i=0;i<ry.Size();i++){
//	 cout<<" z("<<i<<")= "<< z(i) <<endl
//		 <<"ry("<<i<<")= "<<ry(i) <<endl
//		 <<endl;
//   }
//   cout<<endl;
   /* debug */
/**************************************************/

   /* calculate right hand side b = B^T InverseGram F */
   {
		Vector IGF(size_vtest + size_stest);
		InverseGram.Mult(F,IGF);
		B.MultTranspose(IGF,b);
/**************************************************/
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
   //       Vhat = u_normal_jump^T V^-1 u_normal_jump
   //
   //       Vhat = q_jump^T S^-1 q_jump.
   //
   //       One interesting fact:
   //			V0 \approx Mass
   //			S0 \approx Mass
   //
   // We want to approximate them.
/***************************************************************/
   /* the preconditioner here is not working */
   BilinearForm *V0 = new BilinearForm(q0_space);
   SumIntegrator * sumV0 = new SumIntegrator;
   sumV0->AddIntegrator(new VectorMassIntegrator() );
   V0->AddDomainIntegrator(sumV0);
   V0->Assemble();
   V0->Finalize();

   SparseMatrix & AmatV0 = V0->SpMat();

   BilinearForm *S0 = new BilinearForm(u0_space);
   S0->AddDomainIntegrator(new MassIntegrator() );
   S0->Assemble();
   S0->Finalize();

   SparseMatrix & AmatS0 = S0->SpMat();

/**************************************************************/
   

	// the exact form of the diagonal block //
   SparseMatrix * matV00 = RAP(matB_q_weak_div, matSinv, matB_q_weak_div);
   SparseMatrix * matV0  = RAP(matB_mass_q, matVinv, matB_mass_q);
   *matV0 += *matV00;

   SparseMatrix * matS0  = RAP(matB_u_dot_div, matVinv, matB_u_dot_div);

   SparseMatrix * Vhat   = RAP(matB_q_jump, matSinv, matB_q_jump);
   SparseMatrix * Shat   = RAP(matB_u_normal_jump, matVinv, matB_u_normal_jump);


   /* debug */
/****************************************************************/
//   matV0->Add(-1.,AmatV0);
//   AmatS0.Add(-1.,*matS0);
//   cout<<" difference between approximate one and exact one: "<<endl;
//   printf(" V0: %e \n S0: %e \n\n",
//		   AmatV0.MaxNorm(),
//		   AmatS0.MaxNorm()
//		 );

//	ofstream myfileV("./pzc_data/V0.dat");
//	AmatV0.PrintMatlab(myfileV);
//	matV0->PrintMatlab(myfileV);

//	ofstream myfileS("./pzc_data/S0.dat");
//	matS0->PrintMatlab(myfileS);
/**************************************************************/
   
   cout<<endl<<"Preconditioner matrix assembled"<<endl;

#ifndef MFEM_USE_SUITESPARSE
   const double prec_rtol = 1e-3;
   const int prec_maxit = 200;

   CGSolver *V0inv = new CGSolver;
   V0inv->SetOperator( *matV0 );
//   V0inv->SetOperator( AmatV0 );
   V0inv->SetPrintLevel(-1);
   V0inv->SetRelTol(prec_rtol);
   V0inv->SetMaxIter(prec_maxit);

   CGSolver *S0inv = new CGSolver;
//   S0inv->SetOperator( *matS0 );
   S0inv->SetOperator( AmatS0 );
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
   Operator *V0inv = new UMFPackSolver( *matV0);
//   Operator *V0inv = new UMFPackSolver(AmatV0);
   Operator *Vhatinv = new UMFPackSolver(*Vhat);

   Operator *S0inv = new UMFPackSolver(AmatS0);
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

//     CG(A,b,x,solver_print_opt,4000, 1e-16,0.);
//   GMRES(A, P, b, x, solver_print_opt, 1000, 1000, 0., 1e-12);
   PCG(A, P, b, x, solver_print_opt, 1000, 1e-18, 0.0);
//   PCG(A, P, b, x, solver_print_opt, 1000, 1e-16, 0.0);


//
   {
      Vector LSres(F.Size() );
      B.Mult(x, LSres);
      LSres -= F;
      double res = sqrt(matSinv.InnerProduct(LSres, LSres));
      cout << "\n|| Bx - F ||_{S^-1} = " << res << endl;
   }
//
//   // 10b. error 
   cout<< "\n dimension: "<<dim<<endl;
   cout << "\nelement number of the mesh: "<< mesh->GetNE ()<<endl; 
   printf("\n|| u_h - u ||_{L^2} = %e \n",u0.ComputeL2Error(u_coeff) );
   printf("\n|| q_h - u ||_{L^2} = %e \n",q0.ComputeL2Error(q_coeff) );
   cout<<endl;

//   printf("\n\n");
//   printf("\n|| u_h - u ||_{L^inf} = %e \n",u0.ComputeMaxError(u_coeff) );
//   printf("\n|| q_h - u ||_{L^inf} = %e \n",q0.ComputeMaxError(q_coeff) );
//   cout << "\n|| u_h - u ||_{L^2} = " << u0.ComputeL2Error(u_coeff) << '\n' << endl;
//   cout<<endl;


   // 11. Save the refined mesh and the solution. This output can be viewed
   //     later using GLVis: "glvis -m refined.mesh -g sol.gf".
   {
      ofstream mesh_ofs("refined.mesh");
      mesh_ofs.precision(8);
      mesh->Print(mesh_ofs);

      ofstream sol_ofs("sol.gf");
      sol_ofs.precision(8);
      u0.Save(sol_ofs);

	  if(q_visual==1){
         ofstream q_variable_ofs("sol_q.gf");
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
      sol_sock.precision(8);
      sol_sock << "solution\n" << *mesh << u0 << flush;

	  if(q_visual==1){
         socketstream q_sock(vishost, visport);
         q_sock.precision(8);
         q_sock << "solution\n" << *mesh << q0 << "window_title 'Solution q'" <<
                endl;
	  }
   }

//   // 13. Free the used memory.
	/* bilinear form */
   delete B_mass_q;
   delete B_u_dot_div;
   delete B_u_normal_jump;
   delete B_q_weak_div;
   delete B_q_jump;

   delete Vinv;
   delete Sinv; 

//   delete matV00;
//   delete matV0;
//   delete matS0;
//   delete Vhat;
//   delete Shat;

   /* preconditionner */
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
		return 8.*M_PI*M_PI* sin(2.*M_PI*x(0) ) * sin(2.*M_PI*x(1) );/* HDG */
//		return -12.*x(0)-12.*x(1) + 12.;
//		return -4.;
//		return 0.;
		return 2*M_PI*M_PI*sin(M_PI*x(0) ) * sin(M_PI*x(1) );
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
//		return  2.*x(0)*x(0)*x(0) - 3.*x(0)*x(0)
//			   +2.*x(1)*x(1)*x(1) - 3.*x(1)*x(1);
//		return  x(0)*x(0) + x(1) * x(1);
//		return  1.;
//		return  x(0) + x(1);
		return  1. + x(0) + sin(2.*M_PI*x(0) ) * sin(2.* M_PI * x(1) ); /* HDG */
		return  sin(M_PI*x(0) ) * sin( M_PI * x(1) ); /* first index is 0 */
	}
	else if(x.Size() ==3 ){
		return x(0) + sin(2.*M_PI*x(0) ) * sin(2.*M_PI*x(1) ) * sin(2.*M_PI*x(2) ); 
	}
	else if(x.Size() == 1){
//		return atan(alpha_pzc * x(0) );
		return sin(2. * M_PI* x(0) ) ;
	}
	else{
		return 0;
	}

}

/* exact q = -grad u */
void q_exact(const Vector & x,Vector & f){
	if(x.Size() == 2){
//		f(0) = -M_PI*cos(M_PI*x(0) ) * sin(M_PI*x(1) );
//		f(1) = -M_PI*sin(M_PI*x(0) ) * cos(M_PI*x(1) );

//		f(0) = -1.;
//		f(1) = -1.;

//		f(0) = -2.*x(0);
//		f(1) = -2.*x(1);

//		f(0) = 6.*x(0)*(x(0)-1);
//		f(1) = 6.*x(1)*(x(1)-1);

//	    f = 0.;
		f(0) = -1. - 2.*M_PI*cos(2.*M_PI*x(0) ) * sin(2.*M_PI*x(1) );/* HDG */
		f(1) =     - 2.*M_PI*sin(2.*M_PI*x(0) ) * cos(2.*M_PI*x(1) );/* HDG */
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

