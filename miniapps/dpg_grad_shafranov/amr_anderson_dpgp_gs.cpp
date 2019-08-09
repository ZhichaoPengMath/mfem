/**********************************************************************************************
 * Anderson acceleration (mixing) to solve nonlinear Grad-Shafranov equation
 * with DPG method
 *
 *
 * Try to solve:
 *  r \nabla ( 1/r \nabla u ) = F
 *  Scheme:
 *  (rq, v) - (u,div(v) ) + <u_hat, v\cdot n> = 0
 *  -(q,\nabla \tau) + <q_hat, \tau> = (F/r, \tau)
 *  Using DPG method
 *  Trial space:
 *      Interior terms:
 *           q, u \in L^2
 *      Trace terms
 *           \hat{q} \in H^{-1/2} = trace of H(div)
 *           \hat{u} \in H^{1/2}  = trace of H^1
 *  Check paper:
 *      "AN ANALYSIS OF THE PRACTICAL DPG METHOD",
 *      J. GOPALAKRISHNAN AND W. QIU, 2014
 *  for details,
 *  the method can be seen as a generalization to the Grad-Shafaranov equation
 *
 * Set Source terms in SourceAndBoundary.hpp, we need to split source terms into linear
 * and non-linear part, also
 * source = -F(psi,r,z)/r
 * u is psi
 * q = -\grad u/r
 *
 * Adaptive mesh refinement (AMR) based on the error esitmator provided by DPG method is used
 * AMR only works for conforming mesh refinement for now
 *
 * sample run:
 * mpirun -np 4 ./amr_anderson_dpgp_gs -sol_opt 4 -o 3 -r 0 -petscopts rc_anderson -amr_level 5 -amr_abs_tol 1e-7 -amr_max_tol 0. -amr_global_tol 0.
 *
 * *******************************************************************************************/
#include "mfem.hpp"
#include "amr_anderson_reduced_system_operator.hpp"
//#include "RHSCoefficient.hpp"
#include <fstream>
#include <iostream>



using namespace std;
using namespace mfem;

/* r_exact = r */
double r_exact(const Vector & x){
	return x(0);
}
void  zero_fun(const Vector & x, Vector & f);

/****************************************************
 * Error estimator for adaptive mesh refinement
 * **************************************************/
void ElementErrorEstimate( FiniteElementSpace * fes,
						   GridFunction estimator,
						   GridFunction residual,
						   GridFunction &result);

/***********************************************
 * Subroutines to update finite element spaces
 * operators, matrices, after mesh refinement
 ***********************************************/
/* Update finite element spaces and block structure */
void AMRUpdateFEMSpaceAndBlockStructure( 
		/* finite element spaces */
		ParFiniteElementSpace * q0_space,
		ParFiniteElementSpace * u0_space,
		ParFiniteElementSpace * qhat_space,
		ParFiniteElementSpace * uhat_space,
		ParFiniteElementSpace * vtest_space,
		ParFiniteElementSpace * stest_space,
		/* block structures */
		Array<int> & offsets,
		Array<int> & offsets_test);

/* update lienar and bilinear forms */
void AMRUpdateOperators(
		/* lienar form */
		ParLinearForm *linear_source_operator,
		/* bilienar forms */
		ParMixedBilinearForm *B_mass_q,
		ParMixedBilinearForm *B_u_dot_div,
		ParMixedBilinearForm *B_u_normal_jump,
		ParMixedBilinearForm *B_q_weak_div,
		ParMixedBilinearForm *B_q_jump,
		ParBilinearForm * Vinv,
		ParBilinearForm * Sinv,
		/* terms related to boundary conditions */
		Array<int> ess_trace_vdof_list,
		ParGridFunction uhat,
		Vector *F_bc_block
		);

/* update HypreMatrices */
void AMRUpdateHypreMatrices( 
		/* related bilienar forms */
		ParMixedBilinearForm *B_mass_q,
		ParMixedBilinearForm *B_u_dot_div,
		ParMixedBilinearForm *B_u_normal_jump,
		ParMixedBilinearForm *B_q_weak_div,
		ParMixedBilinearForm *B_q_jump,
		ParBilinearForm * Vinv,
		ParBilinearForm * Sinv,
		/* HypreParMatrices */
		HypreParMatrix * &matB_mass_q,
		HypreParMatrix * &matB_u_dot_div,
		HypreParMatrix * &matB_u_normal_jump,
		HypreParMatrix * &matB_q_weak_div,
		HypreParMatrix * &matB_q_jump,
		HypreParMatrix * &matVinv,
		HypreParMatrix * &matSinv);

/* update block operators */
void AMRUpdateBlockOperators(
		/* block structure */
		Array<int> offsets_test,
		Array<int> offsets,
		/* block operators */
		BlockOperator *  B,
		BlockOperator * Jac,
		BlockOperator * InverseGram,
		/* matrices realted to subblocks */
		HypreParMatrix * matB_mass_q,
		HypreParMatrix * matB_u_dot_div,
		HypreParMatrix * matB_u_normal_jump,
		HypreParMatrix * matB_q_weak_div,
		HypreParMatrix * matB_q_jump,
		HypreParMatrix * matVinv,
		HypreParMatrix * matSinv);

int main(int argc, char *argv[])
{
      // 1. Initialize MPI. Parse command-line options
      int num_procs, myid;
      MPI_Init(&argc, &argv);
      MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
      MPI_Comm_rank(MPI_COMM_WORLD, &myid);
   
      const char * mesh_file;
      mesh_file = "../../data/cerfon_iter_quad.mesh";
      int order = 1;
      int ref_levels = -1;
   
      bool visualization = 1;
      bool q_visual = 1;
      bool q_vis_error = false;
   
      int gradgrad_opt = 0;
      int solver_print_opt = 1;
      int h1_trace_opt = 0;/* use lower order h1_trace term */
      int rt_trace_opt = 0;/* use lower order rt_trace term */
   
   
      double c_divdiv = 1.;
      double c_gradgrad = 1.;
   
      double amg_perturbation = 1e-3;
      bool perturb = false;
   
      bool use_petsc = true;
   
      bool vhat_amg = false;
   
      const char *petscrc_file = "";
   
      /* AMR parameters */
      int amr_refine_level = 0;
      /* max number of allowable hanging nodes */
      int num_hanging_node_limit = 3;
      /* thresholds for AMR */
      double max_ref_threshold = 0.25;
      double global_ref_threshold = 0.125;
      double abs_ref_threshold = 1e-5;
   
      bool amr_tri_nonconforming = false;
   
   
      OptionsParser args(argc, argv);
   
   
      /* order and refienment */
      args.AddOption(&order, "-o", "--order",
                     "Finite element order (polynomial degree).");
   
      args.AddOption(&ref_levels, "-r", "--refine",
                     "Number of times to refine the mesh uniformly, -1 by default.");
   
      /* visualization */
      args.AddOption(&visualization, "-vis", "--visualization", "-no_vis",
                     "--no-visualization",
                     "Enable or disable GLVis visualization.");
   
      args.AddOption(&q_vis_error, "-q_vis_error", "--q_vis_fd", "-no_q_vis_error",
                     "--no_q_vis_error",
                     "visualize error of q = -grad(u)/r or not, by default not visualize it ");
   
      args.AddOption(&q_visual, "-q_vis", "--visualization for grad term", "-no-vis",
                     "--no-visualization-for-grad-term",
                     "Enable or disable GLVis visualization for q = -grad(u)/r.");
   
      /* play with test spaces and test norms */
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
   
      /* perturbation or not */
      args.AddOption(&perturb, "-perturb", "--perturb", "-no_perturb",
                     "--not perturb precondiitioner",
                     "Enable perturb preconditioner or not, by derfault not do that");
   
      args.AddOption(&amg_perturbation, "-amg_perturbation", "--amg_perturbation",
   				  " the perturbation for the last diagonal block in the preconditioner");
   
      /* which problem to consider */
      args.AddOption(&sol_opt, "-sol_opt", "--sol_opt",
   				  " exact solution, 0 by default manufactured solution, 1 Cerfon's ITER solution");
   
      /* petsc options */
      args.AddOption(&petsc_linear_solver_rel_tol, "-petsc_ls_rtol", "--petsc_ls_rtol",
   				  " rel tolerance for petsc lienar solver, can be also changed in petscopts");
   
      args.AddOption(&petscrc_file, "-petscopts", "--petscopts",
                     "PetscOptions file to use.");
   
      args.AddOption(&use_petsc, "-petsc", "--petsc", "-no_petsc",
                     "--no petsc",
                     "Enable petsc or not");
   
      
      args.Parse();
      if(sol_opt == 1){
   		mesh_file = "../../data/tri_cerfon_iter_quad.mesh";
      }
      else if(sol_opt == 2){
   		mesh_file = "../../data/tri_cerfon_nstx_quad.mesh";
      }
      else if(sol_opt == 3){
   		mesh_file = "../../data/tri_ITER_double_null.mesh";
//   		mesh_file = "../../data/tri_ITER_double_null.mesh";
      }
      else if(sol_opt == 4){
   		mesh_file = "../../data/tri_cerfon_iter_quad.mesh";
      }
      else if(sol_opt == 5){
   		mesh_file = "../../data/tri_cerfon_iter_quad.mesh";
      }
   
      /* mesh */
      args.AddOption(&mesh_file, "-m", "--mesh",
                     "Mesh file to use.");
      /* AMR options */
      args.AddOption(&amr_refine_level, "-amr_level", "--amr_level",
                     "how many AMR refinement level");
   
      args.AddOption(&max_ref_threshold, "-amr_max_tol", "--amr_max_tol",
                     "relative threshold for adaptive mesh refinement");
   
      args.AddOption(&global_ref_threshold, "-amr_global_tol", "--amr_global_tol",
                     "relative threshold for adaptive mesh refinement");
   
      args.AddOption(&abs_ref_threshold, "-amr_abs_tol", "--amr_abs_tol",
                     "relative threshold for adaptive mesh refinement");
   
   //   args.AddOption(&amr_tri_nonconforming, "-tri_non", "--tri_non", "-tri_conf",
   //                  "--tri_conf",
   //                  "Conforming or non-conforming AConforming or non-conforming AMR for triangle meshh");
   
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
   
      amr_refine_level = max(amr_refine_level,0); /* avoid negative amr_refine_level */
      
      /* initialize visualization */
      socketstream sol_sock;
      socketstream q_sock;
      socketstream q_error_sock;
      char vishost[] = "localhost";
      int  visport   = 19916;
      if (visualization)
      {
         sol_sock.open(vishost, visport);
         sol_sock.precision(8);
   	  if(q_visual){
   	      q_sock.open(vishost, visport);
   		  q_sock.precision(8);
   	  }
   	  if(q_vis_error){
   	      q_error_sock.open(vishost, visport);
             q_error_sock.precision(8);
   	  }
      }
   
      // 1b. Initialize PETSc
      if (use_petsc) { MFEMInitializePetsc(NULL,NULL,petscrc_file,NULL); }
   
      // 2. Read the mesh from the given mesh file. We can handle triangular,
      //    quadrilateral, tetrahedral, hexahedral, surface and volume meshes with
      //    the same code.
      Mesh *serial_mesh = new Mesh(mesh_file, 1, 1);
      int dim = serial_mesh->Dimension();
   
      /* this line is important */
      serial_mesh->EnsureNCMesh( amr_tri_nonconforming );
   
      // 3. Refine the mesh to increase the resolution. In this example we do
      //    'ref_levels' of uniform refinement. We choose 'ref_levels' to be the
      //    largest number that gives a final mesh with no more than 10,000
      //    elements.
      
      // 3a. Coonstruct Parallel mesh and do the parallel refinement
      int par_ref_levels = ref_levels;
      ParMesh *mesh = new ParMesh(MPI_COMM_WORLD, *serial_mesh);
      delete serial_mesh;
      for(int l=0; l < par_ref_levels; l++){
   		mesh->UniformRefinement();
      }
      mesh->ReorientTetMesh();
      int mesh_global_ne = mesh->GetGlobalNE();
      if(myid == 0){
   		cout << "\nelement number of the mesh: "<< mesh_global_ne<<endl; 
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
   
      ParFiniteElementSpace * u0_space   = new ParFiniteElementSpace(mesh, u0_fec);
      ParFiniteElementSpace * q0_space   = new ParFiniteElementSpace(mesh, q0_fec, dim);
      ParFiniteElementSpace * uhat_space = new ParFiniteElementSpace(mesh, uhat_fec);
      ParFiniteElementSpace * qhat_space = new ParFiniteElementSpace(mesh, qhat_fec);
      
      ParFiniteElementSpace * vtest_space = new ParFiniteElementSpace(mesh, vtest_fec,dim);
      ParFiniteElementSpace * stest_space = new ParFiniteElementSpace(mesh, stest_fec);
      
      /* piecewise constant finite element space */
      FiniteElementCollection *piecewise_const_fec = new L2_FECollection(0,dim);
      ParFiniteElementSpace *piecewise_const_space = new ParFiniteElementSpace(mesh, piecewise_const_fec);
   
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
   	   std::cout << "\nTotal Number of Unknowns:\n"<< endl
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
   
      BlockVector x(offsets);
      x = 0.;
   
      BlockVector F(offsets_test); /* block vector for the linear form on the right hand side */
      F = 0.;
   
      ConstantCoefficient one(1.0);          /* coefficients */
      VectorFunctionCoefficient vec_zero(dim, zero_fun);          /* coefficients */
      VectorFunctionCoefficient q_coeff(dim, q_exact);          /* coefficients */
      FunctionCoefficient u_coeff( u_exact );/* coefficients */
   
      FunctionCoefficient r_coeff( r_exact ); /* coefficients */
   
      /* grid function */
      ParGridFunction uhat;
      uhat.MakeTRef(uhat_space, x.GetBlock(uhat_var), 0);
      uhat.ProjectCoefficientSkeletonDG(u_coeff);
   
   
      /*****************************************************
   	* 7. Set up linear form 
   	* ***************************************************/
      /* rhs for -(q,\grad v) + \lgl qhat, v \rgl = (f,v) */
      FunctionCoefficient f_coeff( linear_source );/* coefficients */
      ParLinearForm * linear_source_operator(new ParLinearForm(stest_space) );
      linear_source_operator->AddDomainIntegrator( new DomainLFIntegrator(f_coeff) );
      linear_source_operator->Assemble();
   
   
      // 8. Get essential degree of freedoms for the Dirichlet boundary condition,
      // which is imposed throught trace term  \hat{u}
      Array<int> ess_bdr(mesh->bdr_attributes.Max());
      ess_bdr = 1;
   
      Array<int> ess_trace_vdof_list;/* store the location (index) of  boundary element  */
      Array<int> ess_trace_tdof_list;
      uhat_space->GetEssentialVDofs(ess_bdr, ess_trace_vdof_list);
      uhat_space->GetEssentialTrueDofs(ess_bdr, ess_trace_tdof_list);
   
   
   
      // 9. Set up thebilinear forms 
      //    B_mass_q:    (rq,\tau)
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
      B_mass_q->AddDomainIntegrator(new VectorMassIntegrator(r_coeff) );
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
      B_u_normal_jump->EliminateEssentialBCFromTrialDofs(ess_trace_vdof_list, uhat, F.GetBlock(0) );
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
   
      /********************************************************/
	  /***********************************************************
	  * 10a. Set up the 1x2 block Least Squares DPG operator, 
	  *    the normal equation operator, A = B^t InverseGram B, and
	  *    the normal equation right-hand-size, b = B^t InverseGram F.
	  *
	  *    B = mass_q     -u_dot_div 0        u_normal_jump
	  *        q_weak_div  0         q_jump   0
	  ********************************************************/
	  BlockOperator *B = new BlockOperator (offsets_test, offsets);
	  
	  B->SetBlock(0, q0_var  ,matB_mass_q);
	  B->SetBlock(0, u0_var  ,matB_u_dot_div);
	  B->SetBlock(0, uhat_var,matB_u_normal_jump);
	  
	  B->SetBlock(1, q0_var   ,matB_q_weak_div);
	  B->SetBlock(1, qhat_var ,matB_q_jump);
	
	
	  /*********************************************
	   * 10b. allocate memory to store 
	   * Jac =  mass_q	 -u_dot_div  0		u_normal_jump
	   *        q_weak_div -df/du      q_jump 0
	   *	   = B - DF/DU
	   * Now, we only allocate memory and set 
	   * df/du = 0, and df/du block will be updated
	   * during the nonlinear solve step
	   * *******************************************/
	  BlockOperator * Jac = new BlockOperator(offsets_test,offsets);
	  
	  Jac->SetBlock(0, q0_var  ,matB_mass_q);
	  Jac->SetBlock(0, u0_var  ,matB_u_dot_div);
	  Jac->SetBlock(0, uhat_var,matB_u_normal_jump);
	  
	  Jac->SetBlock(1, q0_var   ,matB_q_weak_div);
	  Jac->SetBlock(1, qhat_var ,matB_q_jump);
	  
	  /* G^{-1} = diag(V^{-1},S^{-1} ) */
	  BlockOperator * InverseGram = new BlockOperator(offsets_test, offsets_test);
	  InverseGram->SetBlock(0,0,matVinv);
	  InverseGram->SetBlock(1,1,matSinv);
     /***************************************************************/
     // 11. Set up a block-diagonal preconditioner for the 4x4 normal equation
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
      HypreParMatrix * matS0 = S0->ParallelAssemble(); 
   
   	// the exact form of the diagonal block //
      HypreParMatrix * matV0  = RAP(matB_mass_q, matVinv, matB_mass_q);
      matV0->Add(1. , *RAP(matB_q_weak_div, matSinv, matB_q_weak_div) );
   
      HypreParMatrix * matVhat   = RAP(matB_q_jump, matSinv, matB_q_jump);
   
      /********************************************************/
      /* perturbed amg preconditioner for the last block */
      ParMixedBilinearForm *Sjump = NULL;
      HypreParMatrix * matSjump = NULL;
      HypreParMatrix * matShat = NULL;
   
	  PetscParMatrix * petsc_matShat = NULL;
	/* By default, we use PETSC, no perturbation needed
	 *
   	 * However, if we use the mfem HpreBoomerAMG, we need to perturb Shat 
   	 * to make it work, due to mfem implementation for BoomerAMG. This is 
	 * Ad-Hoc and not recommended, though it works fine and fast for many
	 * problems*/

//	  HypreParMatrix * matShat_tmp = NULL;
//	  HypreParMatrix * matB_u_normal_jump_Transpose = NULL;

      if(!perturb){
//		  matB_u_normal_jump_Transpose = matB_u_normal_jump->Transpose();
//	      matShat_tmp = ParMult(matVinv, matB_u_normal_jump);
//	      matShat = ParMult(matB_u_normal_jump_Transpose, matShat_tmp);
		  matShat = RAP(matVinv,matB_u_normal_jump);

		  petsc_matShat = new PetscParMatrix(matShat->GetComm(), matShat, Operator::PETSC_MATAIJ);
      }
      else{
   	      amg_perturbation = min(1e-3, amg_perturbation);
   		  cout<<amg_perturbation<<endl;
   		  Sjump = new ParMixedBilinearForm(uhat_space,vtest_space);
      	  Sjump->AddTraceFaceIntegrator(new DGNormalTraceJumpIntegrator() );
      	  Sjump->Assemble();
      	  Sjump->Finalize();
      	  Sjump->SpMat() *= amg_perturbation;
      	  Sjump->SpMat() += B_u_normal_jump->SpMat();
      	  matSjump=Sjump->ParallelAssemble(); 
          matShat = RAP(matSjump, matVinv, matSjump);
      }

	  /*******************************************************************************
	   * 12. pass all the pointer to the infomration interfce,
	   * everything is passed to PETSC by FixedPointReducedSystemOperator,
	   * Jac and A is updated throught the FixedPointReducedSystemOperator
	   * ******************************************************************************/
	  FixedPointReducedSystemOperator * reduced_system_operator = NULL;
	
	  /*******************************************************************************
	   * 13. Adaptive mesh refinement loop
       *     Solve the normal equation system using the PCG iterative solver.
       *     Check the weighted norm of residual for the DPG least square problem.
       *     Wrap the primal variable in a GridFunction for visualization purposes.
	   * ******************************************************************************/
      for(int amr_iter = 0; amr_iter<=amr_refine_level ; amr_iter++){
           ParGridFunction u0(u0_space);
           ParGridFunction q0(q0_space);

       	   StopWatch timer;

		   /**********************************************************************
			* 14. Assemble the block Jacobian preconditioner for the current mesh
			* ********************************************************************/
	       BlockDiagonalPreconditioner prec(offsets);

	       HypreBoomerAMG prec0( *matV0 );
	       prec0.SetPrintLevel(0);

           HypreBoomerAMG prec1( *matS0 );	
	       prec1.SetPrintLevel(0);

	       HypreAMS prec2( *matVhat, qhat_space );
	       prec2.SetPrintLevel(1);
	       prec.SetDiagonalBlock(0,&prec0);
	       prec.SetDiagonalBlock(1,&prec1);
	       prec.SetDiagonalBlock(2,&prec2);

		   PetscLinearSolver prec3(MPI_COMM_WORLD);
		   HypreBoomerAMG mfem_prec3;
	       if( !perturb ){
	           prec3.SetOperator(*petsc_matShat);
	       	   prec3.SetPrintLevel(0);
//	       	   prec3->iterative_mode = true;

	       	   KSP ksp_prec3 = KSP(prec3);
	       	   KSPSetType(ksp_prec3,KSPFCG);
	       	   KSPAppendOptionsPrefix(ksp_prec3,"s3_");
	       	   PC  pc_prec3;
	       	   KSPGetPC(ksp_prec3,&pc_prec3);
	       	   PCSetType(pc_prec3,PCHYPRE);

	       	   prec.SetDiagonalBlock(3,&prec3);
	       }
	       else{
		       mfem_prec3.SetOperator(*matShat);
	       	   mfem_prec3.SetPrintLevel(0);

	       	   prec.SetDiagonalBlock(3,&mfem_prec3);
	       }/* end of if */


		   // 15. Pass the pointers to a ReducedSystemOperator, and then link
		   // with Petsc through the ReducedSystemOperator
		   reduced_system_operator =  new FixedPointReducedSystemOperator(
	  	    										&use_petsc,
	  	    										&perturb,
	  	     									/* finite element spaces */
	  	    										u0_space, q0_space, uhat_space, qhat_space,
	  	    										vtest_space, stest_space,
	  	     									/* linear forms */
	  	    										linear_source_operator,
	  	     									/* vector corresponding to linear forms */
	  	    										F,
	  	     									/* bilinear forms */
	  	     									B_mass_q, B_u_dot_div, B_u_normal_jump, B_q_weak_div, B_q_jump,
	  	     									Vinv, Sinv,
	  	     									/* matrices */
	  	    										matB_mass_q, matB_u_normal_jump, matB_q_weak_div, 
	  	    										matB_q_jump, matVinv, matSinv,
	  	     									/* block sizes */
	  	    										offsets, offsets_test,
	  	     									/* block operators */
	  	    										B,
	  	    										Jac,
	  	    										InverseGram,
	  	     									/* boundary conditions */
	  	    										ess_trace_vdof_list,
	  	     									/* preconditioner */
												    prec
	  	   );

       	   if(!use_petsc){
       		   if(myid == 0){
       				cout<<"Wrong! PETSC is not used!"<<endl<<endl;
       		   }
       	   }
       	   else{
		   // 16. Solve the problem for the current mesh
       		    PetscNonlinearSolver * petsc_anderson = new PetscNonlinearSolver( MPI_COMM_WORLD );
       		    petsc_anderson->SetOperator( *reduced_system_operator );
       		    petsc_anderson->SetRelTol(1e-10);
       		   	petsc_anderson->SetAbsTol(0.);
       			petsc_anderson->SetMaxIter(50);
       			petsc_anderson->SetPrintLevel(1);
       
       			petsc_anderson->iterative_mode = true;
       
       			SNES pn_snes = SNES(*petsc_anderson);
       
       			/* empty vector bb means that we are solving nonlinear_fun(x) = 0 */
       			Vector bb;
       			timer.Start();
       		    petsc_anderson->Mult(bb,x);
       	   }
       	   timer.Stop();
       	   if(myid==0){
       			cout<<"time: "<<timer.RealTime()<<endl;
       	   }
       
       	// 17. error 
       	   u0.Distribute( x.GetBlock(u0_var) );
       	   q0.Distribute( x.GetBlock(q0_var) );
       
       	   double u_l2_error = u0.ComputeL2Error(u_coeff);
       	   double q_l2_error = q0.ComputeL2Error(q_coeff);
       
       	   double u_max_error = u0.ComputeMaxError(u_coeff);
       	   double q_max_error = q0.ComputeMaxError(q_coeff);
       
       	   int global_ne = mesh->GetGlobalNE();
		   if(myid == 0){
       	       cout << "\nelement number of the mesh: "<< global_ne<<endl; 
		   }
       	   if(myid == 0){
			   if(sol_opt <=3 ){
       		       cout << "\nelement number of the mesh: "<< global_ne<<endl; 
       			   cout<< "\n dimension: "<<dim<<endl;
       			   printf("\n|| u_h - u ||_{L^2} = %e \n", u_l2_error );
       			   printf("\n|| q_h - q ||_{L^2} = %e \n", q_l2_error );
       			   cout<<endl;
       			   printf("\n|| u_h - u ||_{L^inf} = %e \n", u_max_error );
       			   printf("\n|| q_h - q ||_{L^inf} = %e \n", q_max_error );
       			   cout<<endl;
			   }
       	   }
       
       	
       	// 18. Visualization
       	//	   Save the refined mesh and the solution. This output can be viewed
       	//     later using GLVis: "glvis -m refined.mesh -g sol.gf".
       	   ParGridFunction * q_projection_minus_num = NULL; 
       	   if(q_vis_error){
       		   q_projection_minus_num = new ParGridFunction(q0_space);
       		   q_projection_minus_num->ProjectCoefficient(q_coeff);
       		   *q_projection_minus_num -= q0;
       	   }
       
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
       		  if(q_vis_error){
       	    	 ostringstream q_error_name;
       	    	 mesh_name<< "q_error."<<setfill('0')<<setw(6)<<myid;
       	         ofstream q_error_variable_ofs(q_error_name.str().c_str() );
       	         q_error_variable_ofs.precision(8);
       	         q_projection_minus_num->Save(q_error_variable_ofs);
       		  }
       	   }
       	  
       	   // 18b. Send the solution by socket to a GLVis server.
       	   if (visualization)
       	   {
       		  sol_sock << "parallel " << num_procs << " " << myid << "\n";
              sol_sock << "solution\n" << *mesh <<  u0 << "window_title 'U' "<<endl;
       	
       		  if(q_visual){
       			 q_sock << "parallel " << num_procs << " " << myid << "\n";
             	 q_sock << "solution\n" << *mesh <<  q0 << "window_title 'Q' "<<endl;
       		  }
       		  /* plot error picture for q */
       		  if(q_vis_error){
       			 q_error_sock << "parallel " << num_procs << " " << myid << "\n";
				 q_error_sock << "solution\n" << *mesh <<  *q_projection_minus_num << "window_title 'Q_error' "<<endl;
       		  }
       	   }
       	   delete q_projection_minus_num;


		  /*************************************************************************
		   * 19. Adaptive mesh refinement
		   * ***********************************************************************/

		  /*********************************************************************
		   * 19a. Error estimator, which is the residual in the dual norm
		   * *******************************************************************/
		  /* 19a1. calculate the estimator and residual vector for DPG */
          linear_source_operator->ParallelAssemble( F.GetBlock(1) );
          
		  /* allocate memory */
          BlockVector BlockEstimator( offsets_test ), BlockResidual( offsets_test );
		  /* get Bx */
          B->Mult(x, BlockEstimator);
          
          /* Bx - linear source */
          BlockEstimator -= F;
          
          /* Bx - nonlinear source */
          F = 0.;
          Vector F1(F.GetData() + offsets_test[1],offsets_test[2]-offsets_test[1]);
          ParGridFunction u0_now;
          Vector u0_vec(x.GetData() + offsets[1], offsets[2] - offsets[1]);
          u0_now.MakeTRef(u0_space, u0_vec, 0);
          u0_now.SetFromTrueVector();
          FUXCoefficient fu_coefficient( &u0_now, &nonlinear_source );
          
          ParLinearForm *fu_mass = new ParLinearForm( stest_space );
          fu_mass->AddDomainIntegrator( new DomainLFIntegrator(fu_coefficient)  );
          fu_mass->Assemble();
          
          fu_mass->ParallelAssemble(F1);
          BlockEstimator -= F;
          
		  /* calculate residual */ 
          InverseGram->Mult(BlockEstimator, BlockResidual);

          double res = sqrt(InnerProduct(BlockEstimator,BlockResidual) );
		  /* subblocks */
		  ParGridFunction UEstimator(vtest_space);
		  UEstimator.SetFromTrueDofs( BlockEstimator.GetBlock(0) );
		  ParGridFunction UResidual(vtest_space);
		  UResidual.SetFromTrueDofs( BlockResidual.GetBlock(0) );

		  ParGridFunction QEstimator(stest_space);
		  QEstimator.SetFromTrueDofs( BlockEstimator.GetBlock(1) );
		  ParGridFunction QResidual(stest_space);
		  QResidual.SetFromTrueDofs( BlockResidual.GetBlock(1) );

		  ParGridFunction local_u_estimator(piecewise_const_space);
		  ParGridFunction local_q_estimator(piecewise_const_space);


		  ElementErrorEstimate( vtest_space, UEstimator, UResidual, local_u_estimator);
		  ElementErrorEstimate( stest_space, QEstimator, QResidual, local_q_estimator);

		  if(myid == 0){
				printf("|| Bx - F ||_{G^-1} || = %e \n ",res);
		  }

		  if(amr_iter>=amr_refine_level){
			break;
		  }
		  /*********************************************************************
		   * 19b. coloring the element needs to be refined
		   * *******************************************************************/
		  Array<int> refine_color;

		  double rank_max_q_error = local_q_estimator.Max(); /* max local error on current rank */
		  double max_q_error = 0.; /* max local error */
		  MPI_Allreduce(&rank_max_q_error,&max_q_error,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);

		  double rank_sum_q_estimator = local_q_estimator.Norml1();
		  double sum_q_estimator = 0.;
		  MPI_Allreduce(&rank_sum_q_estimator,&sum_q_estimator,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);


		  for(int i=0; i <mesh->GetNE(); i++){
		   if(  
			   (  (local_q_estimator(i) > abs_ref_threshold)	
			&& (  (local_q_estimator(i) > max_ref_threshold*max_q_error)
			    &&(local_q_estimator(i) > global_ref_threshold*sum_q_estimator/sqrt(mesh->GetNE() ) ) 
			   ) )
			 ){
				refine_color.Append(i);
			} /* if */
		  }/* loop over all elements */

		  /* 19c. Refine the mesh if needed */
		  int num_ref = mesh->ReduceInt(refine_color.Size() );
		  if(myid==0){
			 cout<<num_ref<<" elements needs to be refined "<<endl;	
		  }
		  if(num_ref>0){
			  mesh->GeneralRefinement( refine_color, -1, num_hanging_node_limit );
		  }
		  else{
			 break;
		  }

		  /* output refinement information */
		  int mesh_num = mesh->GetGlobalNE();
		  if(myid == 0){
		  	cout<<"Refinement "<<amr_iter<<" with "<<mesh_num<<" elements"<<endl;
			cout<<endl;
		  }

		  /**********************************************************************
		  * 20. Update the finite element spaces, linear, bilinear forms
		  * and matrices,
		  * update is defined in FixedPointReducedSystemOperator
		  **********************************************************************/
		  /**********************************************
		   * 20a. update the (i) finite element spaces 
		   *				 (ii) block structures 	
		   ***********************************************/
		  AMRUpdateFEMSpaceAndBlockStructure( 
				/* finite element spaces */
				q0_space, u0_space, qhat_space, uhat_space,
				vtest_space, stest_space,
				/* block structures */
				offsets, offsets_test);

		  piecewise_const_space->Update();
		  /*****************************************************************
		   * 20b. update essential degree of freedoms for the boundary
		   * conditions 
		   ****************************************************************/
		  uhat_space->GetEssentialVDofs(ess_bdr, ess_trace_vdof_list);
		  /****************************************************/
		  /* 20c. update the block vectors and grid functions */
		  /****************************************************/
		  x.Update(offsets);
		  x = 0.;
		  F.Update(offsets_test);
		  F = 0.;
		  
		  uhat.Update();
		  uhat.ProjectCoefficientSkeletonDG(u_coeff);

		  u0.Update();
		  q0.Update();
		  /************************************************************
		   * 20d. update linear and bilinear forms
		   * **********************************************************/
	      AMRUpdateOperators( linear_source_operator, /* linear forms */
						B_mass_q,B_u_dot_div,B_u_normal_jump,B_q_weak_div, B_q_jump, /* bilinear forms */
						Vinv,Sinv, /* bilinear forms */
						ess_trace_vdof_list,uhat,&(F.GetBlock(0) ) ); /* deal with boundary conditions */
		  /***************************************************/
		  /* 20e. Set boundary conditions for block vectors	 */
		  /***************************************************/
		  uhat.GetTrueDofs(x.GetBlock(uhat_var) );

		  /***************************************************/
		  /* 20g. Update HypreParMatrices					 */
		  /***************************************************/
		  AMRUpdateHypreMatrices(/* related bilinear forms */ 
		   					B_mass_q, B_u_dot_div, B_u_normal_jump, B_q_weak_div, B_q_jump,
		  					Vinv, Sinv,
		   					/* matrices */
		  					matB_mass_q, matB_u_dot_div, matB_u_normal_jump, matB_q_weak_div, matB_q_jump,
		  					matVinv, matSinv);

		  /*************************************************/
		  /* 20h. Update Block Operators				    */
		  /*************************************************/
		  /* 20h1. resize the operator */
		  delete B;
		  delete Jac;
		  delete InverseGram;

		  B = new BlockOperator (offsets_test, offsets);
		  Jac = new BlockOperator(offsets_test,offsets);
	      InverseGram = new BlockOperator(offsets_test, offsets_test);

	      /**************************************************
		  * 21. load balancing
		  * ************************************************/
		  if(mesh->Nonconforming() ){
		      mesh->Rebalance();
		      /* 21a. FEM space and block structure */
              AMRUpdateFEMSpaceAndBlockStructure( 
						/* finite element spaces */
              			q0_space, u0_space, qhat_space, uhat_space,
              			vtest_space, stest_space,
              			/* block structures */
              			offsets, offsets_test);

			 piecewise_const_space->Update();
              
		      /* 21b. locate essential dofs for boundary conditions */
              uhat_space->GetEssentialVDofs(ess_bdr, ess_trace_vdof_list);
		      /* 21c. update block vectors and grid functions */
              x.Update(offsets);
              x = 0.;
              F.Update(offsets_test);
              F = 0.;
              
              uhat.Update();
              uhat.ProjectCoefficientSkeletonDG(u_coeff);
              
              u0.Update();
		      /* 21d. linear and bilinear forms */
              AMRUpdateOperators( linear_source_operator, /* linear forms */
              			B_mass_q,B_u_dot_div,B_u_normal_jump,B_q_weak_div, B_q_jump, /* bilinear forms */
              			Vinv,Sinv, /* bilinear forms */
              			ess_trace_vdof_list,uhat,&(F.GetBlock(0) ) ); /* deal with boundary conditions */
		      /* 21e. set boundary conditions */
              uhat.GetTrueDofs(x.GetBlock(uhat_var) );
              
		      /* 21f. update Hypre Matrices */
              AMRUpdateHypreMatrices(/* related bilinear forms */ 
              				B_mass_q, B_u_dot_div, B_u_normal_jump, B_q_weak_div, B_q_jump,
              				Vinv, Sinv,
              				/* matrices */
              				matB_mass_q, matB_u_dot_div, matB_u_normal_jump, matB_q_weak_div, matB_q_jump,
              				matVinv, matSinv);

              /* 21g. Update block operators */
               delete B;
               delete Jac;
               delete InverseGram;
              
               B = new BlockOperator (offsets_test, offsets);
               Jac = new BlockOperator(offsets_test,offsets);
               InverseGram = new BlockOperator(offsets_test, offsets_test);

		  }/* end of load balancing */
		  
		  /* 22. Set subblocks block operators
		   *     Put it here, so no matter the mesh is 
		   *     conforming or non-conforming, 
		   *     the block operators are updated and 
		   *     no extra-updating is wasted due to the 
		   *     rebalancing */
          AMRUpdateBlockOperators( offsets_test, offsets, /* block structure */
           			B, Jac, InverseGram, /* block operators */
           			/* matrices to set small blocks */
           			matB_mass_q, matB_u_dot_div, matB_u_normal_jump, matB_q_weak_div, matB_q_jump,
					matVinv, matSinv);
		   
		  /**********************************************/
		  /* 23. Update Block Jacobian Preconditioners */
		  /**********************************************/
		  S0->Update();	
		  S0->Assemble();
		  S0->Finalize();
		  matS0 = S0->ParallelAssemble();

          matV0  = RAP(matB_mass_q, matVinv, matB_mass_q);
		  matV0->Add(1. , *RAP(matB_q_weak_div, matSinv, matB_q_weak_div) );

          matVhat   = RAP(matB_q_jump, matSinv, matB_q_jump);

	      if(!perturb){
	 	      matShat = RAP(matVinv,matB_u_normal_jump);
		      delete petsc_matShat;
		      petsc_matShat = new PetscParMatrix(matSinv->GetComm(), matShat, Operator::PETSC_MATAIJ);
 	      }
 	      else{
 	   	     amg_perturbation = min(1e-3, amg_perturbation);
 	   	     cout<<amg_perturbation<<endl;
		     delete Sjump;
 	   	     Sjump = new ParMixedBilinearForm(uhat_space,vtest_space);
 	      	 Sjump->AddTraceFaceIntegrator(new DGNormalTraceJumpIntegrator() );
 	      	 Sjump->Assemble();
 	      	 Sjump->Finalize();
 	      	 Sjump->SpMat() *= amg_perturbation;
 	      	 Sjump->SpMat() += B_u_normal_jump->SpMat();
 	      	 matSjump=Sjump->ParallelAssemble(); 
 	         matShat = RAP(matSjump, matVinv, matSjump);
 	      }

		  /************************************************
		   * 24. Free the ReducedOperator for current mesh
		   ************************************************/
		   delete reduced_system_operator;
       } /* end of AMR loop */

//   // 25. Free the used memory.
   /* block operators */
   delete InverseGram;
   delete Jac;
   delete B;
	/* bilinear form */
   delete Vinv;
   delete Sinv; 

   delete B_mass_q;
   delete B_u_dot_div;
   delete B_q_weak_div;
   delete B_q_jump;
   delete B_u_normal_jump;

   delete S0;
   if(perturb){
       delete Sjump;
   }
   /* matrix */
   delete matB_mass_q;
   delete matB_u_dot_div;
   delete matB_u_normal_jump;
   delete matB_q_weak_div;
   delete matB_q_jump;
   delete matVinv;
   delete matSinv;

   delete matS0;
   delete matV0;
   delete matVhat;
   delete matShat;

   delete petsc_matShat;
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


/* vector 0 */
void zero_fun(const Vector & x, Vector & f){
	f = 0.;
}

/*****************************************************
 * Subroutines for updating finite element spaces,
 * opeartors and matrices after adaptive mesh refinement
 * ***************************************************/

/* update finite element spaces and block strcture */
void AMRUpdateFEMSpaceAndBlockStructure( 
		ParFiniteElementSpace * q0_space,
		ParFiniteElementSpace * u0_space,
		ParFiniteElementSpace * qhat_space,
		ParFiniteElementSpace * uhat_space,
		ParFiniteElementSpace * vtest_space,
		ParFiniteElementSpace * stest_space,
		Array<int> & offsets,
		Array<int> & offsets_test)
{
	/****************************************
	* 1. update the finite element space
	****************************************/
	q0_space->Update();
	u0_space->Update();
	qhat_space->Update(false);
	uhat_space->Update(false);
	
	vtest_space->Update();
	stest_space->Update();
	/*******************************************************
	* 2. update the block structures					 
	*******************************************************/
	offsets[0] = 0;
	offsets[1] = q0_space->TrueVSize();
	offsets[2] = offsets[1] + u0_space->TrueVSize();
	offsets[3] = offsets[2] + qhat_space->TrueVSize();
	offsets[4] = offsets[3] + uhat_space->TrueVSize();
	
	offsets_test[0] = 0;
	offsets_test[1] = vtest_space->TrueVSize();
	offsets_test[2] = offsets_test[1] + stest_space->TrueVSize();
}

/* update linear and bilinear forms */
void AMRUpdateOperators(
		ParLinearForm *linear_source_operator,
		ParMixedBilinearForm *B_mass_q,
		ParMixedBilinearForm *B_u_dot_div,
		ParMixedBilinearForm *B_u_normal_jump,
		ParMixedBilinearForm *B_q_weak_div,
		ParMixedBilinearForm *B_q_jump,
		ParBilinearForm * Vinv,
		ParBilinearForm * Sinv,
		Array<int> ess_trace_vdof_list,
		ParGridFunction uhat,
		Vector *F_bc_block
		)
{
	/* 1 update linera form */
	linear_source_operator->Update();
	linear_source_operator->Assemble();
	/* 2 update bilinear forms */
	B_mass_q->Update();
	B_mass_q->Assemble();
	B_mass_q->Finalize();
	
	B_u_dot_div->Update();
	B_u_dot_div->Assemble();
	B_u_dot_div->Finalize();
	B_u_dot_div->SpMat() *= -1.;
	
	B_u_normal_jump->Update();
	B_u_normal_jump->Assemble();
	/* deal with boundary conditions */
	B_u_normal_jump->EliminateEssentialBCFromTrialDofs(ess_trace_vdof_list, uhat, *F_bc_block );
	B_u_normal_jump->Finalize();
	
	B_q_weak_div->Update();
	B_q_weak_div->Assemble();
	B_q_weak_div->Finalize();
	
	B_q_jump->Update();
	B_q_jump->Assemble();
	B_q_jump->Finalize();
	
	Vinv->Update();
	Vinv->Assemble();
	Vinv->Finalize();
	
	Sinv->Update();
	Sinv->Assemble();
	Sinv->Finalize();

}

/* update hypre matrices */
void AMRUpdateHypreMatrices( 
		ParMixedBilinearForm *B_mass_q,
		ParMixedBilinearForm *B_u_dot_div,
		ParMixedBilinearForm *B_u_normal_jump,
		ParMixedBilinearForm *B_q_weak_div,
		ParMixedBilinearForm *B_q_jump,
		ParBilinearForm * Vinv,
		ParBilinearForm * Sinv,
		HypreParMatrix * &matB_mass_q,
		HypreParMatrix * &matB_u_dot_div,
		HypreParMatrix * &matB_u_normal_jump,
		HypreParMatrix * &matB_q_weak_div,
		HypreParMatrix * &matB_q_jump,
		HypreParMatrix * &matVinv,
		HypreParMatrix * &matSinv)
{
	matB_mass_q = B_mass_q->ParallelAssemble();
	matB_u_dot_div = B_u_dot_div->ParallelAssemble();
	matB_u_normal_jump = B_u_normal_jump->ParallelAssemble();
	matB_q_weak_div = B_q_weak_div->ParallelAssemble();
	matB_q_jump = B_q_jump->ParallelAssemble();
	
	matVinv = Vinv->ParallelAssemble();
	matSinv = Sinv->ParallelAssemble();
}

/* update block operators */
void AMRUpdateBlockOperators(
		Array<int> offsets_test,
		Array<int> offsets,
		BlockOperator *  B,
		BlockOperator * Jac,
		BlockOperator * InverseGram,
		HypreParMatrix * matB_mass_q,
		HypreParMatrix * matB_u_dot_div,
		HypreParMatrix * matB_u_normal_jump,
		HypreParMatrix * matB_q_weak_div,
		HypreParMatrix * matB_q_jump,
		HypreParMatrix * matVinv,
		HypreParMatrix * matSinv)
{
   enum {q0_var, u0_var,qhat_var,uhat_var, NVAR};

	B->SetBlock(0, q0_var  ,matB_mass_q);
	B->SetBlock(0, u0_var  ,matB_u_dot_div);
	B->SetBlock(0, uhat_var,matB_u_normal_jump);
	
	B->SetBlock(1, q0_var   ,matB_q_weak_div);
	B->SetBlock(1, qhat_var ,matB_q_jump);
	
	/*************************************/
	Jac->SetBlock(0, q0_var  ,matB_mass_q);
	Jac->SetBlock(0, u0_var  ,matB_u_dot_div);
	Jac->SetBlock(0, uhat_var,matB_u_normal_jump);
	
	Jac->SetBlock(1, q0_var   ,matB_q_weak_div);
	Jac->SetBlock(1, qhat_var ,matB_q_jump);
	/*************************************/
	InverseGram->SetBlock(0,0,matVinv);
	InverseGram->SetBlock(1,1,matSinv);
}

/**************************************************
 * Error estimator
 * ************************************************/
void ElementErrorEstimate( FiniteElementSpace * fes,
						  GridFunction estimator,
						  GridFunction residual,
						  GridFunction &result)
{
	result = 0.;
	const FiniteElement * fe;
	Vector LocVec1,LocVec2;
	
	for(int i=0; i<fes->GetNE(); i++){
		fe = fes->GetFE(i);
		Array<int> dofs;
		fes->GetElementDofs(i, dofs);
		estimator.GetSubVector(dofs, LocVec1);
		residual.GetSubVector(dofs, LocVec2);

//	    result[i] = ( InnerProduct(LocVec1,LocVec2) );
	    result[i] = sqrt( InnerProduct(LocVec1,LocVec2) );
	}
};
