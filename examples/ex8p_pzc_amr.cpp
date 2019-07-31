////                                MFEM Example 8
//
// Compile with: make ex8_pzc
//
// Sample runs:  ex8_pzc -m ../data/square-disc.mesh
//               ex8_pzc -m ../data/star.mesh
//               ex8_pzc -m ../data/star-mixed.mesh
//               ex8_pzc -m ../data/escher.mesh
//               ex8_pzc -m ../data/fichera.mesh
//               ex8_pzc -m ../data/fichera-mixed.mesh
//               ex8_pzc -m ../data/square-disc-p2.vtk
//               ex8_pzc -m ../data/square-disc-p3.mesh
//               ex8_pzc -m ../data/star-surf.mesh -o 2
//               ex8_pzc -m ../data/mobius-strip.mesh
//
// Description:  This example code demonstrates the use of the Discontinuous
//               Petrov-Galerkin (DPG) method in its primal 2x2 block form as a
//               simple finite element discretization of the Laplace problem
//               -Delta u = f with homogeneous Dirichlet boundary conditions. We
//               use high-order continuous trial space, a high-order interfacial
//               (trace) space, and a high-order discontinuous test space
//               defining a local dual (H^{-1}) norm.
//
//               We use the primal form of DPG, see "A primal DPG method without
//               a first-order reformulation", Demkowicz and Gopalakrishnan, CAM
//               2013, DOI:10.1016/j.camwa.2013.06.029.
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
void q_exact(const Vector & x,Vector & f);

void ElementErrorEstimate( FiniteElementSpace * fes,
						   GridFunction estimator,
						   GridFunction residual,
						   GridFunction &result);

void ElementInnerProduct( FiniteElementSpace * fes,
						  GridFunction vec1,
						  GridFunction vec2,
						  GridFunction &result);


double alpha_pzc = 100.;
int sol_opt = 0;

int main(int argc, char *argv[])
{
   // 1. Initialize MPI and  parse command-line options.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   int amr_level = 5;
   /* max number of allowable hanging nodes */
   int num_hanging_node_limit = 3;
//   int num_hanging_node_limit = 2;
   double max_ref_threshold = 0.25;
   double global_ref_threshold = 0.125;
   double abs_ref_threshold = 1e-5;

   const char *mesh_file = "../data/star.mesh";
   int order = 1;
   bool visualization = 1;
   int ref_levels = -1;
   int compute_q = 0;
   int  trace_opt = 0;

   double q_error =-100000.;

   bool amr = true;

   bool bool_triangle_nonconforming = false;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&amr, "-amr", "--amr", "-no-amr",
                  "--no-amr",
                  "Enable or disable adaptive mesh refinement");

   args.AddOption(&alpha_pzc, "-alpha", "--alpha",
                  "arctan( alpha * x) as exact solution");

   args.AddOption(&sol_opt, "-sol_opt", "--sol_opt",
                  "sol_opt=0: sin cos polynomial, else arctan");

   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly");
   args.AddOption(&compute_q, "-cq", "--cq",
                  "compute q = grad(u) or not, 1 use L2 projection of grad(u), 2 use DPG, 0 by default do not compute it ");
   args.AddOption(&trace_opt, "-trace", "--trace",
                  "trace_order = trial_order, by default trace_order = trial_order - 1 ");

   args.AddOption(&bool_triangle_nonconforming, "-tri_non", "--tri_non", "-tri_conf",
                  "-tri_conf",
                  "Conforming or nonconforming triangle mesh");

   args.AddOption(&amr_level, "-amr_level", "--amr_level",
                  "level of adaptive mesh refinement");

   args.AddOption(&max_ref_threshold, "-amr_max_tol", "--amr_max_tol",
                  "relative threshold for adaptive mesh refinement");

   args.AddOption(&global_ref_threshold, "-amr_global_tol", "--amr_global_tol",
                  "relative threshold for adaptive mesh refinement");

   args.AddOption(&abs_ref_threshold, "-amr_abs_tol", "--amr_abs_tol",
                  "relative threshold for adaptive mesh refinement");

   args.AddOption(&num_hanging_node_limit, "-nc_limit", "--nc_limit",
                  "number of hanging node limit");


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
   Mesh *serial_mesh = new Mesh(mesh_file, 1, 1);
   int dim = serial_mesh->Dimension();


   // 2b. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement. We choose 'ref_levels' to be the
   //    largest number that gives a final mesh with no more than 10,000
   //    elements.
   if (ref_levels < 0)
   {
      ref_levels = (int)floor(log(50000./serial_mesh->GetNE())/log(2.)/dim);
   }
   int serial_ref_levels = 0;
//   int serial_ref_levels = min(ref_levels, 5);
   for (int l = 0; l < serial_ref_levels; l++)
   {
      serial_mesh->UniformRefinement();
   }

   /* this line is important */
   serial_mesh->EnsureNCMesh( bool_triangle_nonconforming );

   // 2c. Parioning the mesh on each processor and do the refinement
   ParMesh *mesh = new ParMesh(MPI_COMM_WORLD, *serial_mesh);
   delete serial_mesh;
   int par_ref_levels = ref_levels - serial_ref_levels;
   for( int l=0; l<par_ref_levels; l++){
		mesh->UniformRefinement();
   }
   mesh->ReorientTetMesh(); /* what is this? */


   // 4. Define the trial, interfacial (trace) and test DPG spaces:
   //    - The trial space, x0_space, contains the non-interfacial unknowns and
   //      has the essential BC.
   //    - The interfacial space, xhat_space, contains the interfacial unknowns
   //      and does not have essential BC.
   //    - The test space, test_space, is an enriched space where the enrichment
   //      degree may depend on the spatial dimension of the domain, the type of
   //      the mesh and the trial space order.
   unsigned int trial_order = order;
   unsigned int trace_order = order - 1;
   if(trace_opt){
		trace_order++;
   }
   unsigned int test_order  = order; /* reduced order, full order is
                                        (order + dim - 1) */
   if (dim == 2 && (order%2 == 0 || (mesh->MeshGenerator() & 2 && order > 1)))
   {
      test_order++;
   }
   if (test_order < trial_order){
	   if(myid==0){
			cerr << "Warning, test space not enriched enough to handle primal"
      		     << " trial space\n";
	   }
   }

   FiniteElementCollection *x0_fec, *xhat_fec, *test_fec;

   x0_fec   = new H1_FECollection(trial_order, dim);
   xhat_fec = new RT_Trace_FECollection(trace_order, dim);
   test_fec = new L2_FECollection(test_order, dim);

   ParFiniteElementSpace *x0_space   = new ParFiniteElementSpace(mesh, x0_fec);
   ParFiniteElementSpace *xhat_space = new ParFiniteElementSpace(mesh, xhat_fec);
   ParFiniteElementSpace *test_space = new ParFiniteElementSpace(mesh, test_fec);

   /* piecewise constant finite element space */
   FiniteElementCollection *piecewise_const_fec = new L2_FECollection(0,dim);
   ParFiniteElementSpace *piecewise_const_space = new ParFiniteElementSpace(mesh, piecewise_const_fec);

   HYPRE_Int global_s0 = x0_space->GlobalTrueVSize();
   HYPRE_Int global_s1 = xhat_space->GlobalTrueVSize();
   HYPRE_Int global_s_test = test_space->GlobalTrueVSize();
   if(myid == 0){
	   std::cout << "\nNumber of Unknowns:\n"
	             << " Trial space,     X0   : " << global_s0
	             << " (order " << trial_order << ")\n"
	             << " Interface space, Xhat : " << global_s1
	             << " (order " << trace_order << ")\n"
	             << " Test space,      Y    : " << global_s_test
	             << " (order " << test_order << ")\n\n";
	
	   std::cout<< " \n order of saces \n"
		        << " Trial space, X0   : "<< trial_order<<std::endl
		        << " Trace space, Xhat : "<< trace_order<<std::endl
		        << " Test space,  Y    : "<< test_order<<std::endl;
   }

   // 5. Define the block structure of the problem, by creating the offset
   //    variables. Also allocate two BlockVector objects to store the solution
   //    and rhs.
   enum {x0_var, xhat_var, NVAR};

   int s0 = x0_space->TrueVSize();
   int s1 = xhat_space->TrueVSize();
   int s_test = test_space->TrueVSize();

   Array<int> offsets(NVAR+1);
   offsets[0] = 0;
   offsets[1] = s0;
   offsets[2] = s0+s1;

   Array<int> offsets_test(2);
   offsets_test[0] = 0;
   offsets_test[1] = s_test;

   BlockVector x(offsets), b(offsets);
   b = 0.;
   x = 0.;


   // 6.  Set up the visualization part
   socketstream sol_sock;
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      sol_sock.open(vishost, visport);
      sol_sock.precision(8);
   }

   // 7. Set up the linear form F(.) which corresponds to the right-hand side of
   //    the FEM linear system, which in this case is (f,phi_i) where f=1.0 and
   //    phi_i are the basis functions in the test finite element fespace.
   ConstantCoefficient one(1.0);          /* coefficients */
   FunctionCoefficient f_coeff( f_exact );/* coefficients */
   FunctionCoefficient u_coeff( u_exact );/* coefficients */

   /* initialize Linear and Bilinear form */
   /* (f,v) linear form */
   ParLinearForm * F= new ParLinearForm(test_space);
//   F->AddDomainIntegrator(new DomainLFIntegrator(one));
   F->AddDomainIntegrator(new DomainLFIntegrator(f_coeff));

   /* -(\grad u, \grad v) biliner  form */
   ParMixedBilinearForm *B0 = new ParMixedBilinearForm(x0_space,test_space);
   B0->AddDomainIntegrator(new DiffusionIntegrator(one));


   /* trace term integrator <uhat,v> */
   ParMixedBilinearForm *Bhat = new ParMixedBilinearForm(xhat_space,test_space);
   Bhat->AddTraceFaceIntegrator(new TraceJumpIntegrator());

   /* bilinearform for preconditioner */
   /* mass matrix corresponding to the test norm */
   ParBilinearForm *Sinv = new ParBilinearForm(test_space);
   SumIntegrator *Sum = new SumIntegrator;
   Sum->AddIntegrator(new DiffusionIntegrator(one));
   Sum->AddIntegrator(new MassIntegrator(one));
   Sinv->AddDomainIntegrator(new InverseIntegrator(Sum));

   /* diffusion integrator in trial space */
   ParBilinearForm *S0 = new ParBilinearForm(x0_space);
   S0->AddDomainIntegrator(new DiffusionIntegrator(one));

   /* Hypre Par matrix, Hypre Par vector */
   HypreParMatrix * matB0=NULL;
   HypreParMatrix * matBhat=NULL;
   HypreParMatrix * matSinv=NULL;
   HypreParMatrix * matS0=NULL;
   HypreParVector * VecF=NULL;
	
   double global_error_estimator;
   for(int amr_iter = 0;amr_iter<amr_level+1 ;amr_iter++){
	   /* update the right hand side */
       F->Assemble();

       // 8. Deal with boundary conditions
       Array<int> ess_bdr(mesh->bdr_attributes.Max());
       ess_bdr = 1;
       Array<int> ess_dof;
       x0_space->GetEssentialVDofs(ess_bdr, ess_dof);

       ParGridFunction * x0 = new ParGridFunction(x0_space);
//	   x0->SetFromTrueDofs(x.GetBlock(0));
       x0->MakeTRef(x0_space, x.GetBlock(x0_var) );
	   *x0 = 0.;
	   /* impose boundary condition */
//       x0->ProjectCoefficient(u_coeff);

       ParGridFunction * xhat = new ParGridFunction(xhat_space);
//	   xhat->SetFromTrueDofs(x.GetBlock(1));



       // 9. Set up the mixed bilinear form for the primal trial unknowns, B0,
       //    the mixed bilinear form for the interfacial unknowns, Bhat,
       //    the inverse stiffness matrix on the discontinuous test space, Sinv,
       //    and the stiffness matrix on the continuous trial space, S0.
       
       /* diffusion integrator (trial,test) */
       B0->Assemble();
       B0->EliminateEssentialBCFromTrialDofs(ess_dof, *x0, *F);
	   B0->Finalize();

       /* trace terms */
       Bhat->Assemble();
	   Bhat->Finalize();

	   /* preconditioner */
       Sinv->Assemble();
       Sinv->Finalize();

	   S0->Assemble();
       S0->EliminateEssentialBC(ess_bdr);
       S0->Finalize();

       matB0   = B0->ParallelAssemble();    //delete B0;
       matBhat = Bhat->ParallelAssemble();  //delete Bhat;
       matSinv = Sinv->ParallelAssemble();  //delete Sinv;
       matS0   = S0->ParallelAssemble();    //delete S0;

       // 10. Set up the 1x2 block Least Squares DPG operator, B = [B0  Bhat],
       //    the normal equation operator, A = B^t Sinv B, and
       //    the normal equation right-hand-size, b = B^t Sinv F.
       BlockOperator *B = new BlockOperator(offsets_test, offsets);
       B->SetBlock(0,0,matB0);
       B->SetBlock(0,1,matBhat);
       RAPOperator *A = new RAPOperator(*B, *matSinv, *B);


       VecF = F->ParallelAssemble();
       {
          HypreParVector SinvF(test_space);
          matSinv->Mult(*VecF,SinvF);
          B->MultTranspose(SinvF, b);
       }

       //11. Set up a block-diagonal preconditioner for the 2x2 normal equation
       //
       //        [ S0^{-1}     0     ]
       //        [   0     Shat^{-1} ]      Shat = (Bhat^T Sinv Bhat)
       //
       //     corresponding to the primal (x0) and interfacial (xhat) unknowns.
       //     Since the Shat operator is equivalent to an H(div) matrix reduced to
       //     the interfacial skeleton, we approximate its inverse with one V-cycle
       //     of the ADS preconditioner from the hypre library (in 2D we use AMS for
       //     the rotated H(curl) problem).
       HypreBoomerAMG *S0inv = new HypreBoomerAMG(*matS0);
       S0inv->SetPrintLevel(0);

       HypreParMatrix *Shat = RAP(matSinv, matBhat);
       HypreSolver *Shatinv;
       if (dim == 2) { Shatinv = new HypreAMS(*Shat, xhat_space); }
       else          { Shatinv = new HypreADS(*Shat, xhat_space); }

       BlockDiagonalPreconditioner * P= new BlockDiagonalPreconditioner(offsets);
       P->SetDiagonalBlock(0, S0inv);
       P->SetDiagonalBlock(1, Shatinv);

       // 12. Solve the normal equation system using the PCG iterative solver.
       //     Check the weighted norm of residual for the DPG least square problem.
       //     Wrap the primal variable in a GridFunction for visualization purposes.
       CGSolver * pcg= new CGSolver(MPI_COMM_WORLD);
       pcg->SetOperator(*A);
       pcg->SetPreconditioner(*P);
       pcg->SetRelTol(1e-7);
       pcg->SetMaxIter(2000);
       pcg->SetPrintLevel(3);
       pcg->Mult(b, x);

	   delete Shatinv;
	   delete S0inv;
	   delete Shat;
	   delete P;
	   delete pcg;


	   /*******************************************************************************/
	   // 13 output results on the current mesh
	   // 13a error 
	   x0->Distribute(x.GetBlock(x0_var) );
       double x_error = x0->ComputeL2Error(u_coeff);
       if(myid==0){
		   if(sol_opt == 0){
			   cout<< "\n dimension: "<<dim<<endl;
		       printf("\n|| u_h - u ||_{L^2} =%e\n\n",x_error );
		   }
       }

	    /* res = r^T G^-1 r*/
        HypreParVector VEstimator(test_space), VResidual(test_space);
        B->Mult(x,VResidual);
        VResidual -= *VecF;
        matSinv->Mult( VResidual,VEstimator );
        global_error_estimator = sqrt( InnerProduct(VResidual, VEstimator) );
        if (myid == 0)
        {
           printf( "\n|| B0*x0 + Bhat*xhat - F ||_{S^-1} = %e\n", global_error_estimator);
        }

       // 13b Save the refined mesh and the solution. This output can be viewed
       //     later using GLVis: "glvis -m refined.mesh -g sol.gf".
       {
          ostringstream mesh_name, sol_name;
          mesh_name << "mesh." << setfill('0') << setw(6) << myid;
          sol_name << "sol." << setfill('0') << setw(6) << myid;

          ofstream mesh_ofs(mesh_name.str().c_str());
          mesh_ofs.precision(8);
          mesh->Print(mesh_ofs);

          ofstream sol_ofs(sol_name.str().c_str());
          sol_ofs.precision(8);
          x0->Save(sol_ofs);
       }

       // 13c. Send the solution by socket to a GLVis server.
       if (visualization)
       {
          sol_sock << "parallel " << num_procs << " " << myid << "\n";
          sol_sock << "solution\n" << *mesh << * x0 << flush;
       }

	   if(amr_iter==amr_level){
			break;
	   }
/**    **********************************************************************************/
	   // 14 . Adaptive mesh Refinement
       if(amr)
       {
		   // 14a. get error estimator
          HypreParVector VEstimator(test_space), VResidual(test_space);
          B->Mult(x,VResidual);
          VResidual -= *VecF;
          matSinv->Mult( VResidual,VEstimator );


		  ParGridFunction GResidual(test_space);
		  GResidual.SetFromTrueDofs( VResidual );
		  ParGridFunction GEstimator(test_space);
		  GEstimator.SetFromTrueDofs( VEstimator );

		  ParGridFunction local_error_estimator( piecewise_const_space);
		  ElementErrorEstimate( test_space, GEstimator, GResidual, local_error_estimator);
//		  ElementInnerProduct( test_space, GEstimator, GResidual, local_error_estimator);

		  // stop mesh refien ment when the dofs are too many 
		  HYPRE_Int global_trial_dofs = x0_space->GlobalTrueVSize() + xhat_space->GlobalTrueVSize();
		  if(global_trial_dofs>1000000){
		    	break;
		  }

		  // 14b. refien the mesh
		  /* coloring the element needs to be refined */
		  Array<int> refine_color;
		  double rank_max_local_error = local_error_estimator.Max(); /* max local error on current rank */
		  double max_local_error = 0.; /* max local error */
		  MPI_Allreduce(&rank_max_local_error,&max_local_error,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);

		  double rank_sum_local_error_estimator = local_error_estimator.Norml1();
		  double sum_error_estimator = 0.;
		  MPI_Allreduce(&rank_sum_local_error_estimator,&sum_error_estimator,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);

		  for(int i=0; i <mesh->GetNE(); i++){
		   if(  
			   (  (local_error_estimator(i) > abs_ref_threshold)	
			&& (  (local_error_estimator(i) > max_ref_threshold*max_local_error)
			    &&(local_error_estimator(i) > global_ref_threshold*sum_error_estimator/sqrt(mesh->GetNE() ) ) 
//			    &&(local_error_estimator(i) > global_ref_threshold*global_error_estimator/sqrt(mesh->GetNE() ) ) 
			   ) )
//			  ||(local_error_estimator(i) > 1e-3)	
			 ){
				refine_color.Append(i);
			}
		  }

		  /* refine the mesh */
		  int num_ref = mesh->ReduceInt(refine_color.Size() );
		  if(myid==0){
			 cout<<num_ref<<" elements needs to be refined "<<endl;	
		  }
		  if(num_ref){
			 if(bool_triangle_nonconforming){
				mesh->GeneralRefinement( refine_color, 1, num_hanging_node_limit );
			 }
			 else{
				mesh->GeneralRefinement( refine_color, -1, num_hanging_node_limit );
			 }
			 /* second: 1 non-conforming, 0 conforming, -1 decide by the code itself */
			 int total_element_num = mesh->GetNE();
			 if(myid==0){
				cout<<num_ref<<" elements are refined "<<endl;	
				cout<<"Refinement "<< amr_iter<<" with "<<total_element_num<<" elements "<<endl<<endl; 
			 }
		  }


//		  mesh->UniformRefinement();
//		  cout<<"mesh refined"<<endl;cout<<endl;


		  //14c. Update the finite element space and the binlinear forms 
		 if(num_ref){ 
			/* Get old  GridFunctions to make sure they match with the block vector */
		 	x0->SetFromTrueDofs(x.GetBlock(0));
		 	xhat->SetFromTrueDofs(x.GetBlock(0));
		 	
		 	/* upate finite element spaces */
		 	test_space->Update();
		 	x0_space->Update();
		 	xhat_space->Update(false);

		 	piecewise_const_space->Update();
		 	/* update grid functions */
		 	x0->Update();
		 	xhat->Update();
		 	/* update  linear form */
		 	F->Update();

		 	/* update bilinear form */
		 	B0->Update();
		 	Bhat->Update();
		 	Sinv->Update();
		 	S0->Update();


		 	/* update block sizes */
		 	offsets[0] = 0;
		 	offsets[1] = x0_space->TrueVSize();
   		 	offsets[2] = offsets[1]+xhat_space->TrueVSize();

		 	offsets_test[0] = 0;
		 	offsets_test[1] = test_space->TrueVSize();

		 	/* resize block vectors */
		 	x.Update(offsets);
		 	b.Update(offsets);
		 	x = 0.;
		 	b = 0.;

//			x0->GetTrueDofs(x.GetBlock(0) );
         }
		 else{
		    delete B;
	   	    delete A;
			break;
		 }

		  // 14c. load balancing
		  if( mesh->Nonconforming() ){
			 mesh->Rebalance();
			 /* Get old  GridFunctions to make sure they match with the block vector */
		 	 x0->SetFromTrueDofs(x.GetBlock(0));
		 	 xhat->SetFromTrueDofs(x.GetBlock(0));
		 	 
		 	 /* upate finite element spaces */
		 	 test_space->Update();
		 	 x0_space->Update();
		 	 xhat_space->Update(false);

		 	 piecewise_const_space->Update();
		 	 /* update grid functions */
		 	 x0->Update();
		 	 xhat->Update();
		 	 /* update  linear form */
		 	 F->Update();

		 	 /* update bilinear form */
		 	 B0->Update();
		 	 Bhat->Update();
		 	 Sinv->Update();
		 	 S0->Update();


		 	 /* update block sizes */
		 	 offsets[0] = 0;
		 	 offsets[1] = x0_space->TrueVSize();
   		 	 offsets[2] = offsets[1]+xhat_space->TrueVSize();

		 	 offsets_test[0] = 0;
		 	 offsets_test[1] = test_space->TrueVSize();

		 	 /* resize block vectors */
		 	 x.Update(offsets);
		 	 b.Update(offsets);
		 	 x = 0.;
		 	 b = 0.;

//			x0->GetTrueDofs(x.GetBlock(0) );
		  }
		  /* free memory for block operators */
		  delete B;
	   	  delete A;
       } /* end of mesh refinement loop */
   } /*end of  amr loop */
   // 14. Free the used memory.
   delete VecF;
//   delete S0inv;
//   delete Shatinv;
   delete matB0;
   delete matBhat;
   delete matSinv;
   delete matS0;
   delete test_space;
   delete test_fec;
   delete xhat_space;
   delete xhat_fec;
   delete x0_space;
   delete x0_fec;
   delete mesh;

   delete piecewise_const_fec;
   delete piecewise_const_space;

   MPI_Finalize();

   return 0;
}


/* define the source term on the right hand side */
// The right hand side
//  - u'' = f
double f_exact(const Vector & x){
	if(x.Size() == 2){
		if(sol_opt == 0){
			return 8.*M_PI*M_PI* sin(2.*M_PI*x(0) ) * sin(2.*M_PI*x(1) );/* HDG */
		}
		else{
			return 1.;
		}
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
		if(sol_opt == 0){
			return  sin(2.*M_PI*x(0) ) * sin(2.* M_PI * x(1) ); /* HDG */
//			return  1. + x(0) + sin(2.*M_PI*x(0) ) * sin(2.* M_PI * x(1) ); /* HDG */
		}
		else{
			return 0.;
		}
	}
	else if(x.Size() == 1){
		return atan(alpha_pzc * x(0) );
		return sin(2. * M_PI* x(0) ) ;
	}
	else{
		return 0;
	}

}

/* grad exact solution */
void q_exact( const Vector & x, Vector & f){
	if(x.Size() == 2){
		if(sol_opt ==  0){
			f(0) = +1. + 2.*M_PI*cos(2.*M_PI*x(0) ) * sin(2.*M_PI*x(1) );/* HDG */
			f(1) =     + 2.*M_PI*sin(2.*M_PI*x(0) ) * cos(2.*M_PI*x(1) );/* HDG */
		}
		else{
			f = 0.;
		}
	}
	else if(x.Size() == 1){
		f(0) = 2.*M_PI*cos(2. * M_PI * x(0) );
	}
	else{
		f(0) = 0.;
		f(1) = 0.;
		f(2) = 0.;
	}

}

void ElementInnerProduct( FiniteElementSpace * fes,
						  GridFunction vec1,
						  GridFunction vec2,
						  GridFunction &result)
{
	result = 0.;
	const FiniteElement * fe;
	Vector LocVec1,LocVec2;
	
	for(int i=0; i<fes->GetNE(); i++){
		fe = fes->GetFE(i);
		Array<int> dofs;
		fes->GetElementDofs(i, dofs);
		vec1.GetSubVector(dofs, LocVec1);
		vec2.GetSubVector(dofs, LocVec2);

	    result[i] += InnerProduct(LocVec1,LocVec2);
	}
};

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

	    result[i] = sqrt( InnerProduct(LocVec1,LocVec2) );
	}
};
