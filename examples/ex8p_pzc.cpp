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

double alpha_pzc = 100.;

int main(int argc, char *argv[])
{
   // 1. Initialize MPI and  parse command-line options.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   const char *mesh_file = "../data/star.mesh";
   int order = 1;
   bool visualization = 1;
   int ref_levels = -1;
   int compute_q = 0;
   int  trace_opt = 0;

   double q_error =-100000.;

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
                  "Number of times to refine the mesh uniformly");
   args.AddOption(&compute_q, "-cq", "--cq",
                  "compute q = grad(u) or not, 1 use L2 projection of grad(u), 2 use DPG, 0 by default do not compute it ");
   args.AddOption(&trace_opt, "-trace", "--trace",
                  "trace_order = trial_order, by default trace_order = trial_order - 1 ");


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
   int serial_ref_levels = min(ref_levels, 5);
   for (int l = 0; l < serial_ref_levels; l++)
   {
      serial_mesh->UniformRefinement();
   }

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


   // 6. Set up the linear form F(.) which corresponds to the right-hand side of
   //    the FEM linear system, which in this case is (f,phi_i) where f=1.0 and
   //    phi_i are the basis functions in the test finite element fespace.
   ConstantCoefficient one(1.0);          /* coefficients */
   FunctionCoefficient f_coeff( f_exact );/* coefficients */
   FunctionCoefficient u_coeff( u_exact );/* coefficients */

   ParLinearForm * F= new ParLinearForm(test_space);
   F->AddDomainIntegrator(new DomainLFIntegrator(f_coeff));
   F->Assemble();


   if(myid==0){
		cout<<endl<<"Righthandside assembled"<<endl;
   }

   // 6. Deal with boundary conditions
   Array<int> ess_bdr(mesh->bdr_attributes.Max());
   ess_bdr = 1;
   Array<int> ess_dof;
   x0_space->GetEssentialVDofs(ess_bdr, ess_dof);

   ParGridFunction * x0 = new ParGridFunction(x0_space);
   x0->MakeTRef(x0_space, x.GetBlock(x0_var) );
   x0->ProjectCoefficient(u_coeff);

   // 7. Set up the mixed bilinear form for the primal trial unknowns, B0,
   //    the mixed bilinear form for the interfacial unknowns, Bhat,
   //    the inverse stiffness matrix on the discontinuous test space, Sinv,
   //    and the stiffness matrix on the continuous trial space, S0.
   
   /* diffusion integrator (trial,test) */
   ParMixedBilinearForm *B0 = new ParMixedBilinearForm(x0_space,test_space);
   B0->AddDomainIntegrator(new DiffusionIntegrator(one));
   B0->Assemble();
   B0->EliminateEssentialBCFromTrialDofs(ess_dof, *x0, *F);
   B0->Finalize();

   /* trace terms */
   ParMixedBilinearForm *Bhat = new ParMixedBilinearForm(xhat_space,test_space);
   Bhat->AddTraceFaceIntegrator(new TraceJumpIntegrator());
   Bhat->Assemble();
   Bhat->Finalize();

   /* mass matrix corresponding to the test norm */
   ParBilinearForm *Sinv = new ParBilinearForm(test_space);
   SumIntegrator *Sum = new SumIntegrator;
   Sum->AddIntegrator(new DiffusionIntegrator(one));
   Sum->AddIntegrator(new MassIntegrator(one));
   Sinv->AddDomainIntegrator(new InverseIntegrator(Sum));
   Sinv->Assemble();
   Sinv->Finalize();

   /* diffusion integrator in trial space */
   ParBilinearForm *S0 = new ParBilinearForm(x0_space);
   S0->AddDomainIntegrator(new DiffusionIntegrator(one));
   S0->Assemble();
   S0->EliminateEssentialBC(ess_bdr);
   S0->Finalize();

   HypreParMatrix * matB0   = B0->ParallelAssemble();    delete B0;
   HypreParMatrix * matBhat = Bhat->ParallelAssemble();  delete Bhat;
   HypreParMatrix * matSinv = Sinv->ParallelAssemble();  delete Sinv;
   HypreParMatrix * matS0   = S0->ParallelAssemble();    delete S0;

   cout<< "B0:   "<<myid<<" "<<matB0->Height() <<" X "<<matB0->Width()<<endl
       << "Bhat: "<<myid<<" "<<matBhat->Height() <<" X "<<matBhat->Width()<<endl
	   << "Sinv: "<<myid<<" "<<matSinv->Height()<< " X "<<matSinv->Width()<<endl
	   <<endl;
   // 8. Set up the 1x2 block Least Squares DPG operator, B = [B0  Bhat],
   //    the normal equation operator, A = B^t Sinv B, and
   //    the normal equation right-hand-size, b = B^t Sinv F.
   BlockOperator B(offsets_test, offsets);
   B.SetBlock(0,0,matB0);
   B.SetBlock(0,1,matBhat);
   RAPOperator A(B, *matSinv, B);

   cout<<"rank "<<myid<<" operator assembled "<<endl;

   HypreParVector *VecF = F->ParallelAssemble();
   {
      HypreParVector SinvF(test_space);
      matSinv->Mult(*VecF,SinvF);
      B.MultTranspose(SinvF, b);
   }
   if(myid==0){
		cout<<endl<<"Right Handside calculated"<<endl;
   }

   // 9. Set up a block-diagonal preconditioner for the 2x2 normal equation
   //
   //        [ S0^{-1}     0     ]
   //        [   0     Shat^{-1} ]      Shat = (Bhat^T Sinv Bhat)
   //
   //    corresponding to the primal (x0) and interfacial (xhat) unknowns.
   HypreBoomerAMG *S0inv = new HypreBoomerAMG(*matS0);
   S0inv->SetPrintLevel(0);

   HypreParMatrix *Shat = RAP(matSinv, matBhat);
   HypreSolver *Shatinv;
   if (dim == 2) { Shatinv = new HypreAMS(*Shat, xhat_space); }
   else          { Shatinv = new HypreADS(*Shat, xhat_space); }

   BlockDiagonalPreconditioner P(offsets);
   P.SetDiagonalBlock(0, S0inv);
   P.SetDiagonalBlock(1, Shatinv);

   // 10. Solve the normal equation system using the PCG iterative solver.
   //     Check the weighted norm of residual for the DPG least square problem.
   //     Wrap the primal variable in a GridFunction for visualization purposes.
   CGSolver pcg(MPI_COMM_WORLD);
   pcg.SetOperator(A);
   pcg.SetPreconditioner(P);
   pcg.SetRelTol(1e-7);
   pcg.SetMaxIter(800);
   pcg.SetPrintLevel(1);
   pcg.Mult(b, x);

   {
      HypreParVector LSres(test_space), tmp(test_space);
      B.Mult(x, LSres);
      LSres -= *VecF;
      matSinv->Mult(LSres, tmp);
      double res = sqrt(InnerProduct(LSres, tmp));
      if (myid == 0)
      {
         cout << "\n|| B0*x0 + Bhat*xhat - F ||_{S^-1} = " << res << endl;
      }
   }
//   x0->Distribute(x.GetBlock(x0_var));
  // 10.5: Compute Grad terms based on the solution x
  // Compute throught DPG formulation
  // (q,\tau) - (\grad u,\tau ) = 0

   if( compute_q ){
	   FiniteElementCollection * q0_fec = new L2_FECollection(trial_order, dim);
	   
	   ParFiniteElementSpace * q0_space    = new ParFiniteElementSpace(mesh, q0_fec, dim);
	   ParFiniteElementSpace * vtest_space = new ParFiniteElementSpace(mesh, test_fec, dim);
	
	   ParGridFunction q0(q0_space);
	   q0 = 0.; /* never forget to initialize */

	   if(compute_q == 2){
		   if(myid == 0){
				cout<<" project grad u to optimal test space"<<endl;
		   }
		   /* Bilinear Form */
		   ParBilinearForm * Vinv = new ParBilinearForm(vtest_space);
		   Vinv->AddDomainIntegrator( new InverseIntegrator
		      							(new VectorMassIntegrator() ) );
		   Vinv->Assemble();
		   Vinv->Finalize();
		
		   /* -(\grad u,\tau ) */
		   ParMixedBilinearForm * B_ne_grad_u = new ParMixedBilinearForm(x0_space, vtest_space);
		   B_ne_grad_u->AddDomainIntegrator( new TransposeIntegrator
													(new DGVectorWeakDivergenceIntegrator() )  );
		   B_ne_grad_u->Assemble();
		   B_ne_grad_u->Finalize();
		   B_ne_grad_u->SpMat() *= -1.;
		
		   /* (q,tau) */
		   ParMixedBilinearForm * B_mass_q = new ParMixedBilinearForm(q0_space, vtest_space);
		   B_mass_q->AddDomainIntegrator( new VectorMassIntegrator() );
		   B_mass_q->Assemble();
		   B_mass_q->Finalize();
		
		   HypreParMatrix *matB_mass_q    = B_mass_q->ParallelAssemble(); //delete B_mass_q;
		   HypreParMatrix *matB_Vinv      = Vinv->ParallelAssemble(); //delete Vinv;
		   HypreParMatrix *matB_grad_u = B_ne_grad_u->ParallelAssemble(); //delete B_ne_grad_u;
		
//		   RAPOperator AQ( *matB_mass_q, *matB_Vinv, *matB_mass_q);
		   HypreParMatrix *AQ = RAP(matB_Vinv, matB_mass_q);
		   HypreParMatrix *GU = RAP(matB_mass_q, matB_Vinv, matB_grad_u);
		   RAPOperator GU2( *matB_mass_q, *matB_Vinv, *matB_grad_u);
		
		   HypreParVector rhs_q(q0_space );

		   /* this way not working */
//		   HypreParVector rhs_q2(q0_space );
//		   GU->Mult( *x0, rhs_q);
//		   GU2.Mult( *x0, rhs_q2);

		   HypreParVector tmp_q1(vtest_space );
		   HypreParVector tmp_q2(vtest_space );

		   B_ne_grad_u->Mult(*x0, tmp_q1);
		   Vinv->Mult(tmp_q1, tmp_q2);
		   B_mass_q->MultTranspose(tmp_q2,rhs_q);


			/* debug */
//		    VectorFunctionCoefficient q_coeff(dim, q_exact);
//			ParGridFunction q00(q0_space);
//		    q00.ProjectCoefficient(q_coeff);
//		    HypreParVector ex_q(q0_space);
//			AQ->Mult( q00, ex_q);
//			HypreParVector com_q(q0_space);
//			subtract( ex_q,rhs_q, com_q);
//			cout<<endl<<"rank "<<myid<<": "<<com_q.Norml2()<<endl;
//			cout<<endl<<endl;
//			subtract( rhs_q2,rhs_q, com_q);
//			cout<<endl<<"different way rank "<<myid<<": "<<com_q.Norml2()<<endl;
		
		   HypreBoomerAMG *MQ = new HypreBoomerAMG(*AQ);
		   MQ->SetPrintLevel(0);

		   CGSolver qcg(MPI_COMM_WORLD);
		   qcg.SetOperator(*AQ);
		   qcg.SetPreconditioner(*MQ);
		   qcg.SetRelTol(1e-10);
		   qcg.SetMaxIter(200);
		   qcg.SetPrintLevel(1);
		   qcg.Mult(rhs_q, q0);

		   delete AQ;
		   delete MQ;
		   delete Vinv;
		   delete B_mass_q;
		   delete B_ne_grad_u;
		}	
	   else{
		    cout<<"L2 projection of grad u"<<endl;

			ParBilinearForm *mass_q = new ParBilinearForm(q0_space);
	   		mass_q->AddDomainIntegrator( new VectorMassIntegrator() );
	   		mass_q->Assemble();
	   		mass_q->Finalize();
	
	   		ParMixedBilinearForm * grad_u = new ParMixedBilinearForm(x0_space, q0_space);
	   		grad_u->AddDomainIntegrator( new TransposeIntegrator
	   		 								(new DGVectorWeakDivergenceIntegrator() )    );
	   		grad_u->Assemble();
	   		grad_u->Finalize();
	   		grad_u->SpMat() *= -1.;
			grad_u->ParallelAssemble();

		    HypreParVector rhs_q(q0_space );

			/* not correct */
//			HypreParMatrix * mat_grad_u = grad_u->ParallelAssemble(); //delete grad_u;
//	   		mat_grad_u->Mult(*x0, rhs_q); 

			grad_u->Mult(*x0,rhs_q);


	   		HypreParMatrix *AQ = mass_q->ParallelAssemble(); delete mass_q;
		    HypreBoomerAMG *MQ = new HypreBoomerAMG(*AQ);
			MQ->SetPrintLevel(0);

			/* debug */
//			HypreParMatrix * mat_grad_u = grad_u->ParallelAssemble(); //delete grad_u;
//			HypreParVector debug_q1(q0_space);
//			HypreParVector debug_q2(q0_space);
//	   		mat_grad_u->Mult(*x0, debug_q1); 
//			grad_u->TrueAddMult(*x0,debug_q2);
//			HypreParVector com_q(q0_space);
//			subtract( debug_q1,debug_q2, com_q);
//			cout<<endl<<"rank "<<myid<<": "<<com_q.Norml2()<<endl;
			
			/* debug */
//		    VectorFunctionCoefficient q_coeff(dim, q_exact);
//			ParGridFunction q00(q0_space);
//		    q00.ProjectCoefficient(q_coeff);
//		    HypreParVector ex_q(q0_space);
//			AQ->Mult( q00, ex_q);
//			HypreParVector com_q(q0_space);
//			subtract( ex_q,rhs_q, com_q);
//			cout<<endl<<"rank "<<myid<<": "<<com_q.Norml2()<<endl;


			CGSolver qcg(MPI_COMM_WORLD);
			qcg.SetRelTol(1e-6);
			qcg.SetMaxIter(200);
			qcg.SetPrintLevel(0);
			qcg.SetOperator( *AQ);
			qcg.SetPreconditioner(*MQ);
			qcg.Mult( rhs_q, q0);

			delete AQ;
			delete MQ;
			delete grad_u;
	   }
	   VectorFunctionCoefficient q_coeff(dim, q_exact);
//	   q0.ProjectCoefficient(q_coeff);
	   q_error = q0.ComputeL2Error(q_coeff);
	}	

/*******************************************************************************/
   // 10.75 error 

   cout << "\nelement number of the mesh: "<< mesh->GetNE ()<<endl; 

   double x_error = x0->ComputeL2Error(u_coeff);
   ParGridFunction u_l2(x0_space);
   if(myid==0){
	   cout<< "\n dimension: "<<dim<<endl;
	   printf("\n|| u_h - u ||_{L^2} =%e\n\n",x_error );
	   if(compute_q){
		   printf("\n|| q_h - q ||_{L^2} =%e\n\n",q_error );
	   }
   }





   // 11. Save the refined mesh and the solution. This output can be viewed
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

   // 12. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << num_procs << " " << myid << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n" << *mesh << * x0 << flush;
   }

   // 13. Free the used memory.
   delete VecF;
   delete S0inv;
   delete Shatinv;
   delete Shat;
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

   MPI_Finalize();

   return 0;
}


/* define the source term on the right hand side */
// The right hand side
//  - u'' = f
double f_exact(const Vector & x){
	if(x.Size() == 2){
		return 8.*M_PI*M_PI* sin(2.*M_PI*x(0) ) * sin(2.*M_PI*x(1) );/* HDG */

		return 2*M_PI*M_PI*sin(M_PI*x(0) ) * sin(M_PI*x(1) );
//		double yy = x(1) - 0.5;
//		return   2*alpha_pzc*alpha_pzc*alpha_pzc*yy/
//				(1+alpha_pzc*alpha_pzc*yy*yy )/
//				(1+alpha_pzc*alpha_pzc*yy*yy );
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
		return  1. + x(0) + sin(2.*M_PI*x(0) ) * sin(2.* M_PI * x(1) ); /* HDG */
//		return  sin(M_PI*x(0) ) * sin( M_PI * x(1) ); /* first index is 0 */
	}
	else if(x.Size() == 1){
//		return atan(alpha_pzc * x(0) );
		return sin(2. * M_PI* x(0) ) ;
	}
	else{
		return 0;
	}

}

/* grad exact solution */
void q_exact( const Vector & x, Vector & f){
	if(x.Size() == 2){
		f(0) = +1. + 2.*M_PI*cos(2.*M_PI*x(0) ) * sin(2.*M_PI*x(1) );/* HDG */
		f(1) =     + 2.*M_PI*sin(2.*M_PI*x(0) ) * cos(2.*M_PI*x(1) );/* HDG */

//		f(0) = M_PI*cos(M_PI*x(0) ) * sin( M_PI* x(1) );
//		f(1) = M_PI*sin(M_PI*x(0) ) * cos( M_PI* x(1) );

//		f(0) = 0.;
//		f(1) = alpha_pzc/( 1. + alpha_pzc*alpha_pzc * (x(1)-0.5) * (x(1)-0.5) );
		
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
