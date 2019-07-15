#include "mfem.hpp"
#include "nonlinear_gs_integrator.hpp"
#include "test_RHSCoefficient.hpp"
#include <fstream>
#include <iostream>

using namespace mfem;
using namespace std;

double u_exact(const Vector &x);
double f_exact(const Vector &x);
double f(double u);

int main(int argc, char * argv[])
{
   // 1. Initialize MPI. Parse command-line options
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   // 2. Parse command-line options.
   const char * mesh_file="../../data/inline-quad.mesh";
   int order = 1;
   int ref_levels = -1;
   bool static_cond = false;
   bool pa = false;
   bool visualization = true;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&ref_levels, "-r", "--refine levels","how many refine level we apply");

   args.Parse();

   if (!args.Good())
   {
	  if(myid == 0){
        args.PrintUsage(cout);
	  }
	  MPI_Finalize();
	  return 1;
   }
   if(myid == 0){
		args.PrintOptions(cout);
	}
   // 3. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral, hexahedral, surface and volume meshes with
   //    the same code.
   Mesh *serial_mesh = new Mesh(mesh_file, 1, 1);
   int dim = serial_mesh->Dimension();
   int sdim = serial_mesh->SpaceDimension(); /* pzc:space dimension */
   if(myid==0){
		cout<<endl << "dimensiion: "<<dim<<endl;
	}
   // 4. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement. We choose 'ref_levels' to be the
   //    largest number that gives a final mesh with no more than 50,000
   //    elements.
   int sref_levels = ref_levels;
   {
	 if(ref_levels<0){
		if(dim >=2){
	  	  ref_levels =
	  	  	 (int)floor(log(1000./serial_mesh->GetNE())/log(2.)/dim);
	  	}
	  	else if(dim==1){
	  	  ref_levels =
	  	  	 (int)floor(log(500./serial_mesh->GetNE())/log(2.)/dim);
	  	}
	 }
	 if(sref_levels>=4){
		sref_levels = ref_levels - 2;
	 }
      for (int l = 0; l < sref_levels; l++)
      {
         serial_mesh->UniformRefinement();
      }
   }
   if(myid==0){
		cout<<endl << "mesh assebmled: "<<dim<<endl;
	}

   // 5. Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution. Once the
   //    parallel mesh is defined, the serial mesh can be deleted.
   ParMesh * mesh = new ParMesh(MPI_COMM_WORLD, *serial_mesh);
   delete serial_mesh;
   {
	   int par_ref_levels = ref_levels - sref_levels;
      for (int l = 0; l < par_ref_levels; l++)
      {
         mesh->UniformRefinement();
      }

   }
   if(myid==0){
		cout<<endl << "parallel mesh assebmled: "<<dim<<endl;
	}
   // 6. Define finite element space
   FiniteElementCollection * fec;
   fec = new L2_FECollection(order,dim);

   ParFiniteElementSpace *fespace = new ParFiniteElementSpace(mesh, fec);
   
   /* define grid functions */
   ParGridFunction u;
   ParGridFunction fu;

   Vector U(fespace->TrueVSize() );
   U = 0.;
   Vector FU(fespace->TrueVSize() );
   FU = 0.;

   u.MakeTRef(fespace, U, 0);
   u.SetFromTrueVector();
   fu.MakeTRef(fespace, FU, 0);
   fu.SetFromTrueVector();

   FunctionCoefficient u_coefficient(u_exact);
   FunctionCoefficient fu_coefficient(f_exact);

   u.ProjectCoefficient(u_coefficient);
   fu.ProjectCoefficient(fu_coefficient);

   /* define Bilinear forms */
   ParBilinearForm * mass_u = new ParBilinearForm(fespace);
   mass_u->AddDomainIntegrator( new MassIntegrator() );
   mass_u->Assemble();
   mass_u->Finalize();
   HypreParMatrix * mat_m = mass_u->ParallelAssemble();

   ParBilinearForm * mass_fu = new ParBilinearForm(fespace);
   mass_fu->AddDomainIntegrator( new FUIntegrator( u_coefficient, &f ) );
   mass_fu->Assemble();
   mass_fu->Finalize();
   HypreParMatrix * mat_mf = mass_fu->ParallelAssemble();


   Vector MF_U(FU.Size() );
   Vector M_FU(FU.Size() );

   mat_mf->Mult(u, MF_U);
   mat_m->Mult(fu, M_FU);

   Vector diff(FU.Size() );
   subtract(MF_U,M_FU,diff);

   /* linearform */
   u.ProjectCoefficient(u_coefficient);
   RHSCoefficient lfu_coeff(&u);
   ParLinearForm * lf_u = new ParLinearForm(fespace);
   lf_u->AddDomainIntegrator(new DomainLFIntegrator(lfu_coeff) );
   lf_u->Assemble();
   Vector LF_U(FU.Size() );
   lf_u->ParallelAssemble( LF_U );


//   for(int i = 0;i<u.Size();i++){
//		cout<<i<<": "<<"u "<<u(i)<<" fu "<<fu(i)<<endl
//		       <<"  "<<"U "<<U(i)<<" FU "<<FU(i)<<endl
//			   <<endl;
//   }
//
//   for(int i = 0;i<res1.Size();i++){
//		cout<<i<<": "<<"MF*u "<<res1(i)<<" M*fu "<<res2(i)<<endl;
//   }

   // 14. LinearSolve 
//   GMRESSolver  pcg(MPI_COMM_WORLD);
   CGSolver  pcg(MPI_COMM_WORLD);
   pcg.SetOperator(*mat_m);
   HypreBoomerAMG * prec = new HypreBoomerAMG(*mat_m);
   prec->SetPrintLevel(0);
   pcg.SetPreconditioner( *prec );
   pcg.SetRelTol(1e-20);
   pcg.SetMaxIter(2000);
   pcg.SetPrintLevel(1);
   FU = 0.;
   pcg.Mult(LF_U,FU);
//   pcg.Mult(MF_U,FU);

   fu.Distribute(FU);

   double error_l2 = fu.ComputeL2Error(fu_coefficient);
   if(myid == 0){
		printf("\n|| fu_h - f(u) ||_{L^2} = %e \n",error_l2);
	}
   printf("rank %d difference: %e\n",myid,diff.Norml2() );

   /* plot */
   // 15. Save the refined mesh and the solution in parallel. This output can
   //     be viewed later using GLVis: "glvis -np <np> -m mesh -g sol".
   {
      ostringstream mesh_name, sol_name;
      mesh_name << "mesh." << setfill('0') << setw(6) << myid;
      sol_name << "sol." << setfill('0') << setw(6) << myid;

      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      mesh->Print(mesh_ofs);

      ofstream sol_ofs(sol_name.str().c_str());
      sol_ofs.precision(8);
      u.Save(sol_ofs);
   }

   // 16. Send the solution by socket to a GLVis server.
//   fu.ProjectCoefficient(fu_coefficient);
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << num_procs << " " << myid << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n" << *mesh << fu << flush;
   }
    
   /* finalize the code */
   delete mass_u;
   delete mass_fu;
   delete mat_mf;
   delete mat_m;
   delete fespace;
   delete mesh;
   MPI_Finalize();


}

double u_exact(const Vector &x)
{
	if(x.Size() == 2){
		return sin(4*M_PI*x(0) )*sin(4*M_PI*x(1) );
//		return 1;
		return x(0)*x(1);
	}
	else{
		return 0.;
	}
}

double f(double u)
{
//	return u;
	return u*u;
}

double f_exact(const Vector &x)
{
	if(x.Size() == 2){
		double u;
		u = u_exact(x);
		return f(u);
	}
	else{
		return 0.;
	}
}


