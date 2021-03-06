//                                MFEM Example 1
//
// Compile with: make ex1
//
// Sample runs:  ex1 -m ../data/square-disc.mesh
//               ex1 -m ../data/star.mesh
//               ex1 -m ../data/star-mixed.mesh
//               ex1 -m ../data/escher.mesh
//               ex1 -m ../data/fichera.mesh
//               ex1 -m ../data/fichera-mixed.mesh
//               ex1 -m ../data/toroid-wedge.mesh
//               ex1 -m ../data/square-disc-p2.vtk -o 2
//               ex1 -m ../data/square-disc-p3.mesh -o 3
//               ex1 -m ../data/square-disc-nurbs.mesh -o -1
//               ex1 -m ../data/star-mixed-p2.mesh -o 2
//               ex1 -m ../data/disc-nurbs.mesh -o -1
//               ex1 -m ../data/pipe-nurbs.mesh -o -1
//               ex1 -m ../data/fichera-mixed-p2.mesh -o 2
//               ex1 -m ../data/star-surf.mesh
//               ex1 -m ../data/square-disc-surf.mesh
//               ex1 -m ../data/inline-segment.mesh
//               ex1 -m ../data/amr-quad.mesh
//               ex1 -m ../data/amr-hex.mesh
//               ex1 -m ../data/fichera-amr.mesh
//               ex1 -m ../data/mobius-strip.mesh
//               ex1 -m ../data/mobius-strip.mesh -o -1 -sc
//
// Device sample runs:
//               ex1 -pa -d cuda
//               ex1 -pa -d raja-cuda
//               ex1 -pa -d occa-cuda
//               ex1 -pa -d raja-omp
//               ex1 -pa -d occa-omp
//               ex1 -m ../data/beam-hex.mesh -pa -d cuda
//
// Description:  This example code demonstrates the use of MFEM to define a
//               simple finite element discretization of the Laplace problem
//               -Delta u = 1 with homogeneous Dirichlet boundary conditions.
//               Specifically, we discretize using a FE space of the specified
//               order, or if order < 1 using an isoparametric/isogeometric
//               space (i.e. quadratic for quadratic curvilinear mesh, NURBS for
//               NURBS mesh, etc.)
//
//               The example highlights the use of mesh refinement, finite
//               element grid functions, as well as linear and bilinear forms
//               corresponding to the left-hand side and right-hand side of the
//               discrete linear system. We also cover the explicit elimination
//               of essential boundary conditions, static condensation, and the
//               optional connection to the GLVis tool for visualization.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

double f_exact(const Vector &);
double u_exact(const Vector &);

const double alpha_pzc = 100.;

int main(int argc, char *argv[])
{
   // 1. Initialize MPI
   int num_process, myid;
   MPI_Init(&argc,&argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_process);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   // 2. Parse command-line options.
   const char *mesh_file = "../data/xperiodic.mesh";
   int order = 1;
   int ref_levels = -1;
   bool static_cond = false;
   bool pa = false;
   const char *device = "cpu";
   bool visualization = true;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                  "--no-partial-assembly", "Enable Partial Assembly.");
   args.AddOption(&device, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                  "--no-partial-assembly", "Enable Partial Assembly.");
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


   // 6. Define a finite element space on the mesh. Here we use continuous
   //    Lagrange finite elements of the specified order. If order < 1, we
   //    instead use an isoparametric/isogeometric space.
   FiniteElementCollection *fec;
   if (order > 0)
   {
      fec = new H1_FECollection(order, dim);
   }
   else if (mesh->GetNodes())
   {
      fec = mesh->GetNodes()->OwnFEC();
      cout << "Using isoparametric FEs: " << fec->Name() << endl;
   }
   else
   {
      fec = new H1_FECollection(order = 1, dim);
   }
   ParFiniteElementSpace *fespace = new ParFiniteElementSpace(mesh, fec);
   if(myid == 0 ){
		cout<< " Finite element space assembled " << endl;
   }
   HYPRE_Int size = fespace->GlobalTrueVSize();
   if(myid == 0){
	    cout << "Number of finite element unknowns: "
  	         << size << endl;
	}
   // 7. Determine the list of true (i.e. conforming) essential boundary dofs.
   //    In this example, the boundary conditions are defined by marking all
   //    the boundary attributes from the mesh as essential (Dirichlet) and
   //    converting them to a list of true dofs.
   Array<int> ess_tdof_list;
   if (mesh->bdr_attributes.Size())
   {

      Array<int> ess_bdr(mesh->bdr_attributes.Max());
      ess_bdr = 1;
      fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   // 8. Set up the linear form b(.) which corresponds to the right-hand side of
   //    the FEM linear system, which in this case is (f,phi_i) where phi_i are
   //    the basis functions in the finite element fespace and f is given by function
   
//   VectorFunctionCoefficient f(sdim,f_exact); /* pzc: vector */
   ConstantCoefficient one(1.0); /* pzc: constant */
   FunctionCoefficient f(f_exact); /* pzc: scalar */
   ParLinearForm *b = new ParLinearForm(fespace);
   b->AddDomainIntegrator(new DomainLFIntegrator(f));
   b->Assemble();

   // 9. Define the solution vector x as a finite element grid function
   //    corresponding to fespace. Initialize x by projecting the exact
   //    solution. Note that only values from the boundary edges will be used
   //    when eliminating the non-homogeneous boundary condition to modify the
   //    r.h.s. vector b.
   /* handle none-homogeneous boundary */
   ParGridFunction x(fespace);
   FunctionCoefficient U(u_exact);
   x.ProjectCoefficient(U);

   // 10. Set up the bilinear form a(.,.) on the finite element space
   //    corresponding to the Laplacian operator -Delta, by adding the Diffusion
   //    domain integrator.
   ParBilinearForm *a = new ParBilinearForm(fespace);
   if (pa) { a->SetAssemblyLevel(AssemblyLevel::PARTIAL); } /* pa stands for the parallel or not */
   a->AddDomainIntegrator(new DiffusionIntegrator(one)); /* pzc: DiffusionIntegrator with a scalar coefficient */

   // 10. Assemble the bilinear form and the corresponding linear system,
   //     applying any necessary transformations such as: eliminating boundary
   //     conditions, applying conforming constraints for non-conforming AMR,
   //     static condensation, etc.
   if (static_cond) { a->EnableStaticCondensation(); }
   a->Assemble();

   OperatorPtr A;
   Vector B, X;
   a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B);

   cout << "Size of linear system: " << A->Height() << endl;


   // 11. Solve the linear system A X = B.
   //     * With full assembly, use the BoomerAMG preconditioner from hypre.
   //     * With partial assembly, use no preconditioner, for now.
   Solver *prec = NULL;
   if (!pa) { prec = new HypreBoomerAMG; }
   CGSolver cg(MPI_COMM_WORLD);
   cg.SetRelTol(1e-12);
   cg.SetMaxIter(2000);
   cg.SetPrintLevel(1);
   if (prec) { cg.SetPreconditioner(*prec); }
   cg.SetOperator(*A);
   cg.Mult(B, X);
   delete prec;

   // 12. Recover the solution as a finite element grid function.
   a->RecoverFEMSolution(X, *b, x);

   
   // 12b (pzc): compute and print L2 error */
	 cout << "\nelement number of the mesh: "<< mesh->GetNE ()<<endl; 
	 double error_l2 = x.ComputeL2Error(U);
	 if(myid == 0){
		printf("\n|| u_h - u ||_{L^2} = %e \n",error_l2);
	 }

   // 13. Switch back to the host.
   //Device::Disable();

   // 14. Save the refined mesh and the solution. This output can be viewed later
   //     using GLVis: "glvis -m refined.mesh -g sol.gf".
   {
      ostringstream mesh_name, sol_name;
      mesh_name << "mesh." << setfill('0') << setw(6) << myid;
      sol_name << "sol." << setfill('0') << setw(6) << myid;

      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      mesh->Print(mesh_ofs);

      ofstream sol_ofs(sol_name.str().c_str());
      sol_ofs.precision(8);
      x.Save(sol_ofs);
   }

   // 15. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << num_process << " " << myid << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n" << *mesh << x << flush;
   }

   // 16. Free the used memory.
   delete a;
   delete b;
   delete fespace;
   if (order > 0) { delete fec; }
   delete mesh;

   MPI_Finalize();

   return 0;
}


/* define the source term on the right hand side */
// The right hand side
//  - u'' = f
double f_exact(const Vector & x){
	if(x.Size() == 2){
//		return   2*alpha_pzc*alpha_pzc*alpha_pzc*x(1)/
//				(1+alpha_pzc*alpha_pzc*x(1)*x(1) )/
//				(1+alpha_pzc*alpha_pzc*x(1)*x(1) );
//		return -M_PI * M_PI *sin( M_PI*x(1) ); /* first index is 0 */
		return M_PI * M_PI * ( sin(M_PI*x(0) ) + sin( M_PI*x(1) ) ); /* first index is 0 */
	}
	else if(x.Size() == 1){
//		return   2*alpha_pzc*alpha_pzc*alpha_pzc*x(0)/
//				(1+alpha_pzc*alpha_pzc*x(0)*x(0) )/
//				(1+alpha_pzc*alpha_pzc*x(0)*x(0) );
		return 4.*M_PI*M_PI*cos( 2.*M_PI* x(0) ) ;
	}
	else{
		return 0;
	}

}

/* exact solution */
double u_exact(const Vector & x){
	if(x.Size() == 2){
//		return sin( M_PI * x(1) ); /* first index is 0 */
//		return atan(alpha_pzc * x(1) );
		return sin(M_PI* x(0) ) + sin( M_PI * x(1) ); /* first index is 0 */
	}
	else if(x.Size() == 1){
//		return atan(alpha_pzc * x(0) );
		return cos(2. * M_PI* x(0) ) ;
	}
	else{
		return 0;
	}

}
