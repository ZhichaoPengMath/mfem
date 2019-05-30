//			ldg_pzc

//
//
// Description: this code solving 
//					-\laplace u = f
//				using ldg method, rewrite the whole system as a 1st order one:
//					q - grad u =0
//					div(q)     = -f
//
// The code is based on ex5.cpp, ex18.cpp
//
#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>

using namespace std;
using namespace mfem;

double alpha_pzc = 1.;/* used in atan(alpha * x ) */

void function_zero(const Vector & x, Vector & f);

int main(int argc, char *argv[]){
/* 1. set up options */
	const char *mesh_file = "../data/inline-segment.mesh";
	int order = 1;
	int ref_levels = -1; /* refine levels */
	bool visualization = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly, -1 for auto.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }

   args.AddOption(&alpha_pzc, "-alpha", "--alpha",
                  "arctan( alpha * x) as exact solution");

   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);
   // 2. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral and hexahedral meshes with the same code.
   //    NURBS meshes are projected to second order meshes.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension(); /* space dimension */

   // 3. mesh refinement
   if(ref_levels<0){
	   ref_levels = (int)floor( log(10000./mesh->GetNE() )/log(2.) / dim );
   }
   for(int level = 0;level<ref_levels; ++level){
		mesh->UniformRefinement();
   }

   cout<<"\n dimension: "<<dim<<" element number: "<<mesh->GetNE()<<endl;

   //4. Define the finite element space,
   //   DG finite element space
   FiniteElementCollection *dg_coll( new DG_FECollection(order,dim) );
	/* finite element space for the scalar variable */
   FiniteElementSpace * u_space = new FiniteElementSpace(mesh, dg_coll);
   /* finite element space for the mesh-dim vector unkwon */
   FiniteElementSpace * q_space = new FiniteElementSpace(mesh, dg_coll, dim, Ordering::byNODES);
   
   //5. Define the block structure of the problem
   //	array offsets for each variable
   Array<int> block_offsets(3); /* numer of variable + 1 */
   block_offsets[0] = 0;
   block_offsets[1] = u_space->GetVSize();
   block_offsets[2] = q_space->GetVSize();
   block_offsets.PartialSum();

   std::cout << "***********************************************************\n";
   std::cout << "dim for U = " << block_offsets[1] - block_offsets[0] << "\n";
   std::cout << "dim for grad term Q = " << block_offsets[2] - block_offsets[1] << "\n";
   std::cout << "dim(U+Q) = " << block_offsets.Last() << "\n";
   std::cout << "***********************************************************\n";

   // 6. Define the blocked grid function 
   // (u,q) and linear form (uform, qform)
   BlockVector x(block_offsets), rhs(block_offsets);


   ConstantCoefficient one(1.);
   VectorFunctionCoefficient f_zero( dim, function_zero );

   LinearForm * uform(new LinearForm);
   uform->Update(u_space, rhs.GetBlock(0), 0); /* the last number 0, is the starting position */
   uform->AddDomainIntegrator(new DomainLFIntegrator( one ) );
   uform->Assemble();

   LinearForm * qform(new LinearForm);
   qform->Update(q_space, rhs.GetBlock(1), 0);
   qform->AddDomainIntegrator(new VectorDomainLFIntegrator( f_zero ) );
   qform->Assemble();

   // 7. Set up the linear form b(.)   
   //			S= [ 0     B
   //				-B^T   M]
   //	where B(q,v) = -(q, grad v)  + \hat{q} [v]
   //		  C(u,tau) = -(u, div tau) + \hat{u} [au]
   //		  M(q,v) = (q,tau) 
   BilinearForm *mass_operator( new BilinearForm(q_space) );
//   BilinearForm *b_operator( new BilinearForm(q_space) );
   MixedBilinearForm *b_operator( new MixedBilinearForm(q_space, u_space) ); /* (trial space,test space) */

   /* assemble the L2 projection operator */
//   mass_operator->AddDomainIntegrator(new VectorFEMassIntegrator ); /* not support DG */
   mass_operator->AddDomainIntegrator(new VectorMassIntegrator );
   mass_operator->Assemble();
   mass_operator->Finalize();
   SparseMatrix &Mass(mass_operator->SpMat() );

   cout<<"size of mass matrix: "<< Mass.Height()<<" X "<<Mass.Width()<<endl;
  /* define b(*,*) */
   b_operator->AddDomainIntegrator(new VectorDivergenceIntegrator  );
//   b_operator->AddTraceFaceIntegrator(new DGElasticityIntegrator(0. ,0.) );// central flux
//   b_operator->AddInteriorFaceIntegrator(new DGElasticityIntegrator(0. ,0.) );// central flux
   b_operator->Assemble();
   b_operator->Finalize();
   SparseMatrix &B(b_operator->SpMat() );
//   SparseMatrix *NBT = Transpose(B); /* -B^T */
//   *NBT *= -1.;

   cout<<"size of B matrix: "<< B.Height()<<" X "<<B.Width()<<endl;

//   BlockMatrix DiffusionMatrix(block_offsets);



   // 11. error 
   
   /* visualization */
   // 12. Save the mesh and the solution. This output can be viewed later using
   //     GLVis: "glvis -m ldg_pzc.mesh -g sol_u.gf" or "glvis -m ldg_pzc.mesh -g
   //     sol_q.gf".
//    {
//       ofstream mesh_ofs("ex5.mesh");
//       mesh_ofs.precision(8);
//       mesh->Print(mesh_ofs);
// 
// //      ofstream u_ofs("sol_u.gf");
// //      u_ofs.precision(8);
// //      u.Save(u_ofs);
// //
// //      ofstream p_ofs("sol_p.gf");
// //      p_ofs.precision(8);
// //      p.Save(p_ofs);
//    }
// 
//    // 13. Save data in the VisIt format
// //   VisItDataCollection visit_dc("Example5", mesh);
// //   visit_dc.RegisterField("u", &u);
// //   visit_dc.RegisterField("grad ", &q);
// //   visit_dc.Save();
// 
//    // 14. Send the solution by socket to a GLVis server.
//    if (visualization)
//    {
//       char vishost[] = "localhost";
//       int  visport   = 19916;
//       socketstream u_sock(vishost, visport);
//       u_sock.precision(8);
//       u_sock << "solution\n" << *mesh << u << "window_title 'Velocity'" << endl;
// //      socketstream p_sock(vishost, visport);
// //      p_sock.precision(8);
// //      p_sock << "solution\n" << *mesh << p << "window_title 'Pressure'" << endl;
//    }
   /****************************************************/
	return 0;
}

/* zero function */
void function_zero(const Vector & x, Vector & f)
{
   f = 0.0;
}
