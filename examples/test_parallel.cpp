// 2019-06-19
// pzc: a parallel test to understand the difference between the true degree of freedoms 
// and the degree of freedoms

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

double f2(const Vector & x) { 
	return 2.345 * x[0] + 3.579 * x[1]; 
}
void Grad_f2(const Vector & x, Vector & df)
{
   df.SetSize(2);
   df[0] = 2.345;
   df[1] = 3.579;

}

void F2_pzc(const Vector &x, Vector & v)
{
	v.SetSize(2);
	v[0] = 1.234 * x[0]*x[0] + 2.232 * x[1];
    v[1] = 3.572 * x[0]*x[1] + 3.305 * x[1] * x[1];

}

double DivF2_pzc(const Vector & x)
{
	return (2.468+3.572)*x[0] + 6.610 * x[1];
}


double f3(const Vector & x) { return x[0]*(2.-x[0]) *  x[1]*(3.-x[1]); }
void Grad_f3(const Vector & x, Vector & df)
{
   df.SetSize(2);
   df[0] = (2.-2.*x[0] ) *x[1]*(3.-x[1]);
   df[1] = (3.-2.*x[1] ) *x[0]*(2.-x[0]);
}

int main(int argc, char * argv[])
{
	// 1. initialize MPI
	int num_procs, myid;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);

	//2. Parse command-line options
	const char * mesh_file = "../data/inline-quad.mesh";
	int order = 2;
    int ref_levels = -1;

    OptionsParser args(argc, argv);
    args.AddOption(&mesh_file, "-m", "--mesh",
     			  "Mesh file to use.");
    args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly");
    args.AddOption(&order, "-o", "--order",
                  "order of polynomials");

    args.Parse();
    if (!args.Good())
    {
       args.PrintUsage(cout);
       return 1;
    }
    args.PrintOptions(cout);

	// 2. Mesh
   Mesh *serial_mesh = new Mesh(mesh_file, 1, 1);
   int dim = serial_mesh->Dimension();

   // 2a. serial refinement
   if (ref_levels < 0)
   {
      ref_levels = (int)floor(log(50000./serial_mesh->GetNE())/log(2.)/dim);
   }
   int serial_ref_levels = min(ref_levels, 5);
   for (int l = 0; l < serial_ref_levels; l++)
   {
      serial_mesh->UniformRefinement();
   }
   // 2b. Parioning the mesh on each processor and do the refinement
   ParMesh *mesh = new ParMesh(MPI_COMM_WORLD, *serial_mesh);
   delete serial_mesh;
   int par_ref_levels = ref_levels - serial_ref_levels;
   for( int l=0; l<par_ref_levels; l++){
		mesh->UniformRefinement();
   }
   mesh->ReorientTetMesh(); /* what is this? */

   // 3. Finite element space
   FiniteElementCollection * dg_fec;
   dg_fec = new L2_FECollection(order,dim);

   ParFiniteElementSpace * u_space = new ParFiniteElementSpace(mesh, dg_fec,dim);
   ParFiniteElementSpace * div_space = new ParFiniteElementSpace(mesh, dg_fec);

   ParGridFunction u( u_space );
   ParGridFunction u_div(div_space);

   ParGridFunction v( div_space );
   ParGridFunction v_grad(u_space);

   VectorFunctionCoefficient u_coeff( dim,F2_pzc ); 
   FunctionCoefficient div_coeff( DivF2_pzc );

   VectorFunctionCoefficient grad_coeff( dim,Grad_f3 ); 
   FunctionCoefficient v_coeff(f3 );

   u.ProjectCoefficient(u_coeff);
   u_div.ProjectCoefficient(div_coeff);

   v.ProjectCoefficient(v_coeff);
   v_grad.ProjectCoefficient(grad_coeff);

   // 4. Bilinear form
   ParMixedBilinearForm * blf_div( new ParMixedBilinearForm( u_space, div_space) );
   blf_div->AddDomainIntegrator( new VectorDivergenceIntegrator() );
   blf_div->Assemble();
   blf_div->Finalize();

   ParBilinearForm * blf_mass( new ParBilinearForm( div_space )  );
   blf_mass->AddDomainIntegrator( new MassIntegrator() );
   blf_mass->Assemble();
   blf_mass->Finalize();

   HypreParMatrix * mat_div  = blf_div->ParallelAssemble(); 
   HypreParMatrix * mat_mass = blf_mass->ParallelAssemble();

   ParMixedBilinearForm * blf_grad( new ParMixedBilinearForm( u_space, div_space) );
   blf_grad->AddDomainIntegrator( new DGVectorWeakDivergenceIntegrator() );
   blf_grad->Assemble();
   blf_grad->Finalize();

   ParBilinearForm * blf_v_mass( new ParBilinearForm(u_space) );
   blf_v_mass->AddDomainIntegrator( new VectorMassIntegrator() );
   blf_v_mass->Assemble();
   blf_v_mass->Finalize();

   HypreParMatrix * mat_grad  = blf_grad->ParallelAssemble(); 
   HypreParMatrix * mat_v_mass = blf_v_mass->ParallelAssemble();
  
   // 5. compare the results
   HypreParVector res1(div_space),res2(div_space),
				  res3(div_space),res4(div_space),compare(div_space);
   blf_div->Mult(u,res1);
   blf_mass->Mult(u_div,res2);

   mat_mass->Mult(u_div,res3);
   mat_div->Mult(u,res4);

   subtract(res1,res2,compare);
   cout<<endl<<"compare result of blf_div, blf_mass mult: rank "<< myid <<" "<<compare.Norml2()<<endl
	   <<endl;

   subtract(res1,res3,compare);
   cout<<endl<<"compare blf_div->mult and mat_mass->mult: rank "<< myid <<" "<<compare.Norml2()<<endl
	   <<endl;

   subtract(res4,res3,compare);
   cout<<endl<<"compare mat_div->mult and mat_mass->mult: rank "<< myid <<" "<<compare.Norml2()<<endl
	   <<endl;
   /***********************/
   HypreParVector v_res1(u_space),v_res2(u_space),
				  v_res3(u_space),v_res4(u_space),
				  v_res5(u_space),v_res6(u_space),
				  v_compare(u_space);
   blf_v_mass->Mult(v_grad,v_res1);
   blf_grad->MultTranspose(v,v_res2);

   mat_v_mass->Mult(v_grad,v_res3);
   mat_grad->MultTranspose(v,v_res4);

   blf_v_mass->TrueAddMult(v_grad,v_res5);

   add(v_res1,v_res2,v_compare);
   cout<<endl<<" add bewteen blf grad: rank "<<myid<<" "<< v_compare.Norml2()<<endl
	   <<endl;

   add(v_res3,v_res4,v_compare);
   cout<<endl<<" add bewteen mat_v_mass and mat_grad: rank "<<myid<<" "<< v_compare.Norml2()<<endl
	   <<endl;

   add(v_res1,v_res4,v_compare);
   cout<<endl<<" add bewteen blf_v_mass and mat_grad: rank "<<myid<<" "<< v_compare.Norml2()<<endl
	   <<endl;


   subtract(v_res1,v_res3,v_compare);
   cout<<endl<<" subtract bewteen blf_v_mass  and mat_v_mass: rank "<<myid<<" "<< v_compare.Norml2()<<endl
	   <<endl;

   subtract(v_res2,v_res4,v_compare);
   cout<<endl<<" subtract bewteen blf_v_grad  and mat_v_grad: rank "<<myid<<" "<< v_compare.Norml2()<<endl
	   <<endl;

   subtract(v_res5,v_res1,v_compare);
   cout<<endl<<" subtract bewteen  blf ture and normal: rank "<<myid<<" "<< v_compare.Norml2()<<endl
	   <<endl;



   // 6. Free memory 
   delete mat_grad;
   delete mat_v_mass;
   delete mat_div;
   delete mat_mass;
   delete blf_mass;
   delete blf_div;
   delete blf_grad;
   delete blf_v_mass;
   delete u_space;
   delete div_space;

   MPI_Finalize();


}
