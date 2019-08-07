// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

// Test Added by Zhichao Peng 06/02/2019, to make sure DPG integrators works well
#include "mfem.hpp"
#include "catch.hpp"
#include <fstream>

using namespace mfem;
using namespace std;

namespace bilininteg_pzc{

/*****************************************
 * Manufacture solutions for test
 * ***************************************/
double f2(const Vector & x) { 
	return 2.345 * x[0] + 3.579 * x[1]; 
}
void Grad_f2(const Vector & x, Vector & df)
{
   df.SetSize(2);
   df[0] = 2.345;
   df[1] = 3.579;

}

double f22(const Vector & x){
	return sin(x[0]) + sin(x[1]);
}

void Grad_f22(const Vector & x, Vector &df)
{
	df.SetSize(2);
   df[0] = cos(x[0]);
   df[1] = cos(x[1]);
}

double f3(const Vector & x) { return x[0]*(2.-x[0]) *  x[1]*(3.-x[1]); }
void Grad_f3(const Vector & x, Vector & df)
{
   df.SetSize(2);
   df[0] = (2.-2.*x[0] ) *x[1]*(3.-x[1]);
   df[1] = (3.-2.*x[1] ) *x[0]*(2.-x[0]);
}

double f4(const Vector & x) { return 1.; }
void Grad_f4(const Vector & x, Vector & df)
{
   df.SetSize(2);
   df[0] = 0.;
   df[1] = 0.;
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

void NGradDivF2_pzc(const Vector &x, Vector &v)
{
	v.SetSize(2);
	v[0] = 2.468 + 3.572;
	v[1] = 6.610;
	v *= -1.;
}

void NGradf2DivF2_pzc( const Vector &x, Vector &v)
{
	v.SetSize(2);
	NGradDivF2_pzc(x,v);
	v *= f2(x);

	Vector w1(2);
	Vector w2(2);
	Grad_f2(x,w1);
	w1 *= DivF2_pzc(x);

	v -= w1;
}

/*********************************
 * test 
 * *******************************/
TEST_CASE("Test this file is correctly linked",
		  "[HelloWorld]"
		)
{
	cout<<endl<<" hello world"<<endl;
} /* end of test */

TEST_CASE("Test for the DPG domain integrators",
		  "[DPGDomainIntegrator]"
		  "[pzc]"
		)
{
   cout<<endl<<"Test Domain Integrator"<<endl;

   int order = 2, m = 2, n = 1,  dim = 2;
   double cg_rtol = 1e-14;
   double tol = 1e-9;

   const char *mesh_file = "../../data/amr-quad.mesh";
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   /* mesh [0,2] \times [0,3] */
   cout<<endl<<" mesh formed "<<endl<<endl;


   /* define the finite element space */
//	L2_FECollection fec_l2(order,dim,1);
//	L2_FECollection s_fec_l2(order,dim,1);
	L2_FECollection fec_l2(order,dim);
	L2_FECollection s_fec_l2(order,dim);

//	FiniteElementSpace fespace_l2(mesh, &fec_l2,dim);
//	FiniteElementSpace fespace_scalar_l2(mesh,&s_fec_l2);

	FiniteElementSpace fespace_l2(mesh, &fec_l2,dim);
	FiniteElementSpace fespace_scalar_l2(mesh,&s_fec_l2);

	cout<<endl<<"finite elment space formed"<<endl;

	VectorFunctionCoefficient F2_coef_pzc(dim, F2_pzc);
	FunctionCoefficient DivF2_coef_pzc(DivF2_pzc);
	VectorFunctionCoefficient NGradDivF2_coef_pzc(dim, NGradDivF2_pzc);
	FunctionCoefficient f2_coef(f2);
	VectorFunctionCoefficient Grad_f2_coef(dim, Grad_f2);

	GridFunction f_l2(&fespace_l2); f_l2.ProjectCoefficient(F2_coef_pzc);
	GridFunction divf_l2(&fespace_scalar_l2); divf_l2.ProjectCoefficient(DivF2_coef_pzc);
	GridFunction gradf2_l2(&fespace_l2); gradf2_l2.ProjectCoefficient(Grad_f2_coef);
	GridFunction f2_l2(&fespace_scalar_l2); f2_l2.ProjectCoefficient(f2_coef);

    cout<<endl<<" coefficients obtained "<<endl;
	SECTION("Domain integrators: DGDivDivIntegrator") 
		/* ************************************************
		 * compare (div F2_pzc, div basis) calculated by 
		 * DGDivDivIntegrator(F2_pzc) and  
		 * Transpe os VectorDivergenceIntegrator(DivF2_pzc)
		 * ************************************************/
	{
		SECTION("(div v, div w), v,w in vector DG space")
		{
			Vector tmp1_l2(fespace_l2.GetNDofs()*dim ); /* store the result of DGDivDiv */
			Vector tmp2_l2(fespace_l2.GetNDofs()*dim ); /* store the result of VectorDivergence*/
			Vector diff(fespace_l2.GetNDofs()*dim ); /* store the result of VectorDivergence*/

			GridFunction ex_l2(&fespace_l2); ex_l2.ProjectCoefficient(NGradDivF2_coef_pzc);

			GridFunction g_l2(&fespace_l2);

			BilinearForm blf(&fespace_l2);
			blf.AddDomainIntegrator( new DGDivDivIntegrator() );
			blf.Assemble();
			blf.Finalize();

			cout<<endl<<" Bilinear form DGDivDiv Assembled "<<endl;

			MixedBilinearForm blf_mfem(&fespace_l2, &fespace_scalar_l2);
			blf_mfem.AddDomainIntegrator( new VectorDivergenceIntegrator()  );
			blf_mfem.Assemble();
			blf_mfem.Finalize();

			BilinearForm blf_mass(&fespace_l2);
			blf_mass.AddDomainIntegrator( new VectorMassIntegrator() );
			blf_mass.Assemble();
			blf_mass.Finalize();

			BilinearForm blf_diff(&fespace_scalar_l2);
			blf_diff.AddDomainIntegrator( new VectorDiffusionIntegrator() );
			blf_diff.Assemble();
			blf_diff.Finalize();


			cout<<endl<<" Sizes"<<endl
				<<"DivDiv:    "<< blf.SpMat().Width()<<" X "<<blf.SpMat().Height()<<endl
				<<"VectorDiv: "<< blf_mfem.SpMat().Width()<<" X "<<blf_mfem.SpMat().Height()<<endl
				<<"f_l2:   "<<f_l2.Size()<<endl
				<<"divf_l2:   "<<divf_l2.Size()<<endl
				<<"tmp1_l2: "<<tmp1_l2.Size()<<endl
				<<"tmp2_l2: "<<tmp2_l2.Size()<<endl
				<<endl;

			blf.Mult(f_l2, tmp1_l2);
			blf_mfem.MultTranspose( divf_l2, tmp2_l2);


			subtract(tmp1_l2,tmp2_l2,diff);
			cout<<endl<<"norm of difference: "<<diff.Norml2()<<endl;
			REQUIRE(diff.Norml2()<tol);

			cout<<endl<<endl<<endl
				<<"+++++++++++++++++++++++++++++++++"<<endl
				<<" test for DGDivDiv complete"<<endl
				<<"+++++++++++++++++++++++++++++++++"<<endl
				<<endl;


		} /* end of SECTION("(div v, div w), v,w in vector DG space") */
		SECTION("-(u, grad v), u in vector DG space, v scalar DG space")
		{
			Vector tmp1_l2(fespace_l2.GetNDofs()*dim ); /* store the result of DGMixedVectorWeakDivergence */
			Vector tmp2_l2(fespace_l2.GetNDofs()*dim ); /* store the result of VectorMassIntegrator*/
			Vector diff(fespace_l2.GetNDofs()*dim ); /* store the result of VectorDivergence*/

			GridFunction g_l2(&fespace_l2);

			MixedBilinearForm blf(&fespace_l2,&fespace_scalar_l2);
			blf.AddDomainIntegrator( new DGVectorWeakDivergenceIntegrator() );
			blf.Assemble();
			blf.Finalize();

			cout<<endl<<" DGVectorWeakDivergenceIntegrator assembled"<<endl;
			cout<<endl<<" Sizes"<<endl
				<<"WeakDivergence "<< blf.SpMat().Width()<<" X "<<blf.SpMat().Height()<<endl
				<<"f2_l2:   "<<f2_l2.Size()<<endl
				<<"gradf2_l2:   "<<gradf2_l2.Size()<<endl
				<<"tmp1_l2: "<<tmp1_l2.Size()<<endl
				<<"tmp2_l2: "<<tmp2_l2.Size()<<endl
				<<endl;


			blf.MultTranspose( f2_l2, tmp1_l2);

			BilinearForm blf_mfem(&fespace_l2);
			blf_mfem.AddDomainIntegrator( new VectorMassIntegrator()  );
			blf_mfem.Assemble();
			blf_mfem.Finalize();

			blf_mfem.Mult( gradf2_l2, tmp2_l2);

			add(tmp1_l2,tmp2_l2,diff);
			cout<<endl<<"norm of difference: "<<diff.Norml2()<<endl;
			REQUIRE(diff.Norml2()<tol);

			cout<<endl<<endl<<endl
				<<"+++++++++++++++++++++++++++++++++"<<endl
				<<" test for DGMixedVectorWeakDivergence complete"<<endl
				<<"+++++++++++++++++++++++++++++++++"<<endl
				<<endl;


		} /* end of SECTION("(div v, div w), v,w in vector DG space") */
	} /* end of SECTION("Face integrators") */

}
/**********************************************/
TEST_CASE("Test for the DPG face integrators",
		  "[DPGFaceIntegrator]"
		  "[pzc]"
		)
{
   cout<<endl<<"Test Face Integrator"<<endl;

   int order = 1, n = 2, m=2, dim = 2;
   double cg_rtol = 1e-14;
   double tol = 1e-9;

   /* mesh [0,2] \times [0,3] */
//   const char *mesh_file = "../../data/inline-quad.mesh";
//   const char *mesh_file = "../../data/star.mesh";
//   const char *mesh_file = "../../data/square-disc.mesh";
   const char *mesh_file = "../../data/inline-tri.mesh";
//   const char *mesh_file = "../../data/amr-quad.mesh";
   Mesh *mesh = new Mesh(mesh_file, 1, 1);

   cout<<endl<<" mesh formed "<<endl<<endl;


   /* define the finite element space */
   L2_FECollection fec_l2(order,dim);
   L2_FECollection s_fec_l2(order,dim);
   H1_Trace_FECollection trace_fec(order,dim);
                         		
   FiniteElementSpace fespace_trace(mesh,&trace_fec);
   FiniteElementSpace fespace_l2(mesh, &fec_l2,dim);
   FiniteElementSpace fespace_scalar_l2(mesh,&s_fec_l2);

	cout<<endl<<"finite elment space formed"<<endl;

	FunctionCoefficient f2_coef(f2);
	VectorFunctionCoefficient Grad_f2_coef(dim, Grad_f2);

	GridFunction f2_l2(&fespace_scalar_l2); f2_l2.ProjectCoefficient(f2_coef);
	GridFunction gradf2_l2(&fespace_l2); gradf2_l2.ProjectCoefficient(Grad_f2_coef);
	GridFunction f2_trace(&fespace_trace); f2_trace.ProjectCoefficientSkeletonDG(f2_coef);


   cout<<endl<<"trace number: "<<2*n*(n+1)<<endl
	   <<"trace dof: "<<f2_trace.Size()<<endl
	   <<endl;

    cout<<endl<<" coefficients obtained "<<endl;

	SECTION("Trace Integrator: NormalTraceJump")
	{
		SECTION(" <u,[tau cdot n]>")
		{
			Vector tmp1(fespace_l2.GetNDofs()*dim );
			Vector tmp2(fespace_l2.GetNDofs()*dim );
			Vector tmp3(fespace_l2.GetNDofs()*dim );
			Vector res(fespace_l2.GetNDofs()*dim );
			Vector tmp_sum12(fespace_l2.GetNDofs()*dim );

			BilinearForm blf_mass(&fespace_l2);
			blf_mass.AddDomainIntegrator( new VectorMassIntegrator() );
			blf_mass.Assemble();
			blf_mass.Finalize();

			MixedBilinearForm blf_div(&fespace_scalar_l2,&fespace_l2);
			blf_div.AddDomainIntegrator(
					new TransposeIntegrator(new VectorDivergenceIntegrator() )	);
			blf_div.Assemble();
			blf_div.Finalize();

			MixedBilinearForm blf_trace(&fespace_trace,&fespace_l2);
			blf_trace.AddTraceFaceIntegrator( new DGNormalTraceJumpIntegrator() );
			blf_trace.Assemble();
			blf_trace.Finalize();
			
			cout<<" Integrators assembled"<<endl;

			cout<<"Dimension of integrators"<<endl
				<<"mass:      "<<blf_mass.SpMat().Height()<<" X "<<blf_mass.SpMat().Width()<<endl
				<<"weak grad: "<<blf_div.SpMat().Height()  <<" X "<<blf_div.SpMat().Width()<<endl
				<<"trace:     "<<blf_trace.SpMat().Height()<<" X "<<blf_trace.SpMat().Width()<<endl
				<<endl;

			blf_mass.Mult(gradf2_l2,tmp1);
			blf_div.Mult(f2_l2,tmp2);
			blf_trace.Mult(f2_trace,tmp3);


			add(tmp1,tmp2,tmp_sum12);
			subtract(tmp3,tmp_sum12,res);

			double error = res.Norml2();
			cout<<endl<<" norm of 'zero': "<<error<<endl;
			REQUIRE(error<tol);


			
		}
	}

}

/* end of tesst */
/**********************************************/
}






















