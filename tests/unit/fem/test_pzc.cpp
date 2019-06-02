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

using namespace mfem;
using namespace std;

namespace bilininteg_pzc{

/*****************************************
 * Manufacture solutions for test
 * ***************************************/
double f2(const Vector & x) { return 2.345 * x[0] + 3.579 * x[1]; }
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

void GradDivF2_pzc(const Vector &x, Vector &v)
{
	v.SetSize(2);
	v[0] = 2.468 + 3.572;
	v[1] = 6.610;
}

void Gradf2DivF2_pzc( const Vector &x, Vector &v)
{
	v.SetSize(2);
	GradDivF2_pzc(x,v);
	v *= f2(x);

	Vector w1(2);
	Vector w2(2);
	Grad_f2(x,w1);
	w1 *= DivF2_pzc(x);

	v += w1;
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

TEST_CASE("Test for the DPG integrators",
		  "[DPGIntegrator]"
		  "[pzc]"
		  "[DGDivDivIntegrator]"
		)
{
   cout<<endl<<"Test Domain Integrator DGDivDivIntegrator"<<endl;

   int order = 3, n = 1, dim = 2;
   double cg_rtol = 1e-14;
   double tol = 1e-9;

   Mesh mesh(n, n, Element::QUADRILATERAL, 1, 2.0, 3.0);
   cout<<endl<<" mesh formed "<<endl<<endl;


   /* define the finite element space */
	L2_FECollection fec_l2(order,dim);
	FiniteElementSpace fespace_l2(&mesh, &fec_l2,dim);

	cout<<endl<<"finite elment space formed"<<endl;

	VectorFunctionCoefficient F2_coef_pzc(dim, F2_pzc);
	FunctionCoefficient DivF2_coef_pzc(DivF2_pzc);
	VectorFunctionCoefficient GradDivF2_coef_pzc(dim, GradDivF2_pzc);

	GridFunction f_l2(&fespace_l2); f_l2.ProjectCoefficient(F2_coef_pzc);
	GridFunction g_l2(&fespace_l2);

   cout<<endl<<" coefficients obtained "<<endl;
	SECTION("Face integrators")
	{
		Vector tmp_l2(fespace_l2.GetNDofs()*dim );
		SECTION("(div v, div w), v,w in vector DG space")
		{
			BilinearForm m_l2(&fespace_l2);
			m_l2.AddDomainIntegrator(new VectorMassIntegrator);
			m_l2.Assemble();
			m_l2.Finalize();
			cout<<endl<<" Mass matrix Assembled "<<endl<<endl;

			BilinearForm blf(&fespace_l2);
			blf.AddDomainIntegrator( new DGDivDivIntegrator() );
			blf.Assemble();
			blf.Finalize();
			
			cout<<endl<<" Bilinear form DGDivDiv Assembled "<<endl;

			SparseMatrix matBlf = blf.SpMat();
			cout<<endl<<" Sizes"<<endl
				<<"DivDiv: "<< matBlf.Width()<<" X "<<matBlf.Height()<<endl
				<<"f_l2:   "<< f_l2.Size()<<endl
				<<"tmp_l2: "<<tmp_l2.Size()<<endl
				<<endl;

			blf.Mult(f_l2, tmp_l2);
			cout<<endl<<"Righthand Side Assembled "<<endl;
			for(int i=0;i<tmp_l2.Size();i++){
				cout<<i<<": "<<tmp_l2(i)<<endl;
			}

			g_l2 = 0.;
			CG(m_l2, tmp_l2,  g_l2, 1, 200, cg_rtol*cg_rtol,0.0);

			double error_l2 = g_l2.ComputeL2Error(GradDivF2_coef_pzc);
			cout<<endl<<"error L2: "<<error_l2<<endl;

			cout<<endl<<" calculate GradDivF2_pzc"<<endl<<endl;
//
            REQUIRE( g_l2.ComputeL2Error(GradDivF2_coef_pzc) < tol );
		} /* end of SECTION("(div v, div w), v,w in vector DG space") */
	} /* end of SECTION("Face integrators") */

}
/* end of tesst */

}
