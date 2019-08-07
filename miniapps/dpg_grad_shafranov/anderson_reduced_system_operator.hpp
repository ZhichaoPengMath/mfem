#include "mfem.hpp"
#include "petsc.h"
#include "RHSCoefficient.hpp"
#include "SourceAndBoundary.hpp"
#include <iostream>
#include <fstream>


#if defined(PETSC_HAVE_HYPRE)
#include "petscmathypre.h"
#endif

// Error handling
// Prints PETSc's stacktrace and then calls MFEM_ABORT
// We cannot use PETSc's CHKERRQ since it returns a PetscErrorCode
#define PCHKERRQ(obj,err) do {                                                   \
     if ((err))                                                                  \
     {                                                                           \
        PetscError(PetscObjectComm((PetscObject)(obj)),__LINE__,_MFEM_FUNC_NAME, \
                   __FILE__,(err),PETSC_ERROR_REPEAT,NULL);                      \
        MFEM_ABORT("Error in PETSc. See stacktrace above.");                     \
     }                                                                           \
  } while(0);

using namespace std;
using namespace mfem;

//double petsc_linear_solver_rel_tol = 1e-8;
double petsc_linear_solver_rel_tol = 1e-16;

/**********************************************************/
class AndersonReducedSystemOperator : public Operator
{ 
private:
	bool * use_petsc;
	/****************************************
	 * pointers to the trial spaces
	 *  u0_space: interior u, L2
	 *  q0_space: interior -\grad u /r , L2
	 *  uhat_space: trace of u, trace of H1
	 *  qhat_s[ace: trace q\cdot n, trace of RT
	 ****************************************/
	ParFiniteElementSpace * u0_space, * q0_space, * uhat_space, * qhat_space;
	/*****************************************
	 * pointers to the test spaces:
	 *  vtest_space: vector L2 test space
	 *  stest_space: scalar L2 test space
	 *****************************************/
	ParFiniteElementSpace * vtest_space, * stest_space;

	/*********************************
	 * linear operator for the right hand side
	 *********************************/
    ParLinearForm * linear_source_operator;	

	/*************************************
	 * Pointer to parallel matrix
	 * ***********************************/
    HypreParMatrix * matB_mass_q; 
    HypreParMatrix * matB_u_dot_div;
    HypreParMatrix * matB_u_normal_jump;
    HypreParMatrix * matB_q_weak_div;
    HypreParMatrix * matB_q_jump;
	HypreParMatrix * matVinv;
	HypreParMatrix * matSinv;

	/******************************
	 * block structures
	 * ****************************/
	Array<int> offsets;
	Array<int> offsets_test;
	/***********************************************************
	 * pointer to parallel matrix corresponding to the preconditioner
	 * *********************************************************/
	/*******************************
	 * ``Right hand side"  is
	 *		Jac^T * Ginv F
	 *	Equation to solve
	 *		Jac^T * Ginv B ( x - f(x) ) = 0
	 * *****************************/
	BlockOperator * B;
	BlockOperator * Ginv;
	mutable BlockOperator * Jac;

	/* operator calculate J^T G^-1 Bx */
	mutable Operator * JTGinvB;
	mutable HypreParMatrix * NDfDu;

	/* block vector */
	Vector &F;

	/*******************************
	 * essential vdof for boundary condition
	 * *****************************/
   const Array<int>& ess_trace_vdof_list;
		
   /**********************************
	* solvers 
	* *******************************/
    /* block Jacobian preconditioner */
    BlockDiagonalPreconditioner *P;
    /* lienar solver */
    PetscLinearSolver * solver;
public:
	AndersonReducedSystemOperator(
			/* use petsc or not */
			bool * _use_petsc,
			/* Finite element space */
			ParFiniteElementSpace * _u0_space, ParFiniteElementSpace * _q0_space, ParFiniteElementSpace * _uhat_space, ParFiniteElementSpace * _qhat_space,
			ParFiniteElementSpace * _vtest_space, ParFiniteElementSpace * _stest_space,
			/* linear form */
			ParLinearForm * _linear_source_operator,
			/* matrices */
			HypreParMatrix * _matB_mass_q, HypreParMatrix * _matB_u_normal_jump, HypreParMatrix * _matB_q_weak_div, 
			HypreParMatrix * _matB_q_jump, HypreParMatrix * _matVinv, HypreParMatrix * _matSinv,
			/* block structure */
			Array<int> _offsets, Array<int> _offsets_test,
			/* block operators */
			BlockOperator* _B,
			BlockOperator* _Jac,
			BlockOperator* _Ginv,
			/* bc */
			const Array<int> &_ess_trace_vdof_list,
			/* block vector */
			Vector &_F,
			/* block Jacobian preconditioner */
		    BlockDiagonalPreconditioner * _P
			);
	// dynamically update the small blcok NDfDu  =  - DF(u)/Du //
	virtual void UpdateNDFDU(const Vector &x) const;

	// dynamically update Jac = B - Df(x)/Dx //
	virtual void UpdateJac(const Vector &x) const;

	// Define FF(x) = 0 
	virtual void Mult( const Vector &x, Vector &y) const;  

	virtual ~AndersonReducedSystemOperator();

};

 /******************************************************
 *  Pass the pointers, initialization
 *******************************************************/
AndersonReducedSystemOperator::AndersonReducedSystemOperator(
    /* use petsc or not */
	bool * _use_petsc,
	/* Finite element space */
	ParFiniteElementSpace * _u0_space, ParFiniteElementSpace * _q0_space, ParFiniteElementSpace * _uhat_space, ParFiniteElementSpace * _qhat_space,
	ParFiniteElementSpace * _vtest_space, ParFiniteElementSpace * _stest_space,
	/* lienar form */
	ParLinearForm * _linear_source_operator,
	/* matrices */
	HypreParMatrix * _matB_mass_q, HypreParMatrix * _matB_u_normal_jump, HypreParMatrix * _matB_q_weak_div, 
	HypreParMatrix * _matB_q_jump, HypreParMatrix * _matVinv, HypreParMatrix * _matSinv,
	/* block structures */
	Array<int> _offsets, Array<int> _offsets_test,
	/* block operators */
	BlockOperator* _B,
	BlockOperator* _Jac,
	BlockOperator* _Ginv,
	/* boundary conditions */
	const Array<int> &_ess_trace_vdof_list,
	/* block vector */
	Vector &_F,
	/* block Jacobian preconditioenr */
	BlockDiagonalPreconditioner * _P
	):
	Operator( _B->Width() ), /* size of operator, important !!! */
	u0_space(_u0_space), q0_space(_q0_space), uhat_space(_uhat_space), qhat_space(_qhat_space),
	vtest_space(_vtest_space), stest_space(_stest_space),
	matB_mass_q(_matB_mass_q), matB_u_normal_jump(_matB_u_normal_jump), matB_q_weak_div(_matB_q_weak_div),
	matB_q_jump(_matB_q_jump), matVinv(_matVinv), matSinv(_matSinv), 
	offsets(_offsets), offsets_test(_offsets_test),
	B(_B),
	Jac(_Jac),
	Ginv(_Ginv),
	ess_trace_vdof_list(_ess_trace_vdof_list),
	linear_source_operator(_linear_source_operator),
	F(_F),
	NDfDu(NULL),
	use_petsc(_use_petsc),
	P(_P),
	JTGinvB(NULL)
{
		/* initialize linear solver */
	//	solver = new FGMRESSolver(MPI_COMM_WORLD);
		solver = new PetscLinearSolver(MPI_COMM_WORLD);
		solver->SetRelTol(petsc_linear_solver_rel_tol);
		solver->SetMaxIter(3000);
		solver->SetPreconditioner(*P);
		solver->iterative_mode = true; /* turn on the interative mode to take initial guess */
	}
	/******************************************************
	 *  Update NDfDu = DF(u)/Du
	 *******************************************************/
	void AndersonReducedSystemOperator::UpdateNDFDU(const Vector &x) const
	{
		/* calculate df(u)/du  */
		delete NDfDu;
		ParGridFunction u0_now;

		Vector u0_vec(x.GetData(), x.Size() );
		u0_now.MakeTRef(u0_space,u0_vec , 0);
		u0_now.SetFromTrueVector();
		FUXCoefficient dfu_coefficient( &u0_now, &derivative_of_nonlinear_source );

		ParMixedBilinearForm * mass_u = new ParMixedBilinearForm( u0_space, stest_space);
		mass_u->AddDomainIntegrator( new MixedScalarMassIntegrator(dfu_coefficient) );
		mass_u->Assemble();
		mass_u->Finalize();
		mass_u->SpMat() *= -1.;

		NDfDu = mass_u->ParallelAssemble();
		delete mass_u;
		
	}
	 /******************************************************
	 *  Update Jac = B - Df/dx
	 *******************************************************/
	void AndersonReducedSystemOperator::UpdateJac(const Vector &x) const
	{
		/* calculate df(u)/du  */
	//	UpdateNDFDU(x);

		Jac->SetBlock(1,1,NDfDu );
		
	}
	/******************************************************
	 *  Try to solve F(x) = 0
	 *  Mult gives us y = F(x)
 *******************************************************/
void AndersonReducedSystemOperator::Mult(const Vector &x, Vector &y) const
{
	/* update the Df/Du */
	UpdateNDFDU(x);
	/* update the Jacobian */
	UpdateJac(x);

	/* set linear solver to approximate (J^TG^-1B)^-1 */
	JTGinvB = new RAPOperator(*Jac,*Ginv,*B);
	solver->SetOperator(*JTGinvB);

	/****************************************************************************/
	/* calculate the right hand side J^T G^-1 F(x) */
	/* nonlinear source */
    Vector F1(F.GetData() + offsets_test[1],offsets_test[2]-offsets_test[1]);
	F1 = 0.;
    ParGridFunction u0_now;
	Vector u0_vec(x.GetData() + offsets[1], offsets[2] - offsets[1]);
    u0_now.MakeTRef(u0_space, u0_vec, 0);
	u0_now.SetFromTrueVector();
    FUXCoefficient fu_coefficient( &u0_now, &nonlinear_source );

	/* linear source */
	ParLinearForm *fu_mass = new ParLinearForm( stest_space );
	fu_mass->AddDomainIntegrator( new DomainLFIntegrator(fu_coefficient)  );
	fu_mass->Assemble();

	fu_mass->ParallelAssemble(F1); delete fu_mass;

	Vector F2( F1.Size() );
    linear_source_operator->ParallelAssemble( F2 );

	F1 += F2;

	/* J^T G^-1 (linear source + nonlinear source) */
    BlockVector rhs(offsets); 
    rhs=0.;

    BlockVector IGF(offsets_test);
    Ginv->Mult(F,IGF);
    Jac->MultTranspose(IGF,rhs);

	rhs *= -1.;
	/****************************************************************************/
	y = x; /* given initial guess for the KSP solve */
	solver->Mult(rhs,y); /* calculate  -(J^t G^-1 B)^-1*( J^t G^-1 F(x) ) */
	y += x;			/* y = x - (J^t G^-1 B)^-1*( J^t G^-1 F(x) ) */

	delete JTGinvB;
}

AndersonReducedSystemOperator::~AndersonReducedSystemOperator(){
	delete NDfDu;
	delete solver;
}

