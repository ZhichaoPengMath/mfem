#include "mfem.hpp"
#include "petsc.h"
#include "RHSCoefficient.hpp"
#include <iostream>
#include <fstream>

#include "nonlinear_gs_integrator.hpp"

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

class DFDUOperator : public Operator
{
private:
	ParFiniteElementSpace * u0_space, * stest_space;
public:
	DFDUOperator(ParFiniteElementSpace * _u0_space, ParFiniteElementSpace* _stest_space):
		u0_space(_u0_space), stest_space(_stest_space){};
	
	virtual void Mult( const Vector &x, Vector &y) const;  
};

void DFDUOperator::Mult(const Vector &x, Vector &y) const
{
    ParGridFunction u0_now;

	Vector u0_vec(x.GetData(), x.Size() );
    u0_now.MakeTRef(u0_space,u0_vec , 0);
	u0_now.SetFromTrueVector();
    DFDUCoefficient fu_coefficient( &u0_now );

	ParLinearForm *fu_mass = new ParLinearForm( stest_space );
	fu_mass->AddDomainIntegrator( new DomainLFIntegrator(fu_coefficient)  );
	fu_mass->Assemble();

	fu_mass->ParallelAssemble(y);
	delete fu_mass;
}


/**********************************************************/
class ReducedSystemOperator : public Operator
{ 
private:
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
	/***********************************************************
	 * pointer to parallel matrix corresponding to the preconditioner
	 * *********************************************************/
	HypreParMatrix * matV0;
	HypreParMatrix * matS0;
	HypreParMatrix * matVhat;
	HypreParMatrix * matShat;
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
	Vector *b;
	Vector &F;
//	Operator * A;

//    mutable HypreParMatrix * NDfDu;
	mutable HypreParMatrix * NDfDu;
//	mutable Operator * NDfDu;
	mutable BlockOperator * Jac;
	/**********************************
	 * Operator A = Jac^T G^-1 Jac 
	 *			  = AL + AN,
	 * AL is linear part and AN is nonlinear part
	 *
	 * AL  = [ AL00    AL01   AL02   AL03
	 *	       AL01^T  AL11   0      AL13
	 *         AL02^T  0      AL22   0
	 *	       AL03^T  AL13^T 0      AL33 ]
	 *
	 * AN  = [ 0      AN01   0     0
	 *		   AN01^T AN11   AN12  0
	 *		   0      AN12^T 0     0
	 *		   0      0      0     0 ]
	 *
	 * AN01 = DQ^T S^-1 NDfDu
	 * AN11 = DFDu^T S^-1 NDfDu
	 * AN12 = DF^T S^-1 TQ
	 *
	 * Linear Part is assembled and 
	 * passed to A before iteration 0
	 * in each iteration step
	 * we update AN and A
	 * ********************************/
	mutable BlockOperator * A;

	mutable HypreParMatrix * AL01;
	mutable HypreParMatrix * AL11;

	mutable HypreParMatrix * AN01;
	mutable HypreParMatrix * AN11;
	mutable HypreParMatrix * AN12;
	/*********************************
	 * linear operator for the right hand side
	 *********************************/
    ParLinearForm * f_div;	

	/*******************************
	 * essential vdof for boundary condition
	 * *****************************/
   const Array<int>& ess_trace_vdof_list;
		

	/***************************/
	mutable BlockOperator * Jacobian;
    mutable Vector fu;	

public:
	ReducedSystemOperator(
			ParFiniteElementSpace * _u0_space, ParFiniteElementSpace * _q0_space, ParFiniteElementSpace * _uhat_space, ParFiniteElementSpace * _qhat_space,
			ParFiniteElementSpace * _vtest_space, ParFiniteElementSpace * _stest_space,
//			ParMixedBilinearForm  * _B_mass_q, ParMixedBilinearForm * _B_u_dot_div, ParMixedBilinearForm * _B_u_normal_jump,
//			ParMixedBilinearForm  * _B_q_weak_div, ParMixedBilinearForm * _B_q_jump, ParBilinearForm * _Vinv, ParBilinearForm * _Sinv,
			HypreParMatrix * _matB_mass_q, HypreParMatrix * _matB_u_normal_jump, HypreParMatrix * _matB_q_weak_div, 
			HypreParMatrix * _matB_q_jump, HypreParMatrix * _matVinv, HypreParMatrix * _matSinv,
			/* preconditioner */
			HypreParMatrix * _matV0, HypreParMatrix * _matS0, HypreParMatrix * _matVhat, HypreParMatrix * _matShat,
			Array<int> _offsets, Array<int> _offsets_test,
//			Operator* _A,
			BlockOperator* _A,
			HypreParMatrix * _AL01,
			HypreParMatrix * _AL11,
			BlockOperator* _B,
			BlockOperator* _Jac,
			BlockOperator* _Ginv,
			const Array<int> &_ess_trace_vdof_list,
			Vector *_b,
			Vector &_F,
			ParLinearForm * _f_div
			);
	// dynamically update the small blcok NDfDu  =  - DF(u)/Du //
	virtual void UpdateNDFDU(const Vector &x) const;

	// dynamically update Jac = B - Df(x)/Dx //
	virtual void UpdateJac(const Vector &x) const;

	// Define FF(x) = 0 
	virtual void Mult( const Vector &x, Vector &y) const;  

	virtual Operator &GetGradient(const Vector &x) const;

	virtual ~ReducedSystemOperator();

};

/******************************************************
 *  Pass the pointers, initialization
 *******************************************************/
ReducedSystemOperator::ReducedSystemOperator(
	ParFiniteElementSpace * _u0_space, ParFiniteElementSpace * _q0_space, ParFiniteElementSpace * _uhat_space, ParFiniteElementSpace * _qhat_space,
	ParFiniteElementSpace * _vtest_space, ParFiniteElementSpace * _stest_space,
//	ParMixedBilinearForm  * _B_mass_q, ParMixedBilinearForm * _B_u_dot_div, ParMixedBilinearForm * _B_u_normal_jump,
//	ParMixedBilinearForm  * _B_q_weak_div, ParMixedBilinearForm * _B_q_jump, ParBilinearForm * _Vinv, ParBilinearForm * _Sinv,
	HypreParMatrix * _matB_mass_q, HypreParMatrix * _matB_u_normal_jump, HypreParMatrix * _matB_q_weak_div, 
	HypreParMatrix * _matB_q_jump, HypreParMatrix * _matVinv, HypreParMatrix * _matSinv,
	HypreParMatrix * _matV0, HypreParMatrix * _matS0, HypreParMatrix * _matVhat, HypreParMatrix * _matShat,
	Array<int> _offsets, Array<int> _offsets_test,
//	Operator *_A,
	BlockOperator *_A,
	HypreParMatrix * _AL01,
	HypreParMatrix * _AL11,
	BlockOperator* _B,
	BlockOperator* _Jac,
	BlockOperator* _Ginv,
	const Array<int> &_ess_trace_vdof_list,
	Vector *_b,
	Vector &_F,
	ParLinearForm * _f_div
	):
	Operator( _A->Width(), _A->Height() ), /* size of operator, important !!! */
	u0_space(_u0_space), q0_space(_q0_space), uhat_space(_uhat_space), qhat_space(_qhat_space),
	vtest_space(_vtest_space), stest_space(_stest_space),
//	B_mass_q(_B_mass_q),  B_u_dot_div(_B_u_dot_div), B_u_normal_jump(_B_u_normal_jump),
//	B_q_weak_div(_B_q_weak_div), B_q_jump(_B_q_jump), Vinv(_Vinv), Sinv(_Sinv),
	matB_mass_q(_matB_mass_q), matB_u_normal_jump(_matB_u_normal_jump), matB_q_weak_div(_matB_q_weak_div),
	matB_q_jump(_matB_q_jump), matVinv(_matVinv), matSinv(_matSinv), 
	matV0(_matV0), matS0(_matS0), matVhat(_matVhat), matShat(_matShat),
	offsets(_offsets), offsets_test(_offsets_test),
	A(_A),
	AL01(_AL01),
	AL11(_AL11),
	AN01(NULL),
	AN11(NULL),
	AN12(NULL),
	B(_B),
	Jac(_Jac),
	Ginv(_Ginv),
	ess_trace_vdof_list(_ess_trace_vdof_list),
	f_div(_f_div),
	b(_b),
	F(_F),
	Jacobian(NULL),
	NDfDu(NULL),
    fu(offsets_test[2] - offsets_test[1] ){}
/******************************************************
 *  Update NDfDu = DF(u)/Du
 *******************************************************/
void ReducedSystemOperator::UpdateNDFDU(const Vector &x) const
{
	/* calculate df(u)/du  */
	delete NDfDu;
    ParGridFunction u0_now;

	Vector u0_vec(x.GetData(), x.Size() );
    u0_now.MakeTRef(u0_space,u0_vec , 0);
	u0_now.SetFromTrueVector();
    DFDUCoefficient dfu_coefficient( &u0_now );

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
void ReducedSystemOperator::UpdateJac(const Vector &x) const
{
	/* calculate df(u)/du  */
//    ParGridFunction u0_now;
//
//	Vector u0_vec(x.GetData(), x.Size() );
//    u0_now.MakeTRef(u0_space,u0_vec , 0);
//	u0_now.SetFromTrueVector();
//    DFDUCoefficient dfu_coefficient( &u0_now );
//
//	ParMixedBilinearForm * mass_u = new ParMixedBilinearForm( u0_space, stest_space);
//	mass_u->AddDomainIntegrator( new MixedScalarMassIntegrator(dfu_coefficient) );
//	mass_u->Assemble();
//	mass_u->Finalize();
//	mass_u->SpMat() *= -1.;
//
////	NDfDu = mass_u->SpMat();
//	NDfDu = mass_u->ParallelAssemble();
//	delete mass_u;

//	UpdateNDFDU(x);

	Jac->SetBlock(1,1,NDfDu );
	
}


/******************************************************
 *  Try to solve F(x) = 0
 *  Mult gives us y = F(x)
 *******************************************************/
void ReducedSystemOperator::Mult(const Vector &x, Vector &y) const
{
	/* update the Df/Du */
	UpdateNDFDU(x);
	/* update the Jacobian */
	UpdateJac(x);

//	RAPOperator *oper = new RAPOperator(*Jac,*Ginv,*Jac);
	RAPOperator *oper = new RAPOperator(*Jac,*Ginv,*B);
	oper->Mult(x,y);
    


	/* update -(f(u),v) part */
    Vector F1(F.GetData() + offsets_test[1],offsets_test[2]-offsets_test[1]);
	F1 = 0.;
    ParGridFunction u0_now;
	Vector u0_vec(x.GetData() + offsets[1], offsets[2] - offsets[1]);
    u0_now.MakeTRef(u0_space, u0_vec, 0);
	u0_now.SetFromTrueVector();
    RHSCoefficient fu_coefficient( &u0_now );

	ParLinearForm *fu_mass = new ParLinearForm( stest_space );
	fu_mass->AddDomainIntegrator( new DomainLFIntegrator(fu_coefficient)  );
	fu_mass->Assemble();

	fu_mass->ParallelAssemble(F1); delete fu_mass;
	/* update linear source part */
	Vector F2( F1.Size() );
    f_div->ParallelAssemble( F2 );

	F1 += F2;

    BlockVector rhs(offsets); 
    rhs=0.;

    BlockVector IGF(offsets_test);
    Ginv->Mult(F,IGF);
    Jac->MultTranspose(IGF,rhs);
   
	y-=rhs;

	delete oper;
}

Operator &ReducedSystemOperator::GetGradient(const Vector &x) const
{
	/* update the -Df/Du */
	UpdateNDFDU(x);
	/* update AN01, AN11, AN12 */
	delete AN01;
	delete AN11;
	delete AN12;

	/* nonlinear part */
	AN01 = RAP(matB_q_weak_div, matSinv, NDfDu);
	AN01->Add(1.,*AL01);

	AN11 = RAP(NDfDu, matSinv, NDfDu);
	AN11->Add(1.,*AL11);

	AN12 = RAP(NDfDu, matSinv, matB_q_jump);

	A->SetBlock(0,1,AN01);
	A->SetBlock(1,0,AN01->Transpose() );

	A->SetBlock(1,1,AN11);

	A->SetBlock(1,2,AN12);
	A->SetBlock(2,1,AN12->Transpose() );

	return * A;
}

ReducedSystemOperator::~ReducedSystemOperator(){
	delete Jacobian;
}



/************************************************************/
/************************************************************/
/************************************************************/
//------petsc pcshell preconditioenr------
class MyBlockSolver : public Solver
{
private:
   Mat **sub; 

   // Create internal KSP objects to handle the subproblems
   KSP kspblock[4];

   // Create PetscParVectors as placeholders X and Y
   mutable PetscParVector *X, *Y;

   IS index_set[4];

public:
   MyBlockSolver(const OperatorHandle &oh);

   virtual void SetOperator (const Operator &op)
   { MFEM_ABORT("MyBlockSolver::SetOperator is not supported.");}

   virtual void Mult(const Vector &x, Vector &y) const;
   virtual ~MyBlockSolver();
};



MyBlockSolver::MyBlockSolver(const OperatorHandle &oh) : Solver() { 
   PetscErrorCode ierr; 

   // Get the PetscParMatrix out of oh.       
   PetscParMatrix *PP;
   oh.Get(PP);
   Mat P = *PP; // type cast to Petsc Mat
   
   // update base (Solver) class
   width = PP->Width();
   height = PP->Height();
   X = new PetscParVector(PETSC_COMM_WORLD, *this, true, false);
   Y = new PetscParVector(PETSC_COMM_WORLD, *this, false, false);

   PetscInt M, N;
   ierr=MatNestGetSubMats(P,&N,&M,&sub); PCHKERRQ(sub[0][0], ierr);// sub is an N by M array of matrices
   ierr=MatNestGetISs(P, index_set, NULL);  PCHKERRQ(index_set, ierr);// get the index sets of the blocks


   for (int i=0; i<4;i++)
   {
     ierr=KSPCreate(PETSC_COMM_WORLD, &kspblock[i]);    PCHKERRQ(kspblock[i], ierr);
	 /*                    ksp         A          preconditioner     */
     ierr=KSPSetOperators(kspblock[i], sub[i][i], sub[i][i]);PCHKERRQ(sub[i][i], ierr);

     if (i==0){
         KSPAppendOptionsPrefix(kspblock[i],"s0_");
	 }
     else if(i==1){
         KSPAppendOptionsPrefix(kspblock[i],"s1_");
	 }
     else if(i==2){
         KSPAppendOptionsPrefix(kspblock[i],"s2_");
	 }
     else{
         KSPAppendOptionsPrefix(kspblock[i],"s3_");
	 }
     KSPSetFromOptions(kspblock[i]);
     KSPSetUp(kspblock[i]);
   }
}

// How to solve preconditioner
void MyBlockSolver::Mult(const Vector &x, Vector &y) const
{
   //Mat &mass = sub[0][2];
   Vec blockx, blocky;

   X->PlaceArray(x.GetData()); // no copy, only the data pointer is passed to PETSc
   Y->PlaceArray(y.GetData());

   //solve equations
   for (int i = 0; i<4; i++)
   {
     VecGetSubVector(*X,index_set[i],&blockx);
     VecGetSubVector(*Y,index_set[i],&blocky);

     KSPSolve(kspblock[i],blockx,blocky);

     VecRestoreSubVector(*X,index_set[i],&blockx);
     VecRestoreSubVector(*Y,index_set[i],&blocky);
   }

   X->ResetArray();
   Y->ResetArray();
}

MyBlockSolver::~MyBlockSolver()
{
    for (int i=0; i<4; i++)
    {
        KSPDestroy(&kspblock[i]);
    }
    
    delete X;
    delete Y;
}

/*************************************************************************/
// Auxiliary class to provide preconditioners for matrix-free methods 
// Information interface to exchange information between Hypre and Petsc
class PreconditionerFactory : public PetscPreconditionerFactory
{
private:
   const ReducedSystemOperator& op;

public:
   PreconditionerFactory(const ReducedSystemOperator& op_,
                         const string& name_): PetscPreconditionerFactory(name_), op(op_) {};
   virtual mfem::Solver* NewPreconditioner(const mfem::OperatorHandle &oh)
   { return new MyBlockSolver(oh);}

   virtual ~PreconditionerFactory() {};
};

