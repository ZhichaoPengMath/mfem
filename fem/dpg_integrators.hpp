// SkeletonMassIntegrator and SkeletonMassIntegratorRHS is implemented by people running 
// HDG branch
#include "../config/config.hpp"
#include "lininteg.hpp"

namespace mfem
{


/* < v, [\tau \cdot n ] > */
class DGNormalTraceJumpIntegrator : public BilinearFormIntegrator
{
private:
   Vector shape1, shape2, normal, face_shape;
   DenseMatrix shape1_n, shape2_n;

public:
   using BilinearFormIntegrator::AssembleFaceMatrix;
   virtual void AssembleFaceMatrix(const FiniteElement &trial_face_fe,
                                   const FiniteElement &test_fe1,
                                   const FiniteElement &test_fe2,
                                   FaceElementTransformations &Trans,
                                   DenseMatrix &elmat);


};
/* ( div u, div v ), where u and v  in DG space */
class DGDivDivIntegrator : public BilinearFormIntegrator
{
protected:
   Coefficient *Q;

private:
   Vector divshape;
   DenseMatrix dshape;
   DenseMatrix gshape;
   DenseMatrix Jadj;

public:
   DGDivDivIntegrator() { Q = NULL; }
   DGDivDivIntegrator(Coefficient &q) : Q(&q) { }

   virtual void AssembleElementMatrix(const FiniteElement &el,
                                      ElementTransformation &Trans,
                                      DenseMatrix &elmat);
 };

/* -( u, grad v), u is a vector in DG space and v is a scalar in DG space */
class DGVectorWeakDivergenceIntegrator : public BilinearFormIntegrator
{
protected:
		Coefficient *Q;

private:
	   Vector shape;

	   DenseMatrix Jadj; 
	   DenseMatrix dshape;
	   DenseMatrix gshape;
//	   DenseMatrix vshape;

public:
	   DGVectorWeakDivergenceIntegrator(){ Q = NULL; }
	   DGVectorWeakDivergenceIntegrator(Coefficient &q){ Q = &q; }
       DGVectorWeakDivergenceIntegrator(Coefficient *_q) { Q = _q; }

	   virtual void AssembleElementMatrix2(const FiniteElement &trial_fe,
			                              const FiniteElement &test_fe,
										  ElementTransformation &Trans,
										  DenseMatrix &elmat);

};



/* HDG */
/** Class for local mass matrix assembling a(\lamda,\mu) := <\lambda, \mu>
    It is used for the boundary elimination for skeleton variables */
class SkeletonMassIntegrator : public BilinearFormIntegrator
{
private:
   Vector shape;

public:
   SkeletonMassIntegrator(const IntegrationRule *ir = NULL)
      : BilinearFormIntegrator(ir) { }

   using BilinearFormIntegrator::AssembleFaceMatrix;
   virtual void AssembleFaceMatrix(const FiniteElement &face_fe,
                                   FaceElementTransformations &Trans,
                                   DenseMatrix &elmat);
};

/** Class for local mass RHS vector assembling l(\lamda,u) := <\lambda, u>
    It is used for the boundary elimination */
/* from HDG branch contributed by T. Horvath, S. Rhebergen, A. Sivas
//                 University of Waterloo */
class SkeletonMassIntegratorRHS: public LinearFormIntegrator
{
private:
   Vector shape;
   Coefficient &Q;
   int oa, ob;

public:
   SkeletonMassIntegratorRHS(Coefficient &QF, int a = 2, int b = 0,
                             const IntegrationRule *ir = NULL)
      : LinearFormIntegrator(ir), Q(QF), oa(a), ob(b) { }

   using LinearFormIntegrator::AssembleRHSElementVect;
   virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       FaceElementTransformations &Tr,
                                       Vector &elvect);

   virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       ElementTransformation &Tr,
                                       Vector &elvect);
};


/************************************************/
}
