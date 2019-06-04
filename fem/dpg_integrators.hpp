#include "../config/config.hpp"

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
//	   Vector vshape;
//	   Vector gradshape;

	   DenseMatrix Jadj; 
	   DenseMatrix dshape;
	   DenseMatrix gshape;
	   DenseMatrix vshape;

public:
	   DGVectorWeakDivergenceIntegrator(){ Q = NULL; }
	   DGVectorWeakDivergenceIntegrator(Coefficient &q){ Q = &q; }
       DGVectorWeakDivergenceIntegrator(Coefficient *_q) { Q = _q; }

	   virtual void AssembleElementMatrix2(const FiniteElement &trial_fe,
			                              const FiniteElement &test_fe,
										  ElementTransformation &Trans,
										  DenseMatrix &elmat);

};
/************************************************/
}
