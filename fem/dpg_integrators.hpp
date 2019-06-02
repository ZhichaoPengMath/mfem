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
/* < div u, div v >, where u and v  in DG space */
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


/************************************************/
}
