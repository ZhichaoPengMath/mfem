#include "../config/config.hpp"

namespace mfem
{

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

}
