#include "fem.hpp"
#include <cmath>
#include <algorithm>
//#include "dpg_integrators.hpp"

namespace mfem{

/* < v, [n \cdot \tau ] >, \tau in L2 */
void DGNormalTraceJumpIntegrator::AssembleFaceMatrix(const FiniteElement &trial_face_fe,
                        const FiniteElement &test_fe1,
                        const FiniteElement &test_fe2,
                        FaceElementTransformations &Trans,
                        DenseMatrix &elmat)
{
   // Get DoF from faces and the dimension
   int i, j, face_ndof, ndof1, ndof2, dim;
   int order;

   MFEM_VERIFY(trial_face_fe.GetMapType() == FiniteElement::VALUE, "");

   face_ndof = trial_face_fe.GetDof();
   ndof1 = test_fe1.GetDof();
   dim = test_fe1.GetDim();

   face_shape.SetSize(face_ndof);
   normal.SetSize(dim);
   shape1.SetSize(ndof1);
   shape1_n.SetSize(ndof1,dim);


   if (Trans.Elem2No >= 0)
   {
      ndof2 = test_fe2.GetDof();
      shape2.SetSize(ndof2);
      shape2_n.SetSize(ndof2,dim);
   }
   else
   {
      ndof2 = 0;
   }

   elmat.SetSize( dim*(ndof1 + ndof2) , face_ndof);
   elmat = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      if (Trans.Elem2No >= 0)
      {
         order = fmax(test_fe1.GetOrder(), test_fe2.GetOrder()) - 1;
      }
      else
      {
         order = test_fe1.GetOrder() - 1;
      }
      order += trial_face_fe.GetOrder()+1;
      ir = &IntRules.Get(Trans.FaceGeom, order);
   }

   for (int p = 0; p < ir->GetNPoints(); p++)
   {
      const IntegrationPoint &ip = ir->IntPoint(p);
      IntegrationPoint eip1, eip2;
      // Trace finite element shape function
      trial_face_fe.CalcShape(ip, face_shape);
      Trans.Loc1.Transf.SetIntPoint(&ip);
      CalcOrtho(Trans.Loc1.Transf.Jacobian(), normal);
      // Side 1 finite element shape function
      Trans.Loc1.Transform(ip, eip1);
      test_fe1.CalcShape(eip1, shape1);
      MultVWt(shape1,normal, shape1_n);
      if (ndof2)
      {
         // Side 2 finite element shape function
         Trans.Loc2.Transform(ip, eip2);
         Trans.Loc2.Transf.SetIntPoint(&ip);
         CalcOrtho(Trans.Loc2.Transf.Jacobian(), normal);
         test_fe2.CalcShape(eip2, shape2);
         MultVWt(shape2,normal, shape2_n);
      }
      face_shape *= ip.weight;
	  for( i = 0; i < dim; i++)
	      for (int k=0; k < ndof1; k++)
	         for (j = 0; j < face_ndof; j++)
	         {
	            elmat(i*ndof1+k, j) -= shape1_n(k,i) * face_shape(j);
	         }
      if (ndof2)
      {
         // Subtract contribution from side 2
		 for(i = 0; i < dim; i++)
	         for (int k = 0; k < ndof2; k++)
	            for (j = 0; j < face_ndof; j++)
	            {
	               elmat(dim*ndof1+i*ndof2+k, j) += shape2_n(k,i) * face_shape(j);
	            } /* i,j */
      } /* if */
   } /* p */

}/* end of DGNormalJumpIntegrator */

/* (div v, div w ), v and w are in DG space */
void DGDivDivIntegrator::AssembleElementMatrix(
   const FiniteElement &el,
   ElementTransformation &Trans,
   DenseMatrix &elmat)
{
   int dim = el.GetDim();
   int dof = el.GetDof();
   double c;

   dshape.SetSize (dof, dim);
   gshape.SetSize (dof, dim);
   Jadj.SetSize (dim);
   divshape.SetSize (dim*dof);

   elmat.SetSize (dim*dof, dim*dof);

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int order = 2 * el.GetOrder() -2 ;
      ir = &IntRules.Get(el.GetGeomType(), order);
   }

   elmat = 0.0;

   for (int i = 0; i < ir -> GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

	  /* calculate divshape */
      el.CalcDShape (ip, dshape);

      Trans.SetIntPoint (&ip);
      CalcAdjugate(Trans.Jacobian(), Jadj);

      Mult (dshape, Jadj, gshape);

      gshape.GradToDiv (divshape);

      c = ip.weight / Trans.Weight();
      if (Q)
      {
         c *= Q -> Eval (Trans, ip);
      }

      // elmat += c * divshape * divshape ^ t
      AddMult_a_VVt (c, divshape, elmat);
   }
} /* end of DGDivDivIntegrator */


/* -(u,grad v), u is a vector in DG space an v is a scalar in DG space */
/* only scalar coefficient are considered for now */
void DGVectorWeakDivergenceIntegrator::AssembleElementMatrix2(const FiniteElement &trial_fe,
                      const FiniteElement &test_fe,
					  ElementTransformation &Trans,
					  DenseMatrix &elmat)
{
   int dim  = trial_fe.GetDim();
   int trial_dof = trial_fe.GetDof();
   int test_dof = test_fe.GetDof();
   double c;

   shape.SetSize (trial_dof);
   dshape.SetSize (test_dof, dim);
   gshape.SetSize (test_dof, dim);
   Jadj.SetSize (dim);

   elmat.SetSize (test_dof, dim*trial_dof);

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int order = Trans.OrderGrad(&test_fe) + trial_fe.GetOrder();
      ir = &IntRules.Get(trial_fe.GetGeomType(), order);
   }

   elmat = 0.0;

   for (int i = 0; i < ir -> GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      trial_fe.CalcShape (ip, shape);
      test_fe.CalcDShape (ip, dshape);

      Trans.SetIntPoint (&ip);
      CalcAdjugate(Trans.Jacobian(), Jadj);

      Mult (dshape, Jadj, gshape);
	  /* dense matrix gshape(i,j) = data( dof * j + i) = div(dof * j +i) */
	  /* I guess the order of data for the trial space is the same */
	  gshape.GradToDiv(divshape);

      c = ip.weight;
      if (Q)
      {
         c *= Q -> Eval (Trans, ip);
      }

      // elmat += c * shape * divshape ^ t
      shape *= -c;
      AddMultVWt (shape, divshape, elmat);
   }

} /* end of DGVectorWeakDivergenceIntegrator */


}
