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
   int order=0;
   double w;

//   MFEM_VERIFY(trial_face_fe.GetMapType() == FiniteElement::VALUE, "");

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

	Vector normal2(dim );
    CalcOrtho(Trans.Face->Jacobian(), normal2);
	std::cout<<std::endl
	         <<"Face:"<<std::endl<<" Normal:"<<std::endl;
	for(int qq=0;qq<normal2.Size();qq++){
	  	std::cout<<" "<<normal2(qq);
	}
	std::cout<<std::endl;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      if (Trans.Elem2No >= 0)
      {
         order = fmax(test_fe1.GetOrder(), test_fe2.GetOrder()) ;
      }
      else
      {
         order = test_fe1.GetOrder() ;
      }
      order += trial_face_fe.GetOrder();
      ir = &IntRules.Get(Trans.FaceGeom, order);
   }
   std::cout<< "Quadrature order: "<<order<<std::endl;

   for (int p = 0; p < ir->GetNPoints(); p++)
   {
      const IntegrationPoint &ip = ir->IntPoint(p);
      IntegrationPoint eip1, eip2;
      // Trace finite element shape function
      trial_face_fe.CalcShape(ip, face_shape);
      Trans.Loc1.Transf.SetIntPoint(&ip);
      CalcOrtho(Trans.Loc1.Transf.Jacobian(), normal);
//      CalcOrtho(Trans.Face->Jacobian(), normal);

	 /*********************************************************/
	  /* debug */
	  if(p==0){
		std::cout<<std::endl
	  	         <<"Trans1:"<<std::endl<<" Normal:"<<std::endl;
	  	for(int qq=0;qq<normal.Size();qq++){
	  	  	std::cout<<" "<<normal(qq);
	  	}
//		std::cout<<std::endl<<"Jacobian:"<<Trans.Loc1.Transf.Jacobian().Height()<<" "
//				 <<Trans.Loc1.Transf.Jacobian().Width()<<std::endl;
//	  	for(int qq=0;qq<Trans.Loc1.Transf.Jacobian().Height();qq++){
//			for(int qq2=0;qq2<Trans.Loc1.Transf.Jacobian().Width();qq2++){
//				std::cout<<" "<<Trans.Loc1.Transf.Jacobian()(qq,qq2);
//			}
//			std::cout<<std::endl;
//	  	}

//		Vector normal2(dim );
//        CalcOrtho(Trans.Face->Jacobian(), normal2);
//		std::cout<<std::endl
//	  	         <<"Face:"<<std::endl<<" Normal:"<<std::endl;
//	  	for(int qq=0;qq<normal2.Size();qq++){
//	  	  	std::cout<<" "<<normal2(qq);
//	  	}
//		std::cout<<std::endl<<"Jacobian:"<<Trans.Face->Jacobian().Height()<<" "
//				 <<Trans.Face->Jacobian().Width()<<std::endl;
//	  	for(int qq=0;qq<Trans.Face->Jacobian().Height();qq++){
//			for(int qq2=0;qq2<Trans.Face->Jacobian().Width();qq2++){
//				std::cout<<" "<<Trans.Face->Jacobian()(qq,qq2);
//			}
//		}

//		Vector normal3(dim );
//        CalcOrtho(Trans.Elem1->Jacobian(), normal3);
// 		std::cout<<std::endl
// 	  	         <<"Elem1:"<<std::endl<<" Normal:"<<std::endl;
// //	  	for(int qq=0;qq<normal2.Size();qq++){
// //	  	  	std::cout<<" "<<normal2(qq);
// //	  	}
// 		std::cout<<std::endl<<"Jacobian:"<<Trans.Elem1->Jacobian().Height()<<" "
// 				 <<Trans.Elem1->Jacobian().Width()<<std::endl;
// 	  	for(int qq=0;qq<Trans.Elem1->Jacobian().Height();qq++){
// 			for(int qq2=0;qq2<Trans.Elem1->Jacobian().Width();qq2++){
// 				std::cout<<" "<<Trans.Elem1->Jacobian()(qq,qq2);
// 			}
// 		}
		std::cout<<std::endl;
	}
	 /*********************************************************/

      // Side 1 finite element shape function
      Trans.Loc1.Transform(ip, eip1);
      test_fe1.CalcShape(eip1, shape1);
      MultVWt(shape1,normal, shape1_n);


      w = ip.weight;
 //     if (trial_face_fe.GetMapType() == FiniteElement::VALUE)
 //     {
		 std::cout<<" face weight "<<" "<<Trans.Face->Weight()<<std::endl
			      <<" Trans1 weight: "<< Trans.Elem1->Weight()<<std::endl
//			      <<" integration weight "<<w
				  <<std::endl;
         w *= Trans.Face->Weight();
//	  }
      face_shape *= w;
//      face_shape *= ip.weight;

	  for( i = 0; i < dim; i++)
	      for (int k=0; k < ndof1; k++)
	         for (j = 0; j < face_ndof; j++)
	         {
	            elmat(i*ndof1+k, j) += shape1_n(k,i) * face_shape(j);
	         }
      if (ndof2)
      {
         // Side 2 finite element shape function
         Trans.Loc2.Transform(ip, eip2);
         Trans.Loc2.Transf.SetIntPoint(&ip);
         CalcOrtho(Trans.Loc2.Transf.Jacobian(), normal);
         test_fe2.CalcShape(eip2, shape2);
         MultVWt(shape2,normal, shape2_n);
         // Subtract contribution from side 2
		 for(i = 0; i < dim; i++)
	         for (int k = 0; k < ndof2; k++)
	            for (j = 0; j < face_ndof; j++)
	            {
	               elmat(dim*ndof1+i*ndof2+k, j) -= shape2_n(k,i) * face_shape(j);
	            } /* i,k,j */
//		 for(int l=0;l<shape2.Size();l++){
//			std::cout<<l<<"hehehehehe: "<<shape2(l)<<" "<<shape1(l)<<std::endl;
//		 }
      } /* if */
   } /* p */

   /* out put the final result */
   for(int i=0;i<dim* (ndof1+ndof2);i++){
	  std::cout<<std::endl;
	  for(int j=0;j<face_ndof;j++){
//		std::cout<<" "<< elmat(i,j);
	  }
   }
   std::cout<<std::endl;

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
//   vshape.SetSize (dim, trial_dof*dim); vshape = 0.;
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

//	  for( int i_dim=0; i_dim<dim; i_dim++){
//			 for(int k=0;k<trial_dof;k++){
//				vshape(i_dim, i_dim * trial_dof + k) = shape(k);
//			 }
//	  } /* i_dim */

      test_fe.CalcDShape (ip, dshape);

      Trans.SetIntPoint (&ip);
      CalcAdjugate(Trans.Jacobian(), Jadj);

      Mult (dshape, Jadj, gshape);
	  /* dense matrix gshape(i,j) = data( dof * j + i) = div(dof * j +i) */
//	  gshape.GradToDiv(gradshape);

      c = ip.weight;
      if (Q)
      {
         c *= Q -> Eval (Trans, ip);
      }

      // elmat += c * vshape * divshape ^ t
      gshape *= -c;
//      AddMult ( gshape, vshape, elmat);
	 for(int k_test = 0; k_test<test_dof; k_test++){
		for(int i_dim = 0; i_dim<dim; i_dim++){
			for(int k_trial = 0; k_trial<trial_dof; k_trial++){
				elmat(k_test, i_dim * trial_dof + k_trial) += shape(k_trial) * gshape(k_test,i_dim);
			} /* k_trial */
		} /* i_dim */
	 } /* end of k_test */
   }

} /* end of DGVectorWeakDivergenceIntegrator */


/* HDG */
void SkeletonMassIntegrator::AssembleFaceMatrix(const FiniteElement &face_fe,
                                                FaceElementTransformations &Trans,
                                                DenseMatrix &elmat)
{
   int ndof;
   double w;

   ndof = face_fe.GetDof();
   elmat.SetSize(ndof, ndof);
   elmat = 0.0;
   shape.SetSize(ndof);

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int order = 2 * face_fe.GetOrder();
      order *= 2;

      ir = &IntRules.Get(Trans.FaceGeom, order);
   }

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      face_fe.CalcShape(ip, shape);

      Trans.Face->SetIntPoint(&ip);

      w = Trans.Face->Weight() * ip.weight;

      AddMult_a_VVt(w, shape, elmat);
   }

} /* end of SkeletonMassIntegrator */

/* HDG */
void SkeletonMassIntegratorRHS::AssembleRHSElementVect(const FiniteElement &el,
                                                       FaceElementTransformations &Tr,
                                                       Vector &elvect)
{
   int dof = el.GetDof();

   shape.SetSize(dof);       // vector of size dof
   elvect.SetSize(dof);
   elvect = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      ir = &IntRules.Get(el.GetGeomType(), oa * el.GetOrder() + ob);
   }

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      Tr.Face->SetIntPoint (&ip);
      double val = Tr.Face->Weight() * Q.Eval(*Tr.Face, ip);

      el.CalcShape(ip, shape);

      add(elvect, ip.weight * val, shape, elvect);
   }
} /* end of skeletonmassintegratorrhs */

/* HDG */
void SkeletonMassIntegratorRHS::AssembleRHSElementVect(
   const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
{
   mfem_error("Not implemented \n");
} /* end of skeletonmassintegratorrhs */

/****************************************************/

}
