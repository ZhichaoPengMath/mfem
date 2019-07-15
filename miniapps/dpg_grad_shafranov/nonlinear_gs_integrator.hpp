namespace mfem
{

/******************************************
 * ( f(u), v ), where u and v  in DG space,
 * where u is a scalar 
 * ****************************************/
class FUIntegrator : public BilinearFormIntegrator
{
protected:
   Coefficient *Q;
   double (*Function)(double u);
   
#ifndef MFEM_THREAD_SAFE
   Vector shape, fshape, te_shape; /* shape stores the basis evaluated at quadrature points, 
									  fshape stores f(shape) evaluated at quadrature points,
									  te_shape stores the basis in test space evaluated at quadrature points
									*/
#endif

private:
   

public:
   FUIntegrator() { Q = NULL;Function = NULL; }
   FUIntegrator(Coefficient &q, double (*_F)(double ) ): Q(&q),Function(_F) { }

   virtual void AssembleElementMatrix(const FiniteElement &el,
                                      ElementTransformation &Trans,
                                      DenseMatrix &elmat);

   virtual void AssembleElementMatrix2(const FiniteElement &trial_fe,
                                       const FiniteElement &test_fe,
                                       ElementTransformation &Trans,
                                       DenseMatrix &elmat);
};


void FUIntegrator::AssembleElementMatrix(
   const FiniteElement &el,
   ElementTransformation &Trans,
   DenseMatrix &elmat)
{
   int nd = el.GetDof();
   // int dim = el.GetDim();
   double w;

#ifdef MFEM_THREAD_SAFE
   Vector shape;
   Vector fshape;
#endif
   elmat.SetSize(nd);
   shape.SetSize(nd);
   fshape.SetSize(nd);

   const IntegrationRule *ir = IntRule;
   if(ir == NULL)
   {
//      int ir_order = el.GetOrder() + el.GetOrder() + Trans.OrderW();
      int ir_order = 3*( 2*el.GetOrder() + Trans.OrderW() );
      ir = &IntRules.Get(el.GetGeomType(), ir_order);
   }

   elmat = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      el.CalcShape(ip, shape);

	  for(int j=0;j<shape.Size();j++){
		  fshape(j) = Function(shape(j) );
	  }

      Trans.SetIntPoint (&ip);
//      w = Function(Trans.Weight() * ip.weight);
      w = Trans.Weight() * ip.weight;
//	  w = w*w;
      if (Q)
      {
         w *= Q -> Eval(Trans, ip);
      }

      AddMult_a_VVt(w, shape, elmat);
   }

} /* end of FUIntegrator::AssembleElementMatrix */


void FUIntegrator::AssembleElementMatrix2(
   const FiniteElement &trial_fe, const FiniteElement &test_fe,
   ElementTransformation &Trans, DenseMatrix &elmat)
{
   int tr_nd = trial_fe.GetDof();
   int te_nd = test_fe.GetDof();
   double w;

#ifdef MFEM_THREAD_SAFE
   Vector shape, te_shape;
#endif
   elmat.SetSize(te_nd, tr_nd);
   shape.SetSize(tr_nd);
   te_shape.SetSize(te_nd);

   const IntegrationRule *ir = IntRule;
   if(ir == NULL)
   {
      int ir_order = 2*(trial_fe.GetOrder() + test_fe.GetOrder() + Trans.OrderW() );
      ir = &IntRules.Get(trial_fe.GetGeomType(), ir_order);
   }

   elmat = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      trial_fe.CalcShape(ip, shape);
      test_fe.CalcShape(ip, te_shape);

      Trans.SetIntPoint (&ip);
//	  w = Trans.Weight() * ip.weight;
//	  w = w*w;
      w = Function(Trans.Weight() * ip.weight);
      if (Q)
      {
         w *= Q -> Eval(Trans, ip);
      }

      te_shape *= w;
      AddMultVWt(te_shape, shape, elmat);
   }
}	/* end of FUIntegrator::AssembleElementMatrix2 */

/**********************/
}
