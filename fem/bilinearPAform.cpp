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

// Implementation of class BilinearForm

#include "fem.hpp"
#include "bilininteg.hpp"
#include "kBilinIntegDiffusion.hpp"
#include "kfespace.hpp"
#include "../linalg/kernels/vector.hpp"

#include <cmath>

namespace mfem
{

// ***************************************************************************
// * PABilinearForm
// ***************************************************************************
PABilinearForm::PABilinearForm(FiniteElementSpace* fes) :
   AbstractBilinearForm(fes),
   mesh(fes->GetMesh()),
   trialFes(fes),
   testFes(fes),
   localX(mesh->GetNE() * trialFes->GetFE(0)->GetDof() * trialFes->GetVDim()),
   localY(mesh->GetNE() * testFes->GetFE(0)->GetDof() * testFes->GetVDim()),
   kfes(new kFiniteElementSpace(fes)) { push(); }

// ***************************************************************************
PABilinearForm::~PABilinearForm() { delete kfes; }

// *****************************************************************************
void PABilinearForm::EnableStaticCondensation() { assert(false);}

// ***************************************************************************
// Adds new Domain Integrator.
void PABilinearForm::AddDomainIntegrator(AbstractBilinearFormIntegrator *i)
{
   push();
   integrators.Append(static_cast<BilinearPAFormIntegrator*>(i));
}

// Adds new Boundary Integrator.
void PABilinearForm::AddBoundaryIntegrator(AbstractBilinearFormIntegrator *i)
{
   push();
   assert(false);
   //AddIntegrator(i, BoundaryIntegrator);
}

// Adds new interior Face Integrator.
void PABilinearForm::AddInteriorFaceIntegrator(AbstractBilinearFormIntegrator
                                               *i)
{
   assert(false);
   //AddIntegrator(i, InteriorFaceIntegrator);
}

// Adds new boundary Face Integrator.
void PABilinearForm::AddBoundaryFaceIntegrator(AbstractBilinearFormIntegrator
                                               *i)
{
   assert(false);
   //AddIntegrator(i, BoundaryFaceIntegrator);
}

// *****************************************************************************
// * WARNING DiffusionGetRule Q order
// *****************************************************************************
/*static const IntegrationRule &WARNING_GetRule(const FiniteElement &trial_fe,
                                              const FiniteElement &test_fe)
{
   int order;
   if (trial_fe.Space() == FunctionSpace::Pk)
   {
      order = trial_fe.GetOrder() + test_fe.GetOrder() - 2;
   }
   else
   {
      // order = 2*el.GetOrder() - 2;  // <-- this seems to work fine too
      order = trial_fe.GetOrder() + test_fe.GetOrder() + trial_fe.GetDim() - 1;
   }
   if (trial_fe.Space() == FunctionSpace::rQk)
   {
      return RefinedIntRules.Get(trial_fe.GetGeomType(), order);
   }
   return IntRules.Get(trial_fe.GetGeomType(), order);
   }*/

// ***************************************************************************
void PABilinearForm::Assemble(int skip_zeros)
{
   push();
   assert(integrators.Size()==1);
   //const FiniteElement &fe = *fes->GetFE(0);
//#warning WARNING_GetRule
   //dbg("\033[31;1m[WARNING_GetRule] !!!!");assert(false);
   //const IntegrationRule *ir = &WARNING_GetRule(fe,fe);
   //assert(ir);
   //mfem::Array<mfem::BilinearFormIntegrator*> &dbfi = *bform->GetDBFI();
   const int integratorCount = integrators.Size();
   for (int i = 0; i < integratorCount; ++i)
   {
      const mfem::IntegrationRule *ir = integrators[i]->GetIntRule();
      integrators[i]->Setup(fes,ir);
      integrators[i]->Assemble();
   }
}

// ***************************************************************************
void PABilinearForm::FormOperator(const Array<int> &ess_tdof_list,
                                  Operator &A)
{
   push();
   const Operator* trialP = trialFes->GetProlongationMatrix();
   const Operator* testP  = testFes->GetProlongationMatrix();
   Operator *rap = this;
   if (trialP) { rap = new RAPOperator(*testP, *this, *trialP); }
   const bool own_A = rap!=this;
   assert(rap);
   Operator *CO = new ConstrainedOperator(rap, ess_tdof_list, own_A);
   A = *CO;
}

// ***************************************************************************
void PABilinearForm::FormOperator(const Array<int> &ess_tdof_list,
                                  Operator *&A)
{
   push();
   const Operator* trialP = trialFes->GetProlongationMatrix();
   const Operator* testP  = testFes->GetProlongationMatrix();
   Operator *rap = this;
   if (trialP) { rap = new RAPOperator(*testP, *this, *trialP); }
   A = new ConstrainedOperator(rap, ess_tdof_list, rap != this);
   pop();
}

// ***************************************************************************
void PABilinearForm::FormLinearSystem(const Array<int> &ess_tdof_list,
                                      Vector &x, Vector &b,
                                      Operator *&A, Vector &X, Vector &B,
                                      int copy_interior)
{
   push();
   const Operator* trialP = trialFes->GetProlongationMatrix();
   const Operator* testP  = testFes->GetProlongationMatrix();
   Operator *rap = this;
   if (trialP) { rap = new RAPOperator(*testP, *this, *trialP); }
   const bool own_A = rap!=this;
   assert(rap);

   A = new ConstrainedOperator(rap, ess_tdof_list, own_A);

   const Operator* P = trialFes->GetProlongationMatrix();
   const Operator* R = trialFes->GetRestrictionMatrix();
   if (P)
   {
      // Variational restriction with P
      B.SetSize(P->Width());
      P->MultTranspose(b, B);
      X.SetSize(R->Height());
      R->Mult(x, X);
   }
   else
   {
      // rap, X and B point to the same data as this, x and b
      X.SetSize(x.Size()); X = x;
      B.SetSize(b.Size()); B = b;
   }

   if (!copy_interior and ess_tdof_list.Size()>0)
   {
      const int csz = ess_tdof_list.Size();
      const int xsz = X.Size();
      assert(xsz>=csz);
      Vector subvec(xsz);
      subvec = 0.0;
      kVectorGetSubvector(csz,
                          subvec.GetData(),
                          X.GetData(),
                          ess_tdof_list.GetData());
      X = 0.0;
      kVectorSetSubvector(csz,
                          X.GetData(),
                          subvec.GetData(),
                          ess_tdof_list.GetData());
   }

   ConstrainedOperator *cA = static_cast<ConstrainedOperator*>(A);
   assert(cA);
   if (cA)
   {
      cA->EliminateRHS(X, B);
   }
   else
   {
      mfem_error("BilinearForm::InitRHS expects an ConstrainedOperator");
   }
}

// ***************************************************************************
void PABilinearForm::Mult(const Vector &x, Vector &y) const
{
   push();
   kfes->GlobalToLocal(x, localX);
   localY = 0.0;
   const int iSz = integrators.Size();
   assert(iSz==1);
   for (int i = 0; i < iSz; ++i)
   {
      integrators[i]->MultAdd(localX, localY);
   }
   kfes->LocalToGlobal(localY, y);
}

// ***************************************************************************
void PABilinearForm::MultTranspose(const Vector &x, Vector &y) const
{
   push();
   kfes->GlobalToLocal(x, localX);
   localY = 0.0;
   const int iSz = integrators.Size();
   assert(iSz==1);
   for (int i = 0; i < iSz; ++i)
   {
      integrators[i]->MultTransposeAdd(localX, localY);
   }
   kfes->LocalToGlobal(localY, y);
}

// ***************************************************************************
void PABilinearForm::RecoverFEMSolution(const Vector &X,
                                        const Vector &b,
                                        Vector &x)
{
   push();
   const Operator *P = this->GetProlongation();
   if (P)
   {
      // Apply conforming prolongation
      x.SetSize(P->Height());
      P->Mult(X, x);
      return;
   }
   // Otherwise X and x point to the same data
   x = X;
}

}
