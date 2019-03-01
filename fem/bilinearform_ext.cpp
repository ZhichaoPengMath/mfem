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

// Implementations of classes FABilinearFormExtension, EABilinearFormExtension,
// PABilinearFormExtension and MFBilinearFormExtension.

#include "fem.hpp"
#include "bilininteg.hpp"
#include "bilinearform_ext.hpp"
#include "../linalg/kernels/vector.hpp"

namespace mfem
{

// Data and methods for fully-assembled bilinear forms
FABilinearFormExtension::FABilinearFormExtension(BilinearForm *form) :
   Operator(form->Size()), a(form) { }

// Data and methods for element-assembled bilinear forms
EABilinearFormExtension::EABilinearFormExtension(BilinearForm *form)
   : Operator(form->Size()), a(form) { }

// Data and methods for partially-assembled bilinear forms
PABilinearFormExtension::PABilinearFormExtension(BilinearForm *form) :
   Operator(form->Size()), a(form),
   trialFes(a->fes), testFes(a->fes),
   localX(a->fes->GetNE() * trialFes->GetFE(0)->GetDof() * trialFes->GetVDim()),
   localY(a->fes->GetNE() * testFes->GetFE(0)->GetDof() * testFes->GetVDim()),
   e2l(*a->fes) { }

PABilinearFormExtension::~PABilinearFormExtension()
{
   for (int i = 0; i < integrators.Size(); ++i)
   {
      delete integrators[i];
   }
}

// Adds new Domain Integrator.
void PABilinearFormExtension::AddDomainIntegrator(
   AbstractBilinearFormIntegrator *i)
{
   integrators.Append(static_cast<BilinearPAFormIntegrator*>(i));
}

void PABilinearFormExtension::Assemble()
{
   assert(integrators.Size()==1);
   const int integratorCount = integrators.Size();
   for (int i = 0; i < integratorCount; ++i)
   {
      integrators[i]->Assemble(*a->fes);
   }
}

void PABilinearFormExtension::FormSystemOperator(const Array<int>
                                                 &ess_tdof_list,
                                                 Operator *&A)
{
   const Operator* trialP = trialFes->GetProlongationMatrix();
   const Operator* testP  = testFes->GetProlongationMatrix();
   Operator *rap = this;
   if (trialP) { rap = new RAPOperator(*testP, *this, *trialP); }
   const bool own_A = (rap!=this);
   assert(rap);
   A = new ConstrainedOperator(rap, ess_tdof_list, own_A);
}

void PABilinearFormExtension::FormLinearSystem(const Array<int> &ess_tdof_list,
                                               Vector &x, Vector &b,
                                               Operator *&A, Vector &X, Vector &B,
                                               int copy_interior)
{
   FormSystemOperator(ess_tdof_list, A);

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

   if (!copy_interior && ess_tdof_list.Size()>0)
   {
      const int csz = ess_tdof_list.Size();
      Vector subvec(csz);
      subvec = 0.0;
      kernels::vector::GetSubvector(csz,
                                    subvec.GetData(),
                                    X.GetData(),
                                    ess_tdof_list.GetData());
      X = 0.0;
      kernels::vector::SetSubvector(csz,
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

void PABilinearFormExtension::Mult(const Vector &x, Vector &y) const
{
   e2l.Mult(x, localX);
   localY = 0.0;
   const int iSz = integrators.Size();
   assert(iSz==1);
   for (int i = 0; i < iSz; ++i)
   {
      integrators[i]->MultAdd(localX, localY);
   }
   e2l.MultTranspose(localY, y);
}

void PABilinearFormExtension::MultTranspose(const Vector &x, Vector &y) const
{
   e2l.Mult(x, localX);
   localY = 0.0;
   const int iSz = integrators.Size();
   assert(iSz==1);
   for (int i = 0; i < iSz; ++i)
   {
      integrators[i]->MultTransposeAdd(localX, localY);
   }
   e2l.MultTranspose(localY, y);
}

void PABilinearFormExtension::RecoverFEMSolution(const Vector &X,
                                                 const Vector &b,
                                                 Vector &x)
{
   const Operator *P = a->GetProlongation();
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

// Data and methods for matrix-free bilinear forms
MFBilinearFormExtension::MFBilinearFormExtension(BilinearForm *form)
   : Operator(form->Size()), a(form) { }

// *****************************************************************************
E2LOperator::E2LOperator(const FiniteElementSpace &f)
   :fes(f),
    ne(fes.GetNE()),
    vdim(fes.GetVDim()),
    byvdim(fes.GetOrdering() == Ordering::byVDIM),
    ndofs(fes.GetNDofs()),
    dof(fes.GetFE(0)->GetDof()),
    nedofs(ne*dof),
    offsets(ndofs+1),
    indices(ne*dof)
{
   const FiniteElement *fe = fes.GetFE(0);
   const TensorBasisElement* el = dynamic_cast<const TensorBasisElement*>(fe);
   MFEM_ASSERT(el, "Finite element not supported with partial assembly");
   const Array<int> &dof_map = el->GetDofMap();
   const bool dof_map_is_identity = (dof_map.Size()==0);
   const Table& e2dTable = fes.GetElementToDofTable();
   const int* elementMap = e2dTable.GetJ();
   // We'll be keeping a count of how many local nodes point to its global dof
   for (int i = 0; i <= ndofs; ++i)
   {
      offsets[i] = 0;
   }
   for (int e = 0; e < ne; ++e)
   {
      for (int d = 0; d < dof; ++d)
      {
         const int gid = elementMap[dof*e + d];
         ++offsets[gid + 1];
      }
   }
   // Aggregate to find offsets for each global dof
   for (int i = 1; i <= ndofs; ++i)
   {
      offsets[i] += offsets[i - 1];
   }
   // For each global dof, fill in all local nodes that point   to it
   for (int e = 0; e < ne; ++e)
   {
      for (int d = 0; d < dof; ++d)
      {
         const int did = dof_map_is_identity?d:dof_map[d];
         const int gid = elementMap[dof*e + did];
         const int lid = dof*e + d;
         indices[offsets[gid]++] = lid;
      }
   }
   // We shifted the offsets vector by 1 by using it as a counter
   // Now we shift it back.
   for (int i = ndofs; i > 0; --i)
   {
      offsets[i] = offsets[i - 1];
   }
   offsets[0] = 0;
}

// ***************************************************************************
void E2LOperator::Mult(const Vector& x, Vector& y) const
{
   const int vd = vdim;
   const bool t = byvdim;
   const DeviceArray d_offsets(offsets, ndofs+1);
   const DeviceArray d_indices(indices, nedofs);
   const DeviceMatrix d_x(x, t?vd:ndofs, t?ndofs:vd);
   DeviceMatrix d_y(y, t?vd:nedofs, t?nedofs:vd);
   MFEM_FORALL(i, ndofs,
   {
      const int offset = d_offsets[i];
      const int nextOffset = d_offsets[i+1];
      for (int c = 0; c < vd; ++c)
      {
         const double dofValue = d_x(t?c:i,t?i:c);
         for (int j = offset; j < nextOffset; ++j)
         {
            const int idx_j = d_indices[j];
            d_y(t?c:idx_j,t?idx_j:c) = dofValue;
         }
      }
   });
}

// ***************************************************************************
void E2LOperator::MultTranspose(const Vector& x, Vector& y) const
{
   const int vd = vdim;
   const bool t = byvdim;
   const DeviceArray d_offsets(offsets, ndofs+1);
   const DeviceArray d_indices(indices, nedofs);
   const DeviceMatrix d_x(x, t?vd:nedofs, t?nedofs:vd);
   DeviceMatrix d_y(y, t?vd:ndofs, t?ndofs:vd);
   MFEM_FORALL(i, ndofs,
   {
      const int offset = d_offsets[i];
      const int nextOffset = d_offsets[i + 1];
      for (int c = 0; c < vd; ++c)
      {
         double dofValue = 0;
         for (int j = offset; j < nextOffset; ++j)
         {
            const int idx_j = d_indices[j];
            dofValue +=  d_x(t?c:idx_j,t?idx_j:c);
         }
         d_y(t?c:i,t?i:c) = dofValue;
      }
   });
}

} // namespace mfem
