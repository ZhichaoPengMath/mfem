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

#include "fem.hpp"
#include "kBilinIntegMass.hpp"
#include "kernels/kGeometry.hpp"
#include "kernels/kIntMass.hpp"

namespace mfem
{

// *****************************************************************************
KMassIntegrator::KMassIntegrator(const FiniteElementSpace *f,
                                 const IntegrationRule *i)
   :op(),
    maps(NULL),
    fes(f),
    ir(i) {push();assert(i); assert(fes);}

// *****************************************************************************
void KMassIntegrator::Assemble()
{
   push();
   //const FiniteElement &fe = *(fes->GetFE(0));
   //const Mesh *mesh = fes->GetMesh();
   //const int dim = mesh->Dimension();
   //const int dims = fe.GetDim();
   //const int symmDims = (dims * (dims + 1)) / 2; // 1x1: 1, 2x2: 3, 3x3: 6
   //const int elements = fes->GetNE();
   //assert(elements==mesh->GetNE());
   //const int quadraturePoints = ir->GetNPoints();
   //const int quad1D = IntRules.Get(Geometry::SEGMENT,ir->GetOrder()).GetNPoints();
   //const int size = symmDims * quadraturePoints * elements;
   //vec.SetSize(size);
   //kGeometry *geo = kGeometry::Get(*fes, *ir);
   maps = kDofQuadMaps::Get(*fes, *fes, *ir);
   pop();
}

// *****************************************************************************
void KMassIntegrator::SetOperator(Vector &v)
{
   push();
   op.SetSize(v.Size());
   op = v;
   pop();
}

// *****************************************************************************
void KMassIntegrator::MultAdd(Vector &x, Vector &y)
{
   push();
   ok(maps);
   Mesh *mesh = fes->GetMesh();
   const int dim = mesh->Dimension();
   const int quad1D = IntRules.Get(Geometry::SEGMENT,ir->GetOrder()).GetNPoints();
   const int dofs1D = fes->GetFE(0)->GetOrder() + 1;
   dbg("dim=%d",dim);
   dbg("quad1D=%d",quad1D);
   dbg("dofs1D=%d",dofs1D);
   dbg("NE=%d",mesh->GetNE());
   kMassMultAdd(dim,
                dofs1D,
                quad1D,
                mesh->GetNE(),
                maps->dofToQuad,
                maps->dofToQuadD,
                maps->quadToDof,
                maps->quadToDofD,
                op, x, y);
   pop();
}

} // namespace mfem
