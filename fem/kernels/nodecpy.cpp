// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
// reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed.
//
// The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.

#include "../../general/okina.hpp"

// *****************************************************************************
void rNodeCopyByVDim(const int elements,
                      const int numDofs,
                      const int ndofs,
                      const int dims,
                      const int* eMap,
                      const double* Sx,
                      double* nodes){
   forall(e,elements, {
         for (int dof = 0; dof < numDofs; ++dof)
         {
            const int lid = dof+numDofs*e;
            const int gid = eMap[lid];
            for (int v = 0; v < dims; ++v)
            {
               const int moffset = v+dims*lid;
               const int voffset = gid+v*ndofs;
               nodes[moffset] = Sx[voffset];
            }
         }
      });
}
