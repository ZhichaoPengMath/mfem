// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.googlecode.com.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#ifndef MFEM_IDGENERATOR
#define MFEM_IDGENERATOR

#include "array.hpp"


/** Helper class to generate unique IDs. When IDs are no longer needed, they
 *  can be returned to the class ('Reuse') and they will be returned next time
 *  'Get' is called.
 */
class IdGenerator
{
public:
   IdGenerator(int first_id = 0) : next(first_id) {}

   /// Generate a unique ID.
   int Get()
   {
      if (reusable.Size())
      {
         int id = reusable.Last();
         reusable.DeleteLast();
         return id;
      }
      return next++;
   }

   /// Return an ID previously generated by 'Get'.
   void Reuse(int id)
   {
      reusable.Append(id);
   }

private:
   int next;
   Array<int> reusable;
};


#endif
